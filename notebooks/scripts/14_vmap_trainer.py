# ---
# jupyter:
#   jupytext:
#     formats: ipynb,scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: fx
#     language: python
#     name: python3
# ---

# %%
LOG_LEVEL = "INFO"
NB_PREFIX = "nb10"
DEBUG = False
ENABLE_X64 = False
N_DIM = 2  # TODO: not here


# %%
# %load_ext autoreload
# %autoreload 2

# %%
import logging
import os
from pathlib import Path
import sys

from IPython import get_ipython

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

# redirect stderr (e.g. warnings) to file
stderr_log = sys.stderr = open(f'log/stderr_{NB_PREFIX}.log', 'w')
get_ipython().log.handlers[0].stream = stderr_log 
get_ipython().log.setLevel(LOG_LEVEL)

# %%
import equinox as eqx
import jax
import jax.numpy as jnp 
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax 
import pandas as pd
import seaborn as sns

from feedbax.channel import ChannelState
from feedbax.context import SimpleFeedback, SimpleFeedbackState
import feedbax.loss as fbl
from feedbax.mechanics import Mechanics 
from feedbax.mechanics.muscle import (
    ActivationFilter,
    TodorovLiVirtualMuscle, 
) 
from feedbax.mechanics.muscled_arm import TwoLinkMuscled 
from feedbax.networks import RNN
from feedbax.recursion import Recursion
from feedbax.task import RandomReaches
from feedbax.trainer import TaskTrainer, save, load

from feedbax.plot import (
    plot_mean_losses,
    plot_2D_joint_positions,
    plot_pos_vel_force_2D,
    plot_activity_heatmap,
)

from feedbax.utils import get_model_ensemble, tree_get_idx

# %%
os.environ["FEEDBAX_DEBUG"] = str(DEBUG)
logging.getLogger("jax").setLevel(logging.INFO)

jax.config.update("jax_debug_nans", DEBUG)
jax.config.update("jax_enable_x64", ENABLE_X64)

# not sure if this will work or if I need to use the env variable version
#jax.config.update("jax_traceback_filtering", DEBUG)  

plt.style.use('dark_background')

# %%
# paths

# training checkpoints
chkpt_dir = Path("/tmp/feedbax-checkpoints")
chkpt_dir.mkdir(exist_ok=True)

# tensorboard
tb_logdir = Path("runs")

model_dir = Path("../models/")


# %% [markdown]
# Define the model.

# %%
def get_model(
        key=None,
        dt=0.05, 
        n_hidden=50, 
        n_steps=50, 
        feedback_delay=0, 
        tau=0.01, 
        out_nonlinearity=jax.nn.sigmoid,
):
    if key is None:
        # in case we just want a skeleton model, e.g. for deserializing
        key = jr.PRNGKey(0)
    key1, key2 = jr.split(key)
    
    system = TwoLinkMuscled(
        muscle_model=TodorovLiVirtualMuscle(), 
        activator=ActivationFilter(
            tau_act=tau,  
            tau_deact=tau,
        )
    )
    mechanics = Mechanics(system, dt)
    
    feedback_leaves_func = lambda mechanics_state: (
        mechanics_state.system.theta,
        mechanics_state.system.d_theta,
        mechanics_state.effector,         
    )
    
    # joint state feedback + effector state + target state
    n_input = system.twolink.state_size + 2 * N_DIM + 2 * N_DIM
    cell = eqx.nn.GRUCell(n_input, n_hidden, key=key1)
    net = RNN(
        cell, 
        system.control_size, 
        out_nonlinearity=out_nonlinearity,
        persistence=False, 
        key=key2
    )
    body = SimpleFeedback(
        net, 
        mechanics, 
        delay=feedback_delay,  
        feedback_leaves_func=feedback_leaves_func,
    )
    
    states_includes = SimpleFeedbackState(
        mechanics=True, 
        nn_output=True, 
        hidden=True, 
        feedback=ChannelState(output=True, queue=False)
    )

    return Recursion(body, n_steps, states_includes=states_includes)


# %% [markdown]
# Train the model.

# %%
seed = 5567

n_steps = 50
dt = 0.05 
feedback_delay_steps = 0
workspace = ((-0.15, 0.15), 
             (0.20, 0.50))
n_hidden  = 50
out_nonlinearity = jax.nn.sigmoid
learning_rate = 0.05

loss_term_weights = dict(
    effector_position=1.,
    effector_final_velocity=0.1,
    nn_output=1e-4,
    nn_activity=0.,
)

hyperparams = dict(
    seed=seed,
    n_steps=n_steps,
    workspace=workspace,
    loss_term_weights=loss_term_weights,
    dt=dt,
    n_hidden=n_hidden,
    feedback_delay_steps=feedback_delay_steps,
)


# %%
def setup(
    seed, 
    n_steps, 
    workspace,
    loss_term_weights,
    dt, 
    n_hidden,
    feedback_delay_steps,    
):
    
    key = jr.PRNGKey(seed)

    # #! these assume a particular PyTree structure to the states returned by the model
    # #! which is why we simply instantiate them 
    discount = jnp.linspace(1. / n_steps, 1., n_steps) ** 6
    loss_func = fbl.CompositeLoss(
        dict(
            # these assume a particular PyTree structure to the states returned by the model
            # which is why we simply instantiate them 
            effector_position=fbl.EffectorPositionLoss(discount=discount),
            effector_final_velocity=fbl.EffectorFinalVelocityLoss(),
            nn_output=fbl.NetworkOutputLoss(),
            nn_activity=fbl.NetworkActivityLoss(),
        ),
        weights=loss_term_weights,
    )

    task = RandomReaches(
        loss_func=loss_func,
        workspace=workspace, 
        n_steps=n_steps,
        eval_grid_n=2,
        eval_n_directions=8,
        eval_reach_length=0.05,
    )
    
    model = get_model(
        key, 
        dt=dt,
        n_hidden=n_hidden,
        n_steps=n_steps,
        feedback_delay=feedback_delay_steps,
        tau=0.01,
        out_nonlinearity=jax.nn.sigmoid,
    )
    
    return model, task 


# %%
models, task = setup(**hyperparams)

trainer = TaskTrainer(
    optimizer=optax.inject_hyperparams(optax.adam)(
        learning_rate=learning_rate
    ),
    chkpt_dir=chkpt_dir,
    checkpointing=True,
)

# %%
x = jnp.array([1, 3, 10, 50, 100, 300, 500, 600])
y = jnp.array([23.5, 22, 21.5, 15, 8, 3.2, 2, 1.7])

fig, ax = plt.subplots()
ax.scatter(x, (10000/y)/60, edgecolors="white")
ax.set_xscale("log")
#ax.set_yscale("log")
ax.set_xlim(0.5, 1000)
ax.hlines([8], xmin=0.5, xmax=1000, linestyles=':', lw=0.5)
#ax.set_ylim(0, 25)
ax.set_xlabel("Number of replicates")
ax.set_ylabel("Training time (min)")

# %%
batch_size = 500
n_batches = 500
key_train = jr.PRNGKey(seed + 1)

trainable_leaves_func = lambda model: (
    model.step.net.cell.weight_hh, 
    model.step.net.cell.weight_ih, 
    model.step.net.cell.bias
)

model, losses, losses_terms, learning_rates = trainer.train_ensemble(
    task=task, 
    models=models,
    n_replicates=n_replicates,
    n_batches=n_batches, 
    batch_size=batch_size, 
    log_step=10,
    trainable_leaves_func=trainable_leaves_func,
    key=key_train,
)

plt.loglog(losses.T)

# %%
plot_mean_losses(losses, losses_terms)
plt.show()

# %% [markdown]
# Save the trained model to file

# %%
model_path = save(
    (model, task),
    hyperparams, 
    save_dir=model_dir, 
    suffix=NB_PREFIX,
)

# %% [markdown]
# If we didn't just save a model, we can try to load one

# %%
try:
    model_path
    model, task
except NameError:
    model_path = ''
    model, task = load(model_path, setup)

# %% [markdown]
# Evaluate on a centre-out task

# %%
keys_eval = jr.split(jr.PRNGKey(seed + 2), n_replicates)

loss, loss_terms, states = eqx.filter_vmap(task.eval)(
    models, keys_eval
)

# %%
fig, _ = plot_pos_vel_force_2D(
                    states[1][0][0], states[1][1][0], states[2][0], eval_endpoints[..., :2], 
                    cmap='plasma', workspace=workspace
)

# %%
# plot entire arm trajectory for an example direction
# convert all joints to Cartesian since I only saved the EE state
xy_pos = eqx.filter_vmap(nlink_angular_to_cartesian)(
    models.step.mechanics.system.twolink, 
    states[0].reshape(-1, 2), 
    states[1].reshape(-1, 2)
)[0].reshape(states[0].shape[0], -1, 2, 2)

# %%
ax = plot_2D_joint_positions(xy_pos[0], add_root=True)
plt.show()

# %% [markdown]
# ## Debugging stuff

# %% [markdown]
# ### Get jaxpr for the loss function

# %%
key = jr.PRNGKey(5566)
batch_size = 10
init_state, target_state = uniform_endpoints(key, batch_size, N_DIM, workspace)

filter_spec = jax.tree_util.tree_map(lambda _: False, trained)
filter_spec = eqx.tree_at(
    lambda tree: (tree.step.net.cell.weight_hh, 
                    tree.step.net.cell.weight_ih, 
                    tree.step.net.cell.bias),
    filter_spec,
    replace=(True, True, True)
)     

diff_model, static_model = eqx.partition(trained, filter_spec)

grad_fn = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn, has_aux=True))
jaxpr, _, _ = eqx.filter_make_jaxpr(grad_fn)(diff_model, static_model, init_state, target_state)

# %% [markdown]
# The result is absolutely huge. I guess I should try to repeat this for the un-vmapped model.

# %%
jaxpr