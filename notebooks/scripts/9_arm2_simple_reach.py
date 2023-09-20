# ---
# jupyter:
#   jupytext:
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
import sys
from typing import Optional

import diffrax as dfx
import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp 
import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np
import optax 
import tqdm
from tqdm import tqdm

from feedbax.mechanics.arm import (
    TwoLink, 
    nlink_angular_to_cartesian, 
    twolink_effector_pos_to_angles
)
from feedbax.mechanics.system import System
from feedbax.networks import SimpleMultiLayerNet, RNN
from feedbax.plot import (
    plot_loglog_losses, 
    plot_2D_joint_positions,
    plot_states_forces_2d,
)
from feedbax.task import centreout_endpoints, uniform_endpoints
from feedbax.utils import tree_get_idx, tree_set_idx

# %% [markdown]
# Simple feedback model with a single-layer RNN controlling a two-link arm to reach from a starting position to a target position. 

# %%
DEBUG = False
jax.config.update("jax_debug_nans", True)

N_DIM = 2


# %%
class Mechanics(eqx.Module):
    system: System 
    dt: float = eqx.field(static=True)
    term: dfx.AbstractTerm = eqx.field(static=True)
    solver: Optional[dfx.AbstractSolver] 
    
    def __init__(self, system, dt, solver=None):
        self.system = system
        self.term = dfx.ODETerm(self.system.vector_field)
        if solver is None:
            self.solver = dfx.Tsit5()
        else:
            self.solver = solver
        self.dt = dt        
    
    def __call__(self, state, args):
        inputs, solver_state = args 
        
        # using (0, dt) for (tprev, tnext) seems fine if there's no t dependency in the system
        state, _, _, solver_state, _ = self.solver.step(
            self.term, 0, self.dt, state, inputs, solver_state, made_jump=False
        )
        # #! I don't even return solver state, so apparently it's not important
        return state
    
    def init_solver_state(self, state):
        args = inputs_empty = jnp.zeros((self.system.control_size,))
        return self.solver.init(self.term, 0, self.dt, state, args)
    

class SimpleFeedback(eqx.Module):
    """Simple feedback loop with a single RNN and single mechanical system."""
    net: eqx.Module  
    mechanics: Mechanics 
    delay: int = eqx.field(static=True)
    
    def __init__(self, net, mechanics, delay=0):
        self.net = net
        self.mechanics = mechanics
        self.delay = delay + 1  # indexing: delay=0 -> storage len=1, idx=-1
    
    def __call__(self, state, args):
        mechanics_state, ee_state, _, hidden, solver_state = state
        inputs, feedback_state = args
        
        # mechanics state feedback plus task inputs (e.g. target state)
        control, hidden = self.net((inputs, feedback_state), hidden)
        
        net_force = control  # TODO: noise, perturbations
        
        mechanics_state = self.mechanics(mechanics_state, (net_force, solver_state))
        
        ee_state = tuple(arr[:, -1] for arr in nlink_angular_to_cartesian(
            self.mechanics.system, mechanics_state[0], mechanics_state[1]
        ))
        
        return mechanics_state, ee_state, control, hidden, solver_state
    
    def init_state(self, mechanics_state): 
        
        # #! how to avoid this here?
        ee_state = tuple(arr[:, -1] for arr in nlink_angular_to_cartesian(
            self.mechanics.system, mechanics_state[0], mechanics_state[1]
        ))
        
        return (
            mechanics_state, 
            ee_state, 
            jnp.zeros((self.net.out_size,)),
            jnp.zeros((self.net.hidden_size,)),
            self.mechanics.init_solver_state(mechanics_state),   
        )
    
    
class Recursion(eqx.Module):
    """"""
    step: eqx.Module 
    n_steps: int = eqx.field(static=True)
    
    def __init__(self, step, n_steps):
        self.step = step
        self.n_steps = n_steps        
        
    def _body_func(self, i, x):
        states, args = x 
        # this seems to work, but I'm worried it will break on non-array leaves later
        state = tree_get_idx(states, i)
        
        # #! this ultimately shouldn't be here, but costs less memory than a `SimpleFeedback`-based storage hack:
        # #! could put the concatenate inside of `Network`? & pass any pytree of inputs
        # states[:2] includes both the angular and cartesian state
        feedback = tree_get_idx(states[:2], i - self.step.delay)
        args = (args[0], feedback)  
        
        state = self.step(state, args)
        
        states = tree_set_idx(states, state, i + 1)
        return states, args
    
    def __call__(self, state, args):
        init_state = self.step.init_state(state)
        
        # #! part of the feedback hack
        args = (args, jax.tree_map(jnp.zeros_like, (state, state)))
        
        states = self._init_zero_arrays(init_state, args)
        states = tree_set_idx(states, init_state, 0)
        
        if DEBUG: #! jax.debug doesn't work inside of lax loops  
            for i in range(self.n_steps):
                states, args = self._body_func(i, (states, args))
                
            return states, args   
                 
        return lax.fori_loop(
            0, 
            self.n_steps, 
            self._body_func,
            (states, args),
        )
    
    def _init_zero_arrays(self, state, args):
        return jax.tree_util.tree_map(
            lambda x: jnp.zeros((self.n_steps, *x.shape), dtype=x.dtype),
            eqx.filter_eval_shape(self.step, state, args)
        )


# %%
def get_model(dt, n_hidden, n_steps, key, feedback_delay=0):
    
    system = TwoLink()  # torque control
    mechanics = Mechanics(system, dt)
    n_input = system.state_size * 2 + N_DIM * 2  # angular pos & vel of joints & cartesian EE + target state 
    net = RNN(n_input, system.control_size, n_hidden, key=key)
    body = SimpleFeedback(net, mechanics, delay=feedback_delay)

    return Recursion(body, n_steps)


# %%
LOSS_TERMS = ('position', 'final_velocity', 'control', 'hidden')

def loss_fn(
    diff_model, 
    static_model, 
    init_state, 
    target_state, 
    weights=jnp.array((1., 1., 1e-5, 1e-5)),
    discount=1.,
):  
    """Quadratic in states, controls, and hidden activities.
    
    Assumes the `target_state` is fixed; i.e. this is not a tracking task.
    
    User can apply a temporal discount broadcastable by elementwise multiplication with `(n_batch, n_steps)`.
    """
    model = eqx.combine(diff_model, static_model)  
    batched_model = jax.vmap(model)

    # dataset gives init state in terms of effector position, but we need joint angles
    init_joints_pos = jax.vmap(twolink_effector_pos_to_angles, in_axes=(None, 0))(
        model.step.mechanics.system, init_state
    )
    # #! assumes zero initial velocity; TODO convert initial velocity also
    init_joints_state = (init_joints_pos, jnp.zeros_like(init_joints_pos))

    (joints_states, ee_states, controls, activities, _), _ = batched_model(
        init_joints_state, target_state
    )
    
    states = ee_states  # operational space loss
  
    # sum over xyz, apply temporal discount, sum over time
    position_loss = jnp.sum(discount * jnp.sum((states[0] - target_state[:, None, :2]) ** 2, axis=-1), axis=-1)
    
    loss_terms = dict(
        #final_position=jnp.sum((states[..., -1, :2] - target_state[..., :2]) ** 2, axis=-1).squeeze(),  # sum over xyz
        position=position_loss,  
        final_velocity=jnp.sum((states[1][:, -1] - target_state[..., 2:]) ** 2, axis=-1).squeeze(),  # over xyz
        control=jnp.sum(controls ** 2, axis=(-1, -2)),  # over control variables and time
        hidden=jnp.sum(activities ** 2, axis=(-1, -2)),  # over network units and time
    )
    
    loss_terms = weights * jnp.mean(jnp.stack(list(loss_terms.values()), axis=1), axis=0)  # mean over batch
    
    loss = jnp.sum(loss_terms)  # sum over terms
    return loss, loss_terms


# %%
def train(
    n_steps=100,
    dt=0.1,
    feedback_delay_steps=0,
    workspace = jnp.array([[-0.15, 0.15], [0.20, 0.50]]),
    batch_size=100,
    n_batches=50,
    epochs=1,
    learning_rate=3e-4,
    hidden_size=20,
    seed=5566,
):
    key = jrandom.PRNGKey(seed)

    def get_batch(batch_size, key):
        """Segment endpoints uniformly distributed in a rectangular workspace."""
        return uniform_endpoints(key, batch_size, N_DIM, workspace)
    
    model = get_model(dt, hidden_size, n_steps, key, 
                      feedback_delay=feedback_delay_steps)
    
    # only train the RNN layer (input weights & hidden weights and biases)
    filter_spec = jax.tree_util.tree_map(lambda _: False, model)
    filter_spec = eqx.tree_at(
        lambda tree: (tree.step.net.cell.weight_hh, 
                      tree.step.net.cell.weight_ih, 
                      tree.step.net.cell.bias),
        filter_spec,
        replace=(True, True, True)
    )     
    
    optim = optax.adam(learning_rate)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    position_error_discount = jnp.linspace(1./n_steps, 1., n_steps) ** 6
    
    def train_step(model, init_state, target_state, opt_state):
        diff_model, static_model = eqx.partition(model, filter_spec)
        (loss, loss_terms), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            diff_model, static_model, init_state, target_state, discount=position_error_discount
        )
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, loss_terms, model, opt_state
    
    if not DEBUG:
        train_step = jax.jit(train_step)

    losses = []
    losses_terms = [] 

    for _ in range(epochs):
        for batch in tqdm(range(n_batches)):
            key = jrandom.split(key)[0]
            init_state, target_state = get_batch(batch_size, key)
            
            loss, loss_terms, model, opt_state = train_step(model, init_state, target_state, opt_state)
            
            losses.append(loss)
            losses_terms.append(loss_terms)
            
            if batch % 50 == 0:
                tqdm.write(f"step: {batch}, loss: {loss:.4f}", file=sys.stderr)
    
    losses_terms = jnp.vstack(losses_terms)
    
    return model, losses, losses_terms


# %%
trained, losses, losses_terms = train(
    batch_size=500, 
    dt=0.05, 
    feedback_delay_steps=0,
    n_batches=2600, 
    n_steps=100, 
    hidden_size=100, 
    seed=5566,
    learning_rate=0.01,
)

# %%
plot_loglog_losses(losses, losses_terms, loss_term_labels=LOSS_TERMS)
plt.show()

# %% [markdown]
# Evaluate on a centre-out task

# %%
n_directions = 8
reach_length = 0.1
state_endpoints = centreout_endpoints(
    jnp.array([0., 0.5]), n_directions, 0, reach_length
)
init_joints_pos = jax.vmap(twolink_effector_pos_to_angles, in_axes=(None, 0))(
        trained.step.mechanics.system, state_endpoints[0, :, :2]
)
# #! assumes zero initial velocity; TODO convert initial velocity also
init_states = (init_joints_pos, jnp.zeros_like(init_joints_pos))
target_states = state_endpoints[1]

(states, ee_states, controls, activities, _), _ = jax.vmap(trained)(
    init_states, target_states
)

# %%
test_pos = jax.vmap(nlink_angular_to_cartesian, in_axes=(None, 0, 0))(
    trained.step.mechanics.system, *init_states
)[0]
test_pos[0]

# %%
# plot EE trajectories for all directions
plot_states_forces_2d(ee_states[0], ee_states[1], controls, state_endpoints[..., :2])
plt.show()

# %%
# plot entire arm trajectory for an example direction
# convert all joints to Cartesian since I only saved the EE state
xy_pos = jax.vmap(nlink_angular_to_cartesian, in_axes=(None, 0, 0))(
    trained.step.mechanics.system, states[0].reshape(-1, 2), states[1].reshape(-1, 2)
)[0].reshape(n_directions, -1, 2, 2)

# %%
ax = plot_2D_joint_positions(xy_pos[0], add_root=True)
plt.show()

# %%