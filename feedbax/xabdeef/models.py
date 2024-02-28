"""Pre-built models.

:copyright: Copyright 2023-2024 by Matt Laporte.
:license: Apache 2.0. See LICENSE for details.
"""

from collections.abc import Callable
from functools import partial
import logging
from typing import Optional

import equinox as eqx
import jax
import jax.random as jr
from jaxtyping import PRNGKeyArray

from feedbax import get_ensemble
from feedbax.bodies import SimpleFeedback
from feedbax.mechanics.plant import DirectForceInput
from feedbax.misc import identity_func
from feedbax._model import AbstractModel
from feedbax.iterate import Iterator
from feedbax.mechanics import Mechanics
from feedbax.mechanics.skeleton.pointmass import PointMass
from feedbax.mechanics.muscle import ActivationFilter
from feedbax.nn import SimpleStagedNetwork
from feedbax.task import AbstractTask


logger = logging.getLogger(__name__)


def point_mass_nn(
    task: AbstractTask,
    n_steps: int = 100,
    dt: float = 0.05,
    mass: float = 1.0,
    encoding_size: Optional[int] = None,
    hidden_size: int = 50,
    hidden_type: eqx.Module = eqx.nn.GRUCell,
    out_nonlinearity: Callable = identity_func,
    feedback_delay_steps: int = 0,
    feedback_noise_std: float = 0.0,
    motor_noise_std: float = 0.0,  # TODO
    *,
    key: PRNGKeyArray,
):
    """A point mass controlled by forces output directly by a neural network.

    Arguments:
        task: The task the neural network will be trained to perform by
            controlling the point mass.
        n_steps: The number of time steps in each trial.
        dt: The duration of each time step.
        mass: The mass of the point mass.
        encoding_size: The size of the neural network's encoding layer.
            If `None`, no encoding layer is used.
        hidden_size: The number of units in the network's hidden layer.
        hidden_type: The network type of the hidden layer.
        out_nonlinearity: The nonlinearity to apply to the network output.
        feedback_delay_steps: The number of time steps to delay sensory feedback
            provided to the neural network about the point mass.
        feedback_noise_std: The standard deviation of Gaussian noise added to the
            sensory feedback.
        motor_noise_std: The standard deviation of Gaussian noise added to the
            forces generated by the neural network.
        key: The random key to use for initializing the model.
    """
    key1, key2 = jr.split(key)

    system = PointMass(mass=mass)
    mechanics = Mechanics(DirectForceInput(system), dt)

    feedback_spec = dict(
        where=lambda state: (
            state.plant.skeleton.pos,
            state.plant.skeleton.vel,
        ),
        delay=feedback_delay_steps,
        noise_std=feedback_noise_std,
    )

    # automatically determine network input size
    input_size = SimpleFeedback.get_nn_input_size(
        task, mechanics, feedback_spec=feedback_spec
    )

    net = SimpleStagedNetwork(
        input_size,
        hidden_size,
        out_size=system.input_size,
        encoding_size=encoding_size,
        hidden_type=hidden_type,
        out_nonlinearity=out_nonlinearity,
        key=key1,
    )
    body = SimpleFeedback(net, mechanics, feedback_spec=feedback_spec, key=key2)

    model = Iterator(body, n_steps)

    return model
