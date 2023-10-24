"""

:copyright: Copyright 2023 by Matt L Laporte.
:license: Apache 2.0, see LICENSE for details.
"""

import logging 
from typing import Any, Protocol, TypeVar

import equinox as eqx
from jaxtyping import Array, Float, PyTree


logger = logging.getLogger(__name__)


T = TypeVar("T")


class CartesianState2D(eqx.Module):
    """Cartesian state of a system."""
    pos: Float[Array, "... 2"]
    vel: Float[Array, "... 2"]


# TODO maybe this should be `AbstractSystem(eqx.Module)` instead
class System(Protocol):
    def vector_field(
        self, 
        t: float, 
        y: PyTree[T], 
        args: PyTree,  # controls
) -> PyTree[T]:
        """Vector field of the system."""
        ...

    @property
    def control_size(self) -> int:
        """Number of control inputs."""
        ...
    
    def init(self) -> PyTree[T]:
        """Initial state of the system."""
        ...