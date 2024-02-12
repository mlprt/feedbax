"""Muscle models.

TODO:
- Multiplicative noise option
- Static element (SE) model
- Y and l_eff filters. See `_Y_field` and `_l_eff_field` methods.
- None of the simplified models I'm citing implement these.
  
:copyright: Copyright 2023-2024 by Matt L Laporte.
:license: Apache 2.0, see LICENSE for details.
"""


from abc import abstractmethod
from collections.abc import Callable
from functools import cached_property
import logging 
from typing import Optional, Tuple

import equinox as eqx
from equinox import AbstractVar, field
import jax
import jax.random as jr
import jax.numpy as jnp
from jaxtyping import Float, Array
import numpy as np

from feedbax.dynamics import AbstractDynamicalSystem
from feedbax.state import AbstractState, StateBounds


logger = logging.getLogger(__name__)


class ActivationFilter(AbstractDynamicalSystem):
    """First-order filter to model calcium dynamics of muscle activation.
    """
    tau_act: float 
    tau_deact: float 
    
    def __init__(
        self, 
        tau_act: float = 50,  # [ms]
        tau_deact: float = 66
    ): 
        self.tau_act = tau_act
        self.tau_deact = tau_deact
        self.tau_diff  

    def __call__(
        self, 
        t, 
        state, 
        input
    ):        
        activation = state
        tau = self.tau_deact + self.tau_diff * jnp.where(
            input < activation, input, jnp.zeros(1)
        )
        d_activation = (input - activation) / tau
  
        return d_activation
    
    @cached_property
    def tau_diff(self):
        return self.tau_act - self.tau_deact
    
    def control_size(self):
        return 1

    @property
    def bounds(self):
        ...
    
    def init(self):
        ... 


class AbstractMuscleState(AbstractState):
    activation: AbstractVar[Array]
    length: AbstractVar[Array]
    velocity: AbstractVar[Array]
    tension: AbstractVar[Array]


class VirtualMuscleState(AbstractMuscleState):
    activation: Array
    length: Array 
    velocity: Array
    tension: Array


# TODO: maybe `AbstractFLVFunction` and `AbstractActivationFunction` can be joined
class AbstractFLVFunction(eqx.Module):
    """Base class for muscle total force (force-length-velocity) functions."""
    
    @abstractmethod
    def __call__(self, input: Array, state: AbstractMuscleState) -> Array:
        ...


class AbstractActivationFunction(eqx.Module):
    """Base class for muscle input -> activation functions.
    
    Note that this is not the same as (say) a first-order filter that
    approximates calcium dynamics, which would be applied to `input` before it
    arrives here.
    """
    
    @abstractmethod
    def __call__(self, input: Array, state: AbstractMuscleState) -> Array:
        ...


class AbstractMuscle(eqx.Module):
    """Abstract muscle model.
    
    TODO:
    - Subclass `AbstractStagedModel`
    - Run activation dynamics before `force_func`
    """
    
    n_muscles: AbstractVar[int]
    activation_func: AbstractVar[AbstractActivationFunction]
    force_func: AbstractVar[AbstractFLVFunction]
    noise_func: AbstractVar[Optional[Callable[[Array, Array, Array], Array]]]
    
    @jax.named_scope("fbx.AbstractMuscle")
    def __call__(self, input, state, *, key: Optional[Array] = None):
        """Calculates the force generated by the muscle.
        
        Note that the muscle "input" is `state.activation`, and not part of 
        `input`, which as an argument is meant for task inputs and not for
        state variables determined elsewhere in the model, as the muscle
        activation is by the controller and the activation dynamics 
        (managed by concrete instances of `AbstractPlant`).
        """
        activation = self.activation_func(input, state)
        force = self.force_func(activation, state)
        if self.noise_func is not None:
            force = force + self.noise_func(input, force, key)
        return eqx.tree_at(
            lambda state: (state.activation, state.tension),
            state,
            (activation, force),
        )               
  
    def change_n_muscles(self, n_muscles) -> "AbstractMuscle":
        return eqx.tree_at(
            lambda muscle_model: muscle_model.n_muscles,
            self,
            n_muscles,
        )    


class VirtualMuscle(AbstractMuscle):
    
    n_muscles: int
    activation_func: AbstractActivationFunction
    force_func: AbstractFLVFunction
    noise_func: Optional[Callable[[Array, Array, Array], Array]] = None
    
    def init(self, *, key: Optional[Array] = None):
        state = VirtualMuscleState(
            activation=jnp.zeros(self.n_muscles),
            length=jnp.ones(self.n_muscles),  
            velocity=jnp.zeros(self.n_muscles),
            tension=None,
        )
        
        return self(state.activation, state, key=key)   
    
    @property
    def bounds(self) -> StateBounds[VirtualMuscleState]:
        
        # TODO: this would be more general if n_f behaviour is generalized
        # if n_f := getattr(self.activation_func, "n_f", None) is not None:
        
        if isinstance(self.activation_func, VirtualMuscleActivationFunction):
            n_f = self.activation_func.n_f
            length_ub = 0.95 * n_f[1] / (n_f[1] - n_f[0])
        else:
            length_ub = None
        
        return StateBounds(
            low=VirtualMuscleState(
                activation=0.,
                length=None,
                velocity=None,
                tension=None,
            ),
            high=VirtualMuscleState(
                activation=None,
                length=length_ub,
                velocity=None,
                tension=None,
            ),
        )


class AbstractForceFunction(eqx.Module):
    """Base class for muscle force-length, force-velocity, and passive force functions.
    
    TODO:
    - Maybe this doesn't need to be distinct from `AbstractFLVFunction`
    """
    
    @abstractmethod
    def __call__(self, length: Array, velocity: Array) -> Array:
        ...


class VirtualMuscleForceLength(AbstractForceFunction):
    """Force-length function from Brown et al. 1999."""
    beta: float 
    omega: float
    rho: float
    
    def __call__(self, length: Array, velocity: Array) -> Array:
        return jnp.exp(-jnp.abs(
            (length ** self.beta - 1) / self.omega
        ) ** self.rho)


class AbstractVirtualMuscleShortenFactor(eqx.Module):
    c_v: Tuple[float, float] = field(converter=tuple)
    
    @abstractmethod
    def __call__(self, length: Array) -> Array:
        ...
        

class VirtualMuscleShortenFactor(AbstractVirtualMuscleShortenFactor):
    """Shortening factor from Brown et al. 1999."""
    
    def __call__(self, length: Array) -> Array:
        return self.c_v[0] + self.c_v[1] * length
    

class HillShortenFactor(AbstractVirtualMuscleShortenFactor):
    """Hill-type approximation of the shortening factor."""

    def __call__(self, length: Array) -> Array:
        return self.c_v[0] + self.c_v[1]
    

class VirtualMuscleForceVelocity(AbstractForceFunction):
    """Force-velocity function from Brown et al. 1999."""
    a_v: Tuple[float, float, float]
    b_v: float
    v_max: float
    shorten_denom_factor_func: AbstractVirtualMuscleShortenFactor
    
    def __call__(self, length: Array, velocity: Array) -> Array:
        f_lengthen = (self.b_v - velocity * (
            self.a_v[0] 
            + self.a_v[1] * length 
            + self.a_v[2] * length ** 2
        )) / (self.b_v + velocity)
        
        f_shorten = (self.v_max - velocity) / (
            self.v_max + velocity * self.shorten_denom_factor_func(length)
        ) 
        
        lengthen_idxs = velocity > 0
        
        return lengthen_idxs * f_lengthen + ~lengthen_idxs * f_shorten
    
    
class VirtualMuscleForcePassive1(AbstractForceFunction):
    """Passive force function PE1 from Brown et al. 1999."""
    c1: float
    k1: float
    l_r1: float
    
    def __call__(self, length: Array, velocity: Array) -> Array:
        return self.c1 * self.k1 * jnp.log(1 + jnp.exp(
            (length - self.l_r1) / self.k1
        ))


class VirtualMuscleForcePassive2(AbstractForceFunction):
    """Passive force function PE2 from Brown et al. 1999."""
    c2: float
    k2: float
    l_r2: float
    
    def __call__(self, length: Array, velocity: Array) -> Array:
        # TODO: optional Hill approx without l dep
        return self.c2 * jnp.exp(self.k2 * (length - self.l_r2))
        

class VirtualMuscleActivationFunction(AbstractActivationFunction):
    """Activation for the Virtual Muscle Model (Brown et al. 1999)."""
    a_f: float
    n_f: Tuple[float, float]
    
    def __call__(self, input: Array, state: AbstractMuscleState) -> Array: 
        """Model of the relationship between stimulus frequency `a` and muscle activation.
        
        The notation may be confusing. The input to this function is sometimes called the 
        "activation", hence the name `a`. But this model is based on a muscle being activated
        by stimulation at some frequency by an electrode.
        """
        l_eff = state.length # TODO: l_eff filter option (see method _l_eff_field)
        n_f = self.n_f[0] + self.n_f[1] * (1 / l_eff - 1)  
        Y = 1  # TODO: Y filter option (see method _Y_field)
        A_f = 1 - jnp.exp(-((input * Y) / (self.a_f * n_f)) ** n_f)
        return A_f


class VirtualMuscleFLVFunction(AbstractFLVFunction):
    
    force_length: AbstractForceFunction 
    force_velocity: AbstractForceFunction 
    force_passive_1: AbstractForceFunction 
    force_passive_2: AbstractForceFunction 
    
    def __call__(self, input, state: VirtualMuscleState) -> Array:
        force_l = self.force_length(state.length, state.velocity)
        force_v = self.force_velocity(state.length, state.velocity)
        force_pe1 = self.force_passive_1(state.length, state.velocity)
        force_pe2 = self.force_passive_2(state.length, state.velocity)
        # assumes 100% fibre recruitment, linear factor R=1:
        force = input * (force_l * force_v + force_pe2) + force_pe1  
        
        return force


BROWN_FAST_TWITCH_VIRTUALMUSCLE_PARAMS = dict(
    force_length=dict(
        beta=1.55,
        omega=0.81,
        rho=2.12,
    ),
    force_velocity=dict(
        a_v=(-1.53, 0., 0.),
        b_v=1.05,
        v_max=-7.39,
    ),
    force_passive_1=dict(
        c1=0.,
        k1=1.,  # changed from 0. to avoid div by zero
        l_r1=0.,
    ),
    force_passive_2=dict(
        c2=-0.02,
        k2=-18.7,
        l_r2=0.79,
    ),
    activation=dict(
        n_f=(2.11, 3.31),
        a_f=0.56,
    ),
    shorten=dict(
        c_v=(-3.21, 4.17),
    ),
    
    #! unused
    static=dict(
        c_t=27.8,
        k_t=0.0047,
        l_rt=0.964,
    ),
    y_filter=dict(
        tau_y=0.,  # n/a
        c_y=0.,
        v_y=0.,  # n/a
    ),
    l_eff_filter=dict(
        tau_l=0.088,
    ),
)


BROWN_SLOW_TWITCH_VIRTUALMUSCLE_PARAMS = dict(
    force_length=dict(
        beta=2.30,
        omega=1.26,
        rho=1.62,
    ),
    force_velocity=dict(
        a_v=(-4.70, 8.41, -5.34),
        b_v=0.18,
        v_max=-4.06,
    ),
    force_passive_1=dict(
        c1=0.,
        k1=1.,  # changed from 0. to avoid div by zero
        l_r1=0.,
    ),
    force_passive_2=dict(
        c2=-0.02,
        k2=-18.7,
        l_r2=0.79,
    ),
    activation=dict(
        n_f=(2.11, 5.),
        a_f=0.56,
    ),
    shorten=dict(
        c_v=(5.88, 0),
    ),
    
    #! unused
    static=dict(
        c_t=27.8,
        k_t=0.0047,
        l_rt=0.964,
    ),
    y_filter=dict(
        tau_y=200,  # [ms]
        c_y=0.35,
        v_y=0.1,
    ),
    l_eff_filter=dict(
        tau_l=0.088,
    ),
)
        

BROWN_SLOWFAST_AVG_VIRTUALMUSCLE_PARAMS = jax.tree_map(
    lambda x, y: (x + y) / 2,
    BROWN_SLOW_TWITCH_VIRTUALMUSCLE_PARAMS,
    BROWN_FAST_TWITCH_VIRTUALMUSCLE_PARAMS,
)


# Parameters for Todorov & Li 2004 and Lillicrap & Scott 2013 are generally
# some combination of the slow and fast twitch parameters from 
# Brown et al. 1999, or averages thereof. This is indicated by comments 
# inside the parameter trees.


TODOROV_LI_VIRTUALMUSCLE_PARAMS = dict(
    force_length=dict(
        beta=1.93,  # slow/fast avg
        omega=1.03,  # slow/fast avg is 1.035
        rho=1.87,  # slow/fast avg
    ),
    force_velocity=dict(
        a_v=(-3.12, 4.21, -2.67),  # slow/fast avg
        b_v=0.62,  # slow/fast avg
        v_max=-5.72,  # slow/fast avg is -5.725
    ),
    force_passive_2=dict(  # identical for slow/fast
        c2=-0.02,  
        k2=-18.7,  
        l_r2=0.79,
    ),
    shorten=dict(
        c_v=(1.38, 2.09),  # slow/fast avg is (1.335, 2.085)
    ),
    activation=dict(
        n_f=(2.11, 4.16),  # slow/fast avg (2.11, 4.155),
        a_f=0.56,
    ),
    
    #! unused 
    force_passive_1=dict(
        c1=0.,
        k1=1., 
        l_r1=0.,
    ),
)


LILLICRAP_SCOTT_VIRTUALMUSCLE_PARAMS = dict(
    force_length=dict(
        beta = 1.55,  # fast
        omega = 0.81,  # fast 
        rho = 1.0,  # identity exponent; not specified as such
    ),
    force_velocity=dict(
        a_v=(-3.12, 4.21, -2.67),  # slow/fast avg
        b_v=0.62,  # slow/fast avg
        v_max=-7.39,  # fast
    ),
    shorten=dict(
        c_v=(-3.21, 4.17),  # fast
    ),
    
    #! unused
    force_passive_1=dict(
        c1=0.,
        k1=1.,
        l_r1=0.,
    ),
    force_passive_2=dict(
        c2=0.,
        k2=0.,
        l_r2=0.,
    ),
    activation=dict(
        n_f=0.,
        a_f=0.,
    ),
)
    

def brown_1999_virtualmuscle(
    n_muscles: int = 1, 
    noise_func=None,
    params=BROWN_SLOWFAST_AVG_VIRTUALMUSCLE_PARAMS,
):
    return VirtualMuscle(
        n_muscles,
        activation_func=VirtualMuscleActivationFunction(**params["activation"]),
        force_func=VirtualMuscleFLVFunction(
            force_length=VirtualMuscleForceLength(**params["force_length"]),
            force_velocity=VirtualMuscleForceVelocity(
                **params["force_velocity"],
                shorten_denom_factor_func=VirtualMuscleShortenFactor(**params["shorten"]),
            ),
            force_passive_1=VirtualMuscleForcePassive1(**params["force_passive_1"]),
            # force_passive_1=lambda length, velocity: 0,
            force_passive_2=VirtualMuscleForcePassive2(**params["force_passive_2"]),
        ),
        noise_func=noise_func,
    )
    
    
def todorov_li_2004_virtualmuscle(
    n_muscles: int = 1, 
    noise_func=None,
    params=TODOROV_LI_VIRTUALMUSCLE_PARAMS,
):
    """Muscle model from Todorov & Li 2004.
    
    Simplification of the Brown et al. 1999 Virtual Muscle Model:
    
    1. Omits the first passive element, PE1.
    2. Uses averages of the fast and slow twitch parameters from Brown 1999.
    
    """
    return VirtualMuscle(
        n_muscles,
        activation_func=VirtualMuscleActivationFunction(**params["activation"]),
        force_func=VirtualMuscleFLVFunction(
            force_length=VirtualMuscleForceLength(**params["force_length"]),
            force_velocity=VirtualMuscleForceVelocity(
                **params["force_velocity"],
                shorten_denom_factor_func=VirtualMuscleShortenFactor(**params["shorten"]),
            ),
            force_passive_1=lambda length, velocity: 0,
            force_passive_2=VirtualMuscleForcePassive2(**params["force_passive_2"]),
        ),
        noise_func=noise_func,
    )


class LillicrapScottForceLength(AbstractForceFunction):
    """Force-length function from Lillicrap & Scott 2013 supplement.
    
    Possibly incorrect. The sign inside the exponential is reversed compared
    to that in Brown et al. 1999 and Todorov & Li 2004.
    """
    beta: float 
    omega: float
    
    def __call__(self, length: Array, velocity: Array) -> Array:
        return jnp.exp(jnp.abs(
            (length ** self.beta - 1) / self.omega
        ))


def lillicrap_scott_2013_virtualmuscle(
    n_muscles: int = 1,
    noise_func=None, 
    params=LILLICRAP_SCOTT_VIRTUALMUSCLE_PARAMS,
):
    """Muscle model from Lillicrap & Scott 2013.
    
    Simplification of the Brown et al. 1999 Virtual Muscle Model:
    
    1. Uses the Hill approximation (removes denominator `l` dependency) 
       for FV shortening.
    2. Omits both passive elements, PE1 and PE2.
    3. Uses the activation directly: `A_f = a`, thus also omits any 
       `Y` and `l_eff` filters.
    4. Uses a mixture of fast twitch and averaged slow/fast twitch 
       parameters from Brown 1999; see `LILLICRAP_SCOTT_VIRTUALMUSCLE_PARAMS`.
    """
    
    return VirtualMuscle(
        n_muscles,
        activation_func=lambda input, length: input,
        force_func=VirtualMuscleFLVFunction(
            # force_length=LillicrapScottForceLength(params.force_length),
            force_length=VirtualMuscleForceLength(**params["force_length"]),
            force_velocity=VirtualMuscleForceVelocity(
                **params["force_velocity"],
                shorten_denom_factor=HillShortenFactor(**params["shorten"]),
            ),
            force_passive_1=lambda length, velocity: 0,
            force_passive_2=lambda length, velocity: 0,
        ),
        noise_func=noise_func,
    )
    

#     def _Y_field(self, t, y, args):
#         #! currently unused
#         Y = y
#         v = args 
#         c_Y, tau_Y, v_Y = self.c_y, self.tau_y, self.v_y
#         d_Y = 1 - Y - c_Y * (1 - jnp.exp(-jnp.abs(v) / v_Y)) / tau_Y
#         return d_Y
    
#     def _l_eff_field(self, t, y, args):
#         #! currently unused
#         # TODO: to do this, need to track A_f from last step...
#         l_eff = y
#         l, A_f = args 
#         tau_l = self.tau_l
#         d_l_eff = (l - l_eff) ** 3 / (tau_l * (1 - A_f))
#         return d_l_eff    



    

        
