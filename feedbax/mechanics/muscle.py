"""Muscle models.

NOTE:
- `eqx.Module` is a frozen dataclass, thus its fields are immutable. This has 
  implications for how we structure the simplifications of `VirtualMuscle`.
  For now, I encode the 

TODO:
- ActivationFilter could be 
    - a module applied in serial to the muscle module 
    - a wrapper around other muscle modules; signature `(l, v, a, u)`
    - composed into the muscle module
- It's kind of annoying to keep so many fields for parameters, but it seems
  like the cleanest way to do it. But maybe there's a way to use the default
  dataclass init, at least.
"""


from functools import cached_property
from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.random as jrandom
import jax.numpy as jnp
from jaxtyping import Float, Array


class VirtualMuscle(eqx.Module):
    """Virtual Muscle Model from Brown et al. 1999.
    
    TODO:
    - Multiplicative noise option
    - Static element (SE) model
    - Y and l_eff filters. See `_Y_field` and `_l_eff_field` methods.
        - None of the simplified models I'm citing implement these.
    """
    beta: float 
    omega: float
    rho: float
    v_max: float
    c_v: Tuple[float, float]
    a_v: Tuple[float, float, float]
    b_v: float
    n_f: Tuple[float, float]
    a_f: float
    c1: Optional[float] = None 
    c2: float
    k1: Optional[float] = None
    k2: float
    l_r1: Optional[float] = None
    l_r2: float
    tau_l: Optional[float] = None 
    c_y: Optional[float] = None
    v_y: Optional[float] = None
    tau_y: Optional[float] = None
    hill_shorten_approx: bool  # use Hill approx for f_shorten 
    noise_std: Optional[float] = None  # add noise to force output
    
    def __init__(
        self, 
        beta, 
        omega, 
        rho, 
        v_max, 
        c_v,
        a_v,
        b_v,
        n_f,
        a_f,
        c1,  
        c2,
        k1,  
        k2,
        l_r1,  
        l_r2,
        tau_l,
        c_y,
        v_y,
        tau_y,
        hill_shorten_approx=False,
        noise_std=None,
    ):
        self.beta = beta
        self.omega = omega
        self.rho = rho
        self.v_max = v_max
        self.c_v = c_v
        self.a_v = a_v
        self.b_v = b_v
        self.n_f = n_f
        self.a_f = a_f
        self.c1 = c1
        self.c2 = c2
        self.k1 = k1
        self.k2 = k2
        self.l_r1 = l_r1
        self.l_r2 = l_r2
        self.tau_l = tau_l
        self.c_y = c_y
        self.v_y = v_y
        self.tau_y = tau_y
        self.hill_shorten_approx = hill_shorten_approx
        self.noise_std = noise_std
        
        # initialize cached properties
        self._add_noise
        self._shorten_denom_factor
        self._cv_sum

    def __call__(self, muscle_length, muscle_velocity, activation, key=None):
        """Calculates the force generated by the muscle."""
        
        force_l = self.force_length(muscle_length)
        force_v = self.force_velocity(muscle_length, muscle_velocity)
        force_pe1, force_pe2 = self.force_passive(muscle_length)
        #! technically "frequency" (of stimulation) is the input, here
        A_f = self.activation_frequency(activation, muscle_length)
        # assumes 100% fibre recruitment, linear factor R=1:
        force = A_f * (force_l * force_v + force_pe2) + force_pe1  
        force = self._add_noise(force, activation, key)
        return force

    def force_length(self, l):
        return jnp.exp(-jnp.abs((l ** self.beta - 1) / self.omega) ** self.rho)
    
    def force_velocity(self, l, v):
        a_v, b_v, v_max = self.a_v, self.b_v, self.v_max
        f_lengthen = (b_v - v * (a_v[0] + a_v[1] * l + a_v[2] * l ** 2)) / (b_v + v)
        f_shorten = (v_max - v) / (v_max + v * self._shorten_denom_factor(l)) 
        lengthen_idxs = v > 0
        return lengthen_idxs * f_lengthen + ~lengthen_idxs * f_shorten

    @cached_property    
    def _shorten_denom_factor(self):
        if self.hill_shorten_approx:
            return self._hill_shorten_denom_factor
        else:
            return self._full_shorten_denom_factor
    
    def _full_shorten_denom_factor(self, l):
        return self.c_v[0] + self.c_v[1] * l 

    def _hill_shorten_denom_factor(self, l):
        return self._cv_sum 
    
    @cached_property
    def _cv_sum(self):
        return self.c_v[0] + self.c_v[1]
    
    def force_passive(self, l):
        return self.force_passive_1(l), self.force_passive_2(l)
    
    def force_passive_1(self, l):
        """Model of passive element PE1."""
        c1, k1, l_r1 = self.c1, self.k1, self.l_r1
        f_pe1 = c1 * k1 * jnp.log(jnp.exp((l - l_r1) / k1) + 1)  
        return f_pe1
    
    def force_passive_2(self, l):
        """Model of passive element PE2."""
        c2, k2, l_r2 = self.c2, self.k2, self.l_r2
        f_pe2 = c2 * jnp.exp(k2 * (l - l_r2))  # TODO: optional Hill approx without l dep
        return f_pe2
    
    def activation_frequency(self, a, l): 
        """Model of the relationship between stimulus frequency `a` and muscle activation.
        
        The notation may be confusing. The input to this function is sometimes called the 
        "activation", hence the name `a`. But this model is based on a muscle being activated
        by stimulation at some frequency by an electrode.
        """
        n_f, a_f = self.n_f, self.a_f
        n_f = n_f[0] + n_f[1] * (1 / l - 1)  # TODO: l_eff filter option (see method _l_eff_field)
        Y = 1  # TODO: Y filter option (see method _Y_field)
        A_f = 1 - jnp.exp(-(a / (a_f * n_f)) ** n_f)
        # A_f = 1 - jnp.exp(-((a * Y) / (a_f * n_f)) ** n_f)
        return A_f
  
    @cached_property
    def _add_noise(self):
        if self.noise_std is None:
            return lambda force, *args: force
        else:
            return self.__add_noise 
    
    def __add_noise(self, force, activation, key):
        noise = jrandom.normal(key, shape=force.shape, dtype=force.dtype)
        return force + self.noise_std * noise

    def _Y_field(self, t, y, args):
        #! currently unused
        Y = y
        v = args 
        c_Y, tau_Y, v_Y = self.c_y, self.tau_y, self.v_y
        d_Y = 1 - Y - c_Y * (1 - jnp.exp(-jnp.abs(v) / v_Y)) / tau_Y
        return d_Y
    
    def _l_eff_field(self, t, y, args):
        #! currently unused
        # TODO: to do this, need to track A_f from last step...
        l_eff = y
        l, A_f = args 
        tau_l = self.tau_l
        d_l_eff = (l - l_eff) ** 3 / (tau_l * (1 - A_f))
        return d_l_eff


class TodorovLiVirtualMuscle(VirtualMuscle):
    """Muscle model from Todorov & Li 2004.
    
    Simplification of the Brown et al. 1999 Virtual Muscle Model, omitting the 
    first passive element (PE1) and the Y and l_eff filters.
    """
    
    def __init__(self):
        super().__init__(**_TODOROV_LI_PARAMS)
    
    def force_passive(self, l):
        # omit f_pe1
        return 0, self.force_passive_2(l)


class LillicrapScottVirtualMuscle(VirtualMuscle): 
    """Muscle model from Lillicrap & Scott 2013.
    
    Simplification of the Brown et al. 1999 Virtual Muscle Model:
    
    1. Uses the Hill approximation (removes denominator `l` dependency) 
       for `f_shorten`.
    2. Omits the passive elements, PE1 and PE2.
    3. Uses the activation directly for the activation frequency: `A_f = a`,
       thus also omits the Y and l_eff filters.
    
    TODO:
    - Figure out if they actually use a modified form of `force_length` with 
      the sign inside the exponential reversed.
    """
    hill_shorten_approx: bool = True  # 1. use Hill approx for f_shorten
    
    def __init__(self):
        super().__init__(**_LILLICRAP_SCOTT_PARAMS)
    
    def force_passive(self, l):
        return 0, 0
    
    def activation_frequency(self, a, l):
        return a 

    #! this is the (wrong?) equation reported in the paper supplement    
    # def force_length(self, l):
    #     return jnp.exp(jnp.abs((l ** self.beta - 1) / self.omega))
        

_TODOROV_LI_PARAMS = dict(
    beta=1.93,  # slow/fast avg
    omega=1.03,  # slow/fast avg is 1.035
    rho=1.87,  # slow/fast avg
    v_max=-5.72,  # slow/fast avg is -5.725
    c_v=(1.38, 2.09),  # slow/fast avg is (1.335, 2.085)
    a_v=(-3.12, 4.21, -2.67),  # slow/fast avg
    b_v=0.62,  # slow/fast avg
    n_f=(2.11, 4.16),  # slow/fast avg (2.11, 4.155)
    a_f=0.56,  
    c2=-0.02,  
    k2=-18.7,  
    l_r2=0.79,
    
    #! unused
    c1=0.,
    k1=0., 
    l_r1=0.,
    tau_l=0.,
    c_y=0.,
    v_y=0.,
    tau_y=0.,  # [ms]    
)


_LILLICRAP_SCOTT_PARAMS = dict(
    beta = 1.55,  # fast
    omega = 0.81,  # fast 
    v_max = -7.39,  # fast 
    c_v = (-3.21, 4.17),  # fast 
    a_v = (-3.12, 4.21, -2.67),  # slow/fast avg
    b_v = 0.62,  # slow/fast avg
    
    rho = 1.0,  # identity exponent; not specified as such
    
    #! unused
    n_f=0.,
    a_f=0.,
    c1=0.,
    c2=0.,
    k1=0.,
    k2=0.,
    l_r1=0.,
    l_r2=0.,
    tau_l=0.,
    c_y=0.,
    v_y=0.,
    tau_y=0.,  # [ms]
)

_BROWN_SLOW_TWITCH_PARAMS = dict(
    beta=2.30,
    omega=1.26,
    rho=1.62,
    v_max=-4.06,
    c_v=(5.88, 0),
    a_v=(-4.70, 8.41, -5.34),
    b_v=0.18,
    n_f=(2.11, 5.),
    a_f=0.56,
    c2=-0.02,
    k2=-18.7,
    l_r2=0.79,
    
    #! unused
    # static element
    c_t=27.8,
    k_t=0.0047,
    l_rt=0.964,
    
    # Y and l_eff filters
    tau_l=0.088,
    c_y=0.35,
    v_y=0.1,
    tau_y=200,  # [ms]
    
    # PE1
    c1=0.,
    k1=0., 
    l_r1=0.,
)

_BROWN_FAST_TWITCH_PARAMS = dict(
    beta=1.55,
    omega=0.81,
    rho=2.12,
    v_max=-7.39,
    c_v=(-3.21, 4.17),
    a_v=(-1.53, 0., 0.),
    b_v=1.05,
    n_f=(2.11, 3.31),
    a_f=0.56,
    c2=-0.02,
    k2=-18.7,
    l_r2=0.79,
    
    #! unused
    # static element
    c_t=27.8,
    k_t=0.0047,
    l_rt=0.964,
    
    # Y and l_eff filters
    tau_l=0.088,
    c_y=0.,
    v_y=0.,  # n/a
    tau_y=0.,  # n/a
    
    # PE1
    c1=0.,
    k1=0., 
    l_r1=0.,
)

_BROWN_SLOWFAST_AVG_PARAMS = jax.tree_map(
    lambda x, y: (x + y) / 2,
    _BROWN_SLOW_TWITCH_PARAMS,
    _BROWN_FAST_TWITCH_PARAMS,
)
    
    
class ActivationFilter(eqx.Module):
    """First-order filter to model calcium dynamics of muscle activation.
    """
    tau_act: float = eqx.field(static=True)
    tau_deact: float = eqx.field(static=True)
    
    def __init__(self, tau_act=50, tau_deact=66): # [ms]
        self.tau_act = tau_act
        self.tau_deact = tau_deact
        self.tau_diff  

    def vector_field(self, t, y, args):
        activation = y 
        u = args 
        
        tau = self.tau_deact + self.tau_diff * jnp.where(
            u < activation, u, jnp.zeros(1)
        )
        d_activation = (u - activation) / tau
        
        return d_activation
    
    @cached_property
    def tau_diff(self):
        return self.tau_act - self.tau_deact