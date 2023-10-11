"""

:copyright: Copyright 2023 by Matt L. Laporte.
:license: Apache 2.0. See LICENSE for details.
"""

import logging
import os

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PyTree
from tqdm.auto import tqdm

from feedbax.utils import tree_get_idx, tree_set_idx

logger = logging.getLogger(__name__)


class Recursion(eqx.Module):
    """"""
    step: eqx.Module 
    n_steps: int = eqx.field(static=True)
    
    def __init__(self, step, n_steps):
        self.step = step
        self.n_steps = n_steps        
        
    def _body_func(self, i, x):
        input, states, key = x
        
        key1, key2 = jrandom.split(key)
        
        # #! this ultimately shouldn't be here, but costs less memory than a `SimpleFeedback`-based storage hack:
        feedback = (
            tree_get_idx(states.mechanics.system[:2], i - self.step.delay),  # omit muscle activation
            tree_get_idx(states.ee, i - self.step.delay),  # ee state
        )
        args = feedback
        
        state = tree_get_idx(states, i)
        state = self.step(input, state, args, key1)
        states = tree_set_idx(states, state, i + 1)
        
        return input, states, key2
    
    def __call__(self, input, system_state, key):
        key1, key2, key3 = jrandom.split(key, 3)
        
        state = self.step.init(system_state) #! maybe this should be outside
        
        #! `args` is vestigial. part of the feedback hack
        args = jax.tree_map(jnp.zeros_like, (state.mechanics.system[:2], state.ee))
        
        states = self.init(input, state, args, key2)
        
        if os.environ.get('FEEDBAX_DEBUG', False) == "True": 
            # this tqdm doesn't show except on an exception, which might be useful
            for i in tqdm(range(self.n_steps),
                          desc="steps"):
                input, states, key3 = self._body_func(i, (input, states, key3))
                
            return states
                 
        _, states, _ = lax.fori_loop(
            0, 
            self.n_steps, 
            self._body_func,
            (input, states, key3),
        )
        
        return states
    
    def init(self, input, state, args, key):
        # 1. generate empty trajectories of states 
        outputs = eqx.filter_eval_shape(self.step, input, state, args, key)
        # `eqx.is_array_like` is False for jax.ShapeDtypeSty
        scalars, array_structs = eqx.partition(outputs, eqx.is_array_like)
        asarrays = eqx.combine(jax.tree_map(jnp.asarray, scalars), array_structs)
        states = jax.tree_map(
            lambda x: jnp.zeros((self.n_steps, *x.shape), dtype=x.dtype),
            asarrays,
        )
        # 2. initialize the first state
        states = tree_set_idx(states, state, 0)
        return states