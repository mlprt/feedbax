"""

:copyright: Copyright 2024 by Matt L. Laporte.
:license: Apache 2.0. See LICENSE for details.
"""

from abc import abstractmethod
from collections.abc import Callable
from functools import reduce
import logging

import equinox as eqx
from equinox import AbstractVar
from equinox._pretty_print import tree_pp, bracketed
import jax.numpy as jnp
import jax.random as jr
import jax._src.pretty_printer as pp
from jaxtyping import Array, PRNGKeyArray, Shaped

from feedbax.misc import _simple_module_pprint


logger = logging.getLogger(__name__)


class AbstractNoise(eqx.Module):

    @abstractmethod
    def __call__(
        self, key: PRNGKeyArray, x: Shaped[Array, "*dims"]
    ) -> Shaped[Array, "*dims"]:
        ...

    def __add__(self, other):
        return CompositeNoise(terms=(self, other))


class CompositeNoise(AbstractNoise):
    terms: tuple[AbstractNoise, ...]

    def __call__(self, key: PRNGKeyArray, x: Array) -> Array:
        keys = jr.split(key, len(self.terms))
        return reduce(jnp.add, [
            term(key, x)
            for term, key in zip(self.terms, keys)
        ])

    def __getitem__(self, idx):
        return self.terms[idx]

    # def __tree_pp__(self, **kwargs):
    #     _term_sep = pp.concat([pp.brk(), pp.text("+ ")])
    #     return bracketed(
    #         None,
    #         kwargs['indent'],
    #         [pp.join(_term_sep, [tree_pp(term, **kwargs) for term in self.terms])],
    #         '(',
    #         ')',
    #     )


class Normal(AbstractNoise):
    std: float = 1.0
    mean: float = 0.0
    broadcast: bool = False

    def __call__(self, key: PRNGKeyArray, x: Array) -> Array:
        shape = (1,) if self.broadcast else x.shape
        return self.std * jr.normal(key, shape, x.dtype) + self.mean



class Multiplicative(AbstractNoise):
    noise_func: Callable[[PRNGKeyArray, Array], Array]

    def __call__(self, key: PRNGKeyArray, x: Array) -> Array:
        return x * self.noise_func(key, x)

    # def __getattr__(self, name):
    #     return getattr(self.noise_func, name)

    def __tree_pp__(self, **kwargs):
        return _simple_module_pprint("Multiplicative", self.noise_func, **kwargs)


