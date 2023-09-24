"""Utility functions.

:copyright: Copyright 2023 by Matt L Laporte.
:license: Apache 2.0, see LICENSE for details.
"""

from itertools import zip_longest, chain
import logging 
import math
from pathlib import Path
from shutil import rmtree
from time import perf_counter
from typing import Union 

import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, PyTree


logger = logging.getLogger(__name__)


"""The signs of the i-th derivatives of cos and sin.

TODO: infinite cycle
"""
SINCOS_GRAD_SIGNS = jnp.array([(1, 1), (1, -1), (-1, -1), (-1, 1)])


class catchtime:
    """Context manager for timing code blocks.
    
    From https://stackoverflow.com/a/69156219
    """
    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start
        self.readout = f'Time: {self.time:.3f} seconds'
        print(self.readout)


def delete_contents(path: Union[str, Path]):
    """Delete all subdirectories and files of `path`."""
    for p in Path(path).iterdir():
        if p.is_dir():
            rmtree(p)
        elif p.is_file():
            p.unlink()
    

def exp_taylor(x: float, n: int):
    """First `n` terms of the Taylor series for `exp` at the origin."""
    return [(x ** i) / math.factorial(i) for i in range(n)]


def interleave_unequal(*args):
    """Interleave sequences of different lengths."""
    return (x for x in chain.from_iterable(zip_longest(*args)) 
            if x is not None)
    

def internal_grid_points(bounds, n=2):
    """Generate an even grid of points inside the given bounds.
    
    e.g. if bounds=((0, 9), (0, 9)) and n=2 the return value will be
    Array([[3., 3.], [6., 3.], [3., 6.], [6., 6.]]).
    """
    ticks = jax.vmap(lambda b: jnp.linspace(*b, n + 2)[1:-1])(bounds)
    points = jnp.vstack(jax.tree_map(jnp.ravel, jnp.meshgrid(*ticks))).T
    return points


def normalize(
    tree: PyTree,
    min: float = 0, 
    max: float = 1, 
    axis: int = 0,
):
    """Normalize each input array to [min, max] along the given axis.
    
    Defaults to normalizing columns
    """
    def arr_norm(arr):
        arr_min = jnp.min(arr, axis=axis)
        arr_max = jnp.max(arr, axis=axis)
        return min + (max - min) * (arr - arr_min) / (arr_max - arr_min)
    
    return jax.tree_map(arr_norm, tree)


def tree_get_idx(tree, idx: int):
    """Retrieve the `idx`-th element of each leaf of `tree`."""
    return jax.tree_map(lambda xs: xs[idx], tree)


def tree_set_idx(tree, vals, idx: int):
    """Update the `idx`-th element of each leaf of `tree`.
    
    `vals` should be a pytree with the same structure as `tree`,
    except that each leaf should be missing the first dimension of 
    the corresponding leaf in `tree`.
    """
    return jax.tree_util.tree_map(lambda xs, x: xs.at[idx].set(x), tree, vals)


def tree_stack(trees):
    """Stack the leaves of each tree in `trees`.
    
    All trees should have the same structure.
    
    See https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75?permalink_comment_id=4634557#gistcomment-4634557
    """
    return jax.tree_util.tree_map(lambda *v: jnp.stack(v), *trees)


def tree_sum_squares(tree):
    """Sum the sums of squares of the leaves of a PyTree.
    
    Useful for (say) doing weight decay on model parameters.
    """
    return jax.tree_util.tree_reduce(
        lambda x, y: x + y, 
        jax.tree_map(lambda x: jnp.sum(x ** 2), tree)
    )

def corners_2d(bounds: Float[Array, "ndim=2 2"]):    
    """Generate the corners of a rectangle from its bounds."""
    xy = jax.tree_map(jnp.ravel, jnp.meshgrid(*bounds))
    return jnp.vstack(xy)


def twolink_workspace_test(workspace: Float[Array, "xy=2 bounds=2"], twolink):
    """Tests whether a rectangular workspace is reachable by the two-link arm."""
    r = sum(twolink.l)
    lengths = jnp.sum(corners_2d(workspace) ** 2, axis=0) ** 0.5
    if jnp.any(lengths > r):
        return False 
    return True

# 
# jax.debug.print(''.join([f"{s.shape}\t{p}\n" 
#                             for p, s in jax.tree_util.tree_leaves_with_path(state)]))


