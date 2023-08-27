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
import math

import diffrax as dfx
import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp 
import jax.random as jrandom
from jaxtyping import Float, Array
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import optax 
import seaborn as sns
from tqdm import tqdm

from feedbax.utils import exp_taylor


# %%
def torque(t, y, args):
    return jnp.array([-0.1, 0.1])

class TwoLink:
    l: Float[Array, "2"] = jnp.array((0.30, 0.33))  # [m] lengths of arm segments
    m: Float[Array, "2"] = jnp.array((1.4, 1.0))  # [kg] masses of segments
    I: Float[Array, "2"] = jnp.array((0.025, 0.045))  # [kg m^2] moments of inertia of segments
    s: Float[Array, "2"] = jnp.array((0.11, 0.16))  # [m] distance from joint center to segment COM
    B: Float[Array, "2 2"] = jnp.array(((0.05, 0.025),
                                         (0.025, 0.05))) # [kg m^2 s^-1] joint friction matrix
    inertia_gain: float = 1.    
    
    @property
    def a1(self):
        return self.I[0] + self.I[1] + self.m[1] * self.l[0] ** 2 # + m[1]*s[1]**2 + m[0]*s[0]**2
    
    @property
    def a2(self):
        return self.m[1] * self.l[0] * self.s[1]
    
    @property
    def a3(self):
        return self.I[1]  # + m[1] * s[1] ** 2


def twolink_field(twolink):
    def field(t, y, args):
        theta, dtheta = y 
        
        # centripetal and coriolis forces 
        c_vec = jnp.array((
            -dtheta[1] * (2 * dtheta[0] + dtheta[1]),
            dtheta[0] ** 2
        )) * twolink.a2 * jnp.sin(theta[1])  
        
        cs1 = jnp.cos(theta[1])
        tmp = twolink.a3 + twolink.a2 * cs1
        inertia_mat = jnp.array(((twolink.a1 + 2 * twolink.a2 * cs1, tmp),
                                 (tmp, twolink.a3 * jnp.ones_like(cs1))))
        
        net_torque = torque(t, y, args) - c_vec.T - jnp.matmul(dtheta, twolink.B.T) # - viscosity * state.dtheta
        
        ddtheta = jnp.linalg.inv(inertia_mat) @ net_torque
        
        return dtheta, ddtheta
    
    return field

def solve(y0, dt0, args):
    term = dfx.ODETerm(twolink_field(TwoLink()))
    solver = dfx.Tsit5()
    t0 = 0
    t1 = 1
    saveat = dfx.SaveAt(ts=jnp.linspace(t0, t1, 1000))
    sol = dfx.diffeqsolve(term, solver, t0, t1, dt0, y0, args=args, saveat=saveat)
    return sol

y0 = (jnp.array([np.pi / 5, np.pi / 3]), jnp.array([0., 0.]))
dt0 = 0.01  
args = None
sol = solve(y0, dt0, args)   

# %%
# TODO troubleshoot this. the segment lengths are changing. 

_sincos_derivative_signs = jnp.array([(1, 1), (1, -1), (-1, -1), (-1, 1)]).reshape((4, 1, 1, 2))

def sincos_derivative_signs(i):
    return _sincos_derivative_signs[-i]

def angular_to_cartesian(theta, dtheta, twolink):
    angle_sum = jnp.cumsum(theta)  # links
    sincos_angle_sum = jnp.array([jnp.cos(angle_sum),
                                  jnp.sin(angle_sum)])  # links, dims
    sincos_tmp = twolink.l.reshape((1, -1, 1)) * sincos_angle_sum  # links, dims
    xy_position = jnp.cumsum(sincos_tmp, axis=0)
    
    ang_vel_sum = jnp.cumsum(dtheta)#.unsqueeze(-1)  # links
    xy_velocity = jnp.cumsum(jnp.flip(sincos_tmp, (-1,)) * ang_vel_sum
                             * sincos_derivative_signs(1),
                             axis=0)
    return xy_position, xy_velocity


# %%
xy_pos, xy_vel = jax.vmap(angular_to_cartesian, in_axes=[0, 0, None])(sol.ys[0], sol.ys[1], TwoLink())
xy_pos = np.pad(xy_pos.squeeze(), ((0,0), (1,0), (0,0)))


# %%
def plot_2D_positions(xy, links=True, cmap_func=mpl.cm.viridis,
                      unit='m', ax=None, add_root=True):
    """Plot 2D position trajectories.
    Could also plot the controls on the joints.
    Args:
        xy ():
        links (bool):
        cmap_func ():
    """
    if ax is None:
        fig = plt.figure(figsize=(4,8))
        ax = fig.add_subplot()

    if add_root:
        xy = np.pad(xy, ((0,0), (1,0), (0,0)))

    cmap = cmap_func(np.linspace(0, 0.66, num=xy.shape[0], endpoint=True))
    cmap = mpl.colors.ListedColormap(cmap)

    for j in range(xy.shape[1]):
        # xy[np.array((0, len(xy) // 2, -1))]
        ax.plot(*xy[0].T, c=cmap(0.), lw=1, marker="o")
        ax.plot(*xy[len(xy)//2].T, c=cmap(0.5), lw=1, marker='o')
        ax.plot(*xy[-1].T, c=cmap(1.), lw=1, marker='o')
        # scatter = sns.scatterplot((xy[:, j, 0], xy[:, j, 1]),
        #                           marker='.', linewidth=0, #hue=range(xy.shape[0]),
        #                           ax=ax, legend=False palette=cmap.colors)


    ax.margins(0.1, 0.2)
    ax.set_aspect('equal')
    return ax

# %%
ax = plot_2D_positions(xy_pos, add_root=False)
plt.show()

# %% [markdown]
# Verifying that the segments stay constant length

# %%
distances = np.sqrt(np.sum(np.diff(xy_pos, axis=1)**2, axis=2))
print(distances)
np.std(distances, axis=0)


# %% [markdown]
# Clearly something is wrong in the angular -> Cartesian conversion

# %% [markdown]
# ### Linearization

# %%
class LTISystem(eqx.Module):
    A: jnp.ndarray
    B: jnp.ndarray
    C: jnp.ndarray
