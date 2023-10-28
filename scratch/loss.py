"""

TODO:
- Protocols for all the different `state` types/fields
- Could use another dataclass instead of a dict for loss terms,
  though if they're not referenced in transformations I don't 
  know that it really matters.
- L2 by default, but should allow for other norms

:copyright: Copyright 2023 by Matt L. Laporte.
:license: Apache 2.0. See LICENSE for details.
"""

from abc import abstractmethod
import logging
from typing import (
    ClassVar,
    Dict, 
    Optional, 
    Protocol, 
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

import equinox as eqx
from equinox import AbstractClassVar
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree
import numpy as np

from feedbax.types import CartesianState2D


logger = logging.getLogger(__name__)


class AbstractLoss(eqx.Module):
    """Abstract base class for loss functions.
    
    Enforces that concrete subclasses should have a string label.
    
    TODO: 
    - Time aggregation should happen in every concrete subclass.
      Could add a method/abstractvar to `AbstractLoss`.
    - Should probably allow the user to override with their own label.
    """
    label: AbstractClassVar[Optional[str]]
    
    @abstractmethod
    def __call__(
        self,
        states: PyTree, 
        targets: PyTree, 
        task_inputs: Optional[PyTree] = None,
    ) -> Tuple[float, Dict[str, float]]:
        ...        
        
    def __mul__(self, other):
        if jnp.isscalar(other):
            weights = {label: other * w for label, w in self.weights.items()}
            return CompositeLoss(self.terms, weights)
        else:
            return self * other
    
    def __rmul__(self, other):
        return self.__mul__(self, other)
    
    def __div__(self, other):
        return self.__mul__(self, 1. / other)
    
    def __add__(self, other):
        if isinstance(other, AbstractLoss):
            # this will overwrite any duplicates
            terms = {**self.terms, **other.terms}
            weights = {**self.weights, **other.weights}
            return CompositeLoss(terms, weights)
        else: 
            return self + other 
    
    def __radd__(self, other):
        # addition of loss terms is commutative
        return self.__add__(self, other)


class CompositeLoss(AbstractLoss):
    """Composite of simpler loss functions.
    
    TODO:
    - Different aggregation schemes.
    - Perhaps, nesting of losses. As of now, if any of the component losses
      are themselves instances of `CompositeLoss`, their own component terms
      are not remembered. 
    - Perhaps change the labeling scheme; if we don't allow nesting then 
      it is inappropriate to just take the first label name from each component.
    """
    terms: Dict[AbstractLoss]
    weights: Dict[Float[Array, ""]]
    label: ClassVar[None] = None
    
    def __init__(
        self, 
        terms: Sequence[AbstractLoss] | Dict[str, AbstractLoss], 
        weights: Optional[Sequence[float] | Dict[str, float]] = None,
    ):
        
        if weights is None:
            pass 
        # if weights is None:
        #     weights = [1.] * len(terms)
        # elif not len(terms) == len(weights):
        #     raise ValueError("Mismatch between number of loss terms and term weights")
        # #! assumes all the components are simple losses, and `labels` is a one-tuple
        # if isinstance(terms, dict):
        #     self.labels = tuple(terms.keys())
        #     self.terms = terms 
        # else:
        #     self.labels = [term.labels[0] for term in terms]
        #     self.terms = dict(zip(self.labels, terms))
        # if not isinstance(weights, dict):
        #     self.weights = dict(zip(self.labels, weights))
        # else:
        #     self.weights = weights
    
    @jax.named_scope("fbx.CompositeLoss")
    def __call__(
        self, 
        states: PyTree, 
        targets: PyTree,
        task_inputs: Optional[PyTree] = None,
    ) -> Tuple[float, Dict[str, float]]:
        
        # evaluate loss terms
        loss_terms = jax.tree_map(
            lambda term: term(states, targets, task_inputs)[0], 
            self.terms,
            is_leaf=lambda x: isinstance(x, AbstractLoss),
        )
        # aggregate over time 
        #! this should be done in all the other classes as well! they aren't returning scalars
        loss_terms = jax.tree_map(lambda x: jnp.sum(x, axis=-1), loss_terms)
        # aggregate over batch 
        loss_terms = jax.tree_map(lambda x: jnp.mean(x, axis=0), loss_terms)
        if self.weights is not None:
            # term scaling
            loss_terms = jax.tree_map(
                lambda term, weight: term * weight, 
                loss_terms, 
                self.weights
            )
        
        loss = jax.tree_util.tree_reduce(lambda x, y: x + y, loss_terms)
        
        return loss, loss_terms
    

@runtime_checkable
class HasEffectorState(Protocol):
    effector: CartesianState2D


@runtime_checkable
class HasMechanicsState(Protocol):
    mechanics: HasEffectorState


class EffectorPositionLoss(AbstractLoss):
    """
    
    Note that if discount is shaped such that it gives non-zero weight to the
    position error during the fixation period of (say) a delayed reach task,
    then typically the target will be specified as the fixation point during
    that period, and `EffectorPositionLoss` will also act as a fixation loss.
    However, when we are using certain kinds of goal error discounting (e.g.
    exponential, favouring errors near the end of the trial) then the fixation
    loss may not be weighed into `EffectorPositionLoss`, and it may be
    appropriate to add `EffectorFixationLoss` to the composite loss. However,
    in that case the same result could still be achieved using a single
    instance of `EffectorPositionLoss`, by passing a `discount` that's the sum
    of the goal error discount (say, non-zero only near the end of the trial)
    and the hold signal (non-zero only during the fixation period) scaled by
    the relative weights of the goal and fixation error losses.
    
    TODO: do we handle the temporal discount here? or return the sequence of losses
    """
    
    discount: Optional[Float[Array, "time"]] 
    label: ClassVar[str] = "effector_position"

    def __init__(
        self, 
        discount=None,
    ):
        if discount is not None:
            self.discount = discount[None, :]  # singleton batch dimension
        else:   
            self.discount = None
    
    def __call__(
        self, 
        states: HasEffectorState, 
        # TODO: take AbstractTaskTrialSpec?
        targets: PyTree,
        task_inputs: Optional[PyTree] = None,
    ) -> Tuple[float, Dict[str, float]]:
            
        # sum over xyz
        loss = jnp.sum(
            (states.mechanics.effector.pos - targets.pos) ** 2, 
            axis=-1
        )
        
        # temporal discount
        if self.discount is not None:
            loss = loss * self.discount
        
        return loss, {self.labels[0]: loss}


class EffectorFixationLoss(AbstractLoss):
    """"""
    label: ClassVar[str] = "effector_fixation"
    
    def __call__(
        self, 
        states: PyTree, 
        targets: PyTree,
        task_inputs: PyTree,
    ) -> Tuple[float, Dict[str, float]]:
        
        loss = jnp.sum(
            (states.mechanics.effector.pos - targets.pos) ** 2, 
            axis=-1
        )
        
        loss = loss * jnp.squeeze(task_inputs.hold)
        
        return loss, {self.labels[0]: loss}


class EffectorFinalVelocityLoss(AbstractLoss):
    """
    
    TODO: how do we handle calculating oss for a single timestep only?
    """
    label: ClassVar[str] = "effector_final_velocity"
    
    def __call__(
        self, 
        states: PyTree, 
        targets: PyTree,
        task_inputs: Optional[PyTree] = None,
    ) -> Tuple[float, Dict[str, float]]:
        
        loss = jnp.sum(
            (states.mechanics.effector.vel[:, -1] - targets.vel[:, -1]) ** 2, 
            axis=-1
        )
        
        # so the result plays nicely with time aggregation
        loss = jnp.expand_dims(loss, axis=-1)
        
        return loss, {self.labels[0]: loss}


class NetworkOutputLoss(AbstractLoss):
    """"""
    label: ClassVar[str] = "nn_output"

    def __call__(
        self, 
        states: PyTree, 
        #! **kwargs?
        targets: PyTree,
        task_inputs: Optional[PyTree] = None,
    ) -> Tuple[float, Dict[str, float]]:
        
        loss = jnp.sum(states.network.output ** 2, axis=-1)
        
        return loss, {self.labels[0]: loss}


class NetworkActivityLoss(AbstractLoss):
    """"""
    label: ClassVar[str] = "nn_activity"

    def __call__(
        self, 
        states: PyTree, 
        targets: PyTree,
        task_inputs: Optional[PyTree] = None,
    ) -> Tuple[float, Dict[str, float]]:
        
        loss = jnp.sum(states.network.activity ** 2, axis=-1)
        
        return loss, {self.labels[0]: loss}

        
def simple_reach_loss(
    n_steps: int, 
    loss_term_weights: Optional[Dict[str, float]] = None,
    discount_exp: int = 6,
):
    """A typical loss function for a simple reach task.
    
    Includes power function discounting of position error, back in time from
    the end of trials. If the exponent `discount_exp` is zero, there is no
    discounting.
    
    TODO: 
    - Maybe activity loss shouldn't be included by default.
    """
    if discount_exp == 0:
        discount = 1.
    else: 
        discount = jnp.linspace(1. / n_steps, 1., n_steps) ** discount_exp
    return CompositeLoss(
        dict(
            # these assume a particular PyTree structure to the states returned by the model
            # which is why we simply instantiate them 
            effector_position=EffectorPositionLoss(discount=discount),
            effector_final_velocity=EffectorFinalVelocityLoss(),
            nn_output=NetworkOutputLoss(),  # the "control" loss
            nn_activity=NetworkActivityLoss(),
        ),
        weights=loss_term_weights,
    )