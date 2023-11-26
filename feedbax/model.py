"""Modules that compose other modules into control contexts.

For example, classes defined here could be used to to model a sensorimotor 
loop, a body, or (perhaps) multiple interacting bodies. 

TODO:
- Maybe this should be renamed... mainly because it might be confused with 
`AbstractTask` in terms of its purpose.

:copyright: Copyright 2023 by Matt L. Laporte.
:license: Apache 2.0. See LICENSE for details.
"""

from abc import abstractmethod, abstractproperty
from collections import OrderedDict
import copy
from functools import cached_property
import logging
from typing import Callable, Dict, Generic, Optional, Sequence, Tuple, TypeVar, Union

import equinox as eqx
from equinox import AbstractVar
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PyTree
import numpy as np

from feedbax.channel import Channel, ChannelState
from feedbax.intervene import AbstractIntervenor
from feedbax.mechanics import Mechanics, MechanicsState
from feedbax.networks import NetworkState, wrap_stateless_network 
from feedbax.task import AbstractTask, AbstractTaskInput
from feedbax.state import CartesianState2D
from feedbax.utils import tree_sum_n_features


logger = logging.getLogger(__name__)


N_DIM = 2


class AbstractModelState(eqx.Module):
    ...
    

StateT = TypeVar("StateT", bound=AbstractModelState)


class AbstractModel(eqx.Module, Generic[StateT]):
    """Base class for compositional state-dependent models.
    
    To define a new model, the following should be implemented:
    
        1. A concrete subclass of `AbstractModelState` that defines the PyTree
           structure of the full model state. The fields may be types of 
           `AbstractModelState` as well, in the case of nested models.
        2. A concrete subclass of this class (`AbstractModel`) with:
            i. fields to hold the modules that compose the model, 
            ii. a `model_spec` property that specifies a series of operations 
            on parts of the full model state, using those component modules, 
            and
            iii. an `init` method that returns an initial model state
    
    TODO:
    - Should `init` be in `AbstractModelState` rather than an `AbstractModel`?
    
    - It might make sense to implement `__call__` here, and then get subclasses
      to implement another method (`step`?). 
      - For example, this would allow all the `intervenors` to be called at the 
        start of each `__call__`, regardless of what type of model we're 
        dealing with; we'd need an `intervenors: AbstractVar` field in that 
        case. However, I'm not sure yet that this makes sense.
    """
    intervenors: AbstractVar[Dict[str, Sequence[AbstractIntervenor]]]
    
    @jax.named_scope("fbx.AbstractModel")
    def __call__(
        self, 
        input, # AbstractTaskInput 
        state: StateT, 
        key: jax.Array,
    ) -> StateT:
        
        keys = jr.split(key, len(self._stages))
        
        for (intervenors, module, where, module_input), key \
            in zip(self._stages, keys):
            
            key1, key2 = jr.split(key)
            
            for intervenor in intervenors:
                # whether this modifies the state depends on `input`
                state = intervenor(state, input, key=key1)
                
            state = eqx.tree_at(
                where, 
                state, 
                module(self)(
                    module_input(input, state), 
                    where(state), 
                    key=key2,
                ),
            )
        
        return state

    @abstractproperty
    def stages_spec(self) -> Dict[str, Tuple[
        eqx.Module,
        Callable[[StateT], PyTree],
        Sequence[Callable[[AbstractTaskInput, StateT], PyTree]],
    ]]:
        """Specifies the model in terms of substate operations.
        
        Each tuple in the dict has the following elements:
        
            1. An equinox module that performs a state update
            2. A function that selects the part of the model state PyTree that 
               is updated (i.e. output) by the module
            3. A function that selects the part of the model state PyTree that 
               is passed as input to the module
        """
        ...
  
    @cached_property 
    def _stages(self) -> Tuple[Tuple[
        Sequence[AbstractIntervenor],
        eqx.Module,
        Callable[[StateT], PyTree],
        Sequence[Callable[[AbstractTaskInput, StateT], PyTree]],
    ]]: 
        """Zips up the user-defined intervenors with `stages_spec`."""
        #! should not be referred to in `__init__`, at least before defining `intervenors`
        return tuple(jax.tree_map(
            lambda x, y: (y,) + x, 
            self.stages_spec, 
            jax.tree_map(tuple, self.intervenors, 
                         is_leaf=lambda x: isinstance(x, list)),
            is_leaf=lambda x: isinstance(x, tuple),
        ).values())
    
    def _get_intervenors_dict(
        self, intervenors: Optional[Union[Sequence[AbstractIntervenor],
                                          Dict[str, Sequence[AbstractIntervenor]]]]
    ):
        intervenors_dict = jax.tree_map(
            lambda _: [], 
            self.stages_spec,
            is_leaf=lambda x: isinstance(x, tuple),
        )
        
        if intervenors is not None:
            if isinstance(intervenors, Sequence):
                # By default, place interventions in the first stage.
                intervenors_dict.update({'get_feedback': list(intervenors)})
            elif isinstance(intervenors, dict):
                intervenors_dict.update(
                    jax.tree_map(list, intervenors, 
                                 is_leaf=lambda x: isinstance(x, Sequence)))
            else:
                raise ValueError("intervenors not a sequence or dict of sequences")
        
        return intervenors_dict
        
    @abstractmethod
    def init(
        self,
        **kwargs,
    ) -> StateT:
        """Return an initial state for the model."""
        ...
        
    @abstractproperty
    def memory_spec(self) -> PyTree[bool]:
        """Specifies which states should typically be remembered by callers."""
        ...
        

class SimpleFeedbackState(AbstractModelState):
    mechanics: MechanicsState
    network: NetworkState
    net_readout: Array
    feedback: ChannelState


class SimpleFeedback(AbstractModel[SimpleFeedbackState]):
    """Simple feedback loop with a single RNN and single mechanical system.
    
    TODO:
    - PyTree of force inputs (in addition to control inputs) to mechanics
    - Could insert intervenors as stages in the model, but I think it makes 
    more sense for modules to be designed-in, which then provides the user 
    with labels for stages at which interventions can be applied, resulting 
    in a dict of interventions that mirrors the dict of stages.
    """
    net: eqx.Module  
    net_readout: eqx.Module
    mechanics: Mechanics 
    feedback_channel: Channel
    delay: int 
    feedback_leaves_func: Callable[[PyTree], PyTree]  
    
    intervenors: Dict[str, Sequence[AbstractIntervenor]]
    
    def __init__(
        self, 
        net: eqx.Module, 
        mechanics: Mechanics, 
        net_readout: Optional[eqx.Module] = None,
        delay: int = 0, 
        feedback_leaves_func: Callable[[MechanicsState], PyTree] \
            = lambda mechanics_state: mechanics_state.system,
        intervenors: Optional[Union[Sequence[AbstractIntervenor],
                                    Dict[str, Sequence[AbstractIntervenor]]]] \
            = None,
        key: Optional[jax.Array] = None,
    ):
        if net_readout is None:
            net_readout = wrap_stateless_network(eqx.nn.Linear(
                net.hidden_size, 
                mechanics.system.control_size, 
                use_bias=True,
                key=key,
            ))
        
        self.net = net
        self.net_readout = net_readout
        self.mechanics = mechanics
        self.feedback_leaves_func = feedback_leaves_func
        self.delay = delay + 1  # indexing: delay=0 -> storage len=1
        self.feedback_channel = Channel(delay)
        self.intervenors = self._get_intervenors_dict(intervenors)
        
    @property
    def stages_spec(self):
        """Specifies the stages of the model in terms of substate operations.
        
        Note that the module has to be given as a function that obtains it from `self`, 
        or otherwise it won't use modules that have been updated during training.
        I'm not sure why the references are kept across model updates...
        """
        return OrderedDict({
            'get_feedback': (
                lambda self: self.feedback_channel,  # module that operates on it
                lambda state: state.feedback,  # substate it operates on
                lambda _, state: self.feedback_leaves_func(state.mechanics),  # inputs
            ),
            'nn_step': (
                lambda self: self.net,
                lambda state: state.network,
                lambda input, state: (input, state.feedback.output),
            ),
            'nn_readout': (
                lambda self: self.net_readout,
                lambda state: state.net_readout,
                lambda _, state: state.network.activity,
            ),
            'mechanics_step': (
                lambda self: self.mechanics,
                lambda state: state.mechanics,
                lambda _, state: state.net_readout,
            ),
        })        

    def init(
        self, 
        effector_state: CartesianState2D,
        # mechanics_state: MechanicsState = None, 
        # network_state: NetworkState = None, 
        # feedback_state: ChannelState = None,
    ): 
        """Return an initial state for the model.
        
        TODO:
        - We can probably use `eval_shape` more generally when an init is not 
          available, which may be the case for simpler modules. The 
          `stages_spec` could be used to determine the relevant info flow.
        """
        mechanics_state = self.mechanics.init(effector_state=effector_state)
        network_state = self.net.init()
        readout_dtype = eqx.filter_eval_shape(
            self.net_readout, 
            jnp.zeros(self.net.hidden_size), 
            None,
        )
        readout_state = jnp.zeros(readout_dtype.shape)
        return SimpleFeedbackState(
            mechanics=mechanics_state,
            network=network_state,
            net_readout=readout_state,
            feedback=self.feedback_channel.init(
                self.feedback_leaves_func(mechanics_state)
            ),
        )
    
    @property
    def memory_spec(self) -> SimpleFeedbackState:
        """Specifies which states should typically be remembered by callers.
        
        For example, `fbx.Iterator` stores trajectories of states, however it
        doesn't usually make sense to store `states.feedback.queue` for every
        timestep, because it contains info that is already available to
        `Iterator` if `states.mechanics` is stored at every timestep. If the
        feedback delay is 5 steps, `Iterator` would otherwise end up with 5 
        extra copies of all the parts of `states.mechanics` that are part of 
        the feedback.
        
        In particular, this information will be used by `Iterator`, but will
        be ignored by `SimpleIterator`, which remembers the full state 
        indiscriminately to favour speed over memory usage.
        
        NOTE: It makes sense for this to be here since it has to do with the
        logic of the feedback loop, i.e. that queue is just transient internal 
        memory of another variable in the loop. 
        """
        return SimpleFeedbackState(
            mechanics=True, 
            network=True,
            net_readout=True,
            feedback=ChannelState(output=True, queue=False)
        )

    @staticmethod
    def get_nn_input_size(
        task: AbstractTask, 
        mechanics: Mechanics, 
        feedback_leaves_func=lambda mechanics_state: mechanics_state.system,
    ) -> int:
        """Determine how many scalar input features the neural network needs.
        
        This is a static method because its logic (number of network inputs =
        number of task inputs + number of feedback inputs from `mechanics`) 
        is related to the structure of `SimpleFeedback`. However, it is 
        not an instance method because we want to construct the network
        before we construct `SimpleFeedback`.
        """
        example_feedback = feedback_leaves_func(mechanics.init())
        n_feedback = tree_sum_n_features(example_feedback)
        example_trial_spec = task.get_train_trial(key=jr.PRNGKey(0))[0]
        n_task_inputs = tree_sum_n_features(example_trial_spec.input)
        return n_feedback + n_task_inputs
    
    
def add_intervenors(
    model: SimpleFeedback, 
    intervenors: Union[Sequence[AbstractIntervenor],
                       Dict[str, Sequence[AbstractIntervenor]]], 
    keep_existing: bool = True,
    *,
    key: jax.Array
) -> SimpleFeedback:
    """Add intervenors to a model, returning the updated model.
    
    TODO:
    - Could this be generalized to `AbstractModel[StateT]`?
    """
    if keep_existing:
        if isinstance(intervenors, Sequence):
            # If a sequence is given, append to the first stage.
            first_stage_label = next(iter(model.intervenors))
            intervenors_dict = eqx.tree_at(
                lambda intervenors: intervenors[first_stage_label],
                model.intervenors,
                model.intervenors[first_stage_label] + list(intervenors),
            )
        elif isinstance(intervenors, dict):
            intervenors_dict = copy.deepcopy(model.intervenors)
            for label, new_intervenors in intervenors.items():
                intervenors_dict[label] += list(new_intervenors)
        else:
            raise ValueError("intervenors not a sequence or dict of sequences")

    return SimpleFeedback(
        net=model.net,
        mechanics=model.mechanics,
        net_readout=model.net_readout,
        delay=model.delay,
        feedback_leaves_func=model.feedback_leaves_func,
        intervenors=intervenors_dict,
        key=key,
    )

def remove_intervenors(
    model: SimpleFeedback
) -> SimpleFeedback:
    """Return a model with no intervenors."""
    return add_intervenors(model, intervenors=(), keep_existing=False)