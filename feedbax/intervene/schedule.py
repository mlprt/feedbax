"""

:copyright: Copyright 2023-2024 by Matt Laporte.
:license: Apache 2.0. See LICENSE for details.
"""

from collections import OrderedDict
from collections.abc import Mapping, Sequence, Callable
import copy
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Tuple,
    Type,
    TypeAlias,
    TypeVar,
    Union,
)

import equinox as eqx
from equinox import Module
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree

from feedbax.intervene import AbstractIntervenor, AbstractIntervenorInput, is_intervenor
from feedbax.misc import StrAlwaysLT, get_unique_label, is_module
from feedbax._model import AbstractModel
from feedbax.state import StateT
from feedbax._tree import tree_call

if TYPE_CHECKING:
    from feedbax._staged import AbstractStagedModel


logger = logging.getLogger(__name__)


class InterventionSpec(eqx.Module):
    # TODO: The type of `intervenor` is wrong: each entry will have the same PyTree
    # structure as `AbstractIntervenorInput` but may be filled with callables that
    # specify a trial distribution for the leaves
    intervenor: AbstractIntervenor
    where: Callable[[AbstractModel], "AbstractStagedModel"]
    stage_name: Optional[str]
    default_active: bool


pre_first_stage = StrAlwaysLT("_pre_first_stage")
# fixed = StrAlwaysLT("_fixed")


# StateS = TypeVar("StateS", Module, Array)  # TODO: should be PyTree[Array]?
InputT = TypeVar("InputT", bound=AbstractIntervenorInput)

# This used to be `AbstractIntervenor[StateS, InputT]` but I don't think
# it makes sense to bind all a stage's intervenors, to the same `InputT`
Intervenor = AbstractIntervenor[StateT, AbstractIntervenorInput]
StageNameStr: TypeAlias = str
IntervenorLabelStr: TypeAlias = str

StageIntervenors: TypeAlias = Mapping[IntervenorLabelStr, Intervenor[StateT]]
# Use to type the `intervenors` field of an `AbstractStagedModel`
ModelIntervenors: TypeAlias = Mapping[StageNameStr, StageIntervenors[StateT]]
# Use in constructor parameter lists.
ArgIntervenors: TypeAlias = Union[
    Sequence[Intervenor[StateT]], 
    Mapping[StageNameStr, Sequence[Intervenor[StateT]]]
]

    
def _fixed_intervenor_label(intervenor):
    return f"FIXED_{type(intervenor).__name__}"


def add_fixed_intervenor(
    model: "AbstractStagedModel[StateT]",
    where: Callable[[AbstractModel[StateT]], Any],
    intervenor: AbstractIntervenor,
    stage_name: Optional[StageNameStr] = None,
    intervenor_label: Optional[IntervenorLabelStr] = None,
    **kwargs: Any,
) -> "AbstractStagedModel[StateT]":
    """Return an updated model with an added, fixed intervenor.

    Arguments:
        model: The model to which the intervenor will be added.
        where: Takes `model` and returns the instance of `AbstractStagedModel` within
            it (which may be `model` itself) to which to add the intervenors.
        intervenor: The intervenor to add.
        stage_name: The stage named in `model.model_spec` to which the intervenor will
            be added. The intervenor will execute at the end of the stage.
            If `None`, the intervenor will execute before the first model stage.
        intervenor_label: Custom key for the intervenor, which determines how it will be
            accessed as part of the model PyTree. Note that labels for fixed intervenors
            are prepended with `"FIXED_"` if they are not already.
        kwargs: Additional keyword arguments to
            [`add_intervenors`][feedbax.intervene.add_intervenors].
    """
    if stage_name is None:
        stage_name = pre_first_stage
    
    if intervenor_label is not None:
        if not intervenor_label.startswith("FIXED"):
            intervenor_label = f"FIXED_{intervenor_label}"
            logger.debug("Prepending 'FIXED' to user-supplied intervenor label")
    else:
        intervenor_label = _fixed_intervenor_label(intervenor)
    
    return add_intervenors(
        model, 
        where, 
        {stage_name: {intervenor_label: intervenor}}, 
        **kwargs
    )


def add_intervenors(
    model: "AbstractStagedModel[StateT]",
    where: Callable[
        ["AbstractStagedModel[StateT]"], Any  # "AbstractStagedModel[StateS]"
    ],
    intervenors: Union[
        Sequence[Intervenor[StateT]],
        Mapping[
            StageNameStr,
            Union[
                Sequence[Intervenor[StateT]],
                Mapping[IntervenorLabelStr, Intervenor[StateT]],
            ]
        ],
    ],
    stage_name: Optional[StageNameStr] = None,
    keep_existing: bool = True,
) -> "AbstractStagedModel[StateT]":
    """Return an updated model with added intervenors.

    Arguments:
        model: The model to which the intervenors will be added.
        where: Takes `model` and returns the instance of `AbstractStagedModel` within
            it (which may be `model` itself) to which to add the intervenors.
        intervenors: The intervenors to add. May be 1) a sequence of intervenors to
            execute, by default before the first model stage, or 2) a dict/mapping from
            stage names to a) the sequence of intervenors to execute at the end of a
            respective model stage, or b) another dict/mapping from custom intervenor
            labels to intervenors to execute at the end of that state.
        stage_name: If `intervenors` is supplied as a simple sequence of intervenors 
            (case 1), execute them at the end of this model stage. By default, they 
            will be executed prior to the first model stage.
        keep_existing: Whether to keep the existing intervenors belonging directly to
            the instance of `AbstractStagedModel` to which the new intervenors are added.
            If `True`, the new intervenors are appended to the existing ones; if `False`,
            the old intervenors are replaced.
    """
    if keep_existing:
        existing_intervenors = where(model).intervenors
    else:
        existing_intervenors = {stage_name: {} for stage_name in where(model).model_spec}
    intervenors_dict = copy.deepcopy(existing_intervenors)
    
    if isinstance(intervenors, Sequence):
        # If a simple sequence of intervenors is passed, append them to the list of
        # fixed (unscheduled) intervenors for the specified stage -- or by default, the
        # pre-first stage.

        if stage_name is None:
            stage_name = pre_first_stage
            
        intervenors = {pre_first_stage: intervenors}

    if not isinstance(intervenors, Mapping):
        raise ValueError("intervenors not a sequence or dict of sequences")
    #     stage_intervenors = existing_intervenors.get(stage_name, {})
        
    #     stage_intervenors |= {
    #         type(intervenor).__name__: intervenor for intervenor in intervenors
    #     }

    #     intervenors_dict = (
    #         existing_intervenors | {stage_name: stage_intervenors}
    #     )
    # elif isinstance(intervenors, Mapping):
    for stage_name in intervenors:
        # Note that in some cases, a stage may appear in `existing_intervenors` but
        # not in `model_spec`; e.g. if we use `eqx.tree_at` to deactivate `add_noise`
        # in `Channel`, then any intervenors previously added to the `'add_noise'`
        # stage will still be present in the `intervenors` attribute of the channel,
        # but its `model_spec` will no longer contain that stage. If the stage is
        # reactivated later, so will be its intervenors.
        #
        # Here, we raise an error if the user attempts to add to a stage that either
        # doesn't exist, or is currently deactivated.
        if (
            stage_name not in where(model).model_spec
            and stage_name is not pre_first_stage
        ):
            raise ValueError(
                f"{stage_name} is not a valid model stage for intervention"
            )
            
    for stage_name, stage_intervenors_new in intervenors.items():

        # Use `OrderedDict` to make sure intervenors are executed in the ordered provided.
        stage_intervenors = intervenors_dict.get(stage_name, OrderedDict())

        if isinstance(stage_intervenors_new, Sequence):
            # TODO: unique names?
            stage_intervenors_new = {
                _fixed_intervenor_label(intervenor): intervenor 
                for intervenor in stage_intervenors_new
            }
        
        if any(
            shared_keys := set(stage_intervenors) & set(stage_intervenors_new)
        ):
            logger.warning("Intervenors with the following labels were "
                            "overwritten during call to add_intervenors: "
                            ", ".join(shared_keys))
        
        stage_intervenors |= stage_intervenors_new
        intervenors_dict |= {stage_name: stage_intervenors}

    return eqx.tree_at(
        lambda model: where(model).intervenors,
        model,
        intervenors_dict,
    )


class TimeSeriesParam(Module):
    """Wraps intervenor parameters that should be interpreted as time series.

    Attributes:
        param: The parameter to interpret as a time series.
    """

    param: Array

    def __call__(self):
        """Return the wrapped parameter."""
        return self.param


def _eval_intervenor_param_spec(
    intervention_spec: InterventionSpec,
    trial_spec, #: AbstractTaskTrialSpec,
    key: PRNGKeyArray,
):
    # Unwrap any `TimeSeriesParam` instances:
    return tree_call(
        # Evaluate any trial-generation lambdas:
        tree_call(
            intervention_spec.intervenor.params,
            trial_spec,
            key=key,
            # Don't unwrap `TimeSeriesParam`s yet:
            exclude=lambda x: isinstance(x, TimeSeriesParam),
            is_leaf=lambda x: isinstance(x, TimeSeriesParam),
        ),
        is_leaf=lambda x: isinstance(x, TimeSeriesParam),
    )


def schedule_intervenor(
    tasks: PyTree["AbstractTask"],
    models: PyTree[AbstractModel[StateT]],
    where: Callable[[AbstractModel[StateT]], Any],
    intervenor: AbstractIntervenor | Type[AbstractIntervenor],
    stage_name: Optional[str] = None,
    validation_same_schedule: bool = True,
    intervenor_params: Optional[
        AbstractIntervenorInput
    ] = None,  #! wrong! distribution functions are allowed. only the PyTree structure is the same
    intervenor_params_validation: Optional[AbstractIntervenorInput] = None,
    default_active: bool = False,
) -> Tuple[PyTree["AbstractTask"], PyTree[AbstractModel[StateT]]]:
    """Adds an intervention to a model and a task.

    !!! Note ""
        Accepts either an intervenor instance, or an intervenor class. Passing
        an intervenor instance but no `intervenor_params`, the instance's
        `params` attribute is used as `intervenor_params`. This can be combined
        with the intervenor's `with_params` constructor to define the
        schedule. For example:

        ```python
        schedule_intervenor(
            tasks,
            models,
            lambda model: model.step.mechanics,
            CurlField.with_params(
                amplitude=lambda trial_spec, *, key: jr.normal(key, (1,)),
                active=True,
            ),
            ...
        )
        ```

        Passing an intervenor class and an `intervenor_params`, an instance
        will be constructed from the two.

        Passing an intervenor instance *and* an `intervenor_params`, the
        instance's `params` will be replaced with the `intervenor_params`
        before adding to the model.

        Passing an intervenor class but no `intervenor_params`, an error is
        raised due to insufficient information to schedule the intervention.

        Passing a value for `intervenor_params_validation` allows for separate
        control over the intervention schedule for the task's validation set.

    Arguments:
        tasks: The task(s) in whose trials the intervention will be scheduled
        models: The model(s) to which the intervention will be added
        where: Takes `model` and returns the instance of `AbstractStagedModel` within
            it (which may be `model` itself) to which to add the intervenors.
        intervenor: The intervenor (or intervenor class) to schedule.
        stage_name: The name of the stage in `where(model).model_spec` at the end of
            which the intervenor will be executed. If `None`, executes before the
            first stage of the model.
        validation_same_schedule: Whether the interventions should be scheduled
            in the same way for the validation set as for the training set.
        intervenor_params: The parameters of to the intervenor, which may be
            constants, or callables that are used by `task` to construct the
            parameters for the intervention on each trial.
        intervenor_params_validation: Same as `intervenor_input`, but for the
            task's validation set. Overrides `validation_same_schedule`.
        default_active: If the intervenor added to the model should have
            `active=True` by default, so that the intervention will be
            turned on even if the intervenor doesn't explicitly receive values
            for its parameters.
    """
    if stage_name is None:
        stage_name = pre_first_stage

    intervenor_: AbstractIntervenor
    if isinstance(intervenor, type(AbstractIntervenor)):
        if intervenor_params is None:
            raise ValueError("Must pass intervenor_params if intervenor is a class")
        intervenor_ = intervenor(params=intervenor_params)  # type: ignore
    elif isinstance(intervenor, AbstractIntervenor):
        intervenor_ = intervenor
    else:
        raise ValueError("intervenor must be an AbstractIntervenor instance or class")

    # A unique label is needed because `AbstractTask` uses a single dict to
    # pass intervention parameters for all intervenors in an `AbstractStagedModel`,
    # regardless of where they appear in the model hierarchy.
    #
    # The label should be unique across all models and tasks that the intervenor
    # will be registered with.
    invalid_labels_models = jax.tree_util.tree_reduce(
        lambda x, y: x + y,
        jax.tree_map(
            lambda model: model.step._all_intervenor_labels,
            models,
            is_leaf=is_module,  # AbstractModel
        ),
        is_leaf=lambda x: isinstance(x, tuple),
    )
    invalid_labels_tasks = jax.tree_util.tree_reduce(
        lambda x, y: x + y,
        jax.tree_map(
            lambda task: tuple(task.intervention_specs.keys()),
            tasks,
            is_leaf=is_module,  # AbstractTask
        ),
        is_leaf=lambda x: isinstance(x, tuple),
    )
    invalid_labels = set(invalid_labels_models + invalid_labels_tasks)
    label = get_unique_label(type(intervenor_).__name__, invalid_labels)

    # Construct specification-intervenors
    intervention_specs = {label: InterventionSpec(
        intervenor=intervenor_,
        where=where,
        stage_name=stage_name,
        default_active=default_active,
    )}

    if intervenor_params_validation is not None:
        intervention_specs_validation = {label: InterventionSpec(
            intervenor=eqx.tree_at(
                lambda intervenor: intervenor.params,
                intervenor_,
                intervenor_params_validation,
            ),
            where=where,
            stage_name=stage_name,
            default_active=default_active,
        )}
    elif validation_same_schedule:
        intervention_specs_validation = intervention_specs
    else:
        intervention_specs_validation = dict()

    # Add the spec intervenors to every task in `tasks`
    tasks = jax.tree_map(
        lambda task: eqx.tree_at(
            lambda task: (task.intervention_specs, task.intervention_specs_validation),
            task,
            (
                task.intervention_specs | intervention_specs,
                task.intervention_specs_validation | intervention_specs_validation,
            ),
        ),
        tasks,
        is_leaf=is_module,  # AbstractTask
    )

    # Apply the unique label to the intervenor to be added to the model.
    # intervenor_relabeled = eqx.tree_at(lambda x: x.label, intervenor_, label)

    # Construct the intervenor with default parameters, to add to the model.
    # TODO: Should we let the user pass a `default_intervenor_params`?
    key_example = jax.random.PRNGKey(0)
    # Assume that all the tasks are compatible with the way trial specs are used
    # to generate trial-by-trial intervenor parameters.
    task_example = jax.tree_leaves(
        tasks, is_leaf=is_module  # AbstractTask
    )[0]
    trial_spec_example = task_example.get_train_trial(key_example)
    intervenor_defaults = eqx.tree_at(
        lambda intervenor: intervenor.params,
        intervenor,
        _eval_intervenor_param_spec(
            # Prefer the validation parameters, if they exist.
            (intervention_specs | intervention_specs_validation)[label],
            trial_spec_example,
            key_example,
        )
    )

    intervenor_final = eqx.tree_at(
        lambda intervenor: intervenor.params.active,
        intervenor_defaults,
        default_active,
    )

    # Add the intervenor to all of the models.
    models = jax.tree_map(
        lambda model: add_intervenors(
            model,
            where=where,
            intervenors={stage_name: {label: intervenor_final}},
        ),
        models,
        is_leaf=lambda x: isinstance(x, AbstractModel),
    )

    return tasks, models

