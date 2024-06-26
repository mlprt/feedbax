# The structure of Feedbax

Feedbax includes a number of different types of modules. This page summarizes their responsibilities and relationships.

!!! Note "Abstract base classes"
    Most of the module types we'll discuss here are named `Abstract*`, because they are [abstract base classes](https://docs.python.org/3/library/abc.html) (ABCs).

    When we build and train our models, the actual components are *not* abstract, but derived from an abstract parent. For example, `SimpleReaches` is one task we might use, and it's a subclass of `AbstractTask`. `AbstractTask` defines how *any* task object should behave. When we talk about `AbstractTask`, we're talking about any given task object—not just `SimpleReaches`.

    Likewise, if you see a variable annotated as `task: AbstractTask` in the source code, that means that the value of `task` should be some type of task—not yet specified.

    Note that one abstract class may inherit from another, resulting in a more specific—but still general—type of module. For example, every `AbstractStagedModel` is an `AbstractModel` that specializes in defining its computation as a sequence of named state operations.

    Developers, note that Feedbax is designed according to the [abstract-final pattern](https://docs.kidger.site/equinox/pattern/).

## Models and states

The base class for all types of models is [`AbstractModel`][feedbax.AbstractModel]. It defines how any model subclass should depend on a particular type of state [PyTree](/feedbax/examples/pytrees). Each type of state PyTree is a final subclass of `equinox.Module`.

Note that Feedbax models are immutable. We cannot modify a model in-place: we have to replace it with an altered copy. Therefore a model cannot modify its own attributes from within, and states are not stored as attributes of a model object itself. Instead, an `AbstractModel` is like a function that receives a state PyTree, and returns an altered copy.

Most Feedbax models are *staged* models. Their base class is `AbstractStagedModel`, which inherits from `AbstractModel`. It defines how a model may be built as a sequence of operations ("stages") on the type of state PyTree it depends on. For example, `SimpleFeedback` is an `AbstractStagedModel` that is defined as a series of operations on a `SimpleFeedbackState`.

Generally, an `AbstractStagedModel` defines a single step through a model. However, we're interested in models that operate on their state for multiple time steps. Therefore we wrap our single-step model in an [`AbstractIterator`][feedbax.iterate.AbstractIterator], which is essentially a loop. Usually we'll use [`Iterator`][feedbax.iterate.Iterator], which iterates the model and remembers the history of all of its states.

## Tasks and losses

In Feedbax, models are trained to perform tasks. Typically, this means running the model through trials of the task, then scoring its performance, then getting an updated model that should perform slightly better on the next set of trials.

The base class for all types of tasks is [`AbstractTask`][feedbax.task.AbstractTask]. It provides 1) specifications for training trials, 2) specifications for validation trials, 3) a loss function, which scores a model's performance on a trial, and 4) methods for running a model on a given set of trials.

[Trial specifications][feedbax.task.TaskTrialSpec] are always composed of three things:

1. Data with which to initialize one or more parts of a model's state, prior to a trial;
2. Target data which the loss function will use to score the history of a model's states, over a trial;
3. Data which is provided to the model during a trial. This often overlaps with the target data from 2; for example, on a reaching task, we may score a model based on how close it reached to a target position, but we also need to provide the model with information about the target position so that it can know where to reach.

The loss function is not defined within the `AbstractTask`, but merely assigned to it as an attribute. This is because two tasks that are otherwise identical might vary in terms of their scoring mechanism. Therefore we specify the specific loss function we want to use, when we construct an instance of a particular type of task.

The base class for all loss functions is [`AbstractLoss`][feedbax.loss.AbstractLoss]. A loss computation takes as input the states of a model across a trial, as well as the complete specification for that trial.

Most types of loss are "simple" losses, which define one particular scoring mechanism. For example, [`NetworkActivityLoss`][feedbax.loss.NetworkActivityLoss] defines a penalty for the non-zero activities of the units ("neurons") in a model's neural network. Training on this loss will favour reducing the activity in the network as much as possible, given the other constraints.

The class [`CompositeLoss`][feedbax.loss.CompositeLoss] is used to aggregate (say, sum) multiple loss terms into a single loss function. Scoring of a task is usually based on multiple criteria, so the loss function that is assigned to an `AbstractTask` is usually a `CompositeLoss`.

## Training

A [`TaskTrainer`][feedbax.train.TaskTrainer] is used to [train](/feedbax/examples/1_train#training-the-model) a model to perform a task, over a sequence of many batches of training trials provided by an `AbstractTask`.

At the end of a training run, a `TaskTrainer` returns not just the trained model, but also a [`TaskTrainerHistory`][feedbax.train.TaskTrainerHistory] object. Normally this contains the value of the loss over all the batches. However, depending on the arguments given to `TaskTrainer`, it may also contain other information, like 1) the trial specifications on which the model was trained, or 2) the history of the model's trained parameters.

A `TaskTrainer` may also be used to train a set of [model replicates](/feedbax/examples/4_vmap) in parallel.

## Interventions

A core feature of Feedbax is the ability to modify models and tasks with [interventions](/feedbax/examples/3_intervening). The base class for interventions is `AbstractIntervenor`. Each type of intervention specifies a state operation, which may be inserted into the usual sequence of state operations of an `AbstractStagedModel`.

## Components of feedback models

So far, `SimpleFeedback` is the only top-level single-step model in Feedbax. It defines a feedback loop in which a neural network controls a biomechanical model—such as a model of an arm—based on delayed and noisy sensory feedback it receives about the state of the biomechanical model—such as the current position of the arm.

TODO: Biomechanics can refer to other stuff (e.g. biofluid mech) so we should be clear in the way we speak about limbs

There are three main types of components, here.

### Controllers

In control theory, the neural network is a *controller*. In reinforcement learning, we'd call it an *agent*—or at a given moment, a *policy*.

So far there isn't an `AbstractNetwork` or `AbstractController` that defines how controllers should behave in general. If you want to be able to intervene on the hidden state of your network while also having a separate readout or encoding layers, then [`SimpleStagedNetwork`][feedbax.nn.SimpleStagedNetwork] is suitable. Otherwise, you can [use](/feedbax/examples/5_model_stages#using-non-staged-components) arbitrarily complex neural networks as controllers.

### Channels

[`Channel`][feedbax.channel.Channel] is a model of delayed, noisy transmission of data. It's a modified [queue](https://en.wikipedia.org/wiki/Queue_(abstract_data_type)): it stores a number of samples of data in the order they were received; each time it receives a new sample, the oldest sample in the queue is pushed out the back, and the new one enters at the front. Noise is added to the oldest sample before it is returned.

`Channel` is used by `SimpleFeedback` to model sensory feedback, but it can also be used wherever delayed, noisy transmission is required in a model.

### Biomechanical models

Biomechanical models describe the physics of simplified limbs. Generally, there are two aspects to this:

1. Differential equations, which describe the evolution of the limb state over time depending on applied forces.
2. Kinematic (geometric) relationships between state variables. For example, the force generated by a muscle can be calculated directly from the muscle's current length and the velocity of its contraction, and these in turn can be calculated from the current angles and angular velocities of the joints spanned by the muscle.

Both of these aspects are captured by an [`AbstractPlant`][feedbax.mechanics.plant.AbstractPlant]. In particular, `AbstractPlant` is a subtype of both

1. `AbstractStagedModel`: The stages of an `AbstractPlant` describe its kinematic state operations;
2. [`AbstractDynamicalSystem`][feedbax.dynamics.AbstractDynamicalSystem], which is the base class for all modules that define a `vector_field` returning state derivatives. In particular, `AbstractPlant` can aggregate the vector fields of multiple dynamical components into a single vector field describing the continuous dynamics of the full biomechanical model.

Each type of `AbstractPlant` defines continuous dynamics, which must be discretized and associated with a numerical solver. To do this, we [wrap](/feedbax/examples/1_train#building-the-model-ourselves-using-core-feedbax) an `AbstractPlant` as a [`Mechanics`][feedbax.mechanics.Mechanics] object. `Mechanics` uses [Diffrax](https://github.com/patrick-kidger/diffrax) for numerical integration.

#### Plant types and components

An `AbstractPlant` generally has two components, from which it aggregates its dynamic and kinematic operations.

1. A skeleton model. The base class for skeletons is [`AbstractSkeleton`][feedbax.mechanics.skeleton.AbstractSkeleton]. A skeleton is a type of `AbstractDynamicalSystem`, with a `vector_field` that describes how the state of the skeleton changes depending on applied forces. For example, `TwoLinkArm` is a type of `AbstractSkeleton` that describes how a two-jointed arm (think: double pendulum) moves based on the torques applied to its joints.

    The simplest skeleton is [`PointMass`][feedbax.mechanics.skeleton.PointMass], which has mass but no spatial extent, and obeys Newton's laws of motion.

2. A muscle model. The base class for muscles is [`AbstractMuscle`][feedbax.mechanics.muscle.AbstractMuscle].

The simplest type of `AbstractPlant` is [`DirectForceInput`][feedbax.mechanics.plant.DirectForceInput], which has a skeleton but no muscle model. In that case, the input to the `AbstractPlant` is passed directly to its `AbstractSkeleton`, and the output of the controller/neural network may be forces or torques.

For more complex plant models like [`MuscledArm`][feedbax.mechanics.plant.MuscledArm], the input to the plant is commands sent to the muscle model, which generates the forces which act on the skeleton.