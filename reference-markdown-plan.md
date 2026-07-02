# Reference Markdown Plan

This plan inventories the files under `references/` and points each reference to the closest relevant HTML documentation source when one exists.

## Brain Dynamics

### `references/brain-dynamics/brain-dynamics-delay-protocol.md`
- **Description:** Explains BrainState Brain Dynamics delay APIs and delay-buffer behavior.
- **Sources:** [Delay Protocol](https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/02_synaptic_delays.html)

### `references/brain-dynamics/brain-dynamics-event-driven-operators.md`
- **Description:** Catalogs event-driven sparse spike operators and connectivity patterns for scalable SNNs.
- **Sources:** [Event-Driven Operators](https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/03_event_driven_operators.html)

### `references/brain-dynamics/brain-dynamics-snn-workflows.md`
- **Description:** Routes build, simulate, and train workflows for BrainState-style spiking neural networks.
- **Sources:** [Building a Spiking Neural Network](https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/04_building_an_snn.html), [Training a Spiking Neural Network](https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/05_training_an_snn.html)

## BrainCell

### `references/braincell/braincell-custom-ion-channel-authoring.md`
- **Description:** Custom BrainCell ion/channel authoring notes moved out of the skill list into shared reference markdown.
- **Sources:** [Ions and Channels](https://brainx.chaobrain.com/braincell/concepts/ions_channels.html), [Channels](https://brainx.chaobrain.com/braincell/tutorials/channel.html), [Extending BrainCell](https://brainx.chaobrain.com/braincell/developer/extending.html)

### `references/braincell/braincell-manual-morphology-construction.md`
- **Description:** Manual morphology construction and morphology-preparation notes moved out of the skill list into shared reference markdown.
- **Sources:** [Morphology concept](https://brainx.chaobrain.com/braincell/concepts/morphology.html)

### `references/braincell/morphology-io-loading-validation.md`
- **Description:** Unified reference for morphology import paths, validation reports/options, NeuroMorpho caching, and checkpoints.
- **Sources:** [IO Overview](https://brainx.chaobrain.com/braincell/file_formats/overview.html), [SWC](https://brainx.chaobrain.com/braincell/file_formats/swc.html), [Neurolucida ASC](https://brainx.chaobrain.com/braincell/file_formats/asc.html), [NeuroML2](https://brainx.chaobrain.com/braincell/file_formats/neuroml2.html), [NeuroMorpho.Org](https://brainx.chaobrain.com/braincell/file_formats/neuromorpho.html), [Checkpointing](https://brainx.chaobrain.com/braincell/file_formats/checkpointing.html), [Morphology concept](https://brainx.chaobrain.com/braincell/concepts/morphology.html)

### `references/braincell/probe-reference.md`
- **Description:** Documents BrainCell probes for recording runtime state, mechanism state, currents, and traces.
- **Sources:** [Mechanisms in BrainCell](https://brainx.chaobrain.com/braincell/tutorials/mech.html)

### `references/braincell/topology-building-and-visualization.md`
- **Description:** Guides runtime topology inspection and visualization for multicompartment BrainCell cells.
- **Sources:** [Point Tree Visualization](https://brainx.chaobrain.com/braincell/tutorials/vis.html), [Region and Locset Filters](https://brainx.chaobrain.com/braincell/tutorials/filter.html)

## BrainState

### `references/brainstate/advanced-randomness.md`
- **Description:** Catalogs advanced BrainState RNG streams, stochastic layers, and transformed-randomness patterns.
- **Sources:** [Random Number Generation](https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html)

### `references/brainstate/brainstate-control-flow-patterns.md`
- **Description:** Collects loop and branch patterns that remain valid under BrainState and JAX transformations.
- **Sources:** [Control Flow](https://brainx.chaobrain.com/brainstate/tutorials/transformations/05_control_flow.html)

### `references/brainstate/parameter-constraints-regularization.md`
- **Description:** Conceptual workflow for `ParamState` vs `nn.Param`, constrained values, regularization penalties, `Const`, and training-loss integration.
- **Sources:** [Parameters, Transforms, and Regularization](https://brainx.chaobrain.com/brainstate/tutorials/core/05_parameters_transforms_regularization.html)

### `references/brainstate/parameter-containers-transforms-catalog.md`
- **Description:** Catalog for `Param`, `Const`, and built-in parameter transforms; maps constraints to transform choices.
- **Sources:** [Parameter Containers API](https://brainx.chaobrain.com/brainstate/apis/nn/parameters.html)

### `references/brainstate/regularization-catalog-priors.md`
- **Description:** Catalog for classical, structural, chained, and prior-distribution regularizers; maps modeling goals to `reg=` choices.
- **Sources:** [Standard Regularizations API](https://brainx.chaobrain.com/brainstate/apis/nn/regularization.html)

### `references/brainstate/transformation-grad-expansion.md`
- **Description:** Expands gradient and autodiff teaching for differentiable simulation and parameter fitting.
- **Sources:** [Automatic Differentiation](https://brainx.chaobrain.com/brainstate/tutorials/transformations/02_autodiff.html), [Training and Metrics](https://brainx.chaobrain.com/brainstate/tutorials/core/07_training_and_metrics.html), [Parameters, Transforms, and Regularization](https://brainx.chaobrain.com/brainstate/tutorials/core/05_parameters_transforms_regularization.html)

### `references/brainstate/transformation-jit-expansion.md`
- **Description:** Expands BrainState-aware JIT compilation, state write-back, cache, and static-argument guidance.
- **Sources:** [JIT and Compilation](https://brainx.chaobrain.com/brainstate/tutorials/transformations/01_jit_and_compilation.html), [Transformations, the Essentials](https://brainx.chaobrain.com/brainstate/tutorials/core/06_transformations_essentials.html)

### `references/brainstate/transformation-vmap-expansion.md`
- **Description:** Expands BrainState vectorization, batching, state axes, sweeps, and stochastic vmap patterns.
- **Sources:** [Vectorization](https://brainx.chaobrain.com/brainstate/tutorials/transformations/03_vectorization.html), [Random Number Generation](https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html), [pmap and parallelism](https://brainx.chaobrain.com/brainstate/tutorials/transformations/04_advanced_batching.html)

## Diagnostics

### `references/diagnostics/brainstate-transformed-diagnostics.md`
- **Description:** Collects runtime debugging, checking, and error-handling patterns for transformed BrainState code.
- **Sources:** [Error Handling and Runtime Checks](https://brainx.chaobrain.com/brainstate/tutorials/transformations/06_error_handling_and_checks.html), [Debugging Transformed Code](https://brainx.chaobrain.com/brainstate/tutorials/transformations/07_debugging.html)

### `references/diagnostics/common-failures-index.md`
- **Description:** Maps recurring BrainX failure patterns to the skill or reference that should handle them.
- **Sources:** None - local cross-reference router.

## Libraries

### `references/libraries/channel-library.md`
- **Description:** Catalogs BrainCell channels and channel-modeling patterns.
- **Sources:** [BrainCell channel API](https://brainx.chaobrain.com/braincell/apis/braincell.channel.html), [Channels](https://brainx.chaobrain.com/braincell/tutorials/channel.html)

### `references/libraries/cv-policy-reference.md`
- **Description:** Catalogs BrainCell control-volume policies and discretization tradeoffs.
- **Sources:** [Discretization](https://brainx.chaobrain.com/braincell/concepts/discretization.html)

### `references/libraries/filter-function-library.md`
- **Description:** Catalogs BrainCell region and locset filters for targeting mechanisms, probes, and clamps.
- **Sources:** [BrainCell filter API](https://brainx.chaobrain.com/braincell/apis/filter.html)

### `references/libraries/ion-library.md`
- **Description:** Catalogs BrainCell ion species and ion-related modeling patterns.
- **Sources:** [BrainCell ion API](https://brainx.chaobrain.com/braincell/apis/braincell.ion.html)

### `references/libraries/prebuilt-activation-library.md`
- **Description:** Catalogs BrainState activation and normalization components for module-building tasks.
- **Sources:** [BrainState activation API](https://brainx.chaobrain.com/brainstate/apis/nn/activation.html)

### `references/libraries/prebuilt-layer-library.md`
- **Description:** Catalogs BrainState prebuilt layers so agents reuse existing components before writing custom modules.
- **Sources:** [Linear layers](https://brainx.chaobrain.com/brainstate/apis/nn/linear.html), [Convolutional layers](https://brainx.chaobrain.com/brainstate/apis/nn/conv.html), [Normalization layers](https://brainx.chaobrain.com/brainstate/apis/nn/normalization.html), [Pooling layers](https://brainx.chaobrain.com/brainstate/apis/nn/pooling.html), [Padding layers](https://brainx.chaobrain.com/brainstate/apis/nn/padding.html), [Dropout layers](https://brainx.chaobrain.com/brainstate/apis/nn/dropout.html)

### `references/libraries/solver-library-with-effects.md`
- **Description:** Catalogs BrainCell and BrainState solver and integration choices with modeling consequences.
- **Sources:** [BrainCell integration API](https://brainx.chaobrain.com/braincell/apis/integration.html), [Choosing and Using Solvers](https://brainx.chaobrain.com/braincell/integration/solvers.html), [Advanced Integration](https://brainx.chaobrain.com/braincell/integration/advanced.html)

## Local Index and Policy

### `references/index.md`
- **Description:** Top-level router summarizing available references and when to open each one.
- **Sources:** None - local reference index.

### `references/policy/.gitkeep`
- **Description:** Keeps the reserved `references/policy/` directory present until policy references are added.
- **Sources:** None - placeholder file.
