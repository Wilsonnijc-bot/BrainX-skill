# BrainX Reference Markdown Plan

This is an architecture inventory, not a reference-writing brief. The supplied
BrainX workspace implementation plan is the sole source of truth. Existing
drafts and files are used only to determine whether a declared target already
exists; they do not add routes that the supplied plan does not declare.

## Reference Ownership Summary

| Skill | Markdown/reference inventory | Conditional organization | Main action |
|---|---:|---:|---|
| Bundle router (`SKILL.md`) | Semantic skill routes only | Entire routed graph | Route directly to owning skills; own no references |
| `skills/braincell/` | 15 Markdown routes plus 1 semantic skill route | 8 first-layer Markdown routes, 1 semantic skill route, 6 exclusive multicompartment children, and 1 second-level diagnostic child | Expose the pasted first layer and preserve parent-only multicompartment selection |
| `skills/brainevent/` | 4 | None | Preserve; all four exist |
| `skills/brainmass/` | 14 | Ten package references plus four reusable Braintools references | Add the omitted Braintools branches and use the pasted fitting-reference name |
| `skills/brainpy-state/` | 17 Markdown routes plus 1 semantic skill route | Training owns four Braintools children; NEST workflow owns six compact lookup areas | Add omitted local array, delay/event, and Braintools routes |
| `skills/brainstate/` | 19 Markdown references plus 1 semantic skill route | First-layer references plus parameter, randomness, and diagnostics trees | Consolidate size inference and preserve parent-only selection |
| `skills/braintrace/` | 6 | None declared | Add all six skill-local routes |
| `skills/brainunit/` | 8 | None declared | Add all eight skill-local routes; keep examples inline |
| `skills/brainx-acceleration-audit/` | 5 first-layer references, 1 nested randomness child, plus one skill route | Advanced randomness is selected only by its local parent | Preserve duplicated local transform/randomness files |
| `skills/brainx-general-guard/` | 1 randomness parent, 1 nested child, plus one skill route | Install remains a semantic skill route | Preserve the parent-only randomness route |
| `skills/brainx-install/` | 1 | None declared | Add the one compatibility reference and repair skill frontmatter |

## Bundle Router: `SKILL.md`

The bundle router owns no Markdown references. It routes only to package
skills, and each package skill owns every reference it consumes. When multiple
skills need equivalent reference material, keep a copy under each consuming
skill instead of introducing a shared reference path.

## BrainUnit: `skills/brainunit/SKILL.md`

All eight references are first-layer, skill-local routes. BrainUnit keeps its
workflow examples inline and declares no standalone script.

| Canonical reference | Need |
|---|---|
| `skills/brainunit/references/quantity-inspection-and-conversion.md` | Quantity mantissa, unit, dimension, compatibility, conversion, and extraction |
| `skills/brainunit/references/array-creation.md` | Unit-aware array construction |
| `skills/brainunit/references/array-mechanics.md` | Inspection, indexing, functional updates, shape operations, and backend conversion |
| `skills/brainunit/references/math-function-library.md` | Function choice by unit semantics |
| `skills/brainunit/references/unit-structure-and-definition.md` | Unit structure, combination, and custom definitions |
| `skills/brainunit/references/temperature-conversions.md` | Affine temperature conversions and temperature differences |
| `skills/brainunit/references/prefix-library.md` | Predefined units, prefix names, and scales |
| `skills/brainunit/references/physical-constant-library.md` | Unit-aware physical constants |

## BrainCell: `skills/braincell/SKILL.md`

### First-layer references

| Canonical reference | Need | Crafting source |
|---|---|---|
| `skills/braincell/references/multicompartment/multicompartment-cell-workflow.md` | morphology route: `Cell`, CVs, `paint`, `place`, clamps, probes, and geometry-dependent simulation | [Cell tutorial](https://brainx.chaobrain.com/braincell/tutorials/cell.html), then the nested sources below |
| `skills/braincell/references/area-scaled-hh-pattern.md` | Density-to-total conversion for capacitance, conductance, current, and cell area | The current skill's density-versus-total P0 rule and the existing extracted area-scaled pattern |
| `skills/braincell/references/mixions-for-adaptation.md` | First-layer adaptation, AHP/KCa, rebound, dynamic calcium, and `MixIons(k, ca)` composition | [Adaptation](https://brainx.chaobrain.com/braincell/examples/spike_frequency_adaptation.html), [T-current rebound](https://brainx.chaobrain.com/braincell/examples/t_current_rebound.html), and [thalamic neurons](https://brainx.chaobrain.com/braincell/examples/thalamic_neurons.html) |
| `skills/braincell/references/libraries/ion-library.md` | Built-in ions, fixed/InitNernst/dynamic choices, concentration dynamics, and `MixIons` | [Ions and channels concept](https://brainx.chaobrain.com/braincell/concepts/ions_channels.html), [ion tutorial](https://brainx.chaobrain.com/braincell/tutorials/ion.html), [ion API](https://brainx.chaobrain.com/braincell/apis/braincell.ion.html) |
| `skills/braincell/references/libraries/channel-library.md` | Built-in channel families, dependencies, selection, and the built-in-versus-custom boundary | [Ions and channels concept](https://brainx.chaobrain.com/braincell/concepts/ions_channels.html), [channel tutorial](https://brainx.chaobrain.com/braincell/tutorials/channel.html), [channel API](https://brainx.chaobrain.com/braincell/apis/braincell.channel.html), [channel ablation](https://brainx.chaobrain.com/braincell/examples/channel_ablation.html), [adaptation example](https://brainx.chaobrain.com/braincell/examples/spike_frequency_adaptation.html) |
| `skills/braincell/references/libraries/solver-library-with-effects.md` | Integrator names, cable/composite solvers, speed/accuracy guidance, and numerical effects | [Integration concept](https://brainx.chaobrain.com/braincell/concepts/integration.html), [integration API](https://brainx.chaobrain.com/braincell/apis/integration.html), [solver guide](https://brainx.chaobrain.com/braincell/integration/solvers.html), [advanced integration](https://brainx.chaobrain.com/braincell/integration/advanced.html), [integration-methods example](https://brainx.chaobrain.com/braincell/examples/integration_methods.html) |
| `skills/braincell/references/array-creation.md` | Skill-local unit-aware construction of simulation arrays | The BrainCell routing tree in the supplied plan |
| `skills/braincell/references/braincell/braincell-custom-ion-channel-authoring.md` | Custom channel/ion extension after built-ins are exhausted | [Ions and channels concept](https://brainx.chaobrain.com/braincell/concepts/ions_channels.html), [channel tutorial](https://brainx.chaobrain.com/braincell/tutorials/channel.html), [extending BrainCell](https://brainx.chaobrain.com/braincell/developer/extending.html) |
| `skills/brainevent/SKILL.md` | Semantic route to the event-driven connectivity skill | The BrainCell routing tree in the supplied plan |

### Nested reference tree under the multicompartment parent


| Exclusive nested child | Need | Crafting source |
|---|---|---|
| `skills/braincell/references/braincell/morphology-io-loading-validation.md` | SWC, ASC, NeuroML2, NeuroMorpho, validation, checkpoints, and post-load checks | [IO overview](https://brainx.chaobrain.com/braincell/file_formats/overview.html), [SWC](https://brainx.chaobrain.com/braincell/file_formats/swc.html), [ASC](https://brainx.chaobrain.com/braincell/file_formats/asc.html), [NeuroML2](https://brainx.chaobrain.com/braincell/file_formats/neuroml2.html), [NeuroMorpho](https://brainx.chaobrain.com/braincell/file_formats/neuromorpho.html), [checkpointing](https://brainx.chaobrain.com/braincell/file_formats/checkpointing.html), [morphology concept](https://brainx.chaobrain.com/braincell/concepts/morphology.html) |
| `skills/braincell/references/braincell/topology-building-and-visualization.md` | NodeTree, CV/branch/node views, placement verification, and visualization | [Visualization tutorial](https://brainx.chaobrain.com/braincell/tutorials/vis.html), [filter tutorial](https://brainx.chaobrain.com/braincell/tutorials/filter.html) |
| `skills/braincell/references/braincell/probe-reference.md` | State, mechanism, current, and trace probes plus missing-trace checks | [Mechanisms tutorial](https://brainx.chaobrain.com/braincell/tutorials/mech.html) |
| `skills/braincell/references/libraries/filter-function-library.md` | Region and locset selection for mechanisms, probes, and clamps | [Filter tutorial](https://brainx.chaobrain.com/braincell/tutorials/filter.html), [filter API](https://brainx.chaobrain.com/braincell/apis/filter.html), and the existing selector blueprint |
| `skills/braincell/references/libraries/cv-policy-reference.md` | CV policy selection, discretization effects, resolution, and cost | [Discretization concept](https://brainx.chaobrain.com/braincell/concepts/discretization.html), [Cell tutorial](https://brainx.chaobrain.com/braincell/tutorials/cell.html), and the existing CV blueprint |
| `skills/braincell/references/braincell/braincell-manual-morphology-construction.md` | Manual topology creation before `Cell` construction | Morphology concept, Cell tutorial, and existing blueprint |

`skills/braincell/references/diagnostics/common-failures-index.md` remains a second-level diagnostic child, selected only after the manual-morphology, topology, or probe child identifies a failure mode. It is not an eighth first-level child.

The multicompartment parent may reuse first-layer ion, channel, solver, and MixIons references; reuse does not make those files exclusive children.


## BrainEvent: `skills/brainevent/SKILL.md`

All four required Markdown references are skill-local and already exist. Application-script selection and provenance live directly in the skill body.

| Canonical reference | Need | Crafting source |
|---|---|---|
| `skills/brainevent/references/sparse-formats.md` | COO construction, CSR/CSC storage, conversion, and selection | [Sparse matrices tutorial](https://brainx.chaobrain.com/brainevent/tutorials/data-structures/02_sparse_matrices.html), [sparse-data API](https://brainx.chaobrain.com/brainevent/reference/apis/sparsedata.html), [utilities API](https://brainx.chaobrain.com/brainevent/reference/apis/utilities.html) |
| `skills/brainevent/references/connectivity-variants.md` | JITC distributions/orientations, fixed fan-in/out, and format choice | [JIT connectivity](https://brainx.chaobrain.com/brainevent/tutorials/data-structures/03_jit_connectivity.html), [fixed connections](https://brainx.chaobrain.com/brainevent/tutorials/data-structures/04_fixed_connections.html), [format guide](https://brainx.chaobrain.com/brainevent/how-to/data-structures/choosing-a-sparse-format.html), [sparse-data API](https://brainx.chaobrain.com/brainevent/reference/apis/sparsedata.html), [utilities API](https://brainx.chaobrain.com/brainevent/reference/apis/utilities.html) |
| `skills/brainevent/references/synaptic-plasticity.md` | Pre/post event updates, CSR/dense routing, and STDP overlay | [Plasticity tutorial](https://brainx.chaobrain.com/brainevent/tutorials/data-structures/05_synaptic_plasticity.html), [plasticity how-to](https://brainx.chaobrain.com/brainevent/how-to/data-structures/synaptic-plasticity.html), [operations API](https://brainx.chaobrain.com/brainevent/reference/apis/operations.html) |
| `skills/brainevent/references/custom-operators.md` | Route to Numba, Numba CUDA, Warp, C++, or CUDA extension paths | [Index](https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/index.html), [Numba CPU](https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/01_numba.html), [Numba CUDA](https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/02_numba_cuda.html), [Warp](https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/03_warp.html), [C++](https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/04_cpp.html), [CUDA](https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/05_cuda.html) |



## BrainMass: `skills/brainmass/SKILL.md`

The skill defines ten package references plus four reusable Braintools
references. The fitting and HORN parents select those reusable children only
when the task needs them.

| Canonical reference | Need | Crafting source |
|---|---|---|
| `skills/brainmass/references/modellibrary.md` | Model inventory, categories, state variables, use cases, `list_models()`, and `ModelInfo` | [Models API](https://brainx.chaobrain.com/brainmass/reference/models.html), [utilities API](https://brainx.chaobrain.com/brainmass/reference/utilities.html) |
| `skills/brainmass/references/noiseprocesses.md` | Noise-family inventory, seeding, stochastic runs, and batched ensembles | [Noise API](https://brainx.chaobrain.com/brainmass/reference/noise.html), [noise tutorial](https://brainx.chaobrain.com/brainmass/tutorials/03_noise.html) |
| `skills/brainmass/references/coupling-network-api.md` | Coupling mechanisms, delays, and network variants | [Coupling API](https://brainx.chaobrain.com/brainmass/reference/coupling.html), [network tutorial](https://brainx.chaobrain.com/brainmass/tutorials/04_building_a_network.html), [coupling/delays concept](https://brainx.chaobrain.com/brainmass/concepts/coupling_and_delays.html) |
| `skills/brainmass/references/forward-observation-api.md` | HRFBold, kernels, TemporalAverage, BOLDSignal, EEG/MEG, and lead fields | [Forward API](https://brainx.chaobrain.com/brainmass/reference/forward.html), [observation API](https://brainx.chaobrain.com/brainmass/reference/observation.html), [forward-model tutorial](https://brainx.chaobrain.com/brainmass/tutorials/05_forward_models.html) |
| `skills/brainmass/references/fitting-with-objectives-api.md` | Simulator, Network, Fitter/FitResult, objective functions, and backend boundaries | [Orchestration API](https://brainx.chaobrain.com/brainmass/reference/orchestration.html), [gradient fitting](https://brainx.chaobrain.com/brainmass/tutorials/06_fitting_with_gradients.html), [gradient-free fitting](https://brainx.chaobrain.com/brainmass/tutorials/07_gradient_free_fitting.html), [custom objective](https://brainx.chaobrain.com/brainmass/howto/custom_objective.html) |
| `skills/brainmass/references/datasets-api.md` | Dataset registration/loading, Connectome, Signal, and task containers | [Datasets API](https://brainx.chaobrain.com/brainmass/reference/datasets.html) |
| `skills/brainmass/references/visualization-analysis-api.md` | Plotting, FC/FCD, and spectral analysis | [Visualization API](https://brainx.chaobrain.com/brainmass/reference/viz.html), [analysis how-to](https://brainx.chaobrain.com/brainmass/howto/analyze_results.html) |
| `skills/brainmass/references/batch-transform-acceleration.md` | JIT, transformed loops, `scan`, `vmap`, checkpointing, batched initial conditions, and sweeps | [BrainMass batch and accelerate](https://brainx.chaobrain.com/brainmass/howto/batch_and_accelerate.html), with [BrainTrace batching](https://brainx.chaobrain.com/braintrace/tutorials/batching.html) only for the vmap-per-sample comparison already used by the skill |
| `skills/brainmass/references/horn-task-training.md` | HORN components, task datasets, direct optimizer loops, and held-out metrics | [HORN API](https://brainx.chaobrain.com/brainmass/reference/horn.html), [task-training tutorial](https://brainx.chaobrain.com/brainmass/tutorials/08_training_on_tasks.html), [HORN case study](https://brainx.chaobrain.com/brainmass/gallery/case_studies/horn_cognitive_task.html) |
| `skills/brainmass/references/parameter-sweeps-and-regime-analysis.md` | Regime exploration and sensitivity analysis distinct from fitting | [Parameter-sweeps how-to](https://brainx.chaobrain.com/brainmass/howto/parameter_sweeps.html) |
| `skills/brainmass/references/braintools-cognitive-tasks.md` | Cognitive-task trials for HORN training | [Cognitive-task API](https://brainx.chaobrain.com/braintools/apis/cogtask.html) |
| `skills/brainmass/references/braintools-metrics.md` | Loss and metric selection for fitting and task training | [Metric API](https://brainx.chaobrain.com/braintools/apis/metric.html) |
| `skills/brainmass/references/braintools-optimizer.md` | Optimizers, schedules, Optax bridge, and SciPy/Nevergrad wrappers | [Optimization API](https://brainx.chaobrain.com/braintools/apis/optim.html), [optimization tutorials](https://brainx.chaobrain.com/braintools/optim/index.html) |
| `skills/brainmass/references/braintools-surrogate-gradient.md` | Surrogate gradients for otherwise non-differentiable objective paths | [Surrogate-gradient API](https://brainx.chaobrain.com/braintools/apis/surrogate.html) |

### Nested routes

- `fitting-with-objectives-api.md` selects metrics, optimizer, and surrogate-gradient references only when required, and routes to `skills/brainmass/references/scripts/gradient-free-fitting.py` for the complete derivative-free workflow.
- `horn-task-training.md` selects cognitive tasks, metrics, and optimizer references only when task training requires them; it may route to the bundled HORN case-study script.



## BrainPy: `skills/brainpy-state/SKILL.md`

BrainPy uses a semantic route to the BrainEvent skill for event-driven
connectivity. Its array, delay, and event-protocol Markdown are local copies,
and one compact NEST workflow parent owns all NEST-compatible lookup areas.

| Canonical reference | Need | Crafting source |
|---|---|---|
| `skills/brainevent/SKILL.md` | Semantic binary-event and sparse-connectivity boundary | The BrainPy routing tree in the supplied plan |
| `skills/brainpy-state/references/brainpy-neuron-library.md` | Neuron catalog and selection | [Neuron API](https://brainx.chaobrain.com/brainpy-state/apis/brainpy-neurons.html), [neuron-selection how-to](https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/sim-choose-neuron.html) |
| `skills/brainpy-state/references/brainpy-synapse-library.md` | Synaptic dynamics and receptor filters | [Synapse API](https://brainx.chaobrain.com/brainpy-state/apis/brainpy-synapses.html) |
| `skills/brainpy-state/references/brainpy-synaptic-outputs.md` | COBA/CUBA/MgBlock outputs and current-versus-conductance selection | [Synaptic-output API](https://brainx.chaobrain.com/brainpy-state/apis/brainpy-synouts.html), [COBA/CUBA how-to](https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/sim-coba-cuba-synapses.html) |
| `skills/brainpy-state/references/brainpy-projection-library.md` | Projection APIs, AlignPre/AlignPost, direct-current projections, gap junctions, and delays | [Projection API](https://brainx.chaobrain.com/brainpy-state/apis/brainpy-projections.html), [alignment concept](https://brainx.chaobrain.com/brainpy-state/concepts/alignpre-alignpost.html), [delays how-to](https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/sim-delays.html) |
| `skills/brainpy-state/references/brainpy-plasticity.md` | STP/STD state and projection integration | [Plasticity API](https://brainx.chaobrain.com/brainpy-state/apis/brainpy-plasticity.html), [short-term-plasticity how-to](https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/sim-short-term-plasticity.html) |
| `skills/brainpy-state/references/brainpy-custom-models.md` | Custom Neuron/Synapse anatomy, ODE steps, and paper reproduction | [Paper-reproduction how-to](https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/sim-reproduce-a-paper.html), [BrainPy gallery](https://brainx.chaobrain.com/brainpy-state/examples/brainpy-gallery.html) |
| `skills/brainpy-state/references/brainpy-training.md` | Differentiability, surrogate gradients, ParamState, BPTT, and checkpointed rollouts | [Differentiability concept](https://brainx.chaobrain.com/brainpy-state/concepts/differentiability.html), [train-an-SNN tutorial](https://brainx.chaobrain.com/brainpy-state/brainpy-style/tutorials/04-train-an-snn.html), [surrogate-gradient how-to](https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/train-surrogate-gradients.html), [checkpointing how-to](https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/train-long-rollouts-checkpoint.html) |
| `skills/brainpy-state/references/brainpy-readouts-and-inputs.md` | Readout heads, spike/input generators, Poisson helpers, and encoders | [Readout API](https://brainx.chaobrain.com/brainpy-state/apis/brainpy-readouts.html), [readout how-to](https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/train-readouts.html), [input API](https://brainx.chaobrain.com/brainpy-state/apis/brainpy-inputs.html) |
| `skills/brainpy-state/references/brainstate-dynamics/brain-dynamics-delay-protocol.md` | Skill-local delay APIs and buffer behavior | [delay tutorial](https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/02_synaptic_delays.html) |
| `skills/brainpy-state/references/brainstate-dynamics/brain-dynamics-event-driven-operators.md` | Skill-local sparse event operators and connectivity | [event-driven tutorial](https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/03_event_driven_operators.html) |
| `skills/brainpy-state/references/array-creation.md` | Skill-local unit-aware simulation-array creation | The BrainPy routing tree in the supplied plan |
| `skills/brainpy-state/references/braintools-encoder-library.md` | Experimental/data input encoders and spike operations | [Braintools encoder API](https://brainx.chaobrain.com/braintools/apis/braintools.html) |
| `skills/brainpy-state/references/braintools-initializers.md` | Parameter, weight, and distance-modulated connectivity initializers | [Initializer API](https://brainx.chaobrain.com/braintools/apis/init.html) |
| `skills/brainpy-state/references/braintools-metrics.md` | Classification, regression, spike-train, synchronization, LFP, and connectivity metrics | [Metric API](https://brainx.chaobrain.com/braintools/apis/metric.html) |
| `skills/brainpy-state/references/braintools-optimizer.md` | Optimizers, schedulers, Optax bridges, and external wrappers | [Optimization API](https://brainx.chaobrain.com/braintools/apis/optim.html), [optimization tutorials](https://brainx.chaobrain.com/braintools/optim/index.html) |
| `skills/brainpy-state/references/braintools-surrogate-gradient.md` | Functional and object-style surrogate gradients | [Surrogate-gradient API](https://brainx.chaobrain.com/braintools/apis/surrogate.html) |
| `skills/brainpy-state/NEST-compatible/nest-workflow.md` | Parent route for the NEST-compatible branch | The BrainPy routing tree in the supplied plan |

### Training nested routes

`skills/brainpy-state/references/brainpy-training.md` alone selects the encoder,
metrics, optimizer, and surrogate-gradient Braintools references. The
initializers reference remains a first-layer route because initialization can
be required without training.

### NEST-compatible nested branch


| Nested lookup area | Need | Crafting sources | Disposition |
|---|---|---|---|
| `Model Library.md` | Select NEST-compatible neurons and inspect neuron-model APIs | [Models](https://brainx.chaobrain.com/brainpy-state/nest-style/models.html), [neuron API](https://brainx.chaobrain.com/brainpy-state/apis/nest-neurons.html) | Keep as a compact area in `nest-workflow.md` |
| `Synapse And Connectivity.md` | Static/special synapses, plasticity, connection rules, synapse specs, and realized connectivity | [synapse API](https://brainx.chaobrain.com/brainpy-state/apis/nest-synapses.html), [plasticity API](https://brainx.chaobrain.com/brainpy-state/apis/nest-plasticity.html), [connectivity](https://brainx.chaobrain.com/brainpy-state/nest-style/connectivity.html) | Keep as a compact area in `nest-workflow.md` |
| `Devices` | Generators, recorders, detectors, source semantics, direction, and result readback | [devices guide](https://brainx.chaobrain.com/brainpy-state/nest-style/devices.html), [device API](https://brainx.chaobrain.com/brainpy-state/apis/nest-devices.html) | Keep as a compact area in `nest-workflow.md` |
| `Network Building.md` | `Simulator`, `NodeView`, `SimulationResult`, `SynapseCollection`, projection/connection APIs, and spatial primitives | [network tutorial](https://brainx.chaobrain.com/brainpy-state/nest-style/tutorials/03-connect-network.html), [network API](https://brainx.chaobrain.com/brainpy-state/apis/nest-network.html), [spatial API](https://brainx.chaobrain.com/brainpy-state/apis/nest-spatial.html), [spatial guide](https://brainx.chaobrain.com/brainpy-state/nest-style/spatial.html) | Keep as a compact area in `nest-workflow.md` |
| `Divergence And Parity.md` | Porting differences, STDP parameter placement, recording/stochastic parity, validation, and NEST mismatches | [divergence index](https://brainx.chaobrain.com/brainpy-state/nest-style/divergences/index.html), [validation status](https://brainx.chaobrain.com/brainpy-state/nest-style/validation-status.html), [STDP divergence](https://brainx.chaobrain.com/brainpy-state/nest-style/divergences/stdp.html) | Keep as a compact area in `nest-workflow.md` |
| `Integration Categories.md` | Numerical and integration behavior by NEST-compatible model family | [integration categories](https://brainx.chaobrain.com/brainpy-state/nest-style/integration-categories.html) | Keep as a compact area in `nest-workflow.md` |


## BrainState: `skills/brainstate/SKILL.md`

### First-layer references

| Canonical reference | Need | Crafting source |
|---|---|---|
| `skills/brainstate/references/state-graph-operations.md` | Find, extract, split, replace, and reconstruct State graphs | [JIT tutorial](https://brainx.chaobrain.com/brainstate/tutorials/transformations/01_jit_and_compilation.html), [graph API](https://brainx.chaobrain.com/brainstate/apis/graph.html), [graph editing how-to](https://brainx.chaobrain.com/brainstate/how_to/inspect_and_edit_state_graph.html) |
| `skills/brainstate/references/model-interop-and-migration.md` | Flax/Equinox interop and PyTorch migration | [Interop API](https://brainx.chaobrain.com/brainstate/apis/interop.html), [Flax/Equinox how-to](https://brainx.chaobrain.com/brainstate/how_to/interoperate_with_flax_equinox.html), [PyTorch migration](https://brainx.chaobrain.com/brainstate/how_to/migrate_from_pytorch.html) |
| `skills/brainstate/references/state_collections_and_utilities.md` | Filter, organize, freeze, flatten, configure, and print nested collections | [Utility Toolkit](https://brainx.chaobrain.com/brainstate/how_to/filter_and_organize_states.html) |
| `skills/brainstate/references/collective_model_operations.md` | Initialize, reset, invoke methods, batch lifecycle operations, and restore model-wide State | [Collective Operations](https://brainx.chaobrain.com/brainstate/how_to/collective_operations.html) |
| `skills/brainstate/references/extension_mechanisms.md` | Mixins, descriptors, runtime modes, and State hooks | [Mixin System](https://brainx.chaobrain.com/brainstate/how_to/custom_states_and_mixins.html), [State Hooks](https://brainx.chaobrain.com/brainstate/how_to/state_hooks.html) |
| `skills/brainstate/references/size-inference-variations.md` | Convolution formulas and edge cases, pooling reduction, and flatten-size inference | [Common layers tutorial](https://brainx.chaobrain.com/brainstate/tutorials/core/03_common_layers.html) |
| `skills/brainstate/references/braintools-optimizer-reference.md` | Optimizer, scheduler, and external-wrapper selection | [Braintools optimization](https://brainx.chaobrain.com/braintools/optim/index.html) |
| `skills/brainstate/references/brainstate/parameter-constraints-regularization.md` | `ParamState` versus `nn.Param`, constraints/transforms, `nn.Const`, classical/structural penalties, prior regularizers, and loss integration | [Parameters tutorial](https://brainx.chaobrain.com/brainstate/tutorials/core/05_parameters_transforms_regularization.html), [constraint/regularization how-to](https://brainx.chaobrain.com/brainstate/how_to/constrain_and_regularize_parameters.html), [regularization API](https://brainx.chaobrain.com/brainstate/apis/nn/regularization.html) |
| `skills/brainstate/references/brainstate-randomness-reproducibility/randomness-and-reproducibility.md` | Randomness parent: streams, stochastic transforms, trials, dropout/noise, and checkpointed RNG State | [Randomness tutorial](https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html), [random API](https://brainx.chaobrain.com/brainstate/apis/random.html) |
| `skills/brainstate/references/libraries/prebuilt-layer-library.md` | Full layer catalog | [Linear API](https://brainx.chaobrain.com/brainstate/apis/nn/linear.html), [convolution API](https://brainx.chaobrain.com/brainstate/apis/nn/conv.html), [normalization API](https://brainx.chaobrain.com/brainstate/apis/nn/normalization.html), [pooling API](https://brainx.chaobrain.com/brainstate/apis/nn/pooling.html), [padding API](https://brainx.chaobrain.com/brainstate/apis/nn/padding.html), [dropout API](https://brainx.chaobrain.com/brainstate/apis/nn/dropout.html) |
| `skills/brainstate/references/libraries/prebuilt-activation-library.md` | Activation functions and normalization selection | [Activation API](https://brainx.chaobrain.com/brainstate/apis/nn/activation.html) |
| `skills/brainstate/references/brainstate/transformation-jit-expansion.md` | State write-back, cache/static args, compilation boundaries, and benchmarking | [JIT and Compilation](https://brainx.chaobrain.com/brainstate/tutorials/transformations/01_jit_and_compilation.html), [Transformation Essentials](https://brainx.chaobrain.com/brainstate/tutorials/core/06_transformations_essentials.html) |
| `skills/brainstate/references/brainstate/transformation-grad-expansion.md` | Autodiff, differentiable simulation, fitting, `return_value`, and `has_aux` | [Autodiff](https://brainx.chaobrain.com/brainstate/tutorials/transformations/02_autodiff.html), [Training and Metrics](https://brainx.chaobrain.com/brainstate/tutorials/core/07_training_and_metrics.html), [Parameters tutorial](https://brainx.chaobrain.com/brainstate/tutorials/core/05_parameters_transforms_regularization.html) |
| `skills/brainstate/references/brainstate/transformation-vmap-expansion.md` | State axes, ensembles, sweeps, stochastic vmap, `in_states`, and `out_states` | [Vectorization](https://brainx.chaobrain.com/brainstate/tutorials/transformations/03_vectorization.html), [Randomness](https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html) |
| `skills/brainstate/references/brainstate/brainstate-control-flow-patterns.md` | Transform-safe loops, scans, branches, and checkpointed control flow | [Control Flow](https://brainx.chaobrain.com/brainstate/tutorials/transformations/05_control_flow.html) |
| `skills/brainstate/references/diagnostics/brainstate-transformed-diagnostics.md` | Runtime checks, transformed debugging, NaN/Inf checks, callbacks, and traced values | [Error Handling and Checks](https://brainx.chaobrain.com/brainstate/tutorials/transformations/06_error_handling_and_checks.html), [Debugging](https://brainx.chaobrain.com/brainstate/tutorials/transformations/07_debugging.html) |
| `skills/brainx-acceleration-audit/SKILL.md` | Performance, batching, sweeps, memory, GPU, and multi-device work | The acceleration skill plus the transform sources it conditionally opens |

### Nested parameter

`skills/brainstate/references/brainstate/parameter-constraints-regularization.md` contains the merged regularization catalog and alone selects this exhaustive transform child:

| Nested child | Need | Crafting source |
|---|---|---|
| `skills/brainstate/references/brainstate/parameter-containers-transforms-catalog.md` | Parameter containers and transform-class selection | [parameter-container API](https://brainx.chaobrain.com/brainstate/apis/nn/parameters.html), parameters tutorial, constraint how-to |

### Nested randomness

No skill or index route may select advanced randomness directly. The randomness parent is its only inbound selection route.

| Nested child | Need | Crafting source |
|---|---|---|
| `skills/brainstate/references/brainstate-randomness-reproducibility/advanced-randomness.md` | Advanced streams, mapped randomness, key restoration, and checkpoint behavior | Same randomness corpus as the parent |

### Other parent-selected routes

| Conditional reference | Selected by | Need | Crafting source |
|---|---|---|---|
| `skills/brainstate/references/diagnostics/common-failures-index.md` | Transformed-diagnostics parent | Second-level recurring-failure routing | Existing diagnostic routing content |

- `skills/brainstate/references/diagnostics/brainstate-transformed-diagnostics.md` establishes transformed diagnostics before selecting its second-level common-failures router.

Existing dynamics, training, and split size-inference files may remain on disk,
but they are not routes in this architecture because the supplied BrainState
plan does not select them.



## BrainTrace: `skills/braintrace/SKILL.md`

| Canonical reference | Need | Crafting source |
|---|---|---|
| `skills/braintrace/references/primitive-ops-and-transforms.md` | ETP primitives, matmul/conv/sparse/LoRA/element-wise ops, transform hooks, and custom registration | [Concepts](https://brainx.chaobrain.com/braintrace/quickstart/concepts.html), [compiler internals](https://brainx.chaobrain.com/braintrace/advanced/compiler_internals.html), [ETP primitives](https://brainx.chaobrain.com/braintrace/tutorials/etp_primitives.html), [custom transforms](https://brainx.chaobrain.com/braintrace/tutorials/customizing_primitive_transforms.html), [primitives API](https://brainx.chaobrain.com/braintrace/apis/primitives.html) |
| `skills/braintrace/references/algorithms-and-customization.md` | Algorithm-by-algorithm selection and custom algorithm extension | [Algorithms API](https://brainx.chaobrain.com/braintrace/apis/algorithms.html), [custom algorithms](https://brainx.chaobrain.com/braintrace/advanced/custom_algorithms.html) |
| `skills/braintrace/references/compiler-graph-debugging.md` | `ETraceGraph`, hidden groups, relations, diagnostics, exclusions, limitations, and workarounds | [Compiler internals](https://brainx.chaobrain.com/braintrace/advanced/compiler_internals.html), [limitations](https://brainx.chaobrain.com/braintrace/advanced/limitations.html), [graph visualization](https://brainx.chaobrain.com/braintrace/tutorials/graph_visualization.html) |
| `skills/braintrace/references/state-batching-workflows.md` | Hidden-state variants, initialization/reset, single-sample mode, vmap batching, and multi-step input | [Hidden states](https://brainx.chaobrain.com/braintrace/tutorials/hidden_states.html), [batching](https://brainx.chaobrain.com/braintrace/tutorials/batching.html) |
| `skills/braintrace/references/braintools-metrics.md` | Loss and evaluation-metric selection for online learning | [Metric API](https://brainx.chaobrain.com/braintools/apis/metric.html) |
| `skills/braintrace/references/braintools-optimizer.md` | Optimizer and learning-rate schedule selection for online updates | [Optimization API](https://brainx.chaobrain.com/braintools/apis/optim.html), [optimization tutorials](https://brainx.chaobrain.com/braintools/optim/index.html) |



## Acceleration Audit: `skills/brainx-acceleration-audit/SKILL.md`

This skill owns duplicated local transform and randomness references for the exact semantics it audits.

### First-layer routes

| Route | Open when | Crafting source |
|---|---|---|
| `skills/brainstate/SKILL.md` | Any nontrivial state-aware rewrite | Owning BrainState skill |
| `skills/brainx-acceleration-audit/references/brainstate/transformation-jit-expansion.md` | JIT boundaries, static args, recompilation, or benchmarking | JIT and Transformation Essentials sources |
| `skills/brainx-acceleration-audit/references/brainstate/transformation-vmap-expansion.md` | Batch, trial, ensemble, State-axis, or RNG mapping | Vectorization and Randomness sources |
| `skills/brainx-acceleration-audit/references/brainstate/transformation-grad-expansion.md` | Finite-difference replacement, training gradients, or ParamState differentiation | Autodiff, Training, and Parameters sources |
| `skills/brainx-acceleration-audit/references/brainstate/brainstate-control-flow-patterns.md` | Time/recurrent loops, scan/for-loop/while-loop, or checkpointing | Control Flow source |
| `skills/brainx-acceleration-audit/references/brainstate-randomness-reproducibility/randomness-and-reproducibility.md` | Seed/key restoration or independent mapped randomness | Randomness corpus |

### Nested randomness 
The acceleration skill and its transform references route only to the local randomness parent. That parent alone selects:

| Nested child | Open when | Crafting source |
|---|---|---|
| `skills/brainx-acceleration-audit/references/brainstate-randomness-reproducibility/advanced-randomness.md` | Advanced stream, mapped-key, or restoration behavior | Same randomness corpus |

## General Guard: `skills/brainx-general-guard/SKILL.md`

This skill owns local copies of the randomness references it conditionally opens and keeps installation as a semantic skill route.

### First-layer routes

| Route | Open when | Crafting source |
|---|---|---|
| `skills/brainx-install/SKILL.md` | Setup, import, backend, device, version, or package mismatch | Owning installation skill |
| `skills/brainx-general-guard/references/brainstate-randomness-reproducibility/randomness-and-reproducibility.md` | Stochastic behavior, seed control, random trials, dropout/noise, or reproducibility | Randomness tutorial and API |

The guard's source anchors are [Thinking in BrainState](https://brainx.chaobrain.com/brainstate/getting_started/thinking_in_brainstate.html), [BrainCell units](https://brainx.chaobrain.com/braincell/concepts/units.html), [BrainCell mechanisms](https://brainx.chaobrain.com/braincell/concepts/mechanisms.html), and [BrainState randomness](https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html).


## Installation: `skills/brainx-install/SKILL.md`

Create a skill-local `skills/brainx-install/references/` directory.

| Canonical reference | Need | Crafting source |
|---|---|---|
| `skills/brainx-install/references/compatibility-and-release-matching.md` | Release tuples, exact/partial matching, release drift, historical/yanked releases, and compatibility evidence | [BrainX summary](https://brainx.chaobrain.com/summ/) plus the skill's compatibility-classification sections |

No nested Markdown layer is declared.
