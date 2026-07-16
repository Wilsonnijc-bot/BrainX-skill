# Reference Script Plan

This is the normalized script inventory for the bundle. The current repository structure is authoritative for local paths. Each owning `SKILL.md` and the references it routes are authoritative for script purpose, selection, and upstream source.

## Normalization Rules

Apply these rules before reporting a missing path or duplicate:

1. Use the current repository path, not an older skill slug or pre-consolidation directory name.
2. Resolve every `references/...` path under the owning skill. Duplicate reusable content for every consuming skill; repository-root copies belong only to the root workspace router.
3. Preserve each selected reference subtree inside the consuming skill. For example, scripts selected by `references/brainstate-dynamics/` stay under that same `brainstate-dynamics/scripts/` subtree after duplication.
4. Treat a final/default bundle list as the default. Earlier scripts remain non-default routes unless explicitly excluded.
5. When multiple skills select the same official workflow, keep a separate package-local copy for each consumer.
6. Keep inline `Script` blocks inline unless the skill or a routed reference explicitly names a full bundled script.
7. When both a gallery page and a direct `.py` URL are given, the gallery selects the workflow and the direct `.py` is the code source.
8. Preserve upstream subdirectories, helper imports, data dependencies, filenames, and license headers.

Canonical path normalization used by this plan:

| Older or shorthand route | Canonical repository route |
|---|---|
| `skills/braincell-singlecell/...` | `skills/braincell/...` |
| `skills/braincell-multicompartment/...` | `skills/braincell/references/multicompartment/...` |
| `skills/brainunit-quantity-safety/...` | `skills/brainunit/...` |
| BrainPy `NEST-Compatible.md` | `skills/brainpy/NEST-compatible/nest-workflow.md` |

## Inventory Summary

| Skill | Normalized script requirement |
|---|---|
| Bundle router | 3 existing workspace dynamics-script copies, reached through the workspace dynamics parent reference |
| BrainCell | 8 existing selected scripts; 1 planned custom-channel script; solver/network pages remain routes |
| BrainEvent | 2 existing selected application scripts |
| BrainMass | 11 selected scripts: 1 existing and 10 planned; 1 additional optional baseline |
| BrainPy | 7 planned native-gallery scripts, 7 planned NEST-compatible scripts, and 1 existing local `training-snn.py` copy |
| BrainState | 7 existing selected scripts |
| BrainTrace | Planned: 6 default, 2 non-default quickstarts, 3 on-demand operator scripts, 2 optional heavy scripts, plus imported helpers |
| BrainUnit | No external script; 5 sourced inline examples |
| Acceleration audit | No fixed script |
| General guard | No external script |
| BrainX Install | No external script |

## Workspace Router

The root `SKILL.md` contains no executable workflow, but its `references/brainstate-dynamics/dynamics-and-integration.md` copy routes to these workspace-local scripts. They duplicate the BrainState skill's selected dynamics scripts and preserve the same source subtree.

| Canonical script | Role | Crafting source |
|---|---|---|
| `references/brainstate-dynamics/scripts/building-ei-snn.py` | Workspace E/I SNN simulation copy | [Building an SNN](https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/04_building_an_snn.html) |
| `references/brainstate-dynamics/scripts/training-snn.py` | Workspace surrogate-gradient SNN training copy | [Training an SNN](https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/05_training_an_snn.html) |
| `references/brainstate-dynamics/scripts/hodgkin-huxley-neuron.py` | Workspace biophysical dynamics copy | [Hodgkin-Huxley Neuron](https://brainx.chaobrain.com/brainstate/examples/brain_dynamics/hodgkin_huxley_neuron.html) |

## BrainCell

### Existing selected scripts

| Canonical script | Role | Crafting source |
|---|---|---|
| `skills/braincell/references/scripts/hh_neuron_basics.py` | Canonical point-neuron HH current clamp | [HH Neuron Basics](https://brainx.chaobrain.com/braincell/examples/hh_neuron_basics.html) |
| `skills/braincell/references/scripts/fi_curve.py` | Vectorized FI curve and transient-aware rate extraction | [FI Curve](https://brainx.chaobrain.com/braincell/examples/fi_curve.html) |
| `skills/braincell/references/scripts/channel_ablation.py` | Intact-versus-zero-conductance comparison | [Channel Ablation](https://brainx.chaobrain.com/braincell/examples/channel_ablation.html) |
| `skills/braincell/references/scripts/spike_frequency_adaptation.py` | Dynamic calcium, `MixIons`, and AHP/KCa adaptation | [Spike Frequency Adaptation](https://brainx.chaobrain.com/braincell/examples/spike_frequency_adaptation.html) |
| `skills/braincell/references/scripts/t_current_rebound.py` | T-type calcium rebound workflow | [T Current Rebound](https://brainx.chaobrain.com/braincell/examples/t_current_rebound.html) |
| `skills/braincell/references/scripts/thalamic_neurons.py` | Thalamic phenotype/channel comparison | [Thalamic Neurons](https://brainx.chaobrain.com/braincell/examples/thalamic_neurons.html) |
| `skills/braincell/references/scripts/calcium_channel_gating.py` | Channel gating diagnostic | [Calcium Channel Gating](https://brainx.chaobrain.com/braincell/examples/calcium_channel_gating.html) |
| `skills/braincell/references/multicompartment/references/cell_multicompartment_reference.py` | Routed morphology-to-`Cell` workflow | `skills/braincell/references/multicompartment/multicompartment-cell-workflow.md` and [Cell in BrainCell](https://brainx.chaobrain.com/braincell/tutorials/cell.html) |

### Planned routed custom-channel script

Place the retained full script in the existing BrainCell script namespace rather than creating a second script tree under repository-root references. Keep the custom HH and registry examples inline in the routed reference.

| Canonical target | Role | Crafting source |
|---|---|---|
| `skills/braincell/references/scripts/manual-icat-hp1992-authoring.py` | Direct `Channel` subclass with states, derivatives, and current | `skills/braincell/references/braincell/braincell-custom-ion-channel-authoring.md` and [Channels](https://brainx.chaobrain.com/braincell/tutorials/channel.html) |

Do not create `integration_methods.py` or `ei_network.py` by default. Solver comparison stays at [Integration Methods](https://brainx.chaobrain.com/braincell/examples/integration_methods.html). BrainCell owns no SNN-workflow reference; general E/I network work routes to `skills/brainpy/SKILL.md`.

The manual-morphology reference names future blueprint ideas but no full bundled scripts, so it contributes no script requirement yet.

## BrainEvent

| Canonical script | Role | Crafting source |
|---|---|---|
| `skills/brainevent/references/scripts/102_EI_net_1996.py` | Explicitly selected high-level BrainPy-State E/I application using `EventFixedProb` | `skills/brainevent/SKILL.md` and [raw upstream script](https://raw.githubusercontent.com/chaobrain/brainpy.state/main/examples/brainpy_like/102_EI_net_1996.py) |
| `skills/brainevent/references/scripts/204_joglekar_2018_propagation.py` | Explicitly selected direct `BinaryArray @ JITCScalarC` application under `vmap` | `skills/brainevent/SKILL.md` and [raw upstream script](https://raw.githubusercontent.com/chaobrain/brainpy.state/main/examples/brainpy_like/204_joglekar_2018_propagation.py) |

COO/CSR/CSC, generated-connectivity, fixed-count, plasticity, and custom-operator material remains inline or routed through the existing BrainEvent markdown references. No additional full script is selected.

## BrainMass

Use `skills/brainmass/references/scripts/` for every selected BrainMass script.

| Canonical script | Status and role | Crafting source |
|---|---|---|
| `gradient-free-fitting.py` | Existing; Nevergrad and derivative-free SciPy fitting | [Gradient-Free Fitting](https://brainx.chaobrain.com/brainmass/tutorials/07_gradient_free_fitting.html) |
| `resting-state-meg-whole-brain-pipeline.py` | Planned; connectome-to-MEG whole-brain workflow | [Resting-State MEG](https://brainx.chaobrain.com/brainmass/gallery/case_studies/resting_state_meg.html) |
| `eeg-fitting-with-gradients.py` | Planned; data-facing gradient fitting | [EEG Fitting](https://brainx.chaobrain.com/brainmass/gallery/case_studies/eeg_fitting.html) |
| `seizure-epileptor-case-study.py` | Planned; Epileptor disease dynamics | [Seizure Epileptor](https://brainx.chaobrain.com/brainmass/gallery/case_studies/seizure_epileptor.html) |
| `wong-wang-decision-making.py` | Planned; stochastic decision workflow | [Decision Making](https://brainx.chaobrain.com/brainmass/gallery/case_studies/decision_making.html) |
| `horn-cognitive-task-training.py` | Planned; HORN task-training workflow | [HORN Cognitive Task](https://brainx.chaobrain.com/brainmass/gallery/case_studies/horn_cognitive_task.html) |
| `hopf-bifurcation-single-node.py` | Planned; minimal oscillator anchor | [Hopf Model](https://brainx.chaobrain.com/brainmass/gallery/model_zoo/hopf.html) |
| `wilson-cowan-ei-dynamics.py` | Planned; representative E/I rate model | [Wilson-Cowan Model](https://brainx.chaobrain.com/brainmass/gallery/model_zoo/wilson_cowan.html) |
| `jansen-rit-eeg-proxy.py` | Planned; cortical-column EEG proxy | [Jansen-Rit Model](https://brainx.chaobrain.com/brainmass/gallery/model_zoo/jansen_rit.html) |
| `kuramoto-synchronization.py` | Planned; phase synchronization | [Kuramoto Model](https://brainx.chaobrain.com/brainmass/gallery/model_zoo/kuramoto.html) |
| `wong-wang-dmf-resting-state.py` | Planned; E/I dynamic mean field | [Wong-Wang E/I Model](https://brainx.chaobrain.com/brainmass/gallery/model_zoo/wong_wang_exc_inh.html) |
| `linear-baseline-node.py` | Optional; analytical sanity check | [Linear Model](https://brainx.chaobrain.com/brainmass/gallery/model_zoo/linear.html) |

Quickstart, noise, batching, small-network, forward-model, and gradient-fitting fragments remain in the skill body. They do not create additional files.

## BrainPy

Use `skills/brainpy/references/scripts/`. The gallery selects the seven scripts; each direct GitHub file is the code source.

| Canonical script | Role | Code source |
|---|---|---|
| `103_COBA_2005.py` | Default full E/I COBA simulation | [source](https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/103_COBA_2005.py) |
| `104_CUBA_2005_version2.py` | Explicit E→E, E→I, I→E, I→I pathways | [source](https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/104_CUBA_2005_version2.py) |
| `brunel.py` | Builder/Network script route | [source](https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/brunel.py) |
| `106_COBA_HH_2007.py` | Custom HH neuron in a COBA network | [source](https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/106_COBA_HH_2007.py) |
| `107_gamma_oscillation_1996.py` | Custom neuron/synapse paper reproduction | [source](https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/107_gamma_oscillation_1996.py) |
| `109_fast_global_oscillation.py` | `DeltaProj` with delayed feedback | [source](https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/109_fast_global_oscillation.py) |
| `201_surrogate_grad_lif_fashion_mnist.py` | Real-data surrogate-gradient SNN training | [source](https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/201_surrogate_grad_lif_fashion_mnist.py) |

The inline surrogate-gradient route uses the local duplicate `skills/brainpy/references/brainstate-dynamics/scripts/training-snn.py`, crafted from [Training an SNN](https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/05_training_an_snn.html).

### NEST-compatible branch

`skills/brainpy/NEST-compatible/nest-workflow.md` is the existing NEST workflow reference. Place its selected full scripts under `skills/brainpy/NEST-compatible/references/scripts/`; use the [NEST gallery](https://brainx.chaobrain.com/brainpy-state/examples/nest-gallery.html) as selection context and each direct GitHub file as code authority.

| Canonical script | Role | Code source |
|---|---|---|
| `brunel_alpha.py` | Brunel network with alpha-shaped postsynaptic currents | [source](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/brunel_alpha.py) |
| `brunel_delta.py` | Brunel network with delta-shaped postsynaptic events | [source](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/brunel_delta.py) |
| `brette_et_al_2007.py` | Representative NEST-compatible network benchmark/reproduction | [source](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/brette_et_al_2007.py) |
| `synapsecollection.py` | Realized-connectivity and `SynapseCollection` inspection | [source](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/synapsecollection.py) |
| `evaluate_tsodyks2_synapse.py` | Tsodyks2 short-term synapse evaluation | [source](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/evaluate_tsodyks2_synapse.py) |
| `clopath_synapse_spike_pairing.py` | Clopath plasticity spike-pairing workflow | [source](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/clopath_synapse_spike_pairing.py) |
| `spatial_gaussex.py` | Spatial Gaussian/exponential connectivity workflow | [source](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/spatial_gaussex.py) |

These seven scripts are a separate NEST-compatible selection. Do not merge them with the seven native BrainPy-style scripts above.

## BrainState

| Canonical script | Role | Crafting source |
|---|---|---|
| `skills/brainstate/references/lif_neuron_model.py` | State-management example | [State and PyTrees](https://brainx.chaobrain.com/brainstate/tutorials/core/01_state_and_pytrees.html) |
| `skills/brainstate/references/modern_cnn.py` | CNN module composition | [Activations and Normalization](https://brainx.chaobrain.com/brainstate/tutorials/core/04_activations_and_normalization.html) |
| `skills/brainstate/references/resnet.py` | Residual composition and child registration | Existing local script explicitly selected by `skills/brainstate/SKILL.md` |
| `skills/brainstate/references/deeplearning-training/references/scripts/integrator_rnn.py` | Stateful sequence training | [Integrator RNN](https://brainx.chaobrain.com/brainstate/examples/deep_learning/integrator_rnn.html) |
| `skills/brainstate/references/brainstate-dynamics/scripts/building-ei-snn.py` | E/I SNN simulation | [Building an SNN](https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/04_building_an_snn.html) |
| `skills/brainstate/references/brainstate-dynamics/scripts/training-snn.py` | Surrogate-gradient SNN training | [Training an SNN](https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/05_training_an_snn.html) |
| `skills/brainstate/references/brainstate-dynamics/scripts/hodgkin-huxley-neuron.py` | Biophysical dynamics | [Hodgkin-Huxley Neuron](https://brainx.chaobrain.com/brainstate/examples/brain_dynamics/hodgkin_huxley_neuron.html) |

[Training Spiking Neural Networks](https://brainx.chaobrain.com/brainstate/examples/brain_dynamics/snn_training.html) remains external-only because the deep-learning reference explicitly says no local duplicate is bundled.

## BrainTrace

Use `skills/braintrace/references/scripts/` and preserve the upstream `examples/...` subtree.

All BrainTrace targets and helper dependencies below are planned; none is currently bundled.

### Planned default bundle

| Canonical target | Role | Code source |
|---|---|---|
| `examples/drtrl/09-classification-mnist.py` | D_RTRL classification | [source](https://github.com/chaobrain/braintrace/blob/main/examples/drtrl/09-classification-mnist.py) |
| `examples/pp_prop/12-classification-neuromorphic.py` | pp_prop / ES_D_RTRL classification | [source](https://github.com/chaobrain/braintrace/blob/main/examples/pp_prop/12-classification-neuromorphic.py) |
| `examples/drtrl/02-batching-vmap.py` | Vmap batching | [source](https://github.com/chaobrain/braintrace/blob/main/examples/drtrl/02-batching-vmap.py) |
| `examples/pp_prop/06-batching-batched.py` | Batched primitive | [source](https://github.com/chaobrain/braintrace/blob/main/examples/pp_prop/06-batching-batched.py) |
| `examples/pp_prop/14-knob-vjp-method-contrast.py` | VJP-method contrast | [source](https://github.com/chaobrain/braintrace/blob/main/examples/pp_prop/14-knob-vjp-method-contrast.py) |
| `examples/drtrl/11-knob-fast-solve.py` | `fast_solve` performance knob | [source](https://github.com/chaobrain/braintrace/blob/main/examples/drtrl/11-knob-fast-solve.py) |

### Planned non-default full workflows

| Canonical target | Route | Source |
|---|---|---|
| `rnn-online-learning.py` | Core RNN/D_RTRL quickstart | [RNN Online Learning](https://brainx.chaobrain.com/braintrace/quickstart/rnn_online_learning.html) |
| `snn-online-learning.py` | Core SNN/ES_D_RTRL quickstart | [SNN Online Learning](https://brainx.chaobrain.com/braintrace/quickstart/snn_online_learning.html) |
| `examples/drtrl/07-operator-lora.py` | On-demand LoRA operator | [source](https://github.com/chaobrain/braintrace/blob/main/examples/drtrl/07-operator-lora.py) |
| `examples/pp_prop/09-operator-sparse.py` | On-demand sparse/masked operator | [source](https://github.com/chaobrain/braintrace/blob/main/examples/pp_prop/09-operator-sparse.py) |
| `examples/pp_prop/11-operator-conv.py` | On-demand convolution operator | [source](https://github.com/chaobrain/braintrace/blob/main/examples/pp_prop/11-operator-conv.py) |
| `examples/003-snn-memory-and-speed-evaluation-all.py` | Optional heavy benchmark | [source](https://github.com/chaobrain/braintrace/blob/main/examples/003-snn-memory-and-speed-evaluation-all.py) |
| `examples/pp_prop/04-neurons-coba-ei-rsnn.py` | Optional COBA/EI architecture | [source](https://github.com/chaobrain/braintrace/blob/main/examples/pp_prop/04-neurons-coba-ei-rsnn.py) |

Planned support dependencies for the six-file default bundle:

| Default script | Required helper |
|---|---|
| `examples/drtrl/09-classification-mnist.py` | None |
| `examples/pp_prop/12-classification-neuromorphic.py` | `examples/pp_prop/_shared.py` |
| `examples/drtrl/02-batching-vmap.py` | `examples/drtrl/_shared.py` |
| `examples/pp_prop/06-batching-batched.py` | `examples/pp_prop/_shared.py` |
| `examples/pp_prop/14-knob-vjp-method-contrast.py` | `examples/pp_prop/_shared.py` |
| `examples/drtrl/11-knob-fast-solve.py` | `examples/drtrl/_shared.py` |

Craft the helpers from the exact sibling sources named in the BrainTrace skill: [D_RTRL `_shared.py`](https://github.com/chaobrain/braintrace/blob/main/examples/drtrl/_shared.py) and [pp_prop `_shared.py`](https://github.com/chaobrain/braintrace/blob/main/examples/pp_prop/_shared.py). The four minimal basics explicitly excluded by the skill remain excluded.

## Skills With Inline Examples But No Reference Script

### BrainUnit

Keep its five source-backed examples inline: quantity creation, arithmetic/dimension mismatch, conversion, unit-aware math/JIT, and `@u.check_units`. All are crafted from [BrainUnit Quickstart](https://brainx.chaobrain.com/brainunit/getting_started/quickstart.html). Do not invent `quantity_safety_quickstart.py`.

### BrainX Acceleration Audit

No fixed script. It applies its local transform-reference copies to the user's actual hot path and creates task-specific validation with that rewrite.

### BrainX General Guard

No external script. Its JIT, stateful-loop, unit, and routing fragments remain inline guard examples.



