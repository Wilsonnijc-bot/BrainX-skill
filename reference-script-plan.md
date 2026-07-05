# Reference Script Plan

This plan inventories reusable reference scripts under `skills/` plus planned or external tutorial-script references named by skill bodies, and points each script to the closest relevant HTML documentation source when one exists.

## BrainCell

### `skills/braincell-multicompartment/references/cell_multicompartment_reference.py`
- **Status:** Local script.
- **Description:** Primary full-script reference for turning an SWC morphology into a simulation-ready multicompartment `Cell`; covers `Morphology.from_swc`, `Cell`, CVs, CV policies, `init_state`, `node_tree`, `paint`, `place`, `CurrentClamp`, `StateProbe`, and minimal `run` simulation.
- **Mirror Source URLs:** [Cell in BrainCell](https://brainx.chaobrain.com/braincell/tutorials/cell.html)

### `skills/braincell-singlecell/references/scripts/hh_neuron_basics.py`
- **Status:** Local script.
- **Description:** Default end-to-end HH point-neuron current-clamp script using `SingleCompartment`, Na/K/leak currents, `init_state`, `update(I)`, `for_loop`, and voltage/spike plotting.
- **Mirror Source URLs:** [Your First Hodgkin Huxley Neuron](https://brainx.chaobrain.com/braincell/examples/hh_neuron_basics.html)

### `skills/braincell-singlecell/references/scripts/fi_curve.py`
- **Status:** Local script.
- **Description:** Current-sweep and FI-curve reference for vectorized independent point neurons, warm-up discard, spike counting, and firing-rate extraction.
- **Mirror Source URLs:** [Frequency Current Curve](https://brainx.chaobrain.com/braincell/examples/fi_curve.html)

### `skills/braincell-singlecell/references/scripts/channel_ablation.py`
- **Status:** Local script.
- **Description:** Intact-vs-ablated point-neuron comparison reference, especially for setting channel conductance to zero while preserving the ion/channel structure.
- **Mirror Source URLs:** [Channel Ablation](https://brainx.chaobrain.com/braincell/examples/channel_ablation.html)

### `skills/braincell-singlecell/references/scripts/spike_frequency_adaptation.py`
- **Status:** Local script.
- **Description:** Advanced single-cell reference for calcium-dependent spike-frequency adaptation, dynamic calcium, `MixIons(k, ca)`, and AHP/KCa mechanisms.
- **Mirror Source URLs:** [Spike Frequency Adaptation](https://brainx.chaobrain.com/braincell/examples/spike_frequency_adaptation.html)

### `skills/braincell-singlecell/references/scripts/t_current_rebound.py`
- **Status:** Local script.
- **Description:** Advanced single-cell reference for post-inhibitory rebound, T-type calcium current, thalamic-style rebound bursting, and hyperpolarizing current protocols.
- **Mirror Source URLs:** [T Current Rebound](https://brainx.chaobrain.com/braincell/examples/t_current_rebound.html)

### `skills/braincell-singlecell/references/scripts/thalamic_neurons.py`
- **Status:** Local script.
- **Description:** Advanced phenotype-comparison script for richer thalamic point-neuron variants, multiple channel compositions, calcium dynamics, HCN/AHP/T-type mechanisms, and phenotype comparison.
- **Mirror Source URLs:** [Thalamic Neurons](https://brainx.chaobrain.com/braincell/examples/thalamic_neurons.html)

### `skills/braincell-singlecell/references/scripts/calcium_channel_gating.py`
- **Status:** Local script.
- **Description:** Channel-level diagnostic script for voltage-dependent gating curves, steady-state activation/inactivation, low-threshold vs high-threshold calcium channel comparison, and direct channel-method inspection.
- **Mirror Source URLs:** [Calcium Channel Gating](https://brainx.chaobrain.com/braincell/examples/calcium_channel_gating.html)

## BrainState

### `skills/brainstate-state-management/references/scripts/lif_neuron_model.py`
- **Status:** Local script.
- **Description:** State-management LIF example showing `HiddenState`, `ShortTermState`, `ParamState`, and explicit `.value` reads/writes in one model.
- **Mirror Source URLs:** [State Management](https://brainx.chaobrain.com/brainstate/tutorials/core/01_state_and_pytrees.html)

### `skills/brainstate-module-building/references/scripts/modern_cnn.py`
- **Status:** Local script.
- **Description:** Full module-composition reference using `Conv2d`, `BatchNorm2d`, `GELU`, `MaxPool2d`, `Linear`, `LayerNorm`, and `Dropout`.
- **Mirror Source URLs:** [Activation Functions and Normalization](https://brainx.chaobrain.com/brainstate/tutorials/core/04_activations_and_normalization.html)

### `references/brainstate-dynamics/scripts/building-ei-snn.py`
- **Status:** Local script.
- **Description:** Complete E/I spiking-network simulation workflow with `init_all_states` and compiled trajectory.
- **Mirror Source URLs:** [Building a Spiking Neural Network](https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/04_building_an_snn.html)

### `skills/brainstate-deeplearning-training/references/scripts/integrator_rnn.py`
- **Status:** Local script.
- **Description:** Stateful sequence-training reference with synthetic temporal data, custom `RNNCell`, trainable initial state, `ParamState` collection, Adam optimizer, JIT prediction/training steps, gradient update, L2 regularization, epoch loop, and evaluation plot.
- **Mirror Source URLs:** [Training Recurrent Neural Networks](https://brainx.chaobrain.com/brainstate/examples/deep_learning/integrator_rnn.html)

### `references/brainstate-dynamics/scripts/training-snn.py`
- **Status:** Local script.
- **Description:** Full representative SNN training workflow; use when the task crosses from simulation into optimization.
- **Mirror Source URLs:** [Training a Spiking Neural Network](https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/05_training_an_snn.html)

### `snn-training-example.py`
- **Status:** Planned bundled script reference.
- **Description:** Longer example-gallery SNN training script for training variants.
- **Mirror Source URLs:** [Training Spiking Neural Networks](https://brainx.chaobrain.com/brainstate/examples/brain_dynamics/snn_training.html)

### `references/brainstate-dynamics/scripts/hodgkin-huxley-neuron.py`
- **Status:** Local script.
- **Description:** Complete executable HH neuron example showing biophysical state variables and continuous-time dynamics.
- **Mirror Source URLs:** [Hodgkin-Huxley Neuron Model](https://brainx.chaobrain.com/brainstate/examples/brain_dynamics/hodgkin_huxley_neuron.html)

## BrainX Install

### `brainx-install-verify.py`
- **Status:** Planned bundled script reference.
- **Description:** Executable validation check for BrainX import and `jax.devices()`.
- **Mirror Source URLs:** [Installing the Ecosystem](https://brainx.chaobrain.com/summ/install.html)

### `brainx-install-commands.sh`
- **Status:** Planned bundled script reference.
- **Description:** Official command list for full, CPU, CUDA 12, CUDA 13, TPU, pinned, and source installs.
- **Mirror Source URLs:** [Installing the Ecosystem](https://brainx.chaobrain.com/summ/install.html)

## Skills Without Script References

These skills currently have no explicit reusable script references in their `SKILL.md` bodies:

- `brainstate-transformations-core`
- `brainunit-quantity-safety`
- `brainx-general-guard`
