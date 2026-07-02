# BrainX Reference Index

Shared reference files live here once and may be linked by many skills. Open the smallest reference that answers the immediate question.

## BrainState

- `references/brainstate/advanced-randomness.md` — advanced BrainState RNG, RandomState, dropout, random batches, checkpointed RNG, and transformed randomness; open after the randomness skill establishes that stochastic behavior matters.
- `references/brainstate/brainstate-control-flow-patterns.md` — BrainState control-flow pattern catalog; open when loops or branches must remain valid under `jit`, `grad`, or `vmap`.
- `references/brainstate/parameter-constraints-regularization.md` — conceptual workflow for `ParamState` vs `nn.Param`, constrained values, regularization penalties, `Const`, and training-loss integration.
- `references/brainstate/parameter-containers-transforms-catalog.md` — catalog for `Param`, `Const`, and built-in parameter transforms; open when mapping constraints to transform choices.
- `references/brainstate/regularization-catalog-priors.md` — catalog for classical, structural, chained, and prior-distribution regularizers; open when mapping modeling goals to `reg=` choices.
- `references/brainstate/transformation-grad-expansion.md` — BrainState gradient expansion notes; open for `grad`, differentiable simulation, parameter fitting, or gradient target selection.
- `references/brainstate/transformation-jit-expansion.md` — BrainState JIT expansion notes; open for compilation, cache/static arguments, state write-back, or transformed simulation speedups.
- `references/brainstate/transformation-vmap-expansion.md` — BrainState vectorization expansion notes; open for `vmap`, batching, sweeps, vectorized stochastic functions, or state axes.

## Brain Dynamics

- `references/brain-dynamics/brain-dynamics-delay-protocol.md` — delay APIs and delay-buffer behavior; open for `output_delay`, `prefetch_delay`, `Delay`, `DelayAccess`, `StateWithDelay`, or biological delays.
- `references/brain-dynamics/brain-dynamics-event-driven-operators.md` — event-driven sparse spike operators; open for spike-train input, sparse connectivity, `EventLinear`, `EventFixedProb`, or scalable SNN connectivity.
- `references/brain-dynamics/brain-dynamics-snn-workflows.md` — SNN build/simulate/train workflow routes; open for spiking populations, projections, E/I networks, surrogate-gradient training, or time-unrolled SNN losses.

## BrainCell

- `references/braincell/morphology-io-loading-validation.md` — unified morphology import, validation report/options, NeuroMorpho cache, and checkpoint reference; open for SWC, ASC, NeuroML2, NeuroMorpho, validation, or branch-type inspection.
- `references/braincell/probe-reference.md` — BrainCell probe types and trace checks; open for `StateProbe`, `MechanismProbe`, `CurrentProbe`, trace keys, sampling, or missing traces.
- `references/braincell/topology-building-and-visualization.md` — runtime topology and visualization; open for NodeTree, CV/branch/node views, region/locset coverage, mechanism/probe placement, or unexpected topology behavior.

## Libraries

- `references/libraries/channel-library.md` — BrainCell channel catalog blueprint; open when selecting existing channels or checking channel/ion dependencies before authoring.
- `references/libraries/cv-policy-reference.md` — control-volume policy blueprint; open when choosing or comparing CV policies and discretization effects.
- `references/libraries/filter-function-library.md` — BrainCell spatial selector/filter blueprint; open for region, locset, branch, and morphology selection patterns.
- `references/libraries/ion-library.md` — BrainCell ion catalog blueprint; open for ion species, reversal potentials, concentration dynamics, or `MixIons`.
- `references/libraries/prebuilt-activation-library.md` — BrainState activation catalog; open when choosing prebuilt activation layers.
- `references/libraries/prebuilt-layer-library.md` — BrainState prebuilt layer catalog; open when choosing existing layers before custom modules.
- `references/libraries/solver-library-with-effects.md` — solver and integration tradeoff blueprint; open for solver choice, time-step units, stability, stiffness, or trace comparison.

## Diagnostics

- `references/diagnostics/brainstate-transformed-diagnostics.md` — transformed-code debugging and runtime checks; open for tracers, transformed prints, callbacks, breakpoints, assertions, NaN/Inf checks, or checkify-style errors.
- `references/diagnostics/common-failures-index.md` — common failure router; open when an error pattern appears or the right diagnostic reference is unclear.

## Policy

- `references/policy/` — reserved for future bundle policy and routing references that are not domain-specific.
