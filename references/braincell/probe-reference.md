# Probe Reference Blueprint

## Purpose
mostly refer to here
https://brainx.chaobrain.com/braincell/tutorials/mech.html

Use when the user wants to record, sample, inspect, or debug BrainCell runtime state, mechanism state, or currents.

## Used by

- `skills/braincell-singlecell/SKILL.md`
- `skills/braincell-multicompartment/SKILL.md`
- `references/diagnostics/common-failures-index.md`

## Probe types

- StateProbe.
- MechanismProbe.
- CurrentProbe.

## Core API

- `cell.place(locset, StateProbe(...))`
- `cell.place(locset, MechanismProbe(...))`
- `cell.place(locset, CurrentProbe(...))`
- `cell.init_state()`
- `cell.sample_probe(...)`
- `cell.sample_probes(...)`
- `cell.run(...).traces`

## Debug checks

- Was the probe placed before `init_state`?
- Is the locset correct?
- Is the probe name/key correct?
- Is the mechanism name correct?
- Is the field actually stored as runtime State?
- Is the current being requested from mechanism owner or ion owner?
- Are duplicate names causing ambiguous traces?

## Should eventually cover

- Voltage recording patterns.
- Current recording patterns.
- Gating/mechanism-state recording patterns.
- Probe placement rules.
- Trace naming conventions.
- Sampling versus full run traces.
- Probe placement relative to locsets and runtime topology.

## Common mistakes to document

- Using the wrong probe type.
- Expecting probes to record every variable automatically.
- Placing a probe at an invalid locset.
- Forgetting to sample/read the trace after simulation.
- Confusing mechanism state with cell state.
- Assuming probes are automatic.

## Placeholder examples

- Voltage trace probe.
- Mechanism state probe.
- Current probe.
- Probe trace lookup.
- Empty trace diagnostic checklist.
