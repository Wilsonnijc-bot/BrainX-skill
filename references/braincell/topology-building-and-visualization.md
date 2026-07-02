# Topology Building and Visualization Blueprint
https://brainx.chaobrain.com/braincell/tutorials/vis.html
https://brainx.chaobrain.com/braincell/tutorials/filter.html

## Purpose

Use when building, inspecting, validating, or explaining the runtime topology of a multicompartment BrainCell Cell.

## Used by

- `skills/brainx-debugging-diagnostics/SKILL.md`
- `skills/braincell-multicompartment/SKILL.md`
- `skills/morphology-building/SKILL.md`

## Open when

- The user asks how morphology becomes runtime topology.
- The user asks about NodeTree.
- The user asks about CV / branch / node views.
- The user wants to visualize morphology or runtime layout.
- The user wants to verify region/locset coverage.
- The user wants to inspect where mechanisms or probes are attached.
- The user sees unexpected simulation/probe behavior and topology may be wrong.

## Core flow

1. Build or load Morphology.
2. Choose CV policy.
3. Construct Cell.
4. paint/place declarations.
5. call cell.init_state().
6. inspect cell.node_tree.
7. visualize node / CV / branch topology.
8. optionally color by runtime values.

## Should eventually cover

- Declaration-time morphology versus runtime topology.
- NodeTree/runtime graph concept.
- CV count and CV placement.
- Branch topology after discretization.
- Region/locset target inspection.
- Runtime state, mechanism, and probe placement.
- Visualization-assisted debugging.

## Common mistakes to document

- Assuming morphology points equal runtime CVs.
- Debugging the original morphology when the problem is CV topology.
- Choosing CV policy without inspecting the result.
- Targeting a locset that does not map to the expected runtime node.
- Confusing branch names/types after import.
