# BrainCell Morphology IO, Loading Paths, and Validation

## Purpose

Provide a unified reference for morphology import paths, validation reports/options, NeuroMorpho caching, and checkpoints.

## Routing ownership

This is a nested morphology leaf. Open it only after `skills/braincell/references/multicompartment/multicompartment-cell-workflow.md`, either directly from that parent or through its nested manual-construction reference.

## Used by

- `skills/braincell/references/multicompartment/multicompartment-cell-workflow.md`

## Primary source pages to expand from

- [IO Overview](https://brainx.chaobrain.com/braincell/file_formats/overview.html)
- [SWC](https://brainx.chaobrain.com/braincell/file_formats/swc.html)
- [Neurolucida ASC](https://brainx.chaobrain.com/braincell/file_formats/asc.html)
- [NeuroML2](https://brainx.chaobrain.com/braincell/file_formats/neuroml2.html)
- [NeuroMorpho.Org](https://brainx.chaobrain.com/braincell/file_formats/neuromorpho.html)
- [Checkpointing](https://brainx.chaobrain.com/braincell/file_formats/checkpointing.html)
- [Morphology concept](https://brainx.chaobrain.com/braincell/concepts/morphology.html)

## Open when

- The user needs to load morphology from SWC, ASC, NeuroML2, NeuroMorpho.Org, or a BrainCell checkpoint.
- A loaded morphology needs validation before multicompartment simulation.
- The task involves validation reports, reader options, branch-type checks, cache behavior, or checkpoint compatibility.

## Role

Morphologies are required for multicompartment `Cell` models.

Single-compartment models do not need morphology.

This reference covers file/repository/checkpoint IO and validation. Manual morphology construction belongs in a morphology concepts or tree-building reference unless the task only needs a small note about hand-built `Morphology` objects.

## Loading paths

| Source | Entry point | Notes |
|---|---|---|
| SWC | `Morphology.from_swc(...)` | De-facto standard point/parent table; validate structure and branch types before simulation. |
| Neurolucida ASC | `Morphology.from_asc(...)` | Richer metadata, spines, and markers may appear; inspect report details. |
| NeuroML2 | `NeuroMlReader().read(...)` | Import the morphology portion; do not assume mechanisms or channels are imported. |
| NeuroMorpho.Org | `Morphology.from_neuromorpho(...)`, `io.load_neuromorpho(...)` | Download and cache morphology records and raw files. |
| BrainCell checkpoint | `io.save_morpho(...)`, `io.load_morpho(...)` | Reuse a cleaned or processed morphology and handle checkpoint version errors. |

## Validation and reports

- Use `return_report=True` when supported.
- Use `SwcReadOptions` to control SWC validation behavior.
- Use `SwcReader.check(...)` for validate-only workflows.
- Inspect `AscReport` and `SwcReport` before simulation.
- Treat a successful file load as a parsing result, not proof of biological or simulation validity.

## Post-load checks

- Branch types are mapped correctly.
- Soma, dendrite, and axon labels are reasonable for the model.
- Geometry is plausible.
- Topology is connected or intentionally disconnected.
- The morphology can be visualized.
- Region selectors and locsets cover the intended structure.
- Checkpoint versions are compatible.

## Handoff

After loading and validation:

1. Inspect or visualize the morphology.
2. Discretize into control volumes.
3. Define regions and locsets.
4. Paint density mechanisms and place point mechanisms.
5. Build the `Cell`.
6. Simulate and inspect probes.

## NeuroMorpho notes

Keep NeuroMorpho under this IO workflow. Its documentation covers lookup/search, loading by ID, optional validation reports, downloading raw files, and cache behavior as one import path.

## Checkpoint notes

Checkpointing belongs here because it is another morphology IO path. Use it to save a loaded or edited morphology, reload it later, and catch version-related errors before simulation.

## Common mistakes to document

- Keeping separate loading and validation workflows that duplicate the same source set.
- Expecting NeuroML2 import to include mechanisms automatically.
- Skipping validation reports after a successful parse.
- Misreading SWC structure identifiers or branch types.
- Ignoring cached or processed morphology checkpoints.
- Forgetting to inspect region and locset targeting after import.

## Example prompts this reference should support

- "Load an SWC morphology and validate it before building a Cell."
- "Should I use ASC or SWC for this morphology?"
- "Download a NeuroMorpho morphology and cache it."
- "Save a validated morphology checkpoint and reload it later."
