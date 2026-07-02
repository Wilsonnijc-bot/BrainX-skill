---
name: skill-evolution
description: Guides iterative improvement of the BrainX skill bundle, including routing rules, reference structure, examples, validation checks, common-failure coverage, and documentation-page integration. Use when updating skills, adding new BrainX docs, refining triggers, reducing overlap, or converting repeated user failures into better skill guidance.
---

# Skill Evolution Skill

## Purpose

Guide future expansion, refactoring, and validation of this BrainX skill bundle while preserving progressive disclosure.

## When to use this skill

Use when adding new BrainX documentation, converting scripts into examples, updating references, splitting or merging skills, revising routing rules, or maintaining common failures.

## When not to use this skill

Do not use for answering ordinary user modeling questions. Use the relevant BrainX skill instead.

## What information this skill should eventually cover

- Progressive-disclosure design rules.
- Skill versus reference versus example boundaries.
- Source-page mapping.
- Trigger phrases for routing.
- Common-failure harvesting.
- Validation-check design.
- Bundle refactoring rules.
- Version/update notes.

## Expected workflow

1. Identify the new source or user pattern.
2. Decide whether it belongs in a skill, reference, example, or common-failure entry.
3. Keep skill pages lean.
4. Put detailed catalogs in references.
5. Put runnable scripts in examples/placeholders.
6. Update the top-level router.
7. Add or revise common mistakes.
8. Validate folder structure and links.

## Required / useful reference markdowns

- `references/diagnostics/common-failures-index.md`
- Any reference affected by the change.

## Common mistakes this skill should prevent

- Overfilling `SKILL.md` files with full tutorials.
- Creating duplicate skills for topics that should be references.
- Hiding essential routing rules in deep references.
- Removing shared references because multiple skills use them.
- Forgetting to update the top-level router.
- Ignoring user-provided notes from the current session.

## Placeholder for future examples or validation checks

- Bundle structure validation script.
- Markdown link checker.
- New-skill checklist.
- Common-failure update checklist.
- Source-to-skill mapping table.
