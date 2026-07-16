# BrainState Transformed Diagnostics Blueprint

## Purpose

Collect runtime debugging, checking, and error-handling patterns for BrainState code running under transformations.

## Used by

- `skills/brainstate/SKILL.md`
- `skills/brainx-acceleration-audit/SKILL.md`
- `references/diagnostics/common-failures-index.md`

## Primary source pages to expand from

- https://brainx.chaobrain.com/brainstate/tutorials/transformations/06_error_handling_and_checks.html
- https://brainx.chaobrain.com/brainstate/tutorials/transformations/07_debugging.html

## Should eventually cover

- Why ordinary `print` inside JIT shows trace-time values or tracers.
- Runtime printing with transform-compatible debug tools.
- Callback-based inspection.
- Conditional breakpoints.
- Debugging under JIT.
- Debugging under grad.
- Debugging under vmap.
- Debugging state values inside transformed functions.
- Why Python assertions can fail or become misleading under transforms.
- Runtime checks that work under JIT/grad/vmap.
- NaN/Inf detection.
- Invalid value checks.
- Conditional runtime error patterns.
- Checkify-style transformed error handling.
- When to fail fast versus collect diagnostics.

## Common mistakes to document

- Trusting plain `print` inside transformed code.
- Inspecting only Python objects instead of runtime values.
- Python `assert` on traced values.
- Python `if` for runtime array checks.
- Letting NaNs propagate through long simulations.
- Debugging stochastic NaNs without fixing RNG reproducibility.
- Debugging a large transformed function before making a minimal case.

## Placeholder examples

- Runtime print example.
- Callback inspection example.
- Conditional breakpoint example.
- Transformed state inspection example.
- NaN check.
- Bounds check.
- Invalid index check.
- Transformed assertion blueprint.
