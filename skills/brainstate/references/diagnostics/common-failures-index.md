# BrainState Common Failures Index

Open this second-level router only from `brainstate-transformed-diagnostics.md`, after a transformed diagnostic has exposed the runtime symptom. This file maps the symptom to its first owner; it does not repeat the parent's APIs or scripts.

## Return to the diagnostics parent

| Runtime symptom | Parent-owned route | Source |
|---|---|---|
| Plain `print` runs once or shows a tracer instead of a concrete value | Runtime-value printing | Debugging |
| An ordinary Python `assert` or value-dependent `if` receives an abstract tracer | Inspectable runtime checks, or a runtime exception when the step must stop | Checks |
| A NaN or Inf appears far downstream and the producing primitive is unknown | First-nonfinite-value diagnosis | Checks |
| Division by zero or out-of-bounds indexing silently produces a later meaningless result | Built-in division or index checks | Checks |
| Rich Python-side inspection is needed, or a callback was incorrectly expected to return a value to the computation | Runtime callback inspection | Debugging |
| A breakpoint stops every iteration, never opens, or should stop only on a rare bad value | Predicate-controlled runtime breakpoint | Debugging |
| A `State` buffer drifts, appears not to update, or its transformed value is otherwise unclear | Runtime prints immediately before and after the mutation | Debugging |
| Values inside transformed `grad` or `vmap` are not visible, or mapped prints are mistaken for one global print | Runtime printing inside the relevant transform | Debugging |

## Hand off after the symptom is visible

| Confirmed ownership problem | Correct local owner | Minimal handoff |
|---|---|---|
| The traced Python `if` is model control flow, not merely a failed assertion | [`../brainstate/brainstate-control-flow-patterns.md`](../brainstate/brainstate-control-flow-patterns.md) | Replace the value-dependent Python branch with the matching transform-safe branch or loop pattern. |
| Python configuration that should determine trace-time structure arrives as a tracer | [`../brainstate/transformation-jit-expansion.md`](../brainstate/transformation-jit-expansion.md) | Use the owner's static-specialization path. |
| Runtime inspection confirms the body computes a `State` update, but write-back is lost across a raw `jax.jit` boundary | [`../brainstate/transformation-jit-expansion.md`](../brainstate/transformation-jit-expansion.md) | Move the stateful boundary to BrainState's transform, or make the raw JAX boundary pure with state passed and returned explicitly. |

The checks page establishes that array-dependent Python `assert` and `if` see abstract tracers and that division, indexing, NaN, and Inf failures need runtime-aware checks. The debugging page establishes runtime visibility for concrete values, callbacks, conditional breakpoints, and `State` values before and after mutation. Keep tool selection and usage in the parent.

## Official sources

- Checks: https://brainx.chaobrain.com/brainstate/tutorials/transformations/06_error_handling_and_checks.html
- Debugging: https://brainx.chaobrain.com/brainstate/tutorials/transformations/07_debugging.html
