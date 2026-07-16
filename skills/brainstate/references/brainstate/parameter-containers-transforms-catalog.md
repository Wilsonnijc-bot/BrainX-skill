# BrainState Parameter Containers and Transform Catalog

## Purpose

Catalog `Param`, `Const`, and built-in parameter transforms, and map common constraint needs to transform choices.

## Used by

- `references/brainstate/parameter-constraints-regularization.md`

This catalog has no other selector. The parameter-and-regularization parent establishes the task before opening it.

## Primary source pages to expand from

- [Parameter Containers API](https://brainx.chaobrain.com/brainstate/apis/nn/parameters.html)
- [Parameters, Transforms, and Regularization](https://brainx.chaobrain.com/brainstate/tutorials/core/05_parameters_transforms_regularization.html)
- [Constrain and Regularize Parameters](https://brainx.chaobrain.com/brainstate/how_to/constrain_and_regularize_parameters.html)

## Open when

- A parameter needs positivity, boundedness, ordering, simplex structure, unit norm, masking, or composed transforms.
- The user needs a catalog of available parameter containers or transform classes.
- The agent might otherwise invent a custom transform instead of using BrainState's built-ins.

## Role

`ParamState` is bare trainable storage.

`brainstate.nn.Param` is a semantic parameter container. It can attach a transform with `t=` and a regularizer with `reg=`.

`brainstate.nn.Const` is a fixed parameter-like value.

`brainstate.nn.Transform` classes are parameter transforms. They are not execution transforms like `brainstate.transform.jit`, `brainstate.transform.grad`, or `brainstate.transform.vmap`.

## Important access rule

- Optimizers update `param.val`.
- Model code reads `param.value()`.
- `param.value()` returns the model-space value, after applying the transform when one is attached.
- Use `param.reg_loss()` only when a regularizer is attached and the loss should include that penalty.

## Transform families

| Need | Use |
|---|---|
| No constraint | `IdentityT` |
| Positive value | `SoftplusT`, `ExpT`, `PositiveT`, `ReluT` |
| Negative value | `NegSoftplusT`, `NegativeT` |
| Bounded interval | `SigmoidT`, `ClipT` |
| Symmetric bounded interval | `TanhT`, `SoftsignT`, `ScaledSigmoidT` |
| Probability vector | `SimplexT` |
| Unit vector | `UnitVectorT` |
| Ordered values | `OrderedT` |
| Rescale or reparameterize | `AffineT`, `PowerT`, `LogT` |
| Compose transforms | `ChainT` |
| Transform only some entries | `MaskedT` |

## Brain dynamics examples

- Positive time constant: use a positive transform such as `SoftplusT` with an appropriate lower bound when needed.
- Positive conductance: use `PositiveT` or `SoftplusT`.
- Probability or mixture weights: use `SimplexT`.
- Bounded gating parameter: use `SigmoidT` or another bounded transform.
- Ordered thresholds: use `OrderedT`.

## Selection notes

- Prefer the least surprising transform that matches the model domain.
- Use `ChainT` only when one transform cannot express the constraint clearly.
- Use `MaskedT` when only part of a parameter should be constrained.
- Keep unit-aware BrainCell parameters unit-safe; a transform does not replace correct BrainUnit quantities.

## Common mistakes to document

- Reading `param.val` in model code when the constrained value should come from `param.value()`.
- Treating `nn.Transform` as a program transform.
- Hand-rolling a transform when a built-in BrainState transform exists.
- Choosing a hard clip when a smooth transform is needed for optimization.
- Choosing a regularizer without returning to `parameter-constraints-regularization.md` to integrate `reg_loss()` into the objective.

## Example prompts this reference should support

- "Which transform should I use for a positive time constant?"
- "How do I make a parameter a probability simplex?"
- "What is the difference between `Param`, `Const`, and `ParamState`?"
- "How do I compose transforms for a constrained fitted parameter?"
