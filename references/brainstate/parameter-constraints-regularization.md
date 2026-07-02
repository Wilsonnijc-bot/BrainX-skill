# Parameter Constraints and Regularization Workflow Reference

## Purpose

Explain the conceptual workflow for `ParamState` versus `brainstate.nn.Param`, constrained values, regularization penalties, `Const`, and training-loss integration.

## Used by

- `skills/brainstate-deeplearning-training/SKILL.md`
- `skills/brainstate-brain-dynamics/SKILL.md`
- `skills/brainstate-module-building/SKILL.md`

## Primary source pages to expand from

- [Parameters, Transforms, and Regularization](https://brainx.chaobrain.com/brainstate/tutorials/core/05_parameters_transforms_regularization.html)
- [Constrain and Regularize Parameters](https://brainx.chaobrain.com/brainstate/how_to/constrain_and_regularize_parameters.html)

## Open when

- The user needs the high-level workflow for constrained or regularized learnable parameters.
- A model parameter must be positive, bounded, ordered, simplex-valued, or otherwise constrained.
- A loss should include a regularization penalty or MAP-style prior term.
- The distinction between bare trainable state and semantic parameter containers is unclear.

## Core distinction

`ParamState` is the normal BrainState trainable storage object. Use it for ordinary learnable arrays that do not need parameter-container semantics.

`brainstate.nn.Param` is a semantic parameter container. Use it when a trainable value needs a parameter transform, a regularization object, or parameter-specific access through `value()` and `reg_loss()`.

`brainstate.nn.Const` is a fixed parameter-like value. Use it when a value should participate in the same model structure as parameters but should not be optimized.

Do not confuse `brainstate.nn.Transform` with `brainstate.transform`. Parameter transforms map values between unconstrained optimizer space and constrained model space. `brainstate.transform` contains execution transforms such as `jit`, `grad`, `vmap`, and control-flow utilities.

## Access rule

- Optimizers update the stored trainable value, exposed by `param.val`.
- The model should read `param.value()`.
- `param.value()` may apply the transform attached through `t=`.
- `param.reg_loss()` returns the penalty from the regularization attached through `reg=`.
- A regularization penalty has no effect unless it is added to the training or fitting objective.

## Workflow

1. Start with `ParamState` for ordinary trainable weights.
2. Switch to `nn.Param` when the model-space value needs constraints or regularization.
3. Attach a transform with `t=` when the optimizer should operate in an unconstrained space but the model should see a constrained value.
4. Attach a regularization object with `reg=` when the loss should include a penalty or prior.
5. Use `param.value()` in the model computation.
6. Add `param.reg_loss()` to the objective if regularization is intended.
7. Use `nn.Const` for fixed values that should look like model parameters but not update.

## Route deeper

- Open `references/brainstate/parameter-containers-transforms-catalog.md` when choosing a specific parameter container or transform class.
- Open `references/brainstate/regularization-catalog-priors.md` when choosing a classical penalty, structural penalty, chained penalty, or prior-distribution regularizer.
- Open `references/brainstate/transformation-grad-expansion.md` when the issue is gradient target selection or differentiating a fitting objective.

## Brain dynamics examples

- Positive time constants: use `nn.Param` with a positive transform instead of unconstrained raw values.
- Positive conductances: constrain the model-space value before using it in channel or synapse calculations.
- Bounded probabilities or gates: use a bounded transform and read the constrained value in the model.
- Regularized fitted parameters: add `param.reg_loss()` to the loss term used by `grad`.

## Keep out

- Full transform-class selection tables; put those in `parameter-containers-transforms-catalog.md`.
- Full regularization-class selection tables; put those in `regularization-catalog-priors.md`.
- JIT/grad/vmap execution mechanics except routing notes.

## Common mistakes to document

- Confusing parameter transforms with program transforms.
- Updating or reading the wrong side of a transformed parameter.
- Leaving biological or physical parameters unconstrained when their domain is restricted.
- Calling `reg_loss()` but not adding it to the optimized objective.
- Adding regularization without a clear modeling or fitting reason.
- Ignoring units when learning physical parameters.

## Example prompts this reference should support

- "When should I use `ParamState` versus `nn.Param`?"
- "Make this conductance parameter positive during fitting."
- "Add L2 regularization to a BrainState model parameter."
- "Explain why my transformed parameter has `.val` and `value()`."
