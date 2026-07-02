# BrainState Regularization and Prior Catalog

## Purpose

Catalog classical, structural, chained, and prior-distribution regularizers, and map modeling goals to `reg=` choices.

## Used by

- `references/brainstate/parameter-constraints-regularization.md`
- `skills/brainstate-deeplearning-training/SKILL.md`
- `skills/brainstate-brain-dynamics/SKILL.md`

## Primary source pages to expand from

- [Standard Regularizations API](https://brainx.chaobrain.com/brainstate/apis/nn/regularization.html)
- [Parameters, Transforms, and Regularization](https://brainx.chaobrain.com/brainstate/tutorials/core/05_parameters_transforms_regularization.html)
- [Constrain and Regularize Parameters](https://brainx.chaobrain.com/brainstate/how_to/constrain_and_regularize_parameters.html)

## Open when

- A loss should encode shrinkage, sparsity, smoothness, structural pressure, or a prior distribution.
- The user needs to choose a regularizer for `reg=`.
- The task is parameter fitting, MAP-style estimation, variational inference, or uncertainty-aware modeling.

## Role

A regularization object contributes a scalar penalty.

Attached to `nn.Param`, it is read with `param.reg_loss()`.

That penalty must be added to the training or fitting objective. Creating a regularized parameter is not enough by itself.

## Classical penalties

| Goal | Regularizers |
|---|---|
| Sparsity or shrinkage | `L1Reg`, `L2Reg`, `ElasticNetReg` |
| Robust shrinkage | `HuberReg` |
| Grouped sparsity | `GroupLassoReg` |
| Smoothness or local variation | `TotalVariationReg` |
| Soft magnitude constraint | `MaxNormReg` |
| Distributional entropy pressure | `EntropyReg` |
| Matrix structure | `OrthogonalReg`, `SpectralNormReg` |
| Combine multiple penalties | `ChainedReg` |

## Prior-distribution regularizers

| Parameter belief | Regularizers |
|---|---|
| Gaussian prior | `GaussianReg` |
| Heavy-tailed prior | `StudentTReg`, `CauchyReg` |
| Bounded prior | `UniformReg`, `BetaReg` |
| Positive scale prior | `LogNormalReg`, `ExponentialReg`, `GammaReg` |
| Variance prior | `InverseGammaReg` |
| Scale-invariant prior | `LogUniformReg` |
| Sparse heavy-tail prior | `HorseshoeReg` |
| Variable-selection prior | `SpikeAndSlabReg` |
| Simplex prior | `DirichletReg` |

## When to use

Use regularization when the loss should encode:

- weight decay;
- sparsity;
- smoothness;
- structural constraints;
- biological plausibility;
- MAP-style priors;
- parameter-fitting assumptions.

## When not to use

Do not introduce regularization for ordinary beginner training examples unless the task is explicitly about constrained or regularized learning.

Do not use a prior-distribution regularizer as decoration. It should encode a real modeling assumption or fitting objective.

## Common mistakes to document

- Creating a regularized `Param` but not adding `param.reg_loss()` to the objective.
- Choosing a prior regularizer without matching the parameter domain.
- Combining penalties without checking relative scale.
- Applying strong penalties to physical parameters without considering units and biological ranges.

## Example prompts this reference should support

- "Which regularizer should I use for sparse fitted parameters?"
- "How do I add L2 regularization to an `nn.Param`?"
- "What regularizer represents a positive scale prior?"
- "How do I combine several regularization penalties?"
