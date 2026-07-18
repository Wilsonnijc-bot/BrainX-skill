# BrainState Parameter Containers and Transform Catalog

This is an exhaustive selection reference for BrainState parameter containers and built-in parameter transforms. Open it only from `references/brainstate/parameter-constraints-regularization.md`, after that parent has established why a semantic parameter container is needed. Do not open it directly from the main skill or another reference.

This file does not re-decide `ParamState` versus `nn.Param`, enumerate regularizers, or integrate penalties into a loss. Return to the parent for those tasks. Parameter transforms here are `brainstate.nn` value transforms, not execution transforms such as `brainstate.transform.jit`, `grad`, or `vmap`.

## Container catalog

| Container | Exact documented role | Selection distinction |
|---|---|---|
| `nn.Param` | A neural-network parameter container with optional transform and regularization. | Select only after the parent has established a need for the richer parameter contract. Its transformed value may be cached. |
| `nn.Const` | A non-trainable constant parameter. | It has `fit=False` and is not collected when gathering `ParamState`s; use it for a fixed value that must remain in the module tree. Its transformed value may also be cached. |

Source: https://brainx.chaobrain.com/brainstate/apis/nn/parameters.html

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/05_parameters_transforms_regularization.html

## Transform contract

`nn.Transform` is the abstract base class for bijective parameter transformations. The built-ins map between unconstrained and constrained spaces and implement `forward()`, `inverse()`, and optionally `log_abs_det_jacobian()` for probabilistic applications. With `nn.Param`, the optimizer-facing array is unconstrained and `value()` applies the forward map; `set_value()` applies the inverse when storing a constrained value back.

Constructor notation below is deliberately source-limited. A call form appears only when one of the three routed pages publishes it. **Not published** means to inspect the installed-version API before constructing that class; do not infer arguments from a neighboring transform.

Source: https://brainx.chaobrain.com/brainstate/apis/nn/parameters.html

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/05_parameters_transforms_regularization.html

## Complete built-in transform selection

### Identity, interval, and rescaling transforms

| Class | Routed call form | Exact official selection distinction |
|---|---|---|
| `IdentityT` | Not published | Identity transformation: no constraint. It is the default transform, under which the stored and model values coincide. |
| `ClipT` | `ClipT(lower, upper)` | Clips to specified bounds. The tutorial groups it with bounded-interval transforms; distinguish it from the smooth open-interval mapping documented for `SigmoidT`. |
| `AffineT` | `AffineT(scale, shift)` | Linear scaling and shifting; select for a linear rescale or reparameterization. |
| `SigmoidT` | `SigmoidT(lower, upper)` | Maps unbounded values to the open interval `(lower, upper)`. Large magnitudes saturate toward the bounds without crossing them. |
| `TanhT` | Not published | Maps `(-inf, +inf)` to `(lower, upper)` using tanh; the tutorial groups it with symmetric bounded ranges. |
| `SoftsignT` | Not published | Maps `(-inf, +inf)` to `(lower, upper)` using softsign; the tutorial groups it with symmetric bounded ranges. |
| `ScaledSigmoidT` | Not published | A sigmoid transform with adjustable sharpness/temperature; the tutorial groups it with symmetric bounded ranges. |
| `PowerT` | Not published | A power (Box-Cox) transformation for stabilizing variance; the tutorial classifies it as a reparameterization. |

Source: https://brainx.chaobrain.com/brainstate/apis/nn/parameters.html

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/05_parameters_transforms_regularization.html

Source: https://brainx.chaobrain.com/brainstate/how_to/constrain_and_regularize_parameters.html

### Positive and negative transforms

| Class | Routed call form | Exact official selection distinction |
|---|---|---|
| `SoftplusT` | `SoftplusT(lower)` | Maps the real line to `(lower, infinity)`. Select when the value must remain strictly above a configurable lower bound. |
| `NegSoftplusT` | Not published | Maps unbounded values to a negative semi-infinite interval. |
| `LogT` | Not published | Its documented direction is `(lower, +inf)` to `(-inf, +inf)`; it is the constrained-to-unconstrained direction, unlike the positive forward maps below. |
| `ExpT` | `ExpT(lower)` | Maps `(-inf, +inf)` to `(lower, +inf)`; select for a positive forward value using an exponential map. |
| `ReluT` | Not published | A lower-bounded ReLU transform whose documented forward rule is `forward(x) = relu(x) + lower_bound`. |
| `PositiveT` | Not published | Constrains values to the strictly positive interval `(0, +inf)`. |
| `NegativeT` | Not published | Constrains values to the strictly negative interval `(-inf, 0)`. |

`SoftplusT(lower)` is the documented smooth positive-domain example: even a very negative optimizer value produces a model value just above `lower`. Do not replace that guarantee with a hard clip when the parent requires a smooth bijection.

Source: https://brainx.chaobrain.com/brainstate/apis/nn/parameters.html

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/05_parameters_transforms_regularization.html

Source: https://brainx.chaobrain.com/brainstate/how_to/constrain_and_regularize_parameters.html

### Structured and composite transforms

| Class | Routed call form | Exact official selection distinction |
|---|---|---|
| `OrderedT` | Not published | Produces monotonically increasing entries. |
| `SimplexT` | `SimplexT()` | A stick-breaking transform whose model value has non-negative entries summing to one. The how-to's length-three unconstrained example returns four probabilities, so do not assume it preserves the last-axis length. |
| `UnitVectorT` | Not published | Produces a vector with L2 norm equal to one. |
| `ChainT` | `ChainT(t1, t2, ...)` | Composes multiple transforms and applies them sequentially in the order supplied. |
| `MaskedT` | Not published | Applies a transform selectively under a boolean mask. |

Bundle composition with its documented script:

```python
chained = nn.ChainT(nn.AffineT(scale=2.0, shift=1.0), nn.SoftplusT(lower=0.0))
chained.forward(jnp.array(0.0))
```

The affine transform runs first and `SoftplusT` runs second. Use `ChainT` when sequential composition expresses the target domain; use `MaskedT` when only masked entries should be transformed.

Source: https://brainx.chaobrain.com/brainstate/apis/nn/parameters.html

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/05_parameters_transforms_regularization.html

Source: https://brainx.chaobrain.com/brainstate/how_to/constrain_and_regularize_parameters.html

## Selection safeguards

- Keep the direction of the map explicit: `nn.Param.value()` uses `forward()`. `LogT` is documented in the opposite direction from `ExpT`.
- Do not treat every bounded transform as interchangeable. `SigmoidT` is an open-interval smooth map; `ClipT` is documented as clipping.
- Do not infer unpublished constructor arguments. The routed pages name several classes without publishing their calls.
- Do not assume `SimplexT` preserves vector length; the official example demonstrates a three-value unconstrained input mapping to four simplex entries.
- Use `ChainT` in listed order and `MaskedT` for boolean-mask selection; do not hand-roll composition before checking these built-ins.
- Return to `parameter-constraints-regularization.md` for regularizer selection and objective integration.

## Mirror source URLs

- https://brainx.chaobrain.com/brainstate/apis/nn/parameters.html
- https://brainx.chaobrain.com/brainstate/tutorials/core/05_parameters_transforms_regularization.html
- https://brainx.chaobrain.com/brainstate/how_to/constrain_and_regularize_parameters.html
