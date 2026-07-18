# BrainState Parameter Constraints and Regularization

Use this first-layer reference after `skills/brainstate/SKILL.md` has selected `brainstate.nn.Param` for a parameter that needs a valid-domain transform, a regularization penalty, a prior, or fixed parameter-like storage. It owns advanced constraint and loss-integration decisions; the main skill owns the basic `ParamState` versus `nn.Param` choice.

This is the only reference that may open `references/brainstate/parameter-containers-transforms-catalog.md`. Open that nested catalog only when selecting among the complete parameter-container or transform families. Keep execution transforms such as `brainstate.transform.jit` and `brainstate.transform.grad` conceptually separate from `brainstate.nn` parameter transforms.

## Parameter contract

`brainstate.nn.Param` adds two orthogonal capabilities on top of a bare trainable `ParamState`: a bijective transform maps the unconstrained array seen by the optimizer to the constrained value used by the model, and a regularization term contributes a penalty to the loss.

- `param.val` is the underlying `ParamState`; optimizers update its unconstrained array.
- `param.value()` applies the forward transform and returns the constrained model-space value. With the default `IdentityT`, constrained and unconstrained values coincide.
- `param.set_value(value)` applies the inverse transform when storing a constrained value back.
- `param.reg_loss()` applies the attached `reg=` object to the parameter's current value. Attaching a regularizer does not add the result to an objective; the loss function must do that.
- `nn.Const` is a `Param` with `fit=False`. It is excluded from `model.states(brainstate.ParamState)`, so `grad` and optimizers leave it unchanged.

Official source: https://brainx.chaobrain.com/brainstate/tutorials/core/05_parameters_transforms_regularization.html

The constraint how-to returns dimensionless examples as `brainunit.Quantity` values and uses `brainunit.get_magnitude(...)` when a plain JAX value is needed for printing, comparison, or a dimensionless objective.

Official source: https://brainx.chaobrain.com/brainstate/how_to/constrain_and_regularize_parameters.html

## Constraint patterns: domain rules bundled with code

Use `.value()` in every model computation and mutate `.val.value` only as the optimizer-side, unconstrained representation. The following source-grounded forms establish the important limits:

```python
import jax.numpy as jnp
import brainunit as u

import brainstate
from brainstate import nn

rate = nn.Param(jnp.array(0.5), t=nn.SoftplusT(lower=0.0))
rate.val.value = jnp.array(-10.0)
assert float(u.get_magnitude(rate.value())) > 0.0

mix = nn.Param(
    jnp.array(0.5),
    t=nn.SigmoidT(lower=0.0, upper=1.0),
)

probs = nn.Param(jnp.zeros(3), t=nn.SimplexT())
p = u.get_magnitude(probs.value())
assert jnp.all(p >= 0.0)
assert jnp.isclose(p.sum(), 1.0)
```

- `SoftplusT(lower)` maps the real line to the open interval `(lower, infinity)`; an arbitrarily negative optimizer value cannot cross the lower bound.
- `SigmoidT(lower, upper)` maps to the open interval `(lower, upper)` and approaches, but does not cross, either bound.
- `SimplexT()` returns a non-negative vector whose entries sum to one.
- The documented transform path is a smooth bijection rather than a hard clip, so gradients flow through the unconstrained representation.

Official source: https://brainx.chaobrain.com/brainstate/how_to/constrain_and_regularize_parameters.html

## Regularization access bundled with code

A regularization object contributes a scalar penalty derived from a parameter value. Call `reg.loss(value)` directly, or attach it with `reg=` and call `param.reg_loss()`:

```python
weights = jnp.array([3.0, -4.0])

l1_penalty = nn.L1Reg(0.1).loss(weights)
l2_penalty = nn.L2Reg(0.1).loss(weights)

p = nn.Param(
    weights,
    reg=nn.ElasticNetReg(
        l1_weight=1.0,
        l2_weight=1.0,
        alpha=0.5,
    ),
)
elastic_net_penalty = p.reg_loss()
```

For these values, the official example defines L1 as `0.1 * (abs(3) + abs(-4))`, L2 as `0.1 * (3**2 + 4**2)`, and reports `16.0` for the shown elastic-net configuration. `ChainedReg` is the documented composite for combining multiple regularizations; do not guess its constructor when exact composition arguments matter.

Official source: https://brainx.chaobrain.com/brainstate/tutorials/core/05_parameters_transforms_regularization.html

## End-to-end loss integration

This official workflow keeps the optimizer target in unconstrained `ParamState`s, reads transformed values in the forward pass, and explicitly adds the attached L2 penalty to the data loss:

```python
class ConstrainedLinear(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.w = nn.Param(
            brainstate.random.randn(din, dout) * 0.1,
            reg=nn.L2Reg(1e-3),
        )
        self.gain = nn.Param(
            jnp.array(1.0),
            t=nn.SoftplusT(lower=0.0),
        )

    def __call__(self, x):
        return (x @ self.w.value()) * self.gain.value()


model = ConstrainedLinear(4, 2)
params = model.states(brainstate.ParamState)
x = brainstate.random.randn(16, 4)
y = brainstate.random.randn(16, 2)


def loss_fn():
    mse = jnp.mean((model(x) - y) ** 2)
    penalty = model.w.reg_loss()
    return mse + u.get_magnitude(penalty)


@brainstate.transform.jit
def train_step():
    grads, loss = brainstate.transform.grad(
        loss_fn,
        params,
        return_value=True,
    )()
    for key in params:
        params[key].value -= 0.1 * grads[key]
    return loss
```

The gradient is taken with respect to the unconstrained `ParamState` collection. The transform constrains what `model(...)` sees; the regularizer affects training only because `loss_fn()` includes `model.w.reg_loss()`.

Official source: https://brainx.chaobrain.com/brainstate/how_to/constrain_and_regularize_parameters.html

For a model with several regularized parameters, the tutorial's collection pattern is:

```python
def reg_penalty(self):
    return sum(p.reg_loss() for p in self.nodes(nn.Param).values())
```

This walks the model's `nn.Param` nodes, sums their penalties, and adds the result once to the data loss.

Official source: https://brainx.chaobrain.com/brainstate/tutorials/core/05_parameters_transforms_regularization.html

## Fixed parameter-like values

Use `nn.Const` when a value belongs in the module graph and forward computation but must not be collected as trainable state:

```python
class Scaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Param(jnp.ones(3))
        self.gain = nn.Const(jnp.array(2.0))

    def __call__(self, x):
        return x * self.weight.value() * self.gain.value()


model = Scaler()
trainable = model.states(brainstate.ParamState)
```

`trainable` contains the underlying State for `weight`; `gain` is absent.

Official source: https://brainx.chaobrain.com/brainstate/how_to/constrain_and_regularize_parameters.html

## Classical and structural regularizers

Use the official API categories as selectors, then inspect the exact class API before relying on arguments not demonstrated above.

| Need | Official API |
|---|---|
| Abstract regularization contract | `Regularization` |
| L1/Lasso sparsity | `L1Reg` |
| L2/Ridge shrinkage or smoothness | `L2Reg` |
| Combined L1 and L2 | `ElasticNetReg` |
| Robust regularization | `HuberReg` |
| Group sparsity | `GroupLassoReg` |
| Total variation | `TotalVariationReg` |
| Soft max-norm constraint | `MaxNormReg` |
| Entropy pressure | `EntropyReg` |
| Orthogonal structure | `OrthogonalReg` |
| Spectral-norm structure | `SpectralNormReg` |
| Composite penalties | `ChainedReg` |

These methods add penalty terms that encourage sparsity, smoothness, or structural constraints such as orthogonality and spectral norms.

Official source: https://brainx.chaobrain.com/brainstate/apis/nn/regularization.html

## Prior-distribution regularizers

Prior regularizers contribute a negative log-density penalty and encode distributional assumptions for Bayesian-inspired parameter estimation. The API describes them as useful for variational inference, maximum a posteriori estimation, and uncertainty quantification. Each implements `loss()`, `sample_init()`, and `reset_value()` for prior-based initialization; a `Param` carrying a prior can be redrawn with `param.reset_to_prior()`.

| Parameter assumption | Official API |
|---|---|
| Gaussian prior | `GaussianReg` |
| Student's t prior | `StudentTReg` |
| Cauchy prior | `CauchyReg` |
| Soft bounded uniform prior | `UniformReg` |
| Value in `[0, 1]` under a Beta prior | `BetaReg` |
| Positive value under a log-normal prior | `LogNormalReg` |
| Positive value under an exponential prior | `ExponentialReg` |
| Positive value under a Gamma prior | `GammaReg` |
| Variance under an inverse-Gamma prior | `InverseGammaReg` |
| Scale-invariant log-uniform/Jeffreys prior | `LogUniformReg` |
| Strong sparsity with heavy tails | `HorseshoeReg` |
| Variable selection | `SpikeAndSlabReg` |
| Probability simplex | `DirichletReg` |

Match the prior to the parameter domain and the actual modeling assumption. Use `t=` for a guaranteed valid-domain mapping; `reg=` contributes the prior penalty.

Official source: https://brainx.chaobrain.com/brainstate/apis/nn/regularization.html

## Decision and failure rules

- Need a transform class beyond the positive, interval, or simplex patterns above: open `references/brainstate/parameter-containers-transforms-catalog.md`.
- Need only a penalty: attach `reg=` without inventing a transform.
- Need both: attach `t=` and `reg=` to the same `nn.Param`; read `.value()` in the model and add `.reg_loss()` to the objective.
- Need a fixed value in the graph: use `nn.Const`; do not expect it in a `ParamState` collection.
- Never read `param.val` as the constrained forward value or update `param.value()` as optimizer storage.
- Do not treat `SigmoidT` bounds as inclusive, or `SoftplusT(lower)` as able to attain `lower`.
- Do not attach regularization and assume training includes it automatically.
- Do not choose a prior whose support conflicts with the parameter domain.

## Mirror source URLs

- https://brainx.chaobrain.com/brainstate/tutorials/core/05_parameters_transforms_regularization.html
- https://brainx.chaobrain.com/brainstate/how_to/constrain_and_regularize_parameters.html
- https://brainx.chaobrain.com/brainstate/apis/nn/regularization.html
