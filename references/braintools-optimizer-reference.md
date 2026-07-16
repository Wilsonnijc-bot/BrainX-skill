# Braintools Optimizer Reference

Source mirrored: https://brainx.chaobrain.com/braintools/optim/index.html

Use this shared reference when a BrainState, BrainMass, or BrainPy training/fitting task needs optimizer or learning-rate-scheduler selection beyond the default `braintools.optim.Adam` pattern.

## Used by

- `skills/brainstate/SKILL.md`
- `skills/brainstate/references/deeplearning-training/supervised-training-workflows.md`
- `skills/brainmass/SKILL.md`
- `skills/brainpy/SKILL.md`

Official source phrase: "Optimization guides highlight practical solvers for tuning models and experiments."

## Scope

- Use `braintools.optim` for optimizer objects and learning-rate schedulers in BrainState training loops.
- For direct BrainState or BrainPy loops, keep optimizer state registered against the model trainable states, usually `model.states(brainstate.ParamState)`.
- Keep gradient computation in `brainstate.transform.grad(...)` and apply updates through the optimizer update path already used by the nested training reference.
- For BrainMass fitting, use `skills/brainmass/SKILL.md` for `Fitter`, objective, and backend semantics; use this file only to select optimizer, scheduler, or external-wrapper families.
- Use this reference for selecting optimizer families, scheduler families, or external optimization routes; use `skills/brainstate/references/deeplearning-training/supervised-training-workflows.md` for loss/grad/JIT loop structure.

## Tutorial Routing

- NevergradOptimizer tutorial: use for gradient-free parameter tuning and experiment search.
- ScipyOptimizer tutorial: use for SciPy-based routines rather than BrainState gradient-step loops.
- Getting Started with optax Optimizers: use when selecting Optax-backed optimizers through Braintools.
- Learning Rate Scheduling Strategies: use when the training loop needs schedules such as decay, warmup, cosine, cyclic, or plateau behavior.
- Advanced Optimizers and Techniques: use when a task asks for optimizer variants beyond standard SGD/Adam-style choices.

## Optimizer And Scheduler Families Listed By The Index

- Optimizers include `SGD`, `Momentum`, `MomentumNesterov`, `Adam`, `AdamW`, `Adagrad`, `Adadelta`, `RMSprop`, `Adamax`, `Nadam`, `RAdam`, `Lamb`, `Lars`, `Lookahead`, `Yogi`, `LBFGS`, `Rprop`, `Adafactor`, `AdaBelief`, `Lion`, `SM3`, `Novograd`, `Fromage`, `SOFO`, and `SOFOScan`.
- Scheduler utilities include `LRScheduler`, `StepLR`, `MultiStepLR`, `ConstantLR`, `LinearLR`, `ExponentialLR`, `PolynomialLR`, `ExponentialDecayLR`, `CosineAnnealingLR`, `CosineAnnealingWarmRestarts`, `WarmupCosineSchedule`, `CyclicLR`, `OneCycleLR`, `ReduceLROnPlateau`, `WarmupScheduler`, `PiecewiseConstantSchedule`, `ChainedScheduler`, and `SequentialLR`.
- External optimizer wrappers include `ScipyOptimizer` and `NevergradOptimizer`.

## BrainState Training Pattern Reminder

```python
params = model.states(brainstate.ParamState)
optimizer = braintools.optim.Adam(lr=1e-2)
optimizer.register_trainable_weights(params)

@brainstate.transform.jit
def train_step(x, y):
    def loss_fn():
        ...
    grads, loss = brainstate.transform.grad(loss_fn, params, return_value=True)()
    optimizer.update(grads)
    return loss
```

Keep this as the default loop shape unless the selected optimizer tutorial requires a different optimization interface.
