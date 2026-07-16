# Transformation Grad Expansion Blueprint

## Purpose

Detail how gradients and autodiff should be taught after the Transformations Core skill routes here.

## Used by

- `skills/brainstate/SKILL.md`
- `skills/braincell/SKILL.md`

## Primary source pages to expand from

- BrainState autodiff tutorial.
- BrainState training and metrics tutorial.
- BrainState parameter constraints/regularization tutorial.

## Should eventually cover

- Gradients with respect to function arguments.
- Gradients with respect to BrainState `State`.
- `ParamState` as trainable target.
- Auxiliary outputs.
- Loss-function structure.
- Sensitivity analysis.
- Jacobian, Hessian, or vector-gradient routes.
- Differentiating through BrainCell simulations when appropriate.

## Keep out

- Optimizer details except routing to training.
- JIT cache mechanics.
- Full vmap tutorial.

## Common mistakes to document

- Asking for gradients but leaving trainable values outside State.
- Forgetting whether gradients target args, states, or both.
- Mutating state inside loss in unclear ways.
- Ignoring units in physical losses.

## Placeholder examples

- Grad over function argument.
- Grad over `ParamState`.
- Loss with auxiliary metrics.
- Conductance-fitting blueprint.
