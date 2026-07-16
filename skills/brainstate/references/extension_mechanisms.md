# BrainState Extension Mechanisms and State Observability

Use this reference for two distinct extension layers:

1. class-level reusable behavior and construction metadata through mixins, parameter descriptors, type combinators, and modes;
2. runtime observation or interception of State access through hooks.

Do not use hooks to implement ordinary model computation, and do not use mixins when the requirement is to observe one concrete State instance at runtime.

## Sources

- Mixin System: https://brainx.chaobrain.com/brainstate/how_to/custom_states_and_mixins.html
- Observe and Intercept State Access with Hooks: https://brainx.chaobrain.com/brainstate/how_to/state_hooks.html

## Choose the Extension Layer

| Need | Mechanism |
|---|---|
| Reuse focused behavior across classes | `brainstate.mixin.Mixin` |
| Store a reusable constructor preset | `ParamDesc.desc(...)` / `ParamDescriber` |
| Require all or any of several interfaces | `JointTypes` / `OneOfTypes` |
| Represent training, batching, or combined runtime mode | `Mode`, `Training`, `Batching`, `JointMode` |
| Observe every State operation of one kind | `brainstate.register_state_hook` |
| Observe only one State instance | `state.register_hook` |
| Rewrite or reject a pending State write | `write_before` hook |
| Check values inside compiled code | `brainstate.transform.checkify`, `check`, or `debug_nan` |

## Focused Behavior with Mixins

A BrainState mixin inherits from `brainstate.mixin.Mixin` and contributes optional behavior without defining its own required constructor protocol.

```python
import datetime
import brainstate
from brainstate import mixin

class LoggingMixin(mixin.Mixin):
    def log(self, message: str) -> None:
        stamp = datetime.datetime.now().strftime('%H:%M:%S')
        print(f'[LOG {stamp}] {self.__class__.__name__}: {message}')

class Accumulator(brainstate.nn.Module, LoggingMixin):
    def __init__(self):
        super().__init__()
        self.total = 0.0

    def add(self, value):
        self.total += float(value)
        self.log(f'updated running total to {self.total:.2f}')
        return self.total
```

Mixin rules:

- provide behavior, not new required constructor arguments;
- keep each mixin narrow so multiple behaviors compose;
- document any attributes or methods expected from the host class;
- pair mixins with a core class such as `brainstate.nn.Module` rather than turning the mixin into the model hierarchy itself.

## Reusable Constructor Presets

`ParamDesc.desc(...)` stores constructor arguments in a callable `ParamDescriber`. Calling the descriptor constructs the object, and call-time arguments override stored defaults.

```python
from brainstate import mixin

class DenseBlock(mixin.ParamDesc):
    def __init__(self, in_features, out_features, *, activation='relu'):
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

encoder = DenseBlock.desc(256, 128, activation='gelu')

gelu_block = encoder()
relu_block = encoder(activation='relu')
cache_key = encoder.identifier
```

The descriptor identifier is hashable and can be used as a cache key. Use `mixin.ParamDescriber(SomeClass, ...)` directly when the described class cannot inherit from `ParamDesc`.

## Type Combinators and Runtime Modes

- `JointTypes[A, B]` behaves like an intersection: an instance must satisfy every listed type.
- `OneOfTypes[A, B]` behaves like a union: an instance may satisfy any listed type.
- `Mode` captures runtime context; built-ins include `Training`, `Batching`, and `JointMode`.

```python
from brainstate import mixin

class Persistable:
    pass

class Reportable:
    pass

PersistableReport = mixin.JointTypes[Persistable, Reportable]
OptionalNumber = mixin.OneOfTypes[int, float, type(None)]

mode = mixin.JointMode(
    mixin.Training(),
    mixin.Batching(batch_size=8),
)

if mode.has(mixin.Training):
    ...
```

Use modes to centralize runtime semantics rather than scattering unrelated boolean flags through model code.

## State Hook Operations

State hooks run callbacks without editing the model that owns the State.

| Hook operation | Fires when | May modify or cancel? |
|---|---|---|
| `read` | `state.value` is read | No |
| `write_before` | immediately before `state.value = ...` | Yes |
| `write_after` | immediately after a write | No |
| `restore` | `state.restore_value(...)` runs | No |
| `init` | a State is constructed | No |

Global hooks fire for every State in the program. Per-State hooks fire only for their registered instance.

## Observe Writes

A `write_after` callback receives a context with the State name, old value, new value, operation, timestamp, and metadata.

```python
import jax.numpy as jnp
import brainstate

history = []

def record(ctx):
    history.append((ctx.state_name, ctx.old_value, ctx.value))

handle = brainstate.register_state_hook('write_after', record)

weight = brainstate.State(jnp.array(1.0), name='weight')
weight.value = jnp.array(2.0)
```

For long-lived callbacks, guard access to `ctx.state`: it is a weak-reference target and may be `None` after garbage collection.

## Transform or Reject Writes

A `write_before` hook may substitute `ctx.transformed_value`. Multiple hooks chain in priority order, with each hook seeing the preceding transformed value.

```python
def clip_to_unit(ctx):
    current = (
        ctx.transformed_value
        if ctx.transformed_value is not None
        else ctx.value
    )
    ctx.transformed_value = jnp.clip(current, -1.0, 1.0)

clip_handle = brainstate.register_state_hook('write_before', clip_to_unit)
```

Set `ctx.cancel = True` to reject a write. Optionally set `ctx.cancel_reason`; assignment then raises `brainstate.HookCancellationError` and preserves the previous State value.

Use interception only for genuine cross-cutting policy. Prefer `nn.Param` transforms for declarative parameter domains and normal `.value` updates for model logic.

## Scope and Manage Hook Lifetime

Register a hook on one instance when global scope is unnecessary:

```python
watched = brainstate.State(jnp.array(0.0), name='watched')
watched.register_hook(
    'write_after',
    lambda ctx: print('new value:', ctx.value),
)
```

Every registration returns a `HookHandle`:

```python
handle.disable()
handle.enable()
handle.remove()
removed = handle.is_removed()
```

Registry utilities:

- `brainstate.list_state_hooks(...)`
- `brainstate.has_state_hooks()`
- `brainstate.clear_state_hooks()`

Remove temporary debugging hooks when the observation window ends. Do not leave global interception active accidentally across unrelated tests or experiments.

## Hooks and Compiled Code

Hooks are eager Python callbacks. Inside `brainstate.transform.jit`, they fire once during the initial trace—where `ctx.value` may be an abstract tracer—and once per concrete runtime access.

Therefore:

- do not branch in Python on the contents of a hooked value;
- do not force a tracer through `float(...)`, `bool(...)`, or host NumPy;
- keep hook bodies safe for both abstract tracing and concrete execution;
- use `brainstate.transform.checkify`, `check`, or `debug_nan` for assertions and NaN/bounds checks that must execute inside compiled code.

## Boundaries and Gotchas

- Mixins change class behavior; hooks observe or intercept State operations at runtime.
- `ParamDesc` stores construction presets; it is not a trainable parameter container.
- Type combinators express interface expectations; they do not convert objects.
- Global hooks affect every State and should have tightly managed lifetimes.
- Only `write_before` may transform or cancel a write.
- Hooks are not a substitute for compiled validation tools or ordinary model update logic.
