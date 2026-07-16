# BrainState Collective Model Operations

Use this reference when one lifecycle or maintenance operation must be applied across every relevant Module or State in a model graph: initialization, reset, ordered method calls, batched lifecycle operations, or State restoration.

## Source

- Collective Operations: https://brainx.chaobrain.com/brainstate/how_to/collective_operations.html

The official guide describes these helpers as managing all modules inside a model without manually traversing the Module graph.

## Operation Map

| Need | API |
|---|---|
| Fix the order of collectively invoked methods | `brainstate.nn.call_order(level)` |
| Invoke one method across model nodes | `brainstate.nn.call_all_fns` |
| Vectorize a collective method call | `brainstate.nn.vmap_call_all_fns` |
| Initialize State across the graph | `brainstate.nn.init_all_states` |
| Reset existing State across the graph | `brainstate.nn.reset_all_states` |
| Vectorize initialization or reset | `vmap_init_all_states`, `vmap_reset_all_states` |
| Restore values keyed by absolute State paths | `brainstate.nn.assign_state_values` |

## Ordered Collective Calls

`call_order` attaches an execution level to a method. Lower levels run first when collective utilities visit the graph.

```python
import brainstate

class EncoderDecoder(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = brainstate.nn.Linear((16,), (32,))
        self.decoder = brainstate.nn.Linear((32,), (16,))

    @brainstate.nn.call_order(0)
    def init_state(self):
        self.encoder.init_state()
        self.decoder.init_state()

    @brainstate.nn.call_order(1)
    def reset_state(self):
        self.encoder.reset_state()
        self.decoder.reset_state()
```

Add ordering only when method interactions require it. Otherwise let graph order remain the default.

## Initialize and Reset the Whole Model

`init_all_states` walks the Module graph and calls `init_state` on each applicable node. It accepts lifecycle keyword arguments, supports `node_to_exclude`, and returns the target so construction can be chained.

```python
model = brainstate.nn.Sequential(
    brainstate.nn.Linear((10,), (32,)),
    brainstate.nn.GELU(),
    brainstate.nn.Dropout(prob=0.1),
)

brainstate.nn.init_all_states(model, batch_size=4)

# Exclude a node type when it should not participate.
brainstate.nn.init_all_states(
    model,
    node_to_exclude=brainstate.nn.Dropout,
)
```

For recurrent or dynamical models, initialize once and reset existing State between independent sequences or episodes.

```python
rnn = brainstate.nn.ValinaRNNCell(num_in=8, num_out=16)
brainstate.nn.init_all_states(rnn, batch_size=2)

# ... process one sequence ...

brainstate.nn.reset_all_states(rnn)
```

Do not reconstruct the model merely to clear its hidden State. Reset the graph so parameter identity and structure remain intact.

## Batched Lifecycle Operations

Use `vmap_init_all_states` and `vmap_reset_all_states` for independent ensemble or Monte-Carlo instances. The vectorized helpers manage separate random keys and accept filters for State that should remain shared.

```python
policy = brainstate.nn.Sequential(
    brainstate.nn.Linear((4,), (64,)),
    brainstate.nn.GELU(),
    brainstate.nn.Linear((64,), (2,)),
)

brainstate.nn.vmap_init_all_states(policy, axis_size=8)

# ... run the ensemble ...

brainstate.nn.vmap_reset_all_states(policy, axis_size=8)
```

Pass a `state_to_exclude` filter when selected statistics or buffers should stay shared instead of receiving the mapped axis. Keep `axis_size` consistent across initialization, computation, and reset.

## Invoke Other Methods Across the Graph

`call_all_fns` is the primitive behind initialization and reset. Use it when child Modules expose another common lifecycle or maintenance method. Use `vmap_call_all_fns` for the corresponding operation across independent mapped instances.

Requirements:

- each participating child implements the named method;
- arguments are valid for every participating implementation;
- use node filters when only part of the graph should receive the call;
- use `call_order` when cross-node dependencies make order observable.

Do not replace ordinary forward execution with collective dispatch. It is intended for model-wide lifecycle and maintenance methods.

## Restore State Values

`assign_state_values` maps values back to State objects using absolute paths and returns `(unexpected, missing)` keys. Preserve each State value at the path returned by `model.states()`; if a State value is itself a dictionary, do not silently reinterpret its inner keys as additional Module-graph path segments.

```python
brainstate.nn.init_all_states(model)

snapshot = {
    path: state.value
    for path, state in model.states().items()
}

# ... mutate parameters or runtime State ...

unexpected, missing = brainstate.nn.assign_state_values(model, snapshot)
if unexpected or missing:
    raise ValueError(
        f'checkpoint mismatch: unexpected={unexpected}, missing={missing}'
    )
```

Always inspect both returned collections. A restore that reports mismatches is not a successful checkpoint load.

## Boundaries and Gotchas

- Call `init_all_states` after construction before running a State-dependent Module.
- Use `reset_all_states` between independent sequences; do not reinitialize trainable parameters unless that is intentional.
- `node_to_exclude` filters nodes; `state_to_exclude` filters State in mapped lifecycle operations.
- Mapped lifecycle helpers introduce or manage an ensemble axis; keep the axis contract consistent downstream.
- State restoration is path-sensitive. Preserve the absolute paths emitted by the target graph and fail loudly on unexpected or missing keys.
- These APIs traverse model structure. Use `state_collections_and_utilities.md` when the task only reorganizes mappings or PyTrees.
