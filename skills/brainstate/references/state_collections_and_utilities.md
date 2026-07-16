# BrainState State Collections and Utility Toolkit

Use this reference when a task needs to filter, split, reorganize, freeze, flatten, configure, or pretty-print nested mappings and PyTrees. These are collection and data-structure operations; use `collective_model_operations.md` instead when the task must invoke lifecycle methods across every Module in a model graph.

## Source

- Utility Toolkit: https://brainx.chaobrain.com/brainstate/how_to/filter_and_organize_states.html

The official guide describes `brainstate.util` as helpers for collections, structured PyTrees, pretty-printing, and runtime hygiene.

## Choose the Smallest Utility

| Need | API |
|---|---|
| Filter or split mapping values | `DictManager.subset(...)`, `DictManager.split(...)` |
| Transform every mapping value | `DictManager.map_values(...)` |
| Attribute-style nested configuration | `DotDict` |
| Merge or convert dotted-key dictionaries | `merge_dicts`, `flatten_dict`, `unflatten_dict` |
| Define an immutable JAX-compatible record | `util.struct.dataclass` |
| Mark static dataclass metadata | `util.struct.field(pytree_node=False, ...)` |
| Freeze or unfreeze nested mappings | `util.struct.freeze`, `util.struct.unfreeze` |
| Build reusable path/value predicates | `brainstate.util.filter` |
| Flatten nested mappings to tuple paths | `flat_mapping`, `nest_mapping` |
| Inspect nested mappings and lists readably | `NestedDict`, `PrettyList` |

## Filter and Transform `DictManager`

`DictManager` extends a mapping with filters, splits, combination operators, value transforms, and JAX PyTree support.

```python
from brainstate.util import DictManager

modules = DictManager({
    'encoder': {'params': 32},
    'decoder': {'params': 45},
    'dropout': 0.1,
})

submodules = modules.subset(dict)
matched, remainder = modules.split(dict)
param_counts = submodules.map_values(lambda layer: layer['params'])
```

Use `subset` when only the matching collection is needed. Use `split` when both the matched and unmatched entries must be preserved. `map_values` retains the mapping keys while replacing each value.

## Nested Configuration with `DotDict`

`DotDict` provides attribute access to nested dictionary configuration while retaining conversion back to ordinary dictionaries.

```python
from brainstate.util import DotDict

config = DotDict({
    'model': {'layers': 4, 'hidden': 256},
    'training': {'lr': 3e-4},
})

hidden = config.model.hidden
config.training.dropout = 0.2
plain_config = config.to_dict()
```

Use it for configuration ergonomics, not as a replacement for mutable BrainState `State` inside transformed computation.

## Merge, Flatten, and Unflatten Dictionaries

`merge_dicts` supports nested configuration overrides. `flatten_dict` and `unflatten_dict` convert between nested dictionaries and dotted-key representations, which is useful for logging, command-line overrides, and configuration serialization.

```python
from brainstate.util import flatten_dict, merge_dicts, unflatten_dict

base = {'optimizer': {'lr': 1e-3, 'beta1': 0.9}}
override = {'optimizer': {'lr': 5e-4}, 'seed': 1234}

merged = merge_dicts(base, override)
flat = flatten_dict(merged)
restored = unflatten_dict(flat)
```

Do not confuse this dotted-key representation with `flat_mapping`, which uses tuple paths for nested PyTree-style containers.

## Structured and Frozen PyTrees

The `struct` submodule provides Flax-compatible data structures. `struct.dataclass` registers the class as a PyTree; mark metadata that should stay static with `pytree_node=False`.

```python
import jax
import jax.numpy as jnp
from brainstate.util import struct

@struct.dataclass
class LayerConfig:
    weight: jax.Array
    bias: jax.Array
    name: str = struct.field(pytree_node=False, default='layer')

cfg = LayerConfig(weight=jnp.ones((2, 2)), bias=jnp.zeros(2))
updated = cfg.replace(weight=jnp.full((2, 2), 3.0))

frozen = struct.freeze({'encoder': jnp.arange(3)})
plain = struct.unfreeze(frozen)
```

Use `.replace(...)` for immutable dataclass updates. Static fields are excluded from the dynamic PyTree leaves seen by JAX transforms.

## Declarative Path/Value Filters

`brainstate.util.filter` turns declarative filters into predicates. A predicate receives `(path, value)`, allowing type, tag, and path rules to be composed.

```python
from brainstate.util import filter as util_filter

class TaggedModule:
    def __init__(self, tag):
        self.tag = tag

candidate = TaggedModule('trainable')
type_filter = util_filter.OfType(TaggedModule)
tag_filter = util_filter.WithTag('trainable')
trainable_modules = util_filter.All(type_filter, tag_filter)

if trainable_modules(('encoder',), candidate):
    ...
```

Use `to_predicate(...)` to normalize a supported filter specification into a callable. Compose filters rather than embedding repeated traversal conditionals.

## Pretty Nested Containers

`NestedDict`, `FlattedDict`, and `PrettyList` combine readable representations with PyTree semantics. `flat_mapping` produces tuple-path keys, and `nest_mapping` restores the nested form.

```python
import jax.numpy as jnp
from brainstate.util import NestedDict, PrettyList, flat_mapping, nest_mapping

state = NestedDict({
    'encoder': {'weight': jnp.ones((2, 2)), 'bias': jnp.zeros(2)},
    'decoder': {'weight': jnp.eye(2)},
})

flat_state = flat_mapping(state)
round_trip = nest_mapping(flat_state)
history = PrettyList([{'loss': 0.8}, {'loss': 0.42}])
```

Use these containers to inspect checkpoints, State collections, histories, or structured configuration without discarding PyTree behavior.

## Boundaries and Gotchas

- `DotDict` is configuration data, not transformed mutable `State`.
- `flatten_dict` uses dotted keys; `flat_mapping` uses tuple paths. Choose the representation expected by the downstream API.
- Use `struct.field(pytree_node=False, ...)` for metadata that should not become a JAX leaf.
- Freezing a mapping makes the container immutable; use `struct.unfreeze` before ordinary mutation.
- Collection utilities reorganize data. They do not traverse a Module graph to initialize, reset, or invoke methods; use `collective_model_operations.md` for that.
