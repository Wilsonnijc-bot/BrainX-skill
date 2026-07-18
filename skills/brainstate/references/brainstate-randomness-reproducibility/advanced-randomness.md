# Advanced BrainState Randomness

Open this nested child only from `randomness-and-reproducibility.md`, and only when a task needs direct key control, parallel key preparation, exact key replay, custom-generator state inspection, or checkpoint behavior beyond the parent's ordinary stochastic path. Do not route here directly from `skills/brainstate/SKILL.md` or another reference. The parent owns independent-stream selection, dropout/noise, and normal transformed randomness; this child does not repeat them.

## Snapshot a global key and replay from it

The tutorial's advanced boundary is: "For advanced use cases, you can directly access and manipulate keys." Bundle inspection and restoration with the operation whose randomness must be replayed:

```python
import brainstate
import jax.numpy as jnp

saved_key = brainstate.random.get_key()
v1 = brainstate.random.randn(3)
_ = brainstate.random.randn(3)

brainstate.random.set_key(saved_key)
v1_replayed = brainstate.random.randn(3)

assert jnp.allclose(v1, v1_replayed)
```

The snapshot is the sequence position immediately before `v1`; restoring it makes the next draw reproduce `v1`. Use `get_key()` / `set_key(...)` for this explicit snapshot-and-replay contract. Keep `restore_key()` distinct: the API describes it only as restoring "the default random key to its previous state," so do not silently substitute it for a named checkpoint snapshot.

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html

API definitions: https://brainx.chaobrain.com/brainstate/apis/random.html

## Prepare keys for parallel or mapped work

The API groups these as "Functions for creating independent random keys for parallel computation." The tutorial's documented parallel preparation is:

```python
keys = brainstate.random.split_key(n=4)

for i, key in enumerate(keys):
    print(i, key)
```

Select the narrowest documented operation:

| API | Documented role |
|---|---|
| `split_key` | Create new random key(s) from the current seed. The tutorial uses `split_key(n=4)` for parallel operations. |
| `split_keys` | Create multiple independent random keys from the current seed. |
| `self_assign_multi_keys` | Assign multiple keys to the global random state for parallel access. |

The routed pages do not provide a `vmap` call, mapped key-axis configuration, or per-example key-consumption rule. They only identify `vmap` as a next step and document the parallel key utilities above. Do not invent mapped semantics from these pages; combine these RNG operations with a separately documented BrainState mapping workflow when concrete `vmap` mechanics are required.

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html

API definitions: https://brainx.chaobrain.com/brainstate/apis/random.html

## Inspect custom-stream state and choose advanced generators

After the parent has selected a custom `RandomState`, the API documents its current key through `.value`:

```python
rng = brainstate.random.RandomState(seed=123)
_ = rng.normal(0, 1, size=(10, 10))
current_stream_key = rng.value
```

This is the documented custom-stream state inspection surface. The routed pages demonstrate reseeding a custom generator with `rng.seed(999)`, but they do not demonstrate assigning a saved value back to a custom generator or restoring several custom streams. Do not apply the global `set_key(...)` API to `rng`: its documented target is the global random state.

For generator selection without restating the parent's independent-stream example:

| API | Documented role |
|---|---|
| `default_rng` | Get the default random state or create a new one with a specified seed. |
| `clone_rng` | Create a clone of the random state or a new random state. |
| `seed_context` | Temporarily change the random seed with automatic restoration. |

The same API page's example spells the temporary context as `local_seed(123)`, while its API table lists `seed_context`. Because the routed source is internally inconsistent, verify the installed BrainState version before choosing the callable name.

Source: https://brainx.chaobrain.com/brainstate/apis/random.html

Custom reseeding example: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html

## Checkpoint at the intended sequence boundary

The tutorial's checkpoint pattern stores the current global key beside the model and restores that exact key before stochastic work resumes:

```python
checkpoint = {
    'model': model.state_dict(),
    'rng_key': brainstate.random.get_key(),
}

# Restore model state through the model's checkpoint path first.
brainstate.random.set_key(checkpoint['rng_key'])
```

Capture `rng_key` after every random operation that belongs to the completed portion of the run. Restoring an earlier snapshot intentionally replays draws after that snapshot, as the replay script above demonstrates. A checkpoint with multiple custom streams additionally needs each stream's current key, but these routed pages document only reading a custom key through `rng.value`; they do not supply a multi-stream restoration script. Keep such restoration out of generated code until a source for that specific mechanism is selected.

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html

## Key API distinctions

- `get_key()` returns the current global random key.
- `get_key_data()` returns the current global random key as raw `uint32[2]` data; use it only when raw key data is the required interchange form.
- `set_key(...)` sets a new key on the global random state.
- `restore_key()` restores the default key to its previous state.
- `RandomState.value` is the API page's documented way to inspect the current key of a custom generator.

Source: https://brainx.chaobrain.com/brainstate/apis/random.html

## Official sources

- https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html
- https://brainx.chaobrain.com/brainstate/apis/random.html
