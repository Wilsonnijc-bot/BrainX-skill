# Size Inference with Pooling and Flatten

Source mirrored: https://brainx.chaobrain.com/brainstate/tutorials/core/03_common_layers.html

Use this when a convolutional stack transitions into a dense classifier.

Official source phrase: "Pooling layers downsample feature maps, reducing spatial dimensions." The same tutorial describes `Flatten` as reshaping multi-dimensional inputs.

## Pattern

- Pooling reduces spatial dimensions according to `kernel_size`, `stride`, and padding.
- `MaxPool2d.out_size` keeps the channel count and updates height/width.
- `Flatten` converts the selected dimensions into one feature vector.
- For image classifiers, flatten after the final conv/pool block and feed the flattened size into `Linear`.

```python
import brainstate

pool = brainstate.nn.MaxPool2d(
    in_size=(28, 28, 32),
    kernel_size=(2, 2),
    stride=(2, 2),
    channel_axis=-1,
)

print(pool.in_size)   # (28, 28, 32)
print(pool.out_size)  # (14, 14, 32)

flatten = brainstate.nn.Flatten(in_size=pool.out_size)

print(flatten.in_size)   # (14, 14, 32)
print(flatten.out_size)  # (6272,)
```

## Checks before composing

- Prefer `Flatten(in_size=previous_layer.out_size)` over hand-copying dimensions.
- If a layer uses runtime shape inference, verify with one forward pass before training.
- Keep `channel_axis=-1` for NHWC tensors unless the whole stack uses a different layout.
