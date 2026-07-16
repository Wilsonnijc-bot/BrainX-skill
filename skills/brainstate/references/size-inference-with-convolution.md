# Size Inference with Convolution

Source mirrored: https://brainx.chaobrain.com/brainstate/tutorials/core/03_common_layers.html

Use this when wiring convolutional modules and you need to verify the shape contract before composing the next layer.

Official source phrase: "Convolutional layers extract spatial features using learnable filters." The tutorial also says to supply the expected input shape without the batch dimension.

## Pattern

- `Conv2d.in_size` is the expected non-batch input shape, normally `(height, width, channels)` for NHWC inputs.
- `Conv2d.out_size` is the inferred non-batch output shape after kernel, stride, padding, and output channels are applied.
- `padding='SAME'` with stride `(1, 1)` preserves spatial size.
- `padding='VALID'` removes border positions that cannot fit the full kernel.
- `stride` reduces spatial resolution when it is greater than 1.

```python
import brainstate

conv = brainstate.nn.Conv2d(
    in_size=(28, 28, 3),
    out_channels=32,
    kernel_size=(3, 3),
    stride=(1, 1),
    padding='SAME',
)

print(conv.in_size)   # (28, 28, 3)
print(conv.out_size)  # (28, 28, 32)

conv_valid = brainstate.nn.Conv2d(
    in_size=(28, 28, 3),
    out_channels=32,
    kernel_size=(3, 3),
    stride=(2, 2),
    padding='VALID',
)

print(conv_valid.in_size)   # (28, 28, 3)
print(conv_valid.out_size)  # (13, 13, 32)
```

## Checks before composing

- Pass `conv.out_size` into downstream layers that require an `in_size`.
- Keep the channel axis convention consistent with the layer defaults and input tensor.
- Do not include batch size in `in_size`; test tensors include the batch dimension separately.
