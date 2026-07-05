---
name: brainstate-module-building
description: Guides BrainState Module design, composition, nested graph structure, state traversal, prebuilt layers, activations, normalization, size inference, reusable components, stochastic modules, and random initialization. Use for fixed module dataflow, layer wiring, model organization, and Module-State relationships; route dynamic loops or runtime-dependent branches to BrainState control flow and specialized time-evolving Dynamics patterns to the BrainState dynamics reference.
---

## Slide 12

### Title

Brainstate-Module building

### Dynamics routing

`Dynamics` is a specialized BrainState `Module` pattern for time-evolving systems, not a separate primary skill. If the task involves `Dynamics`, `update()`, time-evolving systems, LIF/SNN populations, delays, before/after update hooks, event-driven spike communication, trajectory simulation, or neural population workflows, open `references/brainstate-dynamics/`.

### Concepts

• Concepts
reusability & composibility: build individual small ones and can combine & nest it

model is a tree, with State object at the leaves

->how to register parameters
params =model.states(brainstate.ParamState)
Basic Neural-network layer explanation -> when to use

### Mini title

Minimal example

### Script

```python
# minimal script using pre-built layer e.g linear
# Create a linear layer
brainstate.random.seed(42)
linear = brainstate.nn.Linear(in_size=(10,), out_size=(5,))
# Forward pass
x = brainstate.random.randn(10)
y = linear(x)
```

#### minimal script of using pre-built activation layer

### Explanation text

Activation functions determine the output of a neuron given its input.

### Script

```python
# Classic Activations
# Sigmoid: 1 / (1 + exp(-x))
sigmoid = brainstate.nn.Sigmoid()
# Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
tanh = brainstate.nn.Tanh()
```

### Mini title

#### Neste Module example -> ResNet script Module Tutorial
Residual Connections
Implement skip connections for deeper networks:

### Script

```python
class ResidualBlock(brainstate.nn.Module):
    """Residual block: y = F(x) + x"""

    def __init__(self, dim):
        super().__init__()

        # Two linear layers with activation in between
        self.linear1 = Linear(dim, dim)
        self.activation = LeakyReLU(0.0)
        self.linear2 = Linear(dim, dim)

    def update(self, x):
        # Compute residual
        residual = x

        # Forward through layers
        out = self.linear1(x)
        out = self.activation(out)
        out = self.linear2(out)

        # Add residual
        return out + residual

class ResNet(brainstate.nn.Module):
    """Simple ResNet with multiple residual blocks."""

    def __init__(self, input_dim, hidden_dim, output_dim, n_blocks=3):
        super().__init__()

        # Input projection
        self.input_proj = Linear(input_dim, hidden_dim)

        # Residual blocks
        self.blocks = []
        for i in range(n_blocks):
            block = ResidualBlock(hidden_dim)
            setattr(self, f'block_{i}', block)
            self.blocks.append(block)

        # Output projection
        self.output_proj = Linear(hidden_dim, output_dim)

    def update(self, x):
        # Project to hidden dimension
        x = self.input_proj(x)

        # Pass through residual blocks
        for block in self.blocks:
            x = block(x)

        # Project to output
        x = self.output_proj(x)
        return x

# Create ResNet
brainstate.random.seed(0)
resnet = ResNet(input_dim=10, hidden_dim=32, output_dim=5, n_blocks=3)

# Forward pass
x = brainstate.random.randn(10)
y = resnet(x)
```

### Reference

#### How Module building work with braincell
#### •HH model script HH Neuron model under Brain Dynamics
https://brainx.chaobrain.com/brainstate/examples/brain_dynamics/hodgkin_huxley_neuron.html

### Other important concepts

• Automatic I/O size inference.
-> Basic size inference concept
• Sequential composition and deep network
concept of .desc() method
• Sequential composition

### Mini title

#### illustrated by -> complexNet script example
Complex Architecture with Mixed Layer Types

### Script

```python
# Build a more complex network with different layer types
class ComplexNet(brainstate.nn.Module):
    """Complex network demonstrating various layer types."""

    def __init__(self, in_size):
        super().__init__()

        self.features = brainstate.nn.Sequential(
            # Initial conv block
            brainstate.nn.Conv2d(in_size, out_channels=16, kernel_size=3, padding='SAME'),
            brainstate.nn.ReLU(),

            # Strided conv (reduces spatial size)
            brainstate.nn.Conv2d.desc(out_channels=32, kernel_size=3, stride=2, padding='SAME'),
            brainstate.nn.ReLU(),

            # Another conv + pool
            brainstate.nn.Conv2d.desc(out_channels=64, kernel_size=3, padding='SAME'),
            brainstate.nn.ReLU(),
            brainstate.nn.MaxPool2d.desc(kernel_size=(2, 2), stride=(2, 2), channel_axis=-1),
        )

        self.classifier = brainstate.nn.Sequential(
            brainstate.nn.Flatten(in_size=self.features.out_size),
            brainstate.nn.Linear.desc(out_size=256),
            brainstate.nn.ReLU(),
            brainstate.nn.Linear.desc(out_size=10),
        )

    def update(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Create network
net = ComplexNet(in_size=(32, 32, 3))

print("Complex Network:")
print(f"Input size: {net.features.in_size}")
print(f"After features: {net.features.out_size}")
print(f"After flatten: {net.classifier.layers[0].out_size}")
print(f"Final output: {net.classifier.out_size}")

# Test
x = brainstate.random.randn(2, 32, 32, 3)
y = net(x)
print(f"\nForward pass: {x.shape} -> {y.shape}")
```

### Script result

```text
Complex Network:
Input size: (32, 32, 3)
After features: (8, 8, 64)
After flatten: (4096,)
Final output: (10,)
Forward pass: (2, 32, 32, 3) -> (2, 10)
```

### Mini title

• Common mistakes -> Fix

## References

Use these references by scope instead of loading every module-building example up front.

### Core layer catalogs

- `references/libraries/prebuilt-layer-library.md`
  Use first for BrainState `Linear`, convolution, pooling, padding, dropout, and utility layers.
  Source mirrored: https://brainx.chaobrain.com/brainstate/tutorials/core/03_common_layers.html

- `references/libraries/prebuilt-activation-library.md`
  Use for activation, normalization, BatchNorm, LayerNorm, and related module choices.
  Source mirrored: https://brainx.chaobrain.com/brainstate/tutorials/core/04_activations_and_normalization.html

### Size inference references

- `references/size-inference-with-convolution.md`
  Small reference for `Conv2d.in_size`, `Conv2d.out_size`, `kernel_size`, `padding`, and `stride` when building convolutional modules.
  Source mirrored: https://brainx.chaobrain.com/brainstate/tutorials/core/03_common_layers.html

- `references/size-inference-with-pooling-flatten.md`
  Small reference for pooling dimension reduction and `Flatten` shape conversion before dense classifiers.
  Source mirrored: https://brainx.chaobrain.com/brainstate/tutorials/core/03_common_layers.html

### Full composition script

- `references/scripts/modern_cnn.py`
  Full reference for manual module composition with `Conv2d`, `BatchNorm2d`, `GELU`, `MaxPool2d`, `Linear`, `LayerNorm`, and `Dropout`.
  Source mirrored: https://brainx.chaobrain.com/brainstate/tutorials/core/04_activations_and_normalization.html

### Optional dynamics-network composition reference

- Building an SNN: https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/04_building_an_snn.html
  Use `references/brainstate-dynamics/` when module composition crosses into BrainState dynamics networks: `LIF` populations, projections, synapses, event communication, and a composed E/I network. Prefer the SNN workflow reference over the Training SNN tutorial when the goal is module organization rather than optimization.
