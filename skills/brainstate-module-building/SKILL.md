---
name: brainstate-module-building
description: Guides BrainState Module design, composition, nested graph structure, state traversal, prebuilt layers, activations, normalization, size inference, reusable components, stochastic modules, and random initialization. Use for fixed module dataflow, layer wiring, model organization, and Module-State relationships; route dynamic loops or runtime-dependent branches to BrainState control flow.
---

## Slide 12

### Title

Brainstate-Module building

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

#### Other important concepts
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

## Slide 13

### Title

BrainState-ModuleBuilding REFERENCES (reference.md)

### Reference markdown

#### Pre-built Activation Functions.md 
reorganize the content inside 
https://brainx.chaobrain.com/brainstate/tutorials/core/04_activations_and_normalization.html

### Reference markdown

#### Pre-built Basic Layers.md  
reference.md
reorganize the content inside https://brainx.chaobrain.com/brainstate/tutorials/core/03_common_layers.html

### Reference markdown

#### Size Inference with Convolution.md  
Convolution layers automatically compute output spatial dimensions based on:

Input spatial size

Kernel size

Stride

Padding mode

### Script

```python
# Create a 2D convolution layer
conv = brainstate.nn.Conv2d(
    in_size=(28, 28, 3),      # (height, width, channels)
    out_channels=32,
    kernel_size=3,
    stride=1,
    padding='SAME'
)

print("Conv2d Layer:")
print(f"  in_size:  {conv.in_size}")
print(f"  out_size: {conv.out_size}")
print(f"\n  Input:  (H, W, C) = {conv.in_size}")
print(f"  Output: (H', W', C') = {conv.out_size}")
print("\nWith 'SAME' padding and stride=1, spatial dimensions are preserved!")

# Test with different padding
conv_valid = brainstate.nn.Conv2d(
    in_size=(28, 28, 3),
    out_channels=32,
    kernel_size=3,
    stride=2,
    padding='VALID'
)

print(f"\nWith 'VALID' padding and stride=2:")
print(f"  in_size:  {conv_valid.in_size}")
print(f"  out_size: {conv_valid.out_size}")
print("  Spatial dimensions are reduced!")
```

### Script result

```text
Conv2d Layer:
  in_size:  (28, 28, 3)
  out_size: (28, 28, 32)

  Input:  (H, W, C) = (28, 28, 3)
  Output: (H', W', C') = (28, 28, 32)

With 'SAME' padding and stride=1, spatial dimensions are preserved!

With 'VALID' padding and stride=2:
  in_size:  (28, 28, 3)
  out_size: (13, 13, 32)
  Spatial dimensions are reduced!
```

### Reference markdown

#### Size Inference with Pooling & Flatten.md  
Size Inference with Pooling and Flatten
Pooling layers reduce spatial dimensions, and Flatten layers convert multi-dimensional tensors to 1D vectors. BrainState tracks all these transformations automatically.

### Script

```python
# MaxPool reduces spatial dimensions
pool = brainstate.nn.MaxPool2d(
    in_size=(28, 28, 32),
    kernel_size=(2, 2),
    stride=(2, 2),
    channel_axis=-1
)

print("MaxPool2d Layer:")
print(f"  in_size:  {pool.in_size}  (H=28, W=28, C=32)")
print(f"  out_size: {pool.out_size}  (H=14, W=14, C=32)")
print("  Spatial dimensions reduced by 2x!")

# Flatten converts to 1D
flatten = brainstate.nn.Flatten(in_size=(14, 14, 32))

print(f"\nFlatten Layer:")
print(f"  in_size:  {flatten.in_size}  (3D tensor)")
print(f"  out_size: {flatten.out_size}  (1D vector)")
print(f"  Total elements: {14 * 14 * 32} = {flatten.out_size[0]}")
```

### Script result

```text
MaxPool2d Layer:
  in_size:  (28, 28, 32)  (H=28, W=28, C=32)
  out_size: (14, 14, 32)  (H=14, W=14, C=32)
  Spatial dimensions reduced by 2x!

Flatten Layer:
  in_size:  (14, 14, 32)  (3D tensor)
  out_size: (6272,)  (1D vector)
  Total elements: 6272 = 6272
```

### Explanation text

-> these illustrate more variations of size inference

### Mini title

#### Modern CNN script from activation function and normalization
Building a complete network with activations and normalization:

### Script

```python
import brainunit as u


class ModernCNN(brainstate.nn.Module):
    """CNN with modern activations and normalization."""

    def __init__(self, num_classes=10):
        super().__init__()

        # Block 1: Conv + BatchNorm + GELU
        self.conv1 = brainstate.nn.Conv2d((32, 32, 3), out_channels=64, kernel_size=(3, 3), padding='SAME')
        self.bn1 = brainstate.nn.BatchNorm2d((32, 32, 64))
        self.act1 = brainstate.nn.GELU()
        self.pool1 = brainstate.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Block 2
        self.conv2 = brainstate.nn.Conv2d((16, 16, 64), out_channels=128, kernel_size=(3, 3), padding='SAME')
        self.bn2 = brainstate.nn.BatchNorm2d((16, 16, 128))
        self.act2 = brainstate.nn.GELU()
        self.pool2 = brainstate.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), in_size=self.bn2.out_size)

        # Classifier
        self.fc1 = brainstate.nn.Linear((128 * 8 * 8,), (256,))
        self.ln = brainstate.nn.LayerNorm((256,))
        self.act3 = brainstate.nn.GELU()
        self.dropout = brainstate.nn.Dropout(prob=0.5)
        self.fc2 = brainstate.nn.Linear((256,), (num_classes,))

    def update(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)

        # Classifier
        x = u.math.flatten(x, start_axis=1)
        x = self.fc1(x)
        x = self.ln(x)
        x = self.act3(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# Create and test
brainstate.random.seed(0)
model = ModernCNN(num_classes=10)

# Forward pass
x = brainstate.random.randn(4, 32, 32, 3)  # 4 images
with brainstate.environ.context(fit=True) as env:
    logits = model(x)

print("Modern CNN with GELU + BatchNorm + LayerNorm:")
print(model)
print()
print("Input:", x.shape)
print("Output:", logits.shape)
print()
print("Logits:", logits[0])
```

### Explanation text

-> illustrate combining pre-built layers

### Mini title

#### Skill for Deeplearning Training

### Script reference

#### Training Spiking Neural Network script
https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/05_training_an_snn.html

### Explanation text

-> under Example - Brain Dynamics

