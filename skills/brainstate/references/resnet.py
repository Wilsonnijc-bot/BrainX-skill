"""Residual Module composition pattern for BrainState.

Use this reference when a model needs skip connections or a dynamically
constructed sequence of registered child Modules.
"""

import brainstate


class ResidualBlock(brainstate.nn.Module):
    """Residual block: y = F(x) + x."""

    def __init__(self, size):
        super().__init__()
        self.linear1 = brainstate.nn.Linear(in_size=size, out_size=size)
        self.activation = brainstate.nn.ReLU()
        self.linear2 = brainstate.nn.Linear(in_size=size, out_size=size)

    def update(self, x):
        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x + residual


class ResNet(brainstate.nn.Module):
    """Small residual network with explicitly registered dynamic blocks."""

    def __init__(self, input_size, hidden_size, output_size, n_blocks=3):
        super().__init__()
        self.input_proj = brainstate.nn.Linear(
            in_size=input_size,
            out_size=hidden_size,
        )

        self.blocks = []
        for index in range(n_blocks):
            block = ResidualBlock(hidden_size)
            # Attribute assignment registers each child in the Module graph.
            setattr(self, f"block_{index}", block)
            self.blocks.append(block)

        self.output_proj = brainstate.nn.Linear(
            in_size=hidden_size,
            out_size=output_size,
        )

    def update(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)


brainstate.random.seed(0)
model = ResNet(
    input_size=(10,),
    hidden_size=(32,),
    output_size=(5,),
    n_blocks=3,
)

x = brainstate.random.randn(10)
y = model(x)
assert y.shape == (5,)

