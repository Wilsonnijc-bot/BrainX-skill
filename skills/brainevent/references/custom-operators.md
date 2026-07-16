# BrainEvent Custom Operators

Use this reference when the required event-driven computation is not covered by BrainEvent's built-in operations and a custom CPU or GPU kernel must be written.

BrainEvent's custom-operator tutorials extend the package from high-level Numba/Warp decorators down to hand-written C++ and CUDA. These are extension paths, not part of the canonical `BinaryArray @ connectivity` workflow.

## Tutorial Routing

| Implementation target | Open this official tutorial |
|---|---|
| Custom CPU operator with Numba | [Custom CPU Operators with Numba](https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/01_numba.html) |
| Custom GPU operator with Numba CUDA | [Custom GPU Operators with Numba CUDA](https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/02_numba_cuda.html) |
| Custom GPU operator with NVIDIA Warp | [Custom GPU Operators with Warp](https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/03_warp.html) |
| Hand-written C++ CPU kernel | [Custom C++ (CPU) kernels](https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/04_cpp.html) |
| Hand-written CUDA GPU kernel | [Custom CUDA (GPU) kernels](https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/05_cuda.html) |

Choose the tutorial matching the implementation technology already required by the task. Read that tutorial before selecting decorators, signatures, launch configuration, compilation, or registration details; the index alone does not define those APIs.

## Source

- Custom operators tutorial index: https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/index.html
