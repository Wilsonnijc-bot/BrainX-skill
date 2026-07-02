---
name: brainx-install
description: Guides BrainX environment setup, package installation, backend selection, and validation checks for BrainUnit, BrainState, and BrainCell. Use when the user asks to install BrainX, configure CPU/GPU/TPU or JAX, validate imports, check the Python environment, or troubleshoot setup issues.
---

# brainx-install/

## Concepts

• BrainX meta-package
BrainX is a meta-package that installs a curated set of BrainX components known to work well together.
**Source:** https://brainx.chaobrain.com/summ/install.html

• Requirements
Official requirements: Python 3.10–3.13, pip 23+, optional GPU/TPU drivers and libraries for accelerators.
**Source:** https://brainx.chaobrain.com/summ/install.html

• Hardware-specific install
Choose one that matches your platform and CUDA/toolchain.
**Source:** https://brainx.chaobrain.com/summ/install.html

• Domain routing
Docs group braincell / brainmass under Modeling, brainunit under Infrastructure, and brainstate under Compilation.
**Source:** https://brainx.chaobrain.com/summ/install.html

## Evoke case / task boundary

• use this skill when the task is “install BrainX / BrainCell / BrainState / BrainUnit”, validate an environment, choose CPU/GPU/TPU backend, or debug package import / accelerator setup.
• do not run install commands blindly.
• first identify: OS, Python version, package manager, existing environment, task domain, compute backend.
• choose domain before install: BrainState for stateful computation / transformations, BrainCell for single-cell or multicompartment cellular simulation, BrainUnit for unit safety, BrainMass only when the official task/domain requires it.
• do not guess CUDA extras; match the user’s platform and CUDA/toolchain.

Ask-before-install checkpoint

#### Script

```text
[NEEDS OFFICIAL SCRIPT SOURCE]
```

#### Explanation text

Official install docs give requirements and backend-specific commands, but they do not provide a single “probe environment before installing” script. Keep the skill rule: ask/identify OS, Python version, pip/conda/uv, virtualenv, CPU/GPU/TPU, and task domain before running commands.

Quick install

**Source:** https://brainx.chaobrain.com/summ/install.html

#### Script

```bash
pip install -U BrainX
```

#### Explanation text

This installs pinned versions of the core packages. Use only when the user wants the full curated BrainX stack.

Hardware-specific installs

**Source:** https://brainx.chaobrain.com/summ/install.html

#### Script

```bash
# CPU only
pip install -U BrainX[cpu]
# NVIDIA GPU (CUDA 12.x)
pip install -U BrainX[cuda12]
# NVIDIA GPU (CUDA 13.x)
pip install -U BrainX[cuda13]
# TPU
pip install -U BrainX[tpu]
```

#### Explanation text

Choose one that matches the platform and CUDA/toolchain. For detailed JAX wheel options and compatible CUDA/cuDNN versions, the install page points to JAX installation docs.
**Source:** https://brainx.chaobrain.com/summ/install.html

Pin exact release

**Source:** https://brainx.chaobrain.com/summ/install.html

#### Script

```bash
pip install BrainX==2025.9.15
```

#### Explanation text

Releases use date-style versions. Use this when the user needs reproducibility or a known component set.

Validate install

**Source:** https://brainx.chaobrain.com/summ/install.html

#### Script

```python
python -c "import BrainX, jax; print('BrainX OK'); print('JAX devices:', jax.devices())"
```

#### Explanation text

If using a GPU, the docs say a CUDA device should be listed.
**Source:** https://brainx.chaobrain.com/summ/install.html

## Full bundled script references

brainx-install-verify.py

**Source:** https://brainx.chaobrain.com/summ/install.html
**Purpose:** executable validation check for BrainX import and jax.devices().

brainx-install-commands.sh

**Source:** https://brainx.chaobrain.com/summ/install.html
**Purpose:** official command list for full, CPU, CUDA 12, CUDA 13, TPU, pinned, and source installs.

## Common mistakes -> Fix

• running install before identifying backend -> ask CPU/GPU/TPU and CUDA/toolchain first.
• unsupported Python -> use Python 3.10–3.13.
• old pip -> upgrade pip/setuptools/wheel.
• GPU install but no CUDA device after validation -> ensure installed JAX wheel matches CUDA toolkit version.
• user only needs BrainUnit -> do not force full BrainX stack unless they want the curated stack.
• user asks BrainCell simulation -> install stack that includes BrainCell and BrainUnit, not just raw JAX.
