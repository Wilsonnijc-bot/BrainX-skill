---

name: brainx-install
description: Use when inspecting, installing, upgrading, repairing, or verifying a BrainX environment. Ensures BrainX subpackages belong to one compatible BrainX release, checks Python and CPU/GPU/TPU requirements, and requires explicit user approval before any environment modification.
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# BrainX Installation and Environment Management

Use this skill when the user asks to:

* install BrainX or a BrainX subpackage;
* inspect an existing BrainX environment;
* resolve BrainX version conflicts;
* upgrade or downgrade BrainX;
* choose CPU, CUDA, or TPU support;
* verify whether BrainX packages are mutually compatible;
* repair a partially installed or mixed-version BrainX environment.

## Core invariant

BrainX compatibility is defined by a **BrainX release tuple**.

Compatible subpackages do not necessarily have the same numeric version. They must match the exact package versions selected by one published BrainX release.

Do not independently install the newest versions of:

```text
brainunit
brainevent
brainstate
braintools
braintrace
braincell
brainpy
brainpy-state
brainmass
```

Prefer installing one BrainX meta-package release with the confirmed hardware extra:

```text
BrainX[cpu]==<release>
BrainX[cuda12]==<release>
BrainX[cuda13]==<release>
BrainX[tpu]==<release>
```

## Safety rule

> Inspect automatically. Never mutate automatically.

Read-only inspection does not require confirmation.

Any action that installs, removes, upgrades, downgrades, creates, rebuilds, or edits an environment requires explicit user approval after the complete proposed plan has been shown.

## Boundaries

### This skill should

* inspect the current project and active Python environment;
* identify the environment manager already used by the project;
* collect installed BrainX, JAX, and related package versions;
* compare the observed package tuple with official BrainX releases;
* check the selected release’s Python requirement;
* inspect CPU, NVIDIA GPU, CUDA, TPU, and JAX device evidence;
* recommend a BrainX release and backend;
* show the exact packages and versions that would change;
* ask for explicit approval before making changes;
* verify the environment after an approved installation or repair.

### This skill should not

* silently install or reinstall packages;
* automatically create a virtual environment;
* assume `.venv`, `pip`, or any specific environment manager;
* silently select CPU, CUDA 12, CUDA 13, or TPU;
* install the latest version of every BrainX subpackage independently;
* upgrade `pip`, `setuptools`, `wheel`, JAX, or unrelated dependencies unless required and approved;
* use broad upgrade flags such as `--upgrade-strategy eager` without a specific reason;
* modify `pyproject.toml`, lockfiles, requirements files, Dockerfiles, or Conda files without approval;
* treat successful imports or `pip check` as proof that the BrainX tuple is officially compatible;
* replace a working historical BrainX release with the newest release unless the user requests an upgrade;
* infer that a GPU should be used merely because one is visible;
* claim compatibility when only an incomplete subset of the installed packages can be matched.

## Canonical workflow

Follow this order:

```text
inspect
→ identify the environment
→ collect installed versions
→ match a BrainX release tuple
→ check Python compatibility
→ inspect hardware and JAX
→ diagnose
→ propose an exact plan
→ ask for approval
→ modify only after approval
→ verify
→ report
```

# Phase 1: Read-only inspection

## 1. Inspect the project environment

Identify the project root and inspect relevant environment files.

Useful read-only commands include:

```bash
pwd
git status --short
```

Look for:

```text
pyproject.toml
uv.lock
poetry.lock
requirements.txt
requirements-dev.txt
environment.yml
conda-lock.yml
Pipfile
Pipfile.lock
.python-version
runtime.txt
Dockerfile
docker-compose.yml
```

Determine:

* the project’s environment manager;
* the intended Python environment;
* the active Python executable;
* whether the environment is virtualenv, uv, Poetry, Conda, Docker, Jupyter, system Python, or cluster-managed;
* whether dependency versions are controlled by a manifest or lockfile.

Do not assume that the shell command `python` points to the project interpreter.

Inspect the active interpreter:

```bash
python -c "import sys, platform; print(sys.executable); print(sys.version); print(platform.platform())"
python -m pip --version
```

When working in Jupyter, inspect the kernel interpreter rather than assuming it matches the terminal.

## 2. Inspect installed packages

Collect the installed versions of:

```text
BrainX
brainunit
brainevent
brainstate
braintools
braintrace
braincell
brainpy
brainpy-state
brainmass
jax
jaxlib
optax
```

A read-only inspection may use Python package metadata:

```bash
python -c "
from importlib import metadata

wanted = {
    'brainx',
    'brainunit',
    'brainevent',
    'brainstate',
    'braintools',
    'braintrace',
    'braincell',
    'brainpy',
    'brainpy-state',
    'brainmass',
    'jax',
    'jaxlib',
    'optax',
}

for dist in metadata.distributions():
    name = dist.metadata.get('Name', '')
    if name.lower() in wanted:
        print(f'{name}=={dist.version}')
"
```

Also inspect dependency consistency:

```bash
python -m pip check
```

Do not install missing inspection tools merely to complete this phase.

## 3. Determine the compatibility tuple

### When the BrainX meta-package is installed

Inspect its package metadata:

* installed BrainX version;
* `Requires-Python`;
* exact BrainX component requirements;
* JAX and JAXlib constraints;
* available hardware extras.

Compare the installed package versions with the exact requirements of that BrainX release.

Classify any changed component as release drift.

### When the BrainX meta-package is absent

Compare the observed BrainX component versions against official BrainX release tuples.

Determine whether the installed packages:

* exactly match one release;
* partially match one release;
* combine packages from different releases;
* cannot be matched confidently.

Do not infer official compatibility merely because:

* every package imports;
* `pip check` succeeds;
* package requirements use broad lower bounds;
* all installed versions are recent;
* the code appears to run.

If official release information is unavailable or ambiguous, report that compatibility could not be verified.

# Phase 2: Python and hardware assessment

## 4. Check Python compatibility

Compare the active interpreter with the `Requires-Python` metadata of the relevant BrainX release.

Do not use one hardcoded Python range for every BrainX version.

Report:

```text
Current Python:
Candidate or installed BrainX release:
Release Python requirement:
Status:
```

If Python is incompatible:

* do not attempt installation in the current interpreter;
* identify any existing compatible interpreter or environment when possible;
* otherwise propose creating or selecting a compatible environment;
* ask for approval before creating or changing environments.

## 5. Inspect the intended execution backend

Backend detection provides evidence, not permission.

The final choice must be shown to the user before installation.

### CPU

Recommend CPU when:

* the user explicitly requests CPU;
* no usable accelerator is confirmed;
* the project or CI is intentionally CPU-only;
* portability is more important than accelerator support;
* GPU or TPU evidence is ambiguous.

Do not silently choose CPU merely because accelerator inspection failed.

### NVIDIA GPU

Useful read-only checks include:

```bash
nvidia-smi
```

When JAX is already installed:

```bash
python -c "import jax; print(jax.__version__); print(jax.devices())"
```

Report:

* GPU model;
* driver version;
* CUDA compatibility reported by the driver;
* installed JAX and JAXlib versions;
* devices currently visible to JAX;
* recommended BrainX backend extra;
* any ambiguity caused by Docker, remote execution, cluster nodes, or environment isolation.

Do not select CUDA 12 or CUDA 13 based only on:

* the existence of a CUDA directory;
* a locally installed toolkit version;
* GPU hardware alone.

Use the actual runtime, driver, container, and supported JAX configuration.

### TPU

Inspect:

* TPU-related environment variables;
* runtime metadata;
* JAX devices when JAX is already available.

Recommend the TPU extra only when the execution environment is intended to run on TPU.

Ask the user to confirm that TPU is the intended target.

# Phase 3: Diagnose and request confirmation

## 6. Assign a compatibility classification

Use one of the following classifications.

### Verified release match

All required BrainX components exactly match one BrainX release.

Also verify:

* Python compatibility;
* JAX constraints;
* dependency consistency;
* intended hardware backend.

Do not reinstall unless the user explicitly requests it.

### Partial release match

The installed subset is consistent with one release, but:

* some BrainX packages are absent;
* the BrainX meta-package is absent;
* the full tuple cannot be verified.

Do not describe this as a complete BrainX installation.

Ask whether the user wants to keep the minimal subset or install the complete release.

### Mixed-version stack

Installed packages correspond to different BrainX releases, or no published release matches the complete tuple.

Identify:

* which packages match which release;
* which versions conflict;
* the smallest reasonable repair;
* the proposed target release.

### BrainX release drift

The BrainX meta-package is installed, but one or more components no longer match its declared requirements.

Prefer repairing the currently selected BrainX release.

Do not upgrade to the newest BrainX release unless requested or clearly justified and approved.

### Python incompatible

The current interpreter does not satisfy the selected release’s Python requirement.

Do not install BrainX in that interpreter.

### Backend uncertain

Hardware is visible, but the intended backend or supported CUDA/JAX path cannot be established confidently.

Ask the user to choose among the supported backends.

### No BrainX installation

No relevant BrainX packages are installed.

Inspect the project, Python version, environment manager, and hardware before proposing an installation.

## 7. Present the complete plan

Before any modification, report:

```text
BrainX environment inspection

Project environment
- Project root:
- Environment manager:
- Interpreter:
- Python version:
- Environment type:
- Dependency manifest or lockfile:

Installed packages
- BrainX:
- brainunit:
- brainevent:
- brainstate:
- braintools:
- braintrace:
- braincell:
- brainpy:
- brainpy-state:
- brainmass:
- jax:
- jaxlib:

Compatibility
- Classification:
- Matching release:
- Conflicts or missing packages:
- Python compatibility:
- Dependency consistency:

Hardware
- Detected devices:
- JAX-visible devices:
- Proposed backend:
- Remaining uncertainty:

Proposed action
- Target BrainX release:
- Packages to install:
- Packages to upgrade:
- Packages to downgrade:
- Packages to remove:
- Files or lockfiles that would change:
- Exact command:
```

End with a direct confirmation request:

```text
This action will modify the current environment. Should I proceed?
```

Stop after asking.

Do not run the proposed command in the same step unless the user has already explicitly approved that exact class of modification.

## Mandatory approval boundary

Explicit approval is required before running or performing:

```text
pip install
pip uninstall
uv add
uv remove
uv pip install
poetry add
poetry remove
poetry install
conda install
conda remove
mamba install
mamba remove
virtual environment creation
Conda environment creation
lockfile generation
lockfile updates
requirements-file edits
pyproject.toml edits
Docker image builds
Docker image rebuilds
package upgrades
package downgrades
package removals
```

Approval for inspection is not approval for installation.

Approval to install BrainX is not automatically approval to:

* upgrade unrelated dependencies;
* replace the project’s environment manager;
* rewrite dependency files;
* delete an existing environment;
* upgrade to a different BrainX release;
* switch from CPU to GPU or TPU;
* change the Python interpreter.

Request additional approval when the required action materially differs from the displayed plan.

# Phase 4: Installation or repair after approval

## 8. Preserve the existing project convention

After approval:

* use the environment manager already adopted by the project;
* target the confirmed interpreter and environment;
* preserve existing manifests and lockfiles unless their modification was approved;
* install one exact BrainX release;
* use the confirmed backend extra;
* avoid independently pinning BrainX components unless performing a narrowly justified repair.

Examples:

```bash
python -m pip install "BrainX[cpu]==<release>"
```

```bash
uv pip install "BrainX[cuda12]==<release>"
```

```bash
poetry add "BrainX[cuda13]==<release>"
```

```bash
python -m pip install "BrainX[tpu]==<release>"
```

Generate the real command from:

* the environment manager;
* the selected release;
* the confirmed backend;
* the project’s dependency-management convention.

Do not copy an example command blindly.

## 9. Repair before upgrading

When a BrainX release is already selected but has drifted:

1. prefer restoring that release’s exact tuple;
2. identify the smallest necessary set of changes;
3. avoid unrelated dependency upgrades;
4. propose a release upgrade separately when beneficial.

When no release matches:

1. determine the closest valid release;
2. consider the project’s existing imports and required BrainX packages;
3. propose exact upgrades and downgrades;
4. explain any compatibility risks;
5. install only after approval.

## 10. Handle optional component extras carefully

If the user needs an optional feature from a subpackage:

* first determine the component version selected by the target BrainX release;
* preserve that exact version;
* do not let installing the optional extra silently upgrade the component beyond the release tuple;
* show the resulting command and expected dependency changes before running it.

# Phase 5: Verification

After an approved installation or repair, verify:

## Package tuple

* the BrainX release is installed;
* every required BrainX component matches its declared version;
* no component has drifted;
* the installed subset is clearly identified when the complete bundle was not requested.

## Python and dependencies

Run the environment manager’s equivalent of:

```bash
python -m pip check
```

Confirm:

* Python satisfies `Requires-Python`;
* JAX and JAXlib satisfy the release requirements;
* dependency resolution did not introduce unrelated conflicts.

## Imports

Test imports for installed BrainX packages.

Do not claim that imports prove complete semantic compatibility; imports are only one verification layer.

## Hardware

When JAX is installed:

```bash
python -c "import jax; print(jax.__version__); print(jax.devices())"
```

Confirm that the visible JAX backend matches the user-approved backend.

A successful CUDA package installation is not sufficient if JAX still sees only CPU devices.

## Project validation

When appropriate:

* run the project’s existing tests;
* run a minimal existing BrainX example;
* avoid creating large custom test suites unless necessary;
* do not alter scientific code merely to make installation validation pass.

## Final report

Report:

```text
BrainX environment result

- Environment:
- Python:
- BrainX release:
- Backend:
- JAX-visible devices:
- Compatibility status:
- Packages changed:
- Files changed:
- Verification performed:
- Remaining warnings:
```

Be explicit about anything that was not verified.

# Decision rules

* Prefer a verified historical release over an unverified mixture of newer package versions.
* Prefer repairing an existing release over silently upgrading it.
* Prefer the project’s existing environment manager over introducing another one.
* Prefer exact BrainX release installation over independent component installation.
* Prefer explicit uncertainty over guessing.
* Prefer a minimal change set over broad dependency upgrades.
* Never treat hardware detection as user consent.
* Never modify the environment before presenting the plan and receiving approval.

# References

Open only the references needed for the current task.

* `compatibility-and-release-matching.md`

  * Source: `https://brainx.chaobrain.com/summ/`
  * Cover BrainX release tuples, exact and partial matching, release drift, historical releases, yanked releases, and compatibility evidence.

* `python-and-environment-inspection.md`

  * Cover interpreter discovery, pip, uv, Poetry, Conda, Docker, Jupyter, project manifests, lockfiles, and `Requires-Python`.

* `hardware-backend-selection.md`

  * Source: `https://brainx.chaobrain.com/summ/install.html`
  * Cover CPU, CUDA 12, CUDA 13, TPU, NVIDIA driver evidence, JAX device inspection, containers, remote machines, and clusters.

* `repair-and-verification.md`

  * Cover confirmation boundaries, minimal repair, exact installation commands, optional extras, snapshots, dependency checks, imports, JAX devices, and final reporting.
