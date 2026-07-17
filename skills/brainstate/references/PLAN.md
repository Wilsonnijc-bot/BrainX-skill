# BrainX Install Skill

## Summary

Create `/Users/nijiachen/Desktop/codex videos/brainx-install` as a confirmation-gated installer skill. It will inspect the machine read-only, select one coherent BrainX release, present an exact installation specification, and install into a Python virtual environment only after explicit confirmation.

The BrainX content will be derived exclusively from the supplied brief and these two fixed sources:

- `https://brainx.chaobrain.com/summ/CHANGELOG.html`
- `https://brainx.chaobrain.com/summ/install.html`

Automatic source-refresh behavior is deferred.

## Skill Structure And Interfaces

- Initialize the skill with `skill-creator/scripts/init_skill.py`, including:
  - `SKILL.md`
  - `agents/openai.yaml`
  - `scripts/inspect_environment.py`
  - `references/brainx-package-version-matrix.md`
- Use the skill name `brainx-install`.
- Set UI metadata:
  - Display name: `BrainX Install`
  - Short description: `Plan and install a compatible BrainX environment`
  - Default prompt: `Use $brainx-install to inspect this machine, propose a compatible BrainX environment, and install it after I confirm.`
- Trigger on BrainX installation, setup, migration, upgrade, package reconciliation, virtual-environment creation, or hardware-extra selection.

## Implementation Workflow

1. **Read-only inspection**
   - Collect OS/version/architecture, CPU model/count/load, total and available RAM, and visible NVIDIA, Apple, or TPU devices where detectable.
   - Detect CUDA toolkit/driver details needed to distinguish `cuda12` and `cuda13`.
   - Record the active Python executable/version, venv or environment-manager state, pip version, and candidate Python interpreters.
   - Require Python `>=3.11`. This follows the user requirement and the changelog’s removal of Python 3.10 support from `v2026.3.12`, despite the installation page’s broader `>=3.10` statement.
   - Inventory installed BrainX ecosystem packages in the active interpreter and an existing target venv, if present. Include BrainX, BrainUnit, BrainState, BrainEvent, BrainTools, BrainTrace/BrainScale, BrainCell, BrainPy, BrainPy-State, BrainMass, PINNx, JAX/JAXlib, and Optax.
   - Emit structured JSON with explicit `unknown`, `not_detected`, and command-error states. Perform no installation or environment mutation.

2. **Release selection**
   - Treat one BrainX release row as an indivisible compatibility set; never assemble independently selected newest subpackages.
   - If no existing BrainX component constrains selection, choose the latest supported non-yanked row, `2026.7.9`.
   - If one installed component exactly matches multiple rows, choose the newest non-yanked matching row.
   - If one installed component has no exact row, automatically choose the nearest declared component version by semantic version distance; break ties in favor of the newer BrainX release and disclose the mismatch.
   - If multiple installed components do not resolve to one coherent row, show the competing candidate rows and require the user to choose.
   - Treat an installed BrainX meta-package as an anchor, but surface drift if its component versions conflict.
   - Do not use `v2025.10.08`; distinguish `—`, `Removed`, and `Not bundled`; recognize BrainTrace replacing BrainScale from `v2025.12.2`.
   - Do not use JAX or Optax alone as a legacy anchor unless the user explicitly asks to preserve them.

3. **User-visible specification and gate**
   - Present the intended run location, chosen Python, venv path, OS/hardware target, selected extra (`cpu`, `cuda12`, `cuda13`, or `tpu`), existing packages, chosen BrainX release, complete declared component set, compatibility rationale, expected changes, and exact commands.
   - End with a direct request to confirm or correct the specification.
   - Do not create or modify the venv until confirmation.
   - Default to `<run-location>/.venv`. Inspect and propose reuse when it already exists and uses Python `>=3.11`; otherwise propose `.venv-brainx`. Never delete or overwrite an environment silently.
   - Use CPU on unsupported or uncertain accelerator platforms; ask before choosing between ambiguous CUDA/TPU targets.

4. **Confirmed installation**
   - Create or reuse only the confirmed venv.
   - Upgrade pip to satisfy the documented pip 23+ requirement.
   - Install the selected meta-package release with its hardware extra, for example:
     `"<venv-python>" -m pip install "BrainX[cpu]==2026.7.9"`
   - Do not install individual ecosystem packages separately.
   - Report pip completion or the exact failure and leave the venv intact for diagnosis.
   - Do not run import, device, or package-consistency verification afterward, matching the selected “install only” scope.

## Compatibility Reference

- Reproduce all supported changelog rows from `v2025.9.15` through `v2026.7.9` as readable Markdown tables.
- Include JAX constraints and every declared BrainX component version.
- Preserve the supplied interpretation notes for missing values, PINNx removal, Optax’s explicit pin, the BrainScale-to-BrainTrace transition, and the yanked release.
- Record source URLs and snapshot date without implementing automatic freshness checks.

## Test Plan

- Validate the folder with `quick_validate.py` and verify `agents/openai.yaml` against the skill body.
- Run the inspection helper locally and test partial-command failures, absent accelerators, missing JAX, non-venv Python, and Python 3.10 rejection.
- Exercise release-selection scenarios:
  - Clean CPU environment selects `2026.7.9`.
  - Exact historical package match selects the newest coherent row.
  - BrainPy `2.6.0` automatically selects its nearest supported row and discloses the change.
  - Conflicting installed components pause with candidate rows.
  - A match pointing only to `v2025.10.08` excludes the yanked release.
  - Existing `.venv` is inspected and never overwritten silently.
- Forward-test the skill in disposable directories for CPU, CUDA, unmatched legacy, and conflicting-package prompts; use mocked installation commands where a real package download is unnecessary.
- Acceptance requires zero mutation before confirmation and no modification of global or unrelated Python environments.
