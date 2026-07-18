# BrainX Agent Skills

## What BrainX installs

`brainx-skill` distributes one canonical BrainX Agent Skill bundle for Claude Code, Codex, and Cursor. The current bundle contains only `brainx-install`. The npm package version is the release version for the CLI and the complete published bundle.

## Requirements

- Node.js 18 or newer
- macOS, Linux, or Windows
- Write access to your user-level skill directories

The installer does not require harness executables to be on `PATH`.

## Install

```bash
npx brainx-skill install
```

The installer detects existing harness directories, lets you keep the detected set or customize it, then installs globally or in the current project. Global installation is the default.

If npm resolves a stale cached package, request the latest release explicitly:

```bash
npx brainx-skill@latest install
```

## Update

```bash
npx brainx-skill update
```

To force npm to resolve the latest published release:

```bash
npx brainx-skill@latest update
```

Update uses the harness destinations recorded by the previous installation and does not prompt again. It replaces only skill directories proven to be owned by `brainx-skill`. If no managed installation exists, it leaves all destinations unchanged.

## Install when the internet connection is low

```bash
npx --registry=https://registry.npmmirror.com brainx-skill install
```

## Installation locations

Depending on the selected scope, the canonical `brainx-install` skill is installed into:

```text
Claude Code: ~/.claude/skills/<skill-name> or <cwd>/.claude/skills/<skill-name>
Codex:       ~/.agents/skills/<skill-name> or <cwd>/.agents/skills/<skill-name>
Cursor:      ~/.cursor/skills/<skill-name> or <cwd>/.cursor/skills/<skill-name>
```

Installation ownership is recorded in `~/.brainx/receipt.json`.

## Managed-files warning

Installed BrainX skill directories are package-managed and may be replaced during install or update. Do not edit installed copies. Edit the canonical skill in the source repository and publish a new package release instead.

Directories that are not proven to be BrainX-managed are never overwritten or deleted.

## Troubleshooting

- Conflict errors identify an existing unowned directory. Move or remove that directory, then retry.
- Permission errors identify the affected agent and path. Restore write access to that user-level directory, then retry.
- An invalid receipt blocks destructive changes. Inspect `~/.brainx/receipt.json` and restore a valid receipt before retrying.
- A partial failure can leave one agent updated and the other unchanged. Resolve the reported destination and run the same command again.

## Development and publishing notes

Run the tests and inspect the package contents without publishing:

```bash
npm test
npm pack --dry-run
npm pack
test_home="$(mktemp -d '/tmp/brainx skills.XXXXXX')"
HOME="$test_home" npx --yes --package ./brainx-skill-1.0.1.tgz brainx install
HOME="$test_home" npx --yes --package ./brainx-skill-1.0.1.tgz brainx update
```

To preview the interaction directly from this checkout:

```bash
npx --yes --package . brainx install
```

Recheck that the npm name is available, authenticate, and publish with:

```bash
npm view brainx-skill
npm publish --access public
```

Do not run the publication command until the package contents and release version have been reviewed.
