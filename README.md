# BrainX Agent Skills

## What BrainX installs

`brainx-skills` distributes a canonical BrainX Agent Skill bundle and copies it into both Codex and Claude Code. The current bundle contains only `brainx-install`. The npm package version is the release version for the CLI and the complete published bundle.

## Requirements

- Node.js 18 or newer
- macOS, Linux, or Windows
- Write access to your user-level skill directories

The installer does not require the `codex` or `claude` executable to be on `PATH`.

## Install

```bash
npx brainx-skills install
```

If npm resolves a stale cached package, request the latest release explicitly:

```bash
npx brainx-skills@latest install
```

## Update

```bash
npx brainx-skills update
```

To force npm to resolve the latest published release:

```bash
npx brainx-skills@latest update
```

Update replaces only skill directories proven to be owned by `brainx-skills`. If no managed installation exists, it leaves both destinations unchanged.

## Installation locations

The canonical `brainx-install` skill is installed into:

```text
Codex:       ~/.agents/skills/<skill-name>
Claude Code: ~/.claude/skills/<skill-name>
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
HOME="$test_home" npx --yes --package ./brainx-skills-1.0.0.tgz brainx install
HOME="$test_home" npx --yes --package ./brainx-skills-1.0.0.tgz brainx update
```

Recheck that the npm name is available, authenticate, and publish with:

```bash
npm view brainx-skills
npm publish --access public
```

Do not run the publication command until the package contents and release version have been reviewed.
