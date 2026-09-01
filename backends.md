# Configurable DeepAgents backends

ChainAgents can use every backend shipped by DeepAgents 0.7.11. Backend
configuration is optional and belongs in `deepagent.toml`. When `[backend]` is
omitted, ChainAgents preserves its historical local composite backend: agent
state is the default, `/workspace/` is the local project, `/memories/` uses the
LangGraph store in stateful mode, and generated files remain local.

## Architecture

An explicit `[backend]` always creates one `CompositeBackend`. The table
describes its default backend, which handles paths not claimed by a route.
ChainAgents installs these managed routes before applying `[[backend.routes]]`:

| Route | Initial backend | Purpose |
| --- | --- | --- |
| `/workspace/` | local virtual filesystem | Agent workspace |
| `/memories/` | LangGraph store | Stateful agent memory |
| `/workspace/.files/deepagent/` | local virtual filesystem | Large tool-result artifacts |
| `/workspace/.files/outputs/` | local virtual filesystem | Downloadable generated files |

An explicit route with the same normalized prefix replaces the managed route.
Other explicit routes are added. More-specific composite prefixes take
precedence during backend dispatch. Route prefixes must be absolute POSIX paths,
cannot be `/`, cannot contain `.` or `..` segments, and are normalized to one
trailing slash. Duplicate normalized prefixes are rejected.

A `state` route cannot replace or be nested beneath
`/workspace/.files/outputs/`. Generated-download requests run outside a graph
state and therefore cannot retrieve state-backed bytes after a graph run.
Non-virtual `filesystem` and `local_shell` nodes are also rejected at or beneath
that route because composite prefix stripping would otherwise turn a download
path into an unrestricted absolute host path.

`artifacts_root` is optional for explicit configurations and defaults to
`/workspace/.files/deepagent`. It controls where DeepAgents offloads large tool
results. It must be an absolute non-root virtual path and cannot equal or be
nested beneath `/workspace/.files/outputs/`, which is intentionally public.

One constructed backend instance is shared by the supervisor, model variants,
local synchronous and nested subagents, skill discovery, and each local
interface. The static LangGraph exports also share one instance. Independently
deployed async subagents keep the backend configured in their own deployment.

## Backend types and fields

Only the canonical type values below are accepted. There are no aliases, Python
factory hooks, or third-party plugin entry points. Unknown or backend-incompatible
fields fail during startup.

### State

`state` stores files in the current LangGraph agent state. It has no additional
fields and is suitable for ephemeral files scoped to graph state.

```toml
[backend]
type = "state"
```

### Store

`store` persists files through the runtime's concrete LangGraph store.
`namespace` is a required non-empty array of native-store-safe segments. Each
segment may contain letters, digits, `_`, `-`, `@`, `+`, `:`, or `~`. Store
backends are rejected when `[agent].state = "stateless"`.

```toml
[agent]
state = "stateful"

[backend]
type = "store"
namespace = ["chainagents", "workspace"]
```

### Filesystem

`filesystem` exposes a host directory. Relative `root_dir` values resolve from
the active `deepagent.toml`. `virtual_mode` defaults to `true`, and
`max_file_size_mb` defaults to `10`.

Setting `virtual_mode = false` allows unrestricted host path resolution. It is
rejected unless the explicit acknowledgement
`allow_unrestricted_host_filesystem = true` is present.

```toml
[backend]
type = "filesystem"
root_dir = "workspace"
virtual_mode = false
max_file_size_mb = 20
allow_unrestricted_host_filesystem = true
```

### Local shell

`local_shell` combines filesystem operations with host command execution.
Filesystem fields are supported together with:

- `timeout`: positive default command timeout in seconds; default `120`.
- `max_output_bytes`: positive captured-output limit; default `100000`.
- `env`: string-to-string command environment overrides; default empty.
- `inherit_env`: inherit the ChainAgents process environment; default `false`.
- `allow_unrestricted_host_execution`: required and must be `true`.

The default local-shell backend also requires
`[agent].execute_tool_enabled = true`. A routed local-shell backend still
requires the acknowledgement, but composite execution is always delegated to
the default backend, so a routed node does not add command execution.

```toml
[agent]
execute_tool_enabled = true

[backend]
type = "local_shell"
root_dir = "."
virtual_mode = true
max_file_size_mb = 10
timeout = 120
max_output_bytes = 100000
inherit_env = false
env = { PATH = "/usr/bin:/bin" }
allow_unrestricted_host_execution = true
```

### Context Hub

`context_hub` attaches a LangChain Context Hub repository by its required
`identifier`. Context Hub uses the standard `LANGSMITH_API_KEY` and
`LANGSMITH_ENDPOINT` environment variables. Repository identifiers and
credentials are never included in runtime status or startup logs.

```toml
[backend]
type = "state"

[[backend.routes]]
path = "/reference/"
type = "context_hub"
identifier = "owner/agent-repo"
```

### LangSmith sandbox

`langsmith_sandbox` attaches an existing sandbox by `sandbox_name`; ChainAgents
does not create, stop, or delete it. `api_endpoint` is optional and otherwise
uses `LANGSMITH_ENDPOINT`. `client_timeout` is a positive number with default
`10`, and `max_retries` is a non-negative integer with default `3`.
Authentication uses `LANGSMITH_API_KEY`. On shutdown, ChainAgents closes the
wrapper and client without stopping or deleting the sandbox. The Agent Server
uses its FastAPI lifespan so async clients close on the serving event loop.
Sandbox names, endpoints, and credentials are omitted from status and logs.
Set `[agent].execute_tool_enabled = true` to expose sandbox command execution;
without it, sandbox filesystem tools remain available but status and prompts
correctly report command execution as disabled.

```toml
[agent]
execute_tool_enabled = true

[backend]
type = "langsmith_sandbox"
sandbox_name = "existing-development-sandbox"
api_endpoint = "https://api.smith.langchain.com"
client_timeout = 10
max_retries = 3
```

## Routing examples

This configuration keeps the normal state default, replaces `/workspace/` with
a Context Hub repository, keeps agent artifacts under their default virtual
root, and adds a separate local reference tree. Explicit routes replace a
managed prefix only when the normalized prefixes are identical.

```toml
[backend]
type = "state"
artifacts_root = "/workspace/.files/deepagent"

[[backend.routes]]
path = "/workspace/"
type = "context_hub"
identifier = "owner/remote-workspace"

[[backend.routes]]
path = "/local-reference/"
type = "filesystem"
root_dir = "reference"
virtual_mode = true
max_file_size_mb = 10
```

A routed store can use a namespace separate from `/memories/`:

```toml
[agent]
state = "stateful"

[backend]
type = "state"

[[backend.routes]]
path = "/shared/"
type = "store"
namespace = ["chainagents", "shared"]
```

In stateless mode, all store nodes and any explicit `/memories/` override are
rejected because no runtime store exists.

## Security model

- Keep `virtual_mode = true` unless host-wide file access is intentional.
- Filesystem host access and local-shell execution require their exact explicit
  acknowledgement fields. A misspelling or omitted acknowledgement fails closed.
- `inherit_env = true` can expose process environment variables to commands. Its
  default is `false`; prefer a minimal `env` table.
- Configuration accepts only native DeepAgents types and fields. It cannot import
  arbitrary Python factories.
- `/api/status` exposes only the default type, route path/type summaries,
  execution capability, and whether `/workspace/` is local. It never returns
  environment values, credentials, repository identifiers, or sandbox names.
- Route traversal, root routes, duplicates, invalid values, directories, backend
  errors, and generated-output traversal all fail closed.

## Persistence and memory

Stateful runtimes initialize the checkpointer and store before constructing the
backend. The same concrete in-memory or PostgreSQL store is injected into every
configured `StoreBackend`, including the managed `/memories/` route.

With `[agent].state = "stateless"`, ChainAgents does not create a checkpointer or
store, does not install `/memories/`, and rejects `store` backends and explicit
`/memories/` routes. Other backends remain available.

## Remote workspaces and skills

When `/workspace/` is replaced by a remote backend, agent workspace instructions
and startup status identify it as configured rather than claiming it maps to a
host path. Skill discovery uses the shared backend and preserves virtual skill
paths. Local synchronous subagents and model variants receive that same backend.

Relative host-resource fields such as backend `root_dir`, prompt files, and MCP
working directories resolve from `deepagent.toml`; they do not resolve inside a
remote workspace. Skill source paths are the exception: relative skill paths are
normalized beneath `/workspace/` and therefore follow the configured workspace
backend.

## RAG remains host-project based

Workspace-document RAG intentionally continues to index the host project. Its
include/exclude globs, persisted index, and uploads are not redirected by a
remote `/workspace/` route. Configure RAG independently, and remember that
Snowflake Cortex or Anthropic chat providers still require an explicit Ollama or
OpenAI-compatible embedding provider instead of `auto`.

## Generated file downloads

Generated downloads use the unchanged URL
`/api/generated-files/{relative_path}`. Only normalized regular files beneath
`/workspace/.files/outputs/` are eligible.

For a local output route, ChainAgents retains the local path fast path. For a
remote output route, it retrieves bytes with the backend download API, supplies
byte-backed Chainlit file elements, and returns the bytes from FastAPI. Remote
file metadata is checked before download and bytes are checked again afterward.
Files are limited to 25 MiB each, at most 12 files are attached per response,
and traversal, directories, missing or unknown-size files, oversized content,
and backend errors are ignored or returned as not found.

## Troubleshooting

**Unknown backend type or field**

Use only the canonical types and fields documented here. Backend configuration
is strict so a typo cannot silently weaken isolation.

**Local shell says `execute_tool_enabled` is required**

Set `[agent].execute_tool_enabled = true` when `local_shell` is the default and
keep `allow_unrestricted_host_execution = true`. A routed local-shell node does
not make composite command execution available.

**Filesystem acknowledgement error**

Either restore `virtual_mode = true` or explicitly set
`allow_unrestricted_host_filesystem = true` after reviewing the host access.

**Store backend rejected**

Use `[agent].state = "stateful"`. Stateless runtimes have no store and cannot
expose `/memories/`.

**Context Hub or sandbox attachment fails**

Verify `LANGSMITH_API_KEY`, optional `LANGSMITH_ENDPOINT`, the repository
identifier or existing sandbox name, and network reachability. ChainAgents does
not provision these resources.

**A generated file is not downloadable**

Write it beneath `/workspace/.files/outputs/`, ensure it is a regular file, keep
remote files at or below 25 MiB, and confirm the configured output route supports
downloads.
