# memo-mcp

An MCP server that gives LLMs tools for writing, validating, running, and explaining [memo](https://github.com/kach/memo) probabilistic programs.

memo is a DSL that compiles recursive multi-agent reasoning models to JAX array programs. This server helps LLMs produce correct memo code and explain models to non-technical audiences.

## Tools

| Tool | Purpose |
|------|---------|
| `validate_memo` | Compile-check memo code without executing it |
| `run_memo` | Execute memo code in an isolated subprocess |
| `explain_memo` | Static analysis producing a plain-English model summary (agents, beliefs, choices, recursive structure) |
| `narrate_results` | Run a memo function and return a labeled table with domain value names |
| `compare_scenarios` | Run two parameter settings side-by-side for comparison |
| `trace_reasoning` | Show the recursive execution flow (entry/exit, depth, timing) |
| `inspect_compiled` | Show the generated JAX code for debugging |
| `search_examples` | Keyword search across ~18 indexed demo programs |
| `get_handbook` | Return memo DSL reference documentation |
| `list_patterns` | Browse 14 common design pattern templates |

## Installation

Requires Python 3.12+ (tested with 3.14).

```bash
git clone --recurse-submodules https://github.com/robert-johansson/memo-mcp.git
cd memo-mcp

python -m venv venv
venv/bin/pip install jax mcp memo-lang numpy matplotlib
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init
```

## Usage with Claude Code

Add to your Claude Code MCP config (`~/.claude/mcp.json` or project `.mcp.json`):

```json
{
  "mcpServers": {
    "memo-dsl": {
      "command": "/absolute/path/to/memo-mcp/venv/bin/python",
      "args": ["/absolute/path/to/memo-mcp/server.py"]
    }
  }
}
```

## Usage with other MCP clients

The server communicates over stdio:

```bash
venv/bin/python server.py
```

Any MCP-compatible client can connect using the stdio transport.

## Repository layout

```
server.py        -- MCP server (all 10 tools)
examples.json    -- Search index for demo programs
memo/            -- Git submodule: the memo DSL, Handbook, and 35+ demos
```

## Links

- [memo](https://github.com/kach/memo) -- the underlying DSL
- [MCP specification](https://modelcontextprotocol.io)
