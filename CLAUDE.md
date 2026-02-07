# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**memo-mcp** is an MCP (Model Context Protocol) server that exposes tools for writing, validating, and running [memo](https://github.com/kach/memo) probabilistic programs. The memo DSL compiles recursive multi-agent reasoning models to JAX array programs. This server helps LLMs produce correct memo code through validation, execution, documentation lookup, pattern templates, and example search.

## Repository Layout

- **server.py** — The entire MCP server. Defines 6 tools: `validate_memo`, `run_memo`, `search_examples`, `get_handbook`, `list_patterns`, `inspect_compiled`. All tool logic, pattern templates (14 patterns), and handbook parsing live in this single file.
- **examples.json** — Pre-indexed metadata for ~18 demo programs, used by `search_examples` for keyword matching.
- **memo/** — Git submodule pointing to `https://github.com/kach/memo`. Contains the core DSL (`memo/memo/`), the Handbook reference (`memo/Handbook.md`), and 35+ demo files (`memo/demo/`).
- **venv/** — Python 3.14 virtual environment with jax, mcp, memo-lang, numpy, matplotlib.

## Running the Server

```bash
venv/bin/python server.py
```

The server communicates over stdio using MCP protocol. It is configured in `.mcp.json` for use with Claude Code.

## Architecture

The server uses `mcp.server.fastmcp.FastMCP` to register tools. Key design decisions:

- **Subprocess isolation for execution**: `run_memo` spawns a subprocess via `venv/bin/python` to prevent JAX OOM or hangs from crashing the server. `validate_memo` uses in-process `importlib` (compile-only, no execution).
- **Temporary files**: Both validate and run write code to `tempfile` before processing.
- **Handbook sectioning**: `get_handbook` parses `memo/Handbook.md` by `##` headers, supporting section aliases (e.g., "anatomy", "statements", "expressions", "running").
- **inspect_compiled**: Injects `debug_print_compiled=True` into `@memo` decorators via regex, then captures the generated JAX code through stdout redirection.

## memo DSL Key Concepts

memo models use `@memo` decorated functions with typed axis variables (`def model[x: X, y: Y](...)`). Agents inside are declared by name and use statements like:
- `chooses` — agent selects from a domain
- `thinks` — recursive beliefs about other agents
- `observes` — Bayesian conditioning
- `wants` — expected utility maximization
- `knows` — shared knowledge between agents

Expressions include `E[]`, `Pr[]`, `H[]`, `KL[]`, `Var[]`, `imagine[]` for probabilistic/information-theoretic queries and counterfactuals.

## Dependencies

The memo submodule must be on `sys.path` for imports. The server handles this automatically (`sys.path.insert(0, str(MEMO_DIR))`). The `PYTHONPATH` is also set when spawning subprocesses for `run_memo`.
