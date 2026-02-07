# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**memo-mcp** is an MCP (Model Context Protocol) server that exposes tools for writing, validating, and running [memo](https://github.com/kach/memo) probabilistic programs. The memo DSL compiles recursive multi-agent reasoning models to JAX array programs. This server helps LLMs produce correct memo code through validation, execution, documentation lookup, pattern templates, and example search.

## Repository Layout

- **server.py** — The entire MCP server. Defines 10 tools (see below). All tool logic, pattern templates (14 patterns), and handbook parsing live in this single file.
- **examples.json** — Pre-indexed metadata for ~18 demo programs, used by `search_examples` for keyword matching.
- **memo/** — Git submodule pointing to `https://github.com/kach/memo`. Contains the core DSL (`memo/memo/`), the Handbook reference (`memo/Handbook.md`), and 35+ demo files (`memo/demo/`).
- **venv/** — Python 3.14 virtual environment with jax, mcp, memo-lang, numpy, matplotlib.

## Running the Server

```bash
venv/bin/python server.py
```

The server communicates over stdio using MCP protocol. It is configured in `.mcp.json` for use with Claude Code.

## Tools

**Authoring & debugging:**
- `validate_memo` — compile-check code without executing
- `run_memo` — execute in an isolated subprocess
- `inspect_compiled` — show generated JAX code (`debug_print_compiled=True`)

**Reference & examples:**
- `get_handbook` — return memo DSL docs by section (anatomy, statements, expressions, running)
- `list_patterns` — browse 14 design pattern templates
- `search_examples` — keyword search across ~18 indexed demos

**Accessibility (explaining models to lay audiences):**
- `explain_memo` — static AST analysis producing plain-English model summary (agents, beliefs, choices, recursive structure)
- `narrate_results` — run a function call with `print_table=True` for labeled output
- `compare_scenarios` — run two parameter settings side-by-side
- `trace_reasoning` — show recursive execution flow (`debug_trace=True`)

## Architecture

The server uses `mcp.server.fastmcp.FastMCP` to register tools. Key design decisions:

- **Subprocess isolation for execution**: `_run_memo_subprocess` is a shared helper that spawns `venv/bin/python` in a subprocess to prevent JAX OOM or hangs from crashing the server. Used by `run_memo`, `narrate_results`, `compare_scenarios`, and `trace_reasoning`. `validate_memo` uses in-process `importlib` (compile-only, no execution).
- **Static analysis**: `explain_memo` uses Python's `ast` module to parse memo code without execution, extracting agents, statements, domains, recursive structure, and docstrings. Inside `thinks[...]` blocks, Python parses `agent: stmt` as `Slice` nodes (not `AnnAssign`).
- **Debug flag injection**: `inspect_compiled` and `trace_reasoning` inject debug flags into `@memo` decorators via regex before running.
- **Temporary files**: Validate and run tools write code to `tempfile` before processing.
- **Handbook sectioning**: `get_handbook` parses `memo/Handbook.md` by `##` headers, supporting section aliases (e.g., "anatomy", "statements", "expressions", "running").

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
