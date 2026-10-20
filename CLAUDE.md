# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**memo-mcp** is an MCP (Model Context Protocol) server that exposes tools for writing, validating, and running [memo](https://github.com/kach/memo) probabilistic programs. The memo DSL compiles recursive multi-agent reasoning models to JAX array programs. This server helps LLMs produce correct memo code through validation, execution, documentation lookup, pattern templates, and example search.

## Repository Layout

- **server.py** — The entire MCP server. Defines 11 tools (see below). All tool logic, pattern templates (23 patterns), and handbook parsing live in this single file.
- **examples.json** — Pre-indexed metadata for demo programs, used by `search_examples` for keyword matching. Covers all 36 memo demos in `memo/demo/`; the only `.py` file there it deliberately skips is `test.py` (memo's own test suite), and the two `.wppl` files are WebPPL reference ports, not memo.
- **memo/** — Git submodule pointing to `https://github.com/kach/memo`, pinned to v1.3.2. Contains the core DSL (`memo/memo/`), the Handbook reference (`memo/Handbook.md`), and 36 demo files (`memo/demo/`).
- **test_server.py** — Regression checks: memo-version-specific syntax, every pattern template runs as-is and passes the critique style checks, the index covers every demo, every Handbook heading is reachable, `explain_memo` knows every statement form, and the stdio stream stays pure JSON-RPC when user code prints. Run `venv/bin/python test_server.py` after any change.
- **venv/** — Python 3.14 virtual environment with jax, mcp, memo-lang, numpy, matplotlib.

## Bumping memo

1. `git -C memo fetch --tags && git -C memo checkout vX.Y.Z`
2. Run memo's own suite against the venv: `cd memo/demo && PYTHONPATH=.. ../../venv/bin/python test.py` (every line must be `[ pass ... ]`).
3. `git -C memo diff OLD NEW -- memo/ Handbook.md demo/` for new syntax: teach `explain_memo` new statements, add a pattern for each new construct, and add `_HEADING_ALIASES` entries for Handbook keywords that are not in a heading.
4. Index new demos in `examples.json` (`test_server.py` fails on an unindexed demo).
5. `venv/bin/python test_server.py`, then update the version and counts in this file and README.md.

## Running the Server

```bash
venv/bin/python server.py
```

The server communicates over stdio using MCP protocol. It is configured in `.mcp.json` for use with Claude Code.

## Tools

**Authoring & debugging:**
- `validate_memo` — compile-check code and report memo's compile-time warnings (module-level code runs in-process; its stdout is captured)
- `critique_memo` — compile verdict + mechanical idiom checks (docstrings, frame naming, `wpp=1` vs `uniformly`, escape-hatch quarantine, decision-theoretic constructs) + the plain-English round-trip readback
- `run_memo` — execute in an isolated subprocess
- `inspect_compiled` — show generated JAX code (`debug_print_compiled=True`)

**Reference & examples:**
- `get_handbook` — return memo DSL docs: whole, by part (anatomy, statements, expressions, running), or by any `##`/`###` heading or keyword (`to_be`, `observes_that`, `cost`, ...)
- `list_patterns` — browse 23 design pattern templates; each is a complete script that runs as-is
- `search_examples` — keyword search across the 36 indexed demos (top 5 hits); call with `query=""` to browse the whole index grouped by category

**Accessibility (explaining models to lay audiences):**
- `explain_memo` — static AST analysis producing plain-English model summary (agents, beliefs, choices, recursive structure)
- `narrate_results` — run a function call with `print_table=True` for labeled output
- `compare_scenarios` — run two parameter settings side-by-side
- `trace_reasoning` — show recursive execution flow (`debug_trace=True`)

## Architecture

The server registers tools with the MCP SDK's decorator-based server class, imported
compatibly across SDK versions: 2.x exposes it as `mcp.server.mcpserver.MCPServer`,
1.x as `mcp.server.fastmcp.FastMCP`. `server.py` tries the 2.x path first and falls
back, aliasing either to the name `FastMCP`. The `.tool()` and `.run()` APIs are
identical across both. Key design decisions:

- **Subprocess isolation for execution**: `_run_memo_subprocess` is a shared helper that spawns `venv/bin/python` in a subprocess to prevent JAX OOM or hangs from crashing the server. Used by `run_memo`, `narrate_results`, `compare_scenarios`, and `trace_reasoning`.
- **In-process compiles**: `validate_memo` and `inspect_compiled` import the code in-process through `_exec_in_process`, which holds a lock (the SDK runs sync tools on worker threads, and stdout, warning filters and memo's state are process-global), captures stdout (this process's stdout *is* the stdio transport; a stray `print` would corrupt the JSON-RPC stream), and records memo's warnings. memo is imported before `catch_warnings`, because importing it replaces `warnings.showwarning`.
- **Static analysis**: `explain_memo` uses Python's `ast` module to parse memo code without execution, extracting agents, statements, domains, recursive structure, and docstrings. Inside `thinks[...]` blocks, Python parses `agent: stmt` as `Slice` nodes (not `AnnAssign`).
- **Debug flag injection**: `inspect_compiled` and `trace_reasoning` inject debug flags into `@memo` decorators via `_inject_memo_kwarg` (handles bare `@memo`, `@memo()`, `@memo(...)` and trailing comments, and keeps line numbers).
- **Temporary files**: Validate and run tools write code to `tempfile` before processing.
- **Handbook sectioning**: `get_handbook` splits `memo/Handbook.md` into its four parts at `---` separators, and otherwise finds a `#`–`###` heading (outside code fences) by exact title, `_HEADING_ALIASES`, whole word, then substring; a `##` section includes its `###` subsections.

## memo DSL Key Concepts

memo models use `@memo` decorated functions with typed axis variables (`def model[x: X, y: Y](...)`). Agents inside are declared by name and use statements like:
- `chooses` — agent selects from a domain
- `thinks` — recursive beliefs about other agents
- `observes` — Bayesian conditioning
- `wants` — expected utility maximization
- `knows` — shared knowledge between agents
- `observes_that[...]` / `observes_event(wpp=...)` — conditioning on a proposition / soft evidence
- `snapshots_self_as`, `forgets_about`, `has_theory` — snapshots, eager marginalization, alternate theories of mind
- `chooses(..., to_be= / such_that= / to_maximize= / to_minimize= / uniformly)` — deterministic and uniform choices

Expressions include `E[]`, `Pr[]`, `H[]`, `KL[]`, `Var[]`, `imagine[]` for probabilistic/information-theoretic queries and counterfactuals, and `time @ f(...)` / `space @` / `size @` (`cost @` = `time @`) for cost reflection. A memo may have several `return` statements; they come back stacked on a leading axis, and `f[i][axes](...)` calls one.

## Dependencies

The memo submodule must be on `sys.path` for imports. The server handles this automatically (`sys.path.insert(0, str(MEMO_DIR))`). The `PYTHONPATH` is also set when spawning subprocesses for `run_memo`.
