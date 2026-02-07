"""memo DSL MCP Server — helps LLMs write correct memo code."""

import ast
import json
import os
import re
import sys
import subprocess
import tempfile
import importlib
import importlib.util
import uuid
from io import StringIO
from pathlib import Path
from contextlib import redirect_stdout

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MEMO_DIR = BASE_DIR / "memo"
VENV_PYTHON = str(BASE_DIR / "venv" / "bin" / "python")
HANDBOOK_PATH = MEMO_DIR / "Handbook.md"
EXAMPLES_PATH = BASE_DIR / "examples.json"

# Ensure memo is importable
if str(MEMO_DIR) not in sys.path:
    sys.path.insert(0, str(MEMO_DIR))

# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------
mcp = FastMCP("memo-dsl")


# ===== Tool 1: validate_memo ==============================================

@mcp.tool()
def validate_memo(code: str) -> str:
    """Compile memo code and return errors or success.

    Parses and compiles a complete Python script containing @memo functions.
    Does NOT execute the code — only checks for compilation errors.

    Args:
        code: Complete Python script with imports, domains, and @memo functions.
    """
    tmp_path = None
    mod_name = f"memo_validate_{uuid.uuid4().hex}"
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="memo_validate_")
        os.close(fd)
        with open(tmp_path, "w") as f:
            f.write(code)

        spec = importlib.util.spec_from_file_location(mod_name, tmp_path)
        if spec is None or spec.loader is None:
            return "Error: could not create module spec for temp file"

        module = importlib.util.module_from_spec(spec)
        module.__dict__["__file__"] = tmp_path
        sys.modules[mod_name] = module

        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            return _format_memo_error(exc)

        # Collect defined memo functions
        memo_funcs = []
        for name, obj in vars(module).items():
            if name.startswith("_"):
                continue
            if callable(obj) and hasattr(obj, "_shape"):
                memo_funcs.append(name)

        if memo_funcs:
            return f"Success! Defined memo functions: {', '.join(memo_funcs)}"
        else:
            return "Code compiled without errors, but no @memo functions were found."

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        sys.modules.pop(mod_name, None)


def _format_memo_error(exc: Exception) -> str:
    """Extract structured error info from MemoError or other exceptions."""
    parts = []

    # Check if it's a MemoError (avoid importing memo.core at top level)
    type_name = type(exc).__name__

    if type_name == "MemoError":
        parts.append(f"MemoError: {getattr(exc, 'message', str(exc))}")

        loc = getattr(exc, "loc", None)
        if loc:
            parts.append(f"  location: line {loc.line}, col {loc.offset}, in @memo {loc.name}")

        hint = getattr(exc, "hint", None)
        if hint:
            parts.append(f"  hint: {hint}")

        notes = getattr(exc, "__notes__", [])
        if notes:
            parts.append("  notes:")
            for note in notes:
                parts.append(f"    {note}")
    else:
        msg = str(exc).strip()
        if msg:
            parts.append(f"{type_name}: {msg}")
        else:
            # Empty exception (e.g. bare raise Exception() from memo parser).
            # Include the traceback so there's something to debug with.
            import traceback
            parts.append(f"{type_name} (no message). Traceback:")
            parts.append(traceback.format_exc())
        notes = getattr(exc, "__notes__", [])
        if notes:
            for note in notes:
                parts.append(f"  {note}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Shared subprocess runner
# ---------------------------------------------------------------------------

def _run_memo_subprocess(code: str, timeout: int = 30) -> str:
    """Run memo code in an isolated subprocess and return formatted output."""
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="memo_run_")
        os.close(fd)
        with open(tmp_path, "w") as f:
            f.write(code)

        env = os.environ.copy()
        python_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(MEMO_DIR) + ((":" + python_path) if python_path else "")

        result = subprocess.run(
            [VENV_PYTHON, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )

        parts = []
        if result.stdout.strip():
            parts.append(result.stdout.strip())
        if result.stderr.strip():
            parts.append(f"[stderr]\n{result.stderr.strip()}")
        if result.returncode != 0:
            parts.append(f"[exit code: {result.returncode}]")

        return "\n".join(parts) if parts else "(no output)"

    except subprocess.TimeoutExpired:
        return f"Execution timed out after {timeout} seconds."
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ===== Tool 2: run_memo ===================================================

@mcp.tool()
def run_memo(code: str, timeout: int = 30) -> str:
    """Execute memo code in an isolated subprocess and return output.

    Runs a complete Python script with @memo functions. Uses a subprocess
    for isolation so JAX OOM or hangs won't crash the server.

    Args:
        code: Complete Python script with imports, domains, @memo functions, and execution code.
        timeout: Maximum execution time in seconds (default 30).
    """
    return _run_memo_subprocess(code, timeout)


# ===== Tool 3: search_examples ============================================

@mcp.tool()
def search_examples(query: str) -> str:
    """Find relevant memo demo patterns by keyword search.

    Searches the example index by word overlap across tags, title,
    description, patterns, and concepts. Returns top 5 matches.

    Args:
        query: Natural language query, e.g. "bayesian belief update", "game theory", "POMDP".
    """
    examples = _load_examples()
    if not examples:
        return "No examples index found."

    query_words = set(query.lower().split())

    scored = []
    for ex in examples:
        searchable = " ".join([
            ex.get("title", ""),
            ex.get("description", ""),
            " ".join(ex.get("tags", [])),
            " ".join(ex.get("patterns", [])),
            " ".join(ex.get("concepts", [])),
            ex.get("category", ""),
        ]).lower()
        search_words = set(searchable.split())

        # Score by word overlap + substring matching
        score = 0
        for qw in query_words:
            if qw in search_words:
                score += 2
            elif any(qw in sw for sw in search_words):
                score += 1

        if score > 0:
            scored.append((score, ex))

    scored.sort(key=lambda x: -x[0])
    top = scored[:5]

    if not top:
        return f"No examples matched query '{query}'. Try broader terms like 'game', 'belief', 'planning', 'language'."

    parts = []
    for score, ex in top:
        parts.append(f"## {ex['title']} ({ex['file']})")
        parts.append(f"Category: {ex.get('category', 'N/A')}")
        parts.append(f"Description: {ex.get('description', 'N/A')}")
        parts.append(f"Patterns: {', '.join(ex.get('patterns', []))}")
        parts.append(f"Concepts: {', '.join(ex.get('concepts', []))}")
        if ex.get("snippet"):
            parts.append(f"\n```python\n{ex['snippet']}\n```")
        parts.append("")

    return "\n".join(parts)


def _load_examples() -> list[dict]:
    """Load examples.json."""
    if not EXAMPLES_PATH.exists():
        return []
    with open(EXAMPLES_PATH) as f:
        return json.load(f)


# ===== Tool 4: get_handbook ===============================================

@mcp.tool()
def get_handbook(section: str = "") -> str:
    """Return memo DSL reference documentation from the Handbook.

    Args:
        section: Which section to return. One of:
            "" — full handbook
            "anatomy" — Anatomy of a memo (structure, axes, parameters)
            "statements" — Statements (chooses, thinks, observes, knows, wants, etc.)
            "expressions" — Expressions (operators, E[], Pr[], H[], imagine[], etc.)
            "running" — Running and configuring memos (execution, options, autodiff)
    """
    if not HANDBOOK_PATH.exists():
        return "Handbook.md not found."

    text = HANDBOOK_PATH.read_text()

    if not section:
        return text

    sections = _split_handbook(text)
    key = section.lower().strip()

    aliases = {
        "anatomy": "anatomy",
        "structure": "anatomy",
        "statements": "statements",
        "statement": "statements",
        "expressions": "expressions",
        "expression": "expressions",
        "running": "running",
        "configuring": "running",
        "config": "running",
    }

    key = aliases.get(key, key)

    if key in sections:
        return sections[key]

    available = ", ".join(f'"{k}"' for k in sections)
    return f"Section '{section}' not found. Available sections: {available}"


def _split_handbook(text: str) -> dict[str, str]:
    """Split handbook into sections by top-level headers."""
    sections = {}

    # Split on "---" separators that divide the major sections
    # The handbook has: intro, Anatomy, Statements, Expressions, Running
    parts = re.split(r"\n---\n", text)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Find section by looking for ## or # headers
        if "Anatomy of a memo" in part:
            sections["anatomy"] = part
        elif part.startswith("# Statements") or "## `chooses`" in part:
            sections["statements"] = part
        elif part.startswith("# Expressions") or "## Literals" in part:
            sections["expressions"] = part
        elif "Running and Configuring" in part:
            sections["running"] = part

    return sections


# ===== Tool 5: list_patterns ==============================================

PATTERNS = {
    "basic_choice": {
        "description": "Simple agent choosing from a domain with softmax weighting",
        "template": """\
from memo import memo
from enum import IntEnum

class Actions(IntEnum):
    LEFT = 0
    RIGHT = 1

@memo
def model[a: Actions](beta):
    agent: chooses(a in Actions, wpp=exp(beta * utility(a)))
    return Pr[agent.a == a]
""",
        "reference": "demo-rsa.py",
    },
    "bayesian_inference": {
        "description": "Observer updating beliefs via thinks + observes pattern",
        "template": """\
from memo import memo
import jax.numpy as np

Hypotheses = np.arange(10)
Observations = np.arange(5)

@memo
def model[h: Hypotheses]():
    observer: thinks[
        world: chooses(h in Hypotheses, wpp=prior(h)),
        world: chooses(obs in Observations, wpp=likelihood(obs, h))
    ]
    observer: observes [world.obs] is obs
    return observer[Pr[world.h == h]]
""",
        "reference": "demo-monty.ipynb",
    },
    "recursive_reasoning": {
        "description": "Agents reasoning about each other recursively (level-k thinking)",
        "template": """\
from memo import memo

@memo
def model[x: X](level):
    listener: thinks[
        speaker: chooses(x in X, wpp=
            1 if level == 0 else model[x](level - 1))
    ]
    listener: observes [speaker.x] is x
    listener: chooses(x in X, wpp=Pr[speaker.x == x])
    return Pr[listener.x == x]
""",
        "reference": "demo-rsa.py",
    },
    "game_theory": {
        "description": "Two-player strategic interaction with utility maximization",
        "template": """\
from memo import memo
from enum import IntEnum

class Actions(IntEnum):
    COOPERATE = 0
    DEFECT = 1

@memo
def game[a1: Actions, a2: Actions](beta):
    alice: thinks[
        bob: chooses(a2 in Actions, wpp=exp(beta * bob_utility(a1, a2)))
    ]
    alice: chooses(a1 in Actions, wpp=exp(beta * alice_utility(a1, alice[E[bob.a2]])))
    bob: thinks[
        alice: chooses(a1 in Actions, wpp=exp(beta * alice_utility(a1, a2)))
    ]
    bob: chooses(a2 in Actions, wpp=exp(beta * bob_utility(bob[E[alice.a1]], a2)))
    return Pr[alice.a1 == a1, bob.a2 == a2]
""",
        "reference": "demo-ultimatum.ipynb",
    },
    "rsa": {
        "description": "Rational Speech Act model — speaker/listener reasoning about language",
        "template": """\
from memo import memo
import jax

@jax.jit
def denotes(u, r):
    return (u & r) != 0

@memo
def L[u: U, r: R](beta, t):
    listener: thinks[
        speaker: given(r in R, wpp=1),
        speaker: chooses(u in U, wpp=
            denotes(u, r) * (1 if t == 0 else exp(beta * L[u, r](beta, t - 1))))
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in R, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]
""",
        "reference": "demo-rsa.py",
    },
    "mdp": {
        "description": "Markov Decision Process — sequential planning over time steps",
        "template": """\
from memo import memo

@memo
def V[s: States](t, gamma):
    agent: chooses(a in Actions, wpp=1)
    agent: chooses(s_next in States, wpp=transition(s, a, s_next))
    agent: wants(utility = reward(s, a) + gamma * V[s_next](t - 1, gamma) if t > 0 else 0)
    agent: chooses(a in Actions, to_maximize=EU[utility])
    return E[agent[EU[utility]]]
""",
        "reference": "demo-mdp.ipynb",
    },
    "pomdp": {
        "description": "Partially Observable MDP — agent reasons under state uncertainty",
        "template": """\
from memo import memo

@memo
def policy[s: States](t, gamma):
    agent: thinks[
        world: chooses(s in States, wpp=prior(s)),
        world: chooses(obs in Obs, wpp=observe(s, obs))
    ]
    agent: observes [world.obs] is obs
    agent: chooses(a in Actions, to_maximize=agent[E[
        reward(world.s, a) + gamma * policy[world.s](t-1, gamma)
    ]] if t > 0 else 0)
    return E[agent.a]
""",
        "reference": "demo-pomdp.ipynb",
    },
    "belief_revision": {
        "description": "Agent updating beliefs based on observed evidence",
        "template": """\
from memo import memo

@memo
def model[h: Hypotheses]():
    agent: thinks[
        world: chooses(h in Hypotheses, wpp=prior(h)),
        world: chooses(evidence in Evidence, wpp=likelihood(evidence, h))
    ]
    agent: observes [world.evidence] is evidence
    return agent[Pr[world.h == h]]
""",
        "reference": "demo-chimp-belief-revision.ipynb",
    },
    "information_gain": {
        "description": "Computing expected information gain of an experiment",
        "template": """\
from memo import memo

@memo
def eig[e: Experiments]():
    scientist: thinks[
        world: chooses(h in Hypotheses, wpp=1),
        world: chooses(outcome in Outcomes, wpp=likelihood(outcome, h, e))
    ]
    return H[world.h] - scientist[E[H[world.h]]]
""",
        "reference": "demo-eig.ipynb",
    },
    "model_fitting": {
        "description": "Fitting memo model parameters to data using JAX autodiff",
        "template": """\
from memo import memo
import jax
import jax.numpy as np

@memo
def model[x: X](param):
    agent: chooses(x in X, wpp=exp(param * utility(x)))
    return Pr[agent.x == x]

data = np.array([0.2, 0.5, 0.3])

@jax.jit
def loss(param):
    return np.mean((model(param) - data) ** 2)

# Gradient descent
vg = jax.value_and_grad(loss)
param = 0.0
for _ in range(100):
    l, grad = vg(param)
    param = param - 0.1 * grad
""",
        "reference": "demo-rsa.py",
    },
    "domain_struct": {
        "description": "Using memo's domain() for structured compound domains",
        "template": """\
from memo import domain

# Compound domain with named components
S = domain(x=5, y=5)  # 25 states, each with .x and .y attributes

@jax.jit
def transition(s_next, action, s):
    x, y = S.x(s), S.y(s)
    # ... compute next state ...
    return s_next == S(new_x, new_y)
""",
        "reference": "demo-empowerment.py",
    },
    "multi_agent_with_knows": {
        "description": "Multiple agents with shared knowledge via knows()",
        "template": """\
from memo import memo

@memo
def model[x: X]():
    alice: chooses(x in X, wpp=1)
    bob: knows(alice.x)
    bob: chooses(y in Y, wpp=exp(alice.x + y))
    return E[bob.y]
""",
        "reference": "demo-sally-anne.ipynb",
    },
    "wants_eu": {
        "description": "Forward-looking utility via wants() + EU[] pattern",
        "template": """\
from memo import memo

@memo
def model[cup: Cups](level):
    alice: wants(win = my_cup != bob.poison)
    alice: thinks[
        bob: wants(kill = alice.my_cup == poison),
        bob: chooses(poison in Cups, to_maximize=EU[kill])
    ]
    alice: chooses(my_cup in Cups, to_maximize=EU[win])
    return Pr[alice.my_cup == cup]
""",
        "reference": "demo-vizzini.py",
    },
    "imagine_counterfactual": {
        "description": "Hypothetical reasoning with imagine[] blocks",
        "template": """\
from memo import memo

@memo
def model[u: U, n: N]():
    listener: thinks[
        speaker: chooses(n in N, wpp=1),
        speaker: chooses(u in U, wpp=imagine[
            listener: knows(u),
            listener: chooses(n in N, wpp=meaning(n, u)),
            Pr[listener.n == n]
        ])
    ]
    listener: observes [speaker.u] is u
    listener: chooses(n in N, wpp=E[speaker.n == n])
    return Pr[listener.n == n]
""",
        "reference": "demo-scalar.py",
    },
    "custom_jax_function": {
        "description": "Integrating custom JAX functions into memo models",
        "template": """\
from memo import memo
import jax
import jax.numpy as np

@jax.jit
def my_function(x, y):
    return np.cos(x) * np.sin(y)

@memo
def model[x: X]():
    agent: chooses(x in X, wpp=1)
    agent: chooses(y in Y, wpp=my_function(x, y))
    return E[agent.y]
""",
        "reference": "demo-7segment.ipynb",
    },
}


@mcp.tool()
def list_patterns(pattern: str = "") -> str:
    """List common memo design pattern templates with example code.

    Returns pattern name, description, template code, and reference demo file.

    Args:
        pattern: Specific pattern name to get details for, or "" to list all.
            Available: basic_choice, bayesian_inference, recursive_reasoning,
            game_theory, rsa, mdp, pomdp, belief_revision, information_gain,
            model_fitting, domain_struct, multi_agent_with_knows, wants_eu,
            imagine_counterfactual, custom_jax_function
    """
    if not pattern:
        parts = ["# memo Design Patterns\n"]
        for name, info in PATTERNS.items():
            parts.append(f"- **{name}**: {info['description']} (ref: {info['reference']})")
        parts.append(f"\nUse list_patterns(pattern='<name>') to get the full template code.")
        return "\n".join(parts)

    key = pattern.lower().strip()
    if key not in PATTERNS:
        available = ", ".join(PATTERNS.keys())
        return f"Pattern '{pattern}' not found. Available: {available}"

    info = PATTERNS[key]
    parts = [
        f"# Pattern: {key}",
        f"Description: {info['description']}",
        f"Reference: {info['reference']}",
        f"\n```python\n{info['template']}```",
    ]
    return "\n".join(parts)


# ===== Tool 6: inspect_compiled ============================================

@mcp.tool()
def inspect_compiled(code: str) -> str:
    """Show the generated JAX code for memo functions (for debugging).

    Injects debug_print_compiled=True into @memo decorators, compiles the code,
    and captures the generated JAX tensor operations.

    Args:
        code: Complete Python script with @memo functions.
    """
    # Inject debug_print_compiled=True into @memo decorators
    modified = _inject_debug_flag(code)

    tmp_path = None
    mod_name = f"memo_inspect_{uuid.uuid4().hex}"
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="memo_inspect_")
        os.close(fd)
        with open(tmp_path, "w") as f:
            f.write(modified)
        spec = importlib.util.spec_from_file_location(mod_name, tmp_path)
        if spec is None or spec.loader is None:
            return "Error: could not create module spec"

        module = importlib.util.module_from_spec(spec)
        module.__dict__["__file__"] = tmp_path
        sys.modules[mod_name] = module

        captured = StringIO()
        try:
            with redirect_stdout(captured):
                spec.loader.exec_module(module)
        except Exception as exc:
            output = captured.getvalue()
            error = _format_memo_error(exc)
            if output:
                return f"Compiled output (before error):\n{output}\n\nError:\n{error}"
            return f"Compilation error:\n{error}"

        output = captured.getvalue()
        if output:
            return f"Generated JAX code:\n\n{output}"
        else:
            return "No compiled output captured. The code may not contain @memo functions."

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        sys.modules.pop(mod_name, None)


def _inject_debug_flag(code: str) -> str:
    """Replace @memo decorators with @memo(debug_print_compiled=True)."""

    def _replace_memo_with_args(m: re.Match[str]) -> str:
        args = m.group(1)
        if "debug_print_compiled" in args:
            return m.group(0)
        return f"@memo({args}, debug_print_compiled=True)"

    # Handle @memo with existing args
    code = re.sub(r"@memo\(([^)]*)\)", _replace_memo_with_args, code)
    # Handle bare @memo (not followed by '(')
    code = re.sub(r"@memo\s*\n", "@memo(debug_print_compiled=True)\n", code)
    return code


# ===== Tool 7: explain_memo ================================================

@mcp.tool()
def explain_memo(code: str) -> str:
    """Parse memo code and return a structured plain-English description of the model's components.

    Statically analyzes the code (no execution) to extract:
    - @memo function signatures (axes, domains, parameters)
    - Agent declarations and their statements (chooses, thinks, observes, wants, knows, given)
    - Recursive reasoning structure (thinks[] nesting, self-referential calls)
    - Docstrings and return expressions

    Args:
        code: Complete Python script with imports, domains, and @memo functions.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"Syntax error: {e}"

    # Collect domain definitions (IntEnum classes and array domains)
    domains = _extract_domains(tree)

    # Find @memo-decorated functions
    memo_funcs = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and _has_memo_decorator(node)
    ]

    if not memo_funcs:
        return "No @memo functions found in the code."

    memo_func_names = {f.name for f in memo_funcs}
    parts = []

    for func in memo_funcs:
        parts.append(_explain_memo_function(func, domains, memo_func_names))

    # Domain summary
    if domains:
        domain_lines = ["Domains:"]
        for name, values in domains.items():
            if values:
                domain_lines.append(f"  {name}: {', '.join(values)}")
            else:
                domain_lines.append(f"  {name}")
        parts.insert(0, "\n".join(domain_lines) + "\n")

    return "\n".join(parts)


def _extract_domains(tree: ast.Module) -> dict[str, list[str]]:
    """Extract domain definitions: IntEnum classes and array-based domains."""
    domains: dict[str, list[str]] = {}
    for node in ast.iter_child_nodes(tree):
        # IntEnum classes
        if isinstance(node, ast.ClassDef):
            bases = [_unparse_node(b) for b in node.bases]
            if "IntEnum" in bases:
                members = []
                for item in node.body:
                    if isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name):
                                members.append(target.id)
                    elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        members.append(item.target.id)
                domains[node.name] = members
        # Simple array domains: X = np.arange(N) or similar assignments
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                src = _unparse_node(node.value)
                if "arange" in src or "array" in src or "domain(" in src:
                    domains[target.id] = []
    return domains


def _has_memo_decorator(node: ast.FunctionDef) -> bool:
    """Check if a function has @memo decorator."""
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "memo":
            return True
        if isinstance(dec, ast.Call):
            func = dec.func
            if isinstance(func, ast.Name) and func.id == "memo":
                return True
    return False


def _explain_memo_function(
    func: ast.FunctionDef,
    domains: dict[str, list[str]],
    memo_func_names: set[str],
) -> str:
    """Generate explanation for a single @memo function."""
    lines = []

    # Function signature with type params
    name = func.name
    type_params = _extract_type_params(func)
    params = _extract_params(func)
    docstring = ast.get_docstring(func)

    # Header
    sig_parts = []
    if type_params:
        axes_str = ", ".join(f"{ax}: {dom}" for ax, dom in type_params)
        sig_parts.append(f"{name}[{axes_str}]")
    else:
        sig_parts.append(name)
    if params:
        sig_parts_str = sig_parts[0] + f"({', '.join(params)})"
    else:
        sig_parts_str = sig_parts[0] + "()"
    lines.append(f"Model: {sig_parts_str}")

    # Axes with domain values
    if type_params:
        for ax, dom in type_params:
            if dom in domains and domains[dom]:
                lines.append(f"  Axis: {ax} over {dom} ({', '.join(domains[dom])})")
            else:
                lines.append(f"  Axis: {ax} over {dom}")

    # Parameters
    if params:
        lines.append(f"  Parameters: {', '.join(params)}")

    # Docstring
    if docstring:
        # Take first line of docstring as summary
        summary = docstring.strip().split("\n")[0]
        lines.append(f"  Purpose: {summary}")

    # Extract agents and their statements
    agents = _extract_agents(func.body, memo_func_names)
    if agents:
        lines.append("")
        lines.append("  Agents:")
        for agent_name, statements in agents.items():
            lines.append(f"    {agent_name}:")
            for stmt in statements:
                lines.append(f"      - {stmt}")

    # Detect recursive calls
    recursive_calls = _find_recursive_calls(func.body, name, memo_func_names)
    if recursive_calls:
        lines.append("")
        for call_desc in recursive_calls:
            lines.append(f"  Recursive: {call_desc}")

    # Return expression
    return_expr = _extract_return(func.body)
    if return_expr:
        lines.append(f"  Returns: {return_expr}")

    lines.append("")
    return "\n".join(lines)


def _extract_type_params(func: ast.FunctionDef) -> list[tuple[str, str]]:
    """Extract type parameters from function (e.g., def f[x: X, y: Y])."""
    params = []
    # Python 3.12+ type_params attribute
    for tp in getattr(func, "type_params", []):
        if hasattr(tp, "name") and hasattr(tp, "bound") and tp.bound is not None:
            params.append((tp.name, _unparse_node(tp.bound)))
        elif hasattr(tp, "name"):
            params.append((tp.name, "?"))
    return params


def _extract_params(func: ast.FunctionDef) -> list[str]:
    """Extract regular function parameters (excluding self)."""
    params = []
    for arg in func.args.args:
        if arg.arg != "self":
            params.append(arg.arg)
    return params


def _extract_agents(
    body: list[ast.stmt],
    memo_func_names: set[str],
    depth: int = 0,
) -> dict[str, list[str]]:
    """Extract agent declarations and their statements from a function body.

    In memo, agents are declared via annotated assignments like:
        agent_name: statement(...)
    or
        agent_name: thinks[...]
    """
    agents: dict[str, list[str]] = {}

    for node in body:
        if not isinstance(node, ast.AnnAssign):
            continue
        if not isinstance(node.target, ast.Name):
            continue

        agent_name = node.target.id
        annotation = node.annotation

        if agent_name not in agents:
            agents[agent_name] = []

        stmt_desc = _describe_statement(annotation, memo_func_names, depth)
        if stmt_desc:
            agents[agent_name].append(stmt_desc)

    return agents


def _describe_statement(
    node: ast.expr,
    memo_func_names: set[str],
    depth: int,
) -> str:
    """Describe a single agent statement node."""
    # Function call: chooses(...), knows(...), wants(...), given(...)
    if isinstance(node, ast.Call):
        func_name = _unparse_node(node.func)

        if func_name == "chooses":
            return _describe_chooses(node)
        elif func_name == "knows":
            args = [_unparse_node(a) for a in node.args]
            return f"knows: {', '.join(args)}"
        elif func_name == "wants":
            # wants(name = expr)
            if node.keywords:
                kw = node.keywords[0]
                return f"wants: {kw.arg} = {_unparse_node(kw.value)}"
            return "wants: (utility)"
        elif func_name == "given":
            return _describe_chooses(node, verb="given")
        elif func_name == "observes":
            args = [_unparse_node(a) for a in node.args]
            return f"observes: {', '.join(args)}"
        else:
            return f"{func_name}({', '.join(_unparse_node(a) for a in node.args)})"

    # Subscript: thinks[...], observes[...] is ...
    if isinstance(node, ast.Subscript):
        func_name = _unparse_node(node.value)

        if func_name == "thinks":
            return _describe_thinks(node, memo_func_names, depth)
        elif func_name == "observes":
            return f"observes: {_unparse_node(node.slice)}"

    # Compare node: observes [...] is ...
    if isinstance(node, ast.Compare):
        left_src = _unparse_node(node.left)
        if "observes" in left_src:
            comparators = [_unparse_node(c) for c in node.comparators]
            return f"observes {left_src.replace('observes', '').strip()} is {', '.join(comparators)}"
        return _unparse_node(node)

    return _unparse_node(node)


def _describe_chooses(node: ast.Call, verb: str = "chooses") -> str:
    """Describe a chooses() or given() call."""
    # Parse positional args for "var in Domain" patterns
    choices = []
    extras = []
    for arg in node.args:
        src = _unparse_node(arg)
        choices.append(src)

    for kw in node.keywords:
        if kw.arg == "wpp":
            extras.append(f"weighted by {_summarize_expr(_unparse_node(kw.value))}")
        elif kw.arg == "to_maximize":
            extras.append(f"to maximize {_unparse_node(kw.value)}")

    desc = f"{verb}: {', '.join(choices)}" if choices else verb
    if extras:
        desc += f" ({'; '.join(extras)})"
    return desc


def _describe_thinks(
    node: ast.Subscript,
    memo_func_names: set[str],
    depth: int,
) -> str:
    """Describe a thinks[...] block, recursing into nested agent statements."""
    # The slice contains the nested agent statements
    inner = node.slice

    # Collect nested agent names
    nested_agents: dict[str, list[str]] = {}

    # thinks[] body is typically a Tuple of statements at the AST level
    stmts = []
    if isinstance(inner, ast.Tuple):
        stmts = inner.elts
    else:
        stmts = [inner]

    for stmt in stmts:
        # Inside thinks[], Python parses "agent: stmt" as Slice(lower=agent, upper=stmt)
        if isinstance(stmt, ast.Slice) and isinstance(stmt.lower, ast.Name):
            agent_name = stmt.lower.id
            if agent_name not in nested_agents:
                nested_agents[agent_name] = []
            if stmt.upper is not None:
                desc = _describe_statement(stmt.upper, memo_func_names, depth + 1)
                if desc:
                    nested_agents[agent_name].append(desc)
        # Also handle AnnAssign in case of different parse contexts
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            agent_name = stmt.target.id
            if agent_name not in nested_agents:
                nested_agents[agent_name] = []
            desc = _describe_statement(stmt.annotation, memo_func_names, depth + 1)
            if desc:
                nested_agents[agent_name].append(desc)

    if nested_agents:
        parts = []
        for ag, stmts_list in nested_agents.items():
            for s in stmts_list:
                parts.append(f"{ag}: {s}")
        return f"thinks about: [{'; '.join(parts)}]"

    return f"thinks: {_unparse_node(inner)}"


def _find_recursive_calls(
    body: list[ast.stmt],
    func_name: str,
    memo_func_names: set[str],
) -> list[str]:
    """Find calls to memo functions (including self-recursive calls) in the body."""
    calls = []
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Subscript):
            val = node.value
            if isinstance(val, ast.Name) and val.id in memo_func_names:
                target = val.id
                args_str = _unparse_node(node.slice)
                if target == func_name:
                    calls.append(f"{func_name} calls itself (self-recursive) with [{args_str}]")
                else:
                    calls.append(f"calls {target}[{args_str}]")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in memo_func_names and func.id != func_name:
                args = [_unparse_node(a) for a in node.args]
                calls.append(f"calls {func.id}({', '.join(args)})")
    # Deduplicate
    return list(dict.fromkeys(calls))


def _extract_return(body: list[ast.stmt]) -> str | None:
    """Extract the return expression from a function body."""
    for node in body:
        if isinstance(node, ast.Return) and node.value is not None:
            return _unparse_node(node.value)
    return None


def _summarize_expr(expr: str) -> str:
    """Shorten long expressions for display."""
    if len(expr) > 80:
        return expr[:77] + "..."
    return expr


def _unparse_node(node: ast.expr) -> str:
    """Convert an AST node back to source code."""
    try:
        return ast.unparse(node)
    except Exception:
        return "<?>"


# ===== Tool 8: narrate_results =============================================

@mcp.tool()
def narrate_results(code: str, call: str, timeout: int = 30) -> str:
    """Run a specific memo function call and return results as a labeled human-readable table.

    Takes model code (imports, domains, @memo definitions) and a function call,
    injects print_table=True for labeled output with domain value names, and
    runs it in an isolated subprocess.

    Args:
        code: Complete Python script with imports, domains, and @memo functions (no execution code).
        call: The function call to execute, e.g. "anxiety_level(0.5, 3)".
        timeout: Maximum execution time in seconds (default 30).
    """
    # Inject print_table=True into the call
    call = call.strip().rstrip(";")
    if call.endswith(")"):
        # Insert print_table=True before the closing paren
        inner = call[:-1].rstrip()
        if inner.endswith("("):
            # No args: func() -> func(print_table=True)
            modified_call = inner + "print_table=True)"
        else:
            modified_call = inner + ", print_table=True)"
    else:
        modified_call = call

    full_code = code.rstrip() + "\n\n" + modified_call + "\n"
    return _run_memo_subprocess(full_code, timeout)


# ===== Tool 9: compare_scenarios ===========================================

@mcp.tool()
def compare_scenarios(
    code: str,
    call_a: str,
    call_b: str,
    label_a: str = "Scenario A",
    label_b: str = "Scenario B",
    timeout: int = 30,
) -> str:
    """Run a model under two different parameter settings and present both results for comparison.

    Takes model code and two function calls, runs both with print_table=True,
    and presents the labeled results side by side.

    Args:
        code: Complete Python script with imports, domains, and @memo functions (no execution code).
        call_a: First function call, e.g. "anxiety_level(0.0, 5)".
        call_b: Second function call, e.g. "anxiety_level(1.5, 5)".
        label_a: Label for first scenario (default "Scenario A").
        label_b: Label for second scenario (default "Scenario B").
        timeout: Maximum execution time in seconds (default 30).
    """
    def _inject_print_table(call: str) -> str:
        call = call.strip().rstrip(";")
        if call.endswith(")"):
            inner = call[:-1].rstrip()
            if inner.endswith("("):
                return inner + "print_table=True)"
            return inner + ", print_table=True)"
        return call

    call_a_mod = _inject_print_table(call_a)
    call_b_mod = _inject_print_table(call_b)

    exec_code = code.rstrip() + "\n\n"
    exec_code += f'print("=== {label_a} ===")\n'
    exec_code += f"{call_a_mod}\n"
    exec_code += f'print()\n'
    exec_code += f'print("=== {label_b} ===")\n'
    exec_code += f"{call_b_mod}\n"

    return _run_memo_subprocess(exec_code, timeout)


# ===== Tool 10: trace_reasoning ============================================

@mcp.tool()
def trace_reasoning(code: str, timeout: int = 30) -> str:
    """Show the step-by-step execution flow of memo functions, revealing the recursive reasoning structure.

    Injects debug_trace=True into @memo decorators and runs the code,
    exposing function entry/exit, recursion depth, and timing information.
    This helps explain how recursive multi-agent reasoning unfolds.

    Args:
        code: Complete Python script with @memo functions and execution code.
        timeout: Maximum execution time in seconds (default 30).
    """
    modified = _inject_trace_flag(code)
    return _run_memo_subprocess(modified, timeout)


def _inject_trace_flag(code: str) -> str:
    """Replace @memo decorators with @memo(debug_trace=True)."""

    def _replace_memo_with_args(m: re.Match[str]) -> str:
        args = m.group(1)
        if "debug_trace" in args:
            return m.group(0)
        return f"@memo({args}, debug_trace=True)"

    # Handle @memo with existing args
    code = re.sub(r"@memo\(([^)]*)\)", _replace_memo_with_args, code)
    # Handle bare @memo (not followed by '(')
    code = re.sub(r"@memo\s*\n", "@memo(debug_trace=True)\n", code)
    return code


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run(transport="stdio")
