"""memo DSL MCP Server — helps LLMs write correct memo code."""

import ast
import json
import os
import re
import sys
import subprocess
import tempfile
import threading
import warnings
import importlib
import importlib.util
import uuid
from io import StringIO
from pathlib import Path
from contextlib import redirect_stdout

try:  # MCP SDK >= 2.0 renamed FastMCP to MCPServer
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:  # MCP SDK 1.x
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

    Parses and compiles a complete Python script containing @memo functions,
    and reports memo's compile-time warnings. memo compiles at decoration
    time, so the script's module-level code does run; its printed output is
    captured and discarded (use run_memo to see it).

    Args:
        code: Complete Python script with imports, domains, and @memo functions.
    """
    module, exc, output, caught = _exec_in_process(code, "memo_validate_")
    if exc is not None:
        return _format_memo_error(exc) + _format_warnings(caught)

    # Collect defined memo functions
    memo_funcs = []
    for name, obj in vars(module).items():
        if name.startswith("_"):
            continue
        if callable(obj) and hasattr(obj, "_shape"):
            memo_funcs.append(name)

    if memo_funcs:
        result = f"Success! Defined memo functions: {', '.join(memo_funcs)}"
    else:
        result = "Code compiled without errors, but no @memo functions were found."
    if output.strip():
        result += "\n(module-level code printed output; use run_memo to see it)"
    return result + _format_warnings(caught)


# In-process compiles share sys.stdout, the warnings filters and memo's own
# global state, and the SDK runs sync tools on worker threads: one at a time.
_COMPILE_LOCK = threading.Lock()


def _exec_in_process(code: str, prefix: str):
    """Import code as a throwaway module in this process.

    Returns (module, exception or None, captured stdout, warning messages).
    stdout is captured because this process's real stdout is the MCP stdio
    transport: a stray print would corrupt the JSON-RPC stream.
    """
    tmp_path = None
    mod_name = f"{prefix}{uuid.uuid4().hex}"
    captured = StringIO()
    caught: list[warnings.WarningMessage] = []
    with _COMPILE_LOCK:
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix=prefix)
            os.close(fd)
            with open(tmp_path, "w") as f:
                f.write(code)

            spec = importlib.util.spec_from_file_location(mod_name, tmp_path)
            if spec is None or spec.loader is None:
                return None, RuntimeError("could not create module spec for temp file"), "", []

            module = importlib.util.module_from_spec(spec)
            module.__dict__["__file__"] = tmp_path
            sys.modules[mod_name] = module

            # Importing memo replaces warnings.showwarning with a print to
            # stderr; do it before catch_warnings, which then records instead.
            try:
                importlib.import_module("memo")
            except Exception as exc:
                return module, exc, "", []

            with warnings.catch_warnings(record=True) as caught, redirect_stdout(captured):
                warnings.simplefilter("always")
                try:
                    spec.loader.exec_module(module)
                except Exception as exc:
                    return module, exc, captured.getvalue(), caught
            return module, None, captured.getvalue(), caught

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            sys.modules.pop(mod_name, None)


def _format_warnings(caught: list[warnings.WarningMessage]) -> str:
    """memo's compile-time warnings (redundant E[...], wpp independent of the choice, ...)."""
    messages = list(dict.fromkeys(str(w.message).strip() for w in caught))
    if not messages:
        return ""
    return "\n\nWarnings:\n" + "\n".join("  - " + m.replace("\n", "\n    ") for m in messages)


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
def search_examples(query: str = "") -> str:
    """Find relevant memo demo patterns by keyword search, or browse the full index.

    Searches the example index by word overlap across tags, title,
    description, patterns, and concepts. Returns top 5 matches.

    Args:
        query: Natural language query, e.g. "bayesian belief update", "game theory", "POMDP".
            Pass "" to list every indexed demo (titles and descriptions, no code).
    """
    examples = _load_examples()
    if not examples:
        return "No examples index found."

    # Browse mode. Search caps at 5 hits, so without this there is no way to
    # discover a demo whose vocabulary you cannot already guess.
    if not query.strip():
        by_category: dict[str, list[dict]] = {}
        for ex in examples:
            by_category.setdefault(ex.get("category", "uncategorized"), []).append(ex)

        parts = [f"# memo demos ({len(examples)} indexed)\n"]
        for category in sorted(by_category):
            parts.append(f"## {category}")
            for ex in sorted(by_category[category], key=lambda e: e["title"]):
                parts.append(f"- **{ex['title']}** ({ex['file']}) — {ex.get('description', '')}")
            parts.append("")
        parts.append("Use search_examples(query='<terms>') to get full code for the top matches.")
        return "\n".join(parts)

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
            or any heading inside them, by name or keyword, e.g. "to_be",
            "forgets_about", "has_theory", "observes_that", "imagine",
            "cost" (time @ / space @ / size @), "decorator options".
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

    headings = _handbook_headings(text)
    match = _find_heading(headings, key)
    if match is not None:
        return match[2]

    available = ", ".join(f'"{k}"' for k in sections)
    titles = ", ".join(f'"{_plain_heading(t)}"' for level, t, _ in headings if level >= 2)
    return f"Section '{section}' not found. Available sections: {available}; or a heading: {titles}"


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


# Keywords that name a construct documented under a heading that does not
# contain the keyword itself.
_HEADING_ALIASES = {
    "observes_that": "advanced observation constructs",
    "observes_event": "advanced observation constructs",
    "sees": "observes",
    "hears": "observes",
    "given": "aliases for chooses",
    "draws": "aliases for chooses",
    "assigned": "aliases for chooses",
    "guesses": "aliases for chooses",
    "such_that": "aliases for chooses",
    "uniformly": "aliases for chooses",
    "eu": "wants and eu",
    "time": "cost reflection",
    "space": "cost reflection",
    "size": "cost reflection",
    "e": "probabilistic operators",
    "pr": "probabilistic operators",
    "var": "probabilistic operators",
    "h": "information-theoretic operators",
    "kl": "information-theoretic operators",
    "entropy": "information-theoretic operators",
    "ffi": "calling custom jax functions",
    "jax": "calling custom jax functions",
    "braces": "referencing python variables",
    "escape": "referencing python variables",
    "cache": "decorator options",
    "debug_trace": "decorator options",
    "print_table": "executing a memo",
    "return_pandas": "executing a memo",
    "return_xarray": "executing a memo",
    "save_comic": "executing a memo",
    "autodiff": "automatic differentiation",
    "gradient": "automatic differentiation",
    "grad": "automatic differentiation",
    "multiple returns": "calling other memos",
}


def _plain_heading(title: str) -> str:
    """'Deterministic value with `to_be`' -> 'deterministic value with to_be'."""
    return re.sub(r"\s+", " ", title.replace("`", "")).strip().lower()


def _handbook_headings(text: str) -> list[tuple[int, str, str]]:
    """(level, title, body) for every heading outside code fences.

    A body runs to the next heading of the same or a higher level, so a ##
    section includes its ### subsections.
    """
    lines = text.splitlines()
    starts: list[tuple[int, int, str]] = []
    in_fence = False
    for i, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        m = re.match(r"(#{1,3}) (.+)", line)
        if m and not in_fence:
            starts.append((i, len(m.group(1)), m.group(2).strip()))

    headings = []
    for j, (i, level, title) in enumerate(starts):
        end = len(lines)
        for i2, level2, _ in starts[j + 1:]:
            if level2 <= level:
                end = i2
                break
        body = "\n".join(lines[i:end]).strip()
        body = re.sub(r"\n---\s*$", "", body).strip()
        headings.append((level, title, body))
    return headings


def _find_heading(headings: list[tuple[int, str, str]], key: str):
    """Exact title, then alias, then a whole word of a title, then a substring."""
    key = _plain_heading(key)
    plain = [(_plain_heading(t), h) for h in headings for t in [h[1]]]
    for title, h in plain:
        if title == key:
            return h
    if key in _HEADING_ALIASES:
        return _find_heading(headings, _HEADING_ALIASES[key])
    for title, h in plain:
        if key in re.split(r"[^\w]+", title):
            return h
    for title, h in plain:
        if key in title:
            return h
    return None


# ===== Tool 5: list_patterns ==============================================

PATTERNS = {
    "basic_choice": {
        "description": "One agent choosing from a domain with softmax (wpp=exp(beta * utility))",
        "template": """\
import jax
import jax.numpy as np
from memo import memo
from enum import IntEnum

class Snack(IntEnum):
    APPLE = 0
    CAKE = 1
    NOTHING = 2

@jax.jit
def utility(s):
    return np.array([1.0, 2.0, 0.0])[s]

@memo
def snack_choice[s: Snack](beta):
    \"\"\"How likely is a softmax-rational snacker to pick each snack?\"\"\"
    snacker: chooses(s in Snack, wpp=exp(beta * utility(s)))
    return Pr[snacker.s == s]

snack_choice(1.0, print_table=True)
""",
        "reference": "Memonomicon.ipynb",
    },
    "bayesian_inference": {
        "description": "Observer inferring a hidden cause: thinks[...] a generative model, observes the data, queries the posterior",
        "template": """\
import jax
import jax.numpy as np
from memo import memo

Bias = np.linspace(0, 1, 11)  # hypotheses: the coin's bias
Heads = np.arange(6)          # data: number of heads in 5 flips

@jax.jit
def binom5(k, p):
    comb = np.array([1, 5, 10, 10, 5, 1])[k]
    return comb * p ** k * (1 - p) ** (5 - k)

@memo
def posterior[heads: Heads, b: Bias]():
    \"\"\"After seeing `heads` heads in 5 flips, how likely is each bias?\"\"\"
    observer: thinks[
        coin: given(b in Bias, uniformly),
        coin: given(heads in Heads, wpp=binom5(heads, b))
    ]
    observer: observes [coin.heads] is heads   # `heads` is the axis value
    observer: knows(b)                         # needed to query Pr[... == b]
    return observer[Pr[coin.b == b]]

print(posterior()[4])  # posterior over Bias after 4 heads out of 5
""",
        "reference": "demo-monty.ipynb",
    },
    "recursive_reasoning": {
        "description": "Agents modeling each other to a depth (level-k): two memos calling each other with depth - 1",
        "template": """\
import jax
import jax.numpy as np
from memo import memo

Bar = np.arange(2)

@jax.jit
def prior(b):
    return np.array([0.55, 0.45])[b]

@memo
def alice[b: Bar](depth):
    \"\"\"Where does Alice go, if she wants to meet Bob and models him to `depth`?\"\"\"
    alice: thinks[ bob: chooses(b in Bar, wpp=bob[b](depth - 1)) ]
    alice: chooses(b in Bar, wpp=prior(b) * Pr[b == bob.b])
    return Pr[alice.b == b]

@memo
def bob[b: Bar](depth):
    \"\"\"Where does Bob go, if he wants to meet Alice and models her to `depth`?\"\"\"
    bob: thinks[ alice: chooses(b in Bar, wpp=alice[b](depth) if depth > 0 else 1) ]
    bob: chooses(b in Bar, wpp=prior(b) * Pr[b == alice.b])
    return Pr[bob.b == b]

for depth in range(1, 4):
    print(depth, alice(depth), bob(depth))
""",
        "reference": "demo-schelling.ipynb",
    },
    "game_theory": {
        "description": "Strategic interaction: a proposer imagines the responder's reply (imagine[...]) before choosing (ultimatum game)",
        "template": """\
import jax
import jax.numpy as np
from memo import memo
from enum import IntEnum

Offer = np.linspace(0, 1, 11)  # share offered to the responder

class Reply(IntEnum):
    ACCEPT = 0
    REJECT = 1

@jax.jit
def payout_proposer(offer, reply):
    return np.array([1 - offer, 0.0])[reply]

@jax.jit
def payout_responder(offer, reply):
    return np.array([offer, 0.0])[reply]

@memo
def responder[offer: Offer, reply: Reply](beta):
    \"\"\"How likely is the responder to accept or reject each offer?\"\"\"
    responder: knows(offer)
    responder: chooses(reply in Reply, wpp=exp(beta * payout_responder(offer, reply)))
    return Pr[responder.reply == reply]

@memo
def proposer[offer: Offer](beta):
    \"\"\"Which share does a proposer who anticipates the responder offer?\"\"\"
    proposer: chooses(offer in Offer, wpp=exp(beta * imagine[
        responder: knows(offer),
        responder: chooses(reply in Reply, wpp=responder[offer, reply](beta)),
        E[payout_proposer(offer, responder.reply)]
    ]))
    return Pr[proposer.offer == offer]

print(proposer(20.0))
""",
        "reference": "demo-ultimatum.ipynb",
    },
    "rsa": {
        "description": "Rational Speech Acts: a listener reasons about a speaker who reasons about a literal listener",
        "template": """\
import jax
from memo import memo
from enum import IntEnum

class U(IntEnum):  # utterances, as feature bits
    GREEN = 0b0001
    PINK = 0b0010
    SQUARE = 0b0100
    ROUND = 0b1000

class R(IntEnum):  # referents: the features they have
    GREEN_SQUARE = U.GREEN | U.SQUARE
    GREEN_CIRCLE = U.GREEN | U.ROUND
    PINK_CIRCLE = U.PINK | U.ROUND

@jax.jit
def denotes(u, r):
    return (u & r) != 0

@memo
def L[u: U, r: R](beta, t):
    \"\"\"Which referent does a depth-t pragmatic listener infer from utterance u?\"\"\"
    listener: thinks[
        speaker: given(r in R, uniformly),
        speaker: chooses(u in U, wpp=
            denotes(u, r) * (1 if t == 0 else exp(beta * L[u, r](beta, t - 1))))
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in R, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]

L(1.0, 1, print_table=True)
""",
        "reference": "demo-rsa.py",
    },
    "mdp": {
        "description": "Markov decision process: Q-value iteration as a recursive memo, plus inverse planning of the goal",
        "template": """\
import jax
import jax.numpy as np
from memo import memo

S = np.arange(5)          # corridor cells 0..4
A = np.array([0, 1])      # 0 = left, 1 = right
G = np.array([0, 4])      # goals: either end

@jax.jit
def Tr(s, a, s_):
    return 1.0 * (np.clip(s + 2 * a - 1, 0, 4) == s_)

@jax.jit
def R(s, a, g):
    return 1.0 * (s == g) - 0.1

@memo(cache=True)
def Q[s: S, a: A, g: G](t):
    \"\"\"Long-run value of action a in cell s toward goal g, with t steps to go.\"\"\"
    planner: knows(s, a, g)
    planner: given(s_ in S, wpp=Tr(s, a, s_))
    planner: chooses(a_ in A, to_maximize=0.0 if t < 1 else Q[s_, a_, g](t - 1))
    return E[R(s, a, g) + (0.0 if t < 1 else
                           0.0 if s == g else
                           Q[planner.s_, planner.a_, g](t - 1))]

@memo
def goal_inference[s: S, a: A, g: G](t):
    \"\"\"Seeing a planner take action a in cell s, which goal are they heading for?\"\"\"
    observer: knows(s, a, g)
    observer: thinks[
        planner: chooses(g in G, uniformly),
        planner: knows(s),
        planner: chooses(a in A, wpp=exp(2.0 * Q[s, a, g](t)))
    ]
    observer: observes [planner.a] is a
    return observer[Pr[planner.g == g]]

print(goal_inference(5)[2])  # in the middle cell: P(goal | moved left/right)
""",
        "reference": "demo-mdp.ipynb",
    },
    "pomdp": {
        "description": "POMDP over a discretized belief space: snapshots_self_as + imagine a future observation + forgets_about",
        "template": """\
import jax
import jax.numpy as np
from memo import memo
from enum import IntEnum

class S(IntEnum):
    HUNGRY = 0
    SATED = 1

class A(IntEnum):
    FEED = 0
    SING = 1
    IGNORE = 2

class O(IntEnum):
    CRYING = 0
    QUIET = 1

B = np.linspace(0, 1, 50)  # belief states: P(hungry)

@jax.jit
def get_belief(b, s):
    return np.array([b, 1 - b])[s]

@jax.jit
def Tr(s, a, s_):
    z = np.array([[0.0, 1.0, 1.0], [0.0, 0.1, 0.1]])[s, a]  # P(hungry next | s, a)
    return np.array([z, 1 - z])[s_]

@jax.jit
def Obs(o, s, a):
    z = np.array([[0.8, 0.9, 0.8], [0.1, 0.0, 0.1]])[s, a]  # P(crying | s, a)
    return np.array([z, 1 - z])[o]

@jax.jit
def R(s, a):
    return np.array([-10.0, 0.0])[s] + np.array([-5.0, -0.5, 0.0])[a]

@memo(cache=True)
def Q[b: B, a: A](t):
    \"\"\"Value of action a for a parent who believes P(hungry) = b, t steps from the end.\"\"\"
    parent: knows(b, a)
    parent: thinks[
        baby: knows(b, a),
        baby: chooses(s in S, wpp=get_belief(b, s)),
        baby: chooses(s_ in S, wpp=Tr(s, a, s_)),
        baby: chooses(o in O, wpp=Obs(o, s_, a))
    ]
    parent: snapshots_self_as(future_parent)
    return parent[E[R(baby.s, a)] + (0.0 if t <= 0 else 0.9 * imagine[
        future_parent: observes [baby.o] is baby.o,
        future_parent: chooses(b_ in B, wpp=exp(-100.0 * abs(E[baby.s_ == 0] - b_))),
        baby: forgets_about(s, s_, o),
        future_parent: chooses(a_ in A, to_maximize=Q[b_, a_](t - 1)),
        E[future_parent[Q[b_, a_](t - 1)]]
    ])]

q = Q(10)
print(q.argmax(axis=1))  # best action at each belief state
""",
        "reference": "demo-pomdp.ipynb",
    },
    "belief_revision": {
        "description": "Revising a belief as evidence of different strengths arrives (observes_that on boolean evidence, then choose)",
        "template": """\
import jax.numpy as np
from memo import memo
from enum import IntEnum
from jax.scipy.stats.bernoulli import pmf as ber

class Box(IntEnum):
    FIRST = 0
    SECOND = 1

class Food(IntEnum):
    EMPTY = 0
    FOODY = 1

class Evidence(IntEnum):
    CRUMBS = 0
    NOISES = 1
    VISUAL = 2

def p_foody_evidence(et, fs):
    return np.array([[0.5, 0.6], [0.3, 0.6], [0.01, 0.9]])[et, fs]

@memo
def second_choice[et1: Evidence, et2: Evidence, b: Box]():
    \"\"\"Which box does the chimp pick after foody evidence of type et1 (box 1) and et2 (box 2)?\"\"\"
    chimp: knows(et1, et2)
    chimp: wants(food=(world.box1 if b == {Box.FIRST} else world.box2) == {Food.FOODY})
    chimp: thinks[
        world: chooses(box1 in Food, uniformly),
        world: chooses(box2 in Food, uniformly),
        world: knows(et1, et2),
        world: chooses(e1 in Food, wpp=ber(e1, p_foody_evidence(et1, box1))),
        world: chooses(e2 in Food, wpp=ber(e2, p_foody_evidence(et2, box2)))
    ]
    chimp: observes_that [world.e1 == {Food.FOODY}]
    chimp: observes_that [world.e2 == {Food.FOODY}]
    chimp: chooses(b in Box, wpp=exp(10.0 * EU[food]))
    return Pr[chimp.b == b]

z = second_choice()
print("strong then weak -> P(second box):", z[Evidence.VISUAL, Evidence.NOISES, Box.SECOND])
print("weak then strong -> P(second box):", z[Evidence.NOISES, Evidence.VISUAL, Box.SECOND])
""",
        "reference": "demo-chimp-belief-revision.ipynb",
    },
    "information_gain": {
        "description": "Expected information gain of a question: H[...] now minus expected H[...] after imagining the answer",
        "template": """\
import jax
import jax.numpy as np
from memo import memo

N = np.arange(1, 7)       # a die roll
Q = np.arange(3)          # questions: is it 6? is it even? is it > 3?
A = np.array([0, 1])      # answers: no / yes

@jax.jit
def answer(q, n):
    return np.array([n == 6, n % 2 == 0, n > 3])[q]

@memo
def eig[q: Q]():
    \"\"\"How many bits does asking question q tell the asker about the roll?\"\"\"
    asker: knows(q)
    asker: thinks[
        roller: chooses(n in N, uniformly),
        roller: knows(q),
        roller: chooses(a in A, wpp=answer(q, n) == a)
    ]
    asker: snapshots_self_as(future_asker)
    return asker[imagine[
        future_asker: observes [roller.a] is roller.a,
        H[roller.n] - E[future_asker[H[roller.n]]]
    ]]

print(eig())  # nats; "is it even?" beats "is it 6?"
""",
        "reference": "demo-eig.ipynb",
    },
    "model_fitting": {
        "description": "Fitting a memo parameter to data with JAX: grid search with jax.vmap, then gradient descent with jax.value_and_grad",
        "template": """\
import jax
import jax.numpy as np
from memo import memo
from enum import IntEnum

class Option(IntEnum):
    LOW = 0
    MID = 1
    HIGH = 2

@jax.jit
def utility(o):
    return np.array([0.0, 1.0, 2.0])[o]

@memo
def choice[o: Option](beta):
    \"\"\"Choice probabilities of a softmax chooser with inverse temperature beta.\"\"\"
    chooser: chooses(o in Option, wpp=exp(beta * utility(o)))
    return Pr[chooser.o == o]

data = np.array([0.1, 0.3, 0.6])  # observed choice frequencies

@jax.jit
def loss(beta):
    return np.mean((choice(beta) - data) ** 2)

betas = np.linspace(0, 3, 31)
print("grid search:", betas[np.argmin(jax.vmap(loss)(betas))])

vg = jax.value_and_grad(loss)
beta = 0.0
for _ in range(200):
    l, g = vg(beta)
    beta = beta - 5.0 * g
print("gradient descent:", beta, l)
""",
        "reference": "demo-rsa.py",
    },
    "domain_struct": {
        "description": "Compound states with memo's domain(): pack/unpack named fields (here a 2-D gridworld)",
        "template": """\
import jax
import jax.numpy as np
from memo import memo, domain

Cell = domain(x=4, y=3)        # 12 states; Cell(x, y) packs, Cell.x(c) unpacks
Move = np.arange(4)            # left, right, up, down
DX = np.array([-1, 1, 0, 0])
DY = np.array([0, 0, -1, 1])

@jax.jit
def x_of(c):  # domain field accessors work in JAX code, not in memo expressions
    return Cell.x(c)

@jax.jit
def step(c, m, c_):
    x = np.clip(Cell.x(c) + DX[m], 0, 3)
    y = np.clip(Cell.y(c) + DY[m], 0, 2)
    return c_ == Cell(x, y)

@memo
def next_x[c: Cell]():
    \"\"\"Where (in x) does a uniformly random walker end up after one step from c?\"\"\"
    walker: knows(c)
    walker: chooses(m in Move, uniformly)
    walker: given(c_ in Cell, wpp=step(c, m, c_))
    return E[x_of(walker.c_)]

print(Cell)
print(next_x().reshape(4, 3))
""",
        "reference": "demo-empowerment.py",
    },
    "multi_agent_with_knows": {
        "description": "Who knows what: knows(...) shares a value, and a missed observation leaves a false belief (Sally-Anne)",
        "template": """\
import jax
import jax.numpy as np
from memo import memo
from enum import IntEnum

class Loc(IntEnum):
    BOX = 0
    BASKET = 1

class Action(IntEnum):
    STAY = 0
    MOVE = 1

class Obs(IntEnum):  # what Sally sees
    NONE = -1
    SAW_STAY = Action.STAY
    SAW_MOVE = Action.MOVE

@jax.jit
def do(l, a):
    return np.array([[0, 1], [1, 0]])[a, l]

@memo
def sally_looks[start: Loc, obs: Obs, look: Loc]():
    \"\"\"Where does the child think Sally will look, given what Sally saw?\"\"\"
    child: knows(start, obs, look)
    child: thinks[
        sally: knows(start),
        sally: thinks[
            anne: knows(start),
            anne: chooses(a in Action, wpp=0.01 if a == {Action.MOVE} else 0.99),
            anne: chooses(end in Loc, wpp=do(start, a) == end),
            anne: chooses(o in Obs, wpp=1 if o == {Obs.NONE} or o == a else 0)
        ],
        sally: observes [anne.o] is obs,
        sally: chooses(look in Loc, wpp=Pr[anne.end == look])
    ]
    return child[Pr[sally.look == look]]

sally_looks(print_table=True)
""",
        "reference": "demo-sally-anne.ipynb",
    },
    "wants_eu": {
        "description": "Forward-looking goals: wants(goal=...) names a utility over choices not yet made; to_maximize=EU[goal] optimizes it",
        "template": """\
from memo import memo
from enum import IntEnum

class Cups(IntEnum):
    NEAR_VIZZINI = 0
    NEAR_WESTLEY = 1

@memo
def vizzini_pick[cup: Cups](level):
    \"\"\"Which cup does Vizzini drink from, reasoning `level` deep about Westley?\"\"\"
    vizzini: wants(survive=my_cup != westley.poison)
    vizzini: thinks[
        westley: wants(kill=vizzini.my_cup == poison),
        westley: chooses(poison in Cups, to_maximize=EU[kill]),
        westley: thinks[
            vizzini: chooses(my_cup in Cups, wpp=
                vizzini_pick[my_cup](level - 1) if level > 0
                else my_cup == {Cups.NEAR_VIZZINI})
        ]
    ]
    vizzini: chooses(my_cup in Cups, to_maximize=EU[survive])
    return Pr[vizzini.my_cup == cup]

for level in range(4):
    print(level, vizzini_pick(level))
""",
        "reference": "demo-vizzini.py",
    },
    "imagine_counterfactual": {
        "description": "Hypothetical reasoning with imagine[...]: a speaker imagines how a literal listener would read each utterance",
        "template": """\
import jax
import jax.numpy as np
from memo import memo
from enum import IntEnum

NN = 10
N = np.arange(NN + 1)  # how many of NN people are nice

class U(IntEnum):
    NONE = 0
    SOME = 1
    ALL = 2

@jax.jit
def meaning(n, u):
    return np.array([n == 0, n > 0, n == NN])[u]

@memo
def scalar[n: N, u: U]():
    \"\"\"Hearing u, how many people does a pragmatic listener think are nice?\"\"\"
    listener: thinks[
        speaker: chooses(n in N, uniformly),
        speaker: chooses(u in U, wpp=imagine[
            listener: knows(u),
            listener: chooses(n in N, wpp=meaning(n, u)),
            Pr[listener.n == n]
        ])
    ]
    listener: observes [speaker.u] is u
    listener: chooses(n in N, wpp=E[speaker.n == n])
    return Pr[listener.n == n]

print(scalar()[:, U.SOME])  # "some" now excludes "all"
""",
        "reference": "demo-scalar.py",
    },
    "custom_jax_function": {
        "description": "Calling your own @jax.jit functions (scalars in, one scalar out) from inside a memo",
        "template": """\
import jax
import jax.numpy as np
from memo import memo

Angle = np.linspace(0, np.pi / 2, 10)

@jax.jit
def distance(theta):  # projectile range at unit speed
    return np.sin(2 * theta)

@memo
def throw[theta: Angle](beta):
    \"\"\"How likely is a thrower who wants distance to pick each angle?\"\"\"
    thrower: chooses(theta in Angle, wpp=exp(beta * distance(theta)))
    return Pr[thrower.theta == theta]

print(Angle[throw(20.0).argmax()])  # close to pi/4
""",
        "reference": "demo-7segment.ipynb",
    },
    "deterministic_choice": {
        "description": "Choices without noise: to_maximize / to_minimize (argmax/argmin), to_be (a fixed value), such_that (uniform over a subset)",
        "template": """\
import jax
import jax.numpy as np
from memo import memo

Num = np.arange(10)

@jax.jit
def is_prime(n):
    return np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 0])[n] == 1

@memo
def choices():
    \"\"\"Four deterministic choice rules side by side.\"\"\"
    picker: chooses(p in Num, such_that=is_prime(p))       # uniform over primes
    picker: chooses(big in Num, to_maximize=big)           # argmax
    picker: chooses(near in Num, to_minimize=abs(near - 6.4))  # argmin
    picker: chooses(copy in Num, to_be=p)                  # deterministic copy of p
    return E[picker.p]                      # 4.25 (mean of 2, 3, 5, 7)
    return E[picker.big]                    # 9
    return E[picker.near]                   # 6
    return Pr[picker.copy == picker.p]      # 1

print(choices())  # several returns come back stacked along a leading axis
""",
        "reference": "demo-smith.ipynb",
    },
    "conditioning": {
        "description": "Conditioning beyond observes: observes_that[cond] (hard evidence, like WebPPL condition) and observes_event(wpp=p) (soft evidence, like factor)",
        "template": """\
from memo import memo
from enum import IntEnum

class Kid(IntEnum):
    BOY = 0
    GIRL = 1

class Which(IntEnum):
    OLDER = 0
    YOUNGER = 1

@memo
def at_least_one_boy():
    \"\"\"Told 'at least one is a boy': how likely are both boys?\"\"\"
    reader: thinks[ family: given(c1 in Kid, c2 in Kid, uniformly) ]
    reader: observes_that[family.c1 == {Kid.BOY} or family.c2 == {Kid.BOY}]
    return reader[Pr[family.c1 == {Kid.BOY}, family.c2 == {Kid.BOY}]]

@memo
def saw_a_boy():
    \"\"\"Shown one random child, a boy: how likely are both boys?\"\"\"
    reader: thinks[
        family: given(c1 in Kid, c2 in Kid, uniformly),
        family: chooses(which in Which, uniformly),
        family: chooses(shown in Kid, to_be=c1 if which == {Which.OLDER} else c2)
    ]
    reader: observes_that[family.shown == {Kid.BOY}]
    return reader[Pr[family.c1 == {Kid.BOY}, family.c2 == {Kid.BOY}]]

@memo
def unreliable_report(reliability):
    \"\"\"A witness who is right with probability `reliability` says the older child is a boy.\"\"\"
    reader: thinks[ family: given(c1 in Kid, c2 in Kid, uniformly) ]
    reader: observes_event(wpp=reliability if family.c1 == {Kid.BOY} else 1 - reliability)
    return reader[Pr[family.c1 == {Kid.BOY}]]

print(at_least_one_boy(), saw_a_boy(), unreliable_report(0.8))  # 1/3, 1/2, 0.8
""",
        "reference": "demo-smith.ipynb",
    },
    "snapshot_self": {
        "description": "snapshots_self_as: keep a copy of an agent's beliefs before an update, to compare prior with posterior (surprise, belief change)",
        "template": """\
import jax.numpy as np
from memo import memo

Die = np.arange(1, 7)

@memo
def belief_change[r: Die]():
    \"\"\"How much does hearing 'it's even' raise a guesser's belief in each roll?\"\"\"
    guesser: thinks[ die: chooses(n in Die, uniformly) ]
    guesser: knows(r)                          # before the snapshot, so both selves know r
    guesser: snapshots_self_as(past_guesser)
    guesser: observes_that[die.n % 2 == 0]     # past_guesser is unaffected
    return guesser[Pr[die.n == r] - past_guesser[Pr[die.n == r]]]

print(belief_change())  # +1/6 for even rolls, -1/6 for odd ones
""",
        "reference": "demo-eig.ipynb",
    },
    "forgets_about": {
        "description": "forgets_about(x, ...): eagerly marginalize a choice no longer referenced, so later E[...]/Var[...] run faster",
        "template": """\
import jax.numpy as np
from memo import memo

Coin = np.array([0, 1])
Count = np.arange(4)

@memo
def tally():
    \"\"\"Mean and variance of the number of heads in three fair flips.\"\"\"
    flipper: chooses(f1 in Coin, f2 in Coin, f3 in Coin, uniformly)
    flipper: chooses(heads in Count, to_be=f1 + f2 + f3)
    flipper: forgets_about(f1, f2, f3)   # referencing flipper.f1 after this is an error
    return E[flipper.heads]
    return Var[flipper.heads]

print(tally())  # [1.5, 0.75]
""",
        "reference": "demo-pomdp.ipynb",
    },
    "has_theory": {
        "description": "has_theory(CHILD): an agent with an alternate theory of mind (cannot model others' beliefs via thinks)",
        "template": """\
from memo import memo
from enum import IntEnum

class Treat(IntEnum):
    BROCCOLI = 0
    GOLDFISH = 1

class Face(IntEnum):
    SMILE = 0
    SCOWL = 1

@memo
def offer[face: Face, treat: Treat]():
    \"\"\"Seeing the adult's face at the broccoli, which treat does the toddler offer them?\"\"\"
    toddler: has_theory(CHILD)
    toddler: thinks[
        adult: given(likes in Treat, uniformly),
        adult: chooses(face in Face, such_that=
            (face == {Face.SMILE}) if likes == {Treat.BROCCOLI} else (face == {Face.SCOWL}))
    ]
    toddler: observes [adult.face] is face
    toddler: chooses(treat in Treat, to_maximize=Pr[treat == adult.likes])
    return Pr[toddler.treat == treat]

offer(print_table=True)
# Other variants: NO_THINKS, NO_KNOWS, NO_WANTS; force one for a whole model with
# `from memo import Variant` and @memo(variant=Variant.CHILD).
""",
        "reference": "demo-broccoli.ipynb",
    },
    "multiple_returns": {
        "description": "Several return statements in one memo; pick one with f[i][axes](args), and call another memo's output by index",
        "template": """\
import jax.numpy as np
from memo import memo

Die = np.arange(1, 7)

@memo
def die_stats[k: Die]():
    \"\"\"Mean and variance of a fair roll known to be at most k.\"\"\"
    roller: knows(k)
    roller: chooses(n in Die, such_that=n <= k)
    return E[roller.n]       # return value 0
    return Var[roller.n]     # return value 1

@memo
def spread_per_mean[k: Die]():
    \"\"\"Use one return value of another memo: name[i][axes](args).\"\"\"
    return die_stats[1][k]() / die_stats[0][k]()

print(die_stats())        # shape (2, 6): row i holds return value i
print(spread_per_mean())
""",
        "reference": "demo-liu.ipynb",
    },
    "cost_reflection": {
        "description": "Measuring a memo's computational cost inside another memo: time @ f(...) (FLOPs; alias cost @), space @ (bytes), size @ (program size)",
        "template": """\
import jax
import jax.numpy as np
from memo import memo

N = np.arange(20)

@memo
def guess[n: N](depth):
    \"\"\"A guesser who thinks about a hider `depth` levels deep.\"\"\"
    guesser: thinks[ hider: chooses(n in N, wpp=1 if depth == 0 else guess[n](depth - 1)) ]
    guesser: chooses(n in N, wpp=Pr[hider.n == n])
    return Pr[guesser.n == n]

@memo
def thinking_cost(depth):
    \"\"\"What does reasoning `depth` levels deep cost?\"\"\"
    return time @ guess(depth)     # FLOPs; `cost @` is the same
    return space @ guess(depth)    # bytes of scratch memory
    return size @ guess(depth)     # syntax-tree nodes of the program

for depth in range(3):
    print(depth, thinking_cost(depth))   # [time, space, size]
# Outside a memo: guess(2, return_cost=True).aux.cost -> Cost(flops=..., bytes=..., program_size=...)
# Numbers depend on the memo and JAX versions; compare them, do not hard-code them.
""",
        "reference": "Handbook.md#cost-reflection",
    },
    "python_escape": {
        "description": "{...} escape hatch: reference Python globals (enum members, constants) or host expressions over parameters inside a memo",
        "template": """\
import jax.numpy as np
from memo import memo
from enum import IntEnum

class Weather(IntEnum):
    SUN = 0
    RAIN = 1

THRESHOLD = 0.3   # a plain Python constant

@memo
def take_umbrella(p_rain):
    \"\"\"Does a forecaster's listener take an umbrella when P(rain) is p_rain?\"\"\"
    listener: thinks[ sky: chooses(w in Weather, wpp=p_rain if w == {Weather.RAIN} else 1 - p_rain) ]
    return listener[Pr[sky.w == {Weather.RAIN}]] > {THRESHOLD}  # {THRESHOLD}: a host constant

print(take_umbrella(0.2), take_umbrella(0.5))
# Braces may hold any Python expression, even over parameters: {p_rain * 2}.
# They cannot mention agents or choices; prefer plain memo syntax where it exists.
""",
        "reference": "demo-chimp-belief-revision.ipynb",
    },
}


@mcp.tool()
def list_patterns(pattern: str = "") -> str:
    """List common memo design pattern templates with example code.

    Returns pattern name, description, template code, and reference demo file.
    Every template is a complete script that runs as-is (checked by
    test_server.py), so it can be copied and then modified.

    Args:
        pattern: Specific pattern name to get details for, or "" to list all.
            Available: basic_choice, bayesian_inference, recursive_reasoning,
            game_theory, rsa, mdp, pomdp, belief_revision, information_gain,
            model_fitting, domain_struct, multi_agent_with_knows, wants_eu,
            imagine_counterfactual, custom_jax_function, deterministic_choice,
            conditioning, snapshot_self, forgets_about, has_theory,
            multiple_returns, cost_reflection, python_escape
    """
    if not pattern:
        parts = ["# memo Design Patterns\n"]
        for name, info in PATTERNS.items():
            parts.append(f"- **{name}**: {info['description']} (ref: {info['reference']})")
        parts.append(f"\nUse list_patterns(pattern='<name>') to get the full template code (a complete, runnable script).")
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
    modified = _inject_memo_kwarg(code, "debug_print_compiled")

    _, exc, output, _ = _exec_in_process(modified, "memo_inspect_")
    if exc is not None:
        error = _format_memo_error(exc)
        if output:
            return f"Compiled output (before error):\n{output}\n\nError:\n{error}"
        return f"Compilation error:\n{error}"

    if output:
        return f"Generated JAX code:\n\n{output}"
    else:
        return "No compiled output captured. The code may not contain @memo functions."


def _inject_memo_kwarg(code: str, kwarg: str) -> str:
    """Add `kwarg=True` to every @memo decorator, bare or called.

    Edits in place, so line numbers in memo's error messages still match the
    user's code.
    """

    def _replace_memo_with_args(m: re.Match[str]) -> str:
        args = m.group(1)
        if kwarg in args:
            return m.group(0)
        sep = ", " if args.strip() else ""
        return f"@memo({args}{sep}{kwarg}=True)"

    # Handle @memo(...) with or without existing args
    code = re.sub(r"@memo\(([^)]*)\)", _replace_memo_with_args, code)
    # Handle bare @memo, possibly followed by a comment
    code = re.sub(r"@memo(?=[ \t]*(?:#[^\r\n]*)?\r?$)", f"@memo({kwarg}=True)", code, flags=re.M)
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
        for name, desc in domains.items():
            if desc:
                domain_lines.append(f"  {name}: {desc}")
            else:
                domain_lines.append(f"  {name}")
        parts.insert(0, "\n".join(domain_lines) + "\n")

    return "\n".join(parts)


def _extract_domains(tree: ast.Module) -> dict[str, str]:
    """Extract domain definitions and a short description of each: IntEnum
    classes (their members), array domains and domain(...) compounds (their
    defining expression)."""
    domains: dict[str, str] = {}
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
                domains[node.name] = ", ".join(members)
        # Simple array domains: X = np.arange(N) or similar assignments
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                src = _unparse_node(node.value)
                if any(s in src for s in ("arange", "linspace", "array", "domain(")):
                    domains[target.id] = _summarize_expr(src)
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
    domains: dict[str, str],
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

    # Decorator options, e.g. @memo(cache=True)
    options = _extract_options(func)
    if options:
        lines.append(f"  Options: {', '.join(options)}")

    # Axes with domain values, then any `forall: x in X` query variables
    axes = [(ax, dom, "") for ax, dom in type_params]
    axes += [(ax, dom, " [forall]") for ax, dom in _extract_foralls(func.body)]
    for ax, dom, how in axes:
        values = f" ({domains[dom]})" if domains.get(dom) else ""
        lines.append(f"  Axis: {ax} over {dom}{values}{how}")

    # Parameters
    if params:
        lines.append(f"  Parameters: {', '.join(params)}")

    # Docstring
    if docstring:
        # Take first line of docstring as summary
        summary = docstring.strip().split("\n")[0]
        lines.append(f"  Purpose: {summary}")

    # `cast: [alice, bob]` restricts which agents may appear
    cast = _extract_cast(func.body)
    if cast is not None:
        lines.append(f"  Cast: {', '.join(cast)}")

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

    # Return expression(s); a memo with several returns yields one value per return
    return_exprs = _extract_returns(func.body)
    if len(return_exprs) == 1:
        lines.append(f"  Returns: {return_exprs[0]}")
    elif return_exprs:
        lines.append(f"  Returns {len(return_exprs)} values (stacked on a leading axis; "
                     f"call one as {name}[i][...](...)):")
        for i, expr in enumerate(return_exprs):
            lines.append(f"    [{i}] {expr}")

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
    """Extract regular function parameters (excluding self), marking defaults
    and array-valued (`x: ...`) parameters."""
    args = func.args.args
    defaults = [None] * (len(args) - len(func.args.defaults)) + list(func.args.defaults)
    params = []
    for arg, default in zip(args, defaults):
        if arg.arg == "self":
            continue
        desc = arg.arg
        if isinstance(arg.annotation, ast.Constant) and arg.annotation.value is Ellipsis:
            desc += " (array)"
        if default is not None:
            desc += f"={_unparse_node(default)}"
        params.append(desc)
    return params


def _extract_options(func: ast.FunctionDef) -> list[str]:
    """Keyword options of an @memo(...) decorator."""
    for dec in func.decorator_list:
        if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id == "memo":
            return [f"{kw.arg}={_unparse_node(kw.value)}" for kw in dec.keywords if kw.arg]
    return []


def _extract_cast(body: list[ast.stmt]) -> list[str] | None:
    """The `cast: [alice, bob]` declaration, if the memo starts with one."""
    for node in body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                and node.target.id == "cast" and isinstance(node.annotation, ast.List):
            return [_unparse_node(e) for e in node.annotation.elts]
    return None


def _extract_foralls(body: list[ast.stmt]) -> list[tuple[str, str]]:
    """`forall: x in X` statements, which add query axes like def f[x: X]."""
    out = []
    for node in body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                and node.target.id == "forall" and isinstance(node.annotation, ast.Compare):
            out.append((_unparse_node(node.annotation.left),
                        _unparse_node(node.annotation.comparators[0])))
    return out


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
        if node.target.id in ("cast", "forall"):  # declarations, not agents
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
        args = [_unparse_node(a) for a in node.args]

        if func_name in _CHOICE_STATEMENTS:
            return _describe_chooses(node, verb=func_name)
        elif func_name == "knows":
            return f"knows: {', '.join(args)}"
        elif func_name == "forgets_about":
            return f"forgets about (marginalizes out): {', '.join(args)}"
        elif func_name == "wants":
            # wants(name = expr)
            if node.keywords:
                kw = node.keywords[0]
                return f"wants: {kw.arg} = {_unparse_node(kw.value)}"
            return "wants: (utility)"
        elif func_name == "snapshots_self_as":
            return f"snapshots its current beliefs as: {', '.join(args)}"
        elif func_name == "has_theory":
            return f"has an alternate theory of mind: {', '.join(args)}"
        elif func_name == "observes_event":
            weights = [_summarize_expr(_unparse_node(kw.value)) for kw in node.keywords if kw.arg == "wpp"]
            return f"observes soft evidence (an event with probability proportional to {', '.join(weights)})"
        elif func_name == "INSPECT":
            return "INSPECT (prints this frame's state at compile time)"
        elif func_name == "observes":
            return f"observes: {', '.join(args)}"
        else:
            return f"{func_name}({', '.join(args)})"

    # Subscript: thinks[...], observes_that[...]
    if isinstance(node, ast.Subscript):
        func_name = _unparse_node(node.value)

        if func_name == "thinks":
            return _describe_thinks(node, memo_func_names, depth)
        elif func_name == "observes_that":
            return f"observes that {_unparse_node(node.slice)} (conditions on it being true)"
        elif func_name in _OBSERVE_VERBS:
            return f"{func_name}: {_unparse_node(node.slice)}"

    # Compare node: observes [...] is ... (also sees / hears)
    if isinstance(node, ast.Compare):
        if isinstance(node.left, ast.Subscript) and isinstance(node.left.value, ast.Name) \
                and node.left.value.id in _OBSERVE_VERBS:
            comparators = [_unparse_node(c) for c in node.comparators]
            return f"{node.left.value.id} {_unparse_node(node.left.slice)} is {', '.join(comparators)}"
        return _unparse_node(node)

    return _unparse_node(node)


_OBSERVE_VERBS = {"observes", "sees", "hears"}


def _describe_chooses(node: ast.Call, verb: str = "chooses") -> str:
    """Describe a chooses() call or one of its aliases (given, draws, ...)."""
    # Parse positional args for "var in Domain" patterns
    choices = []
    extras = []
    for arg in node.args:
        if isinstance(arg, ast.Name) and arg.id == "uniformly":
            extras.append("uniformly")
            continue
        choices.append(_unparse_node(arg))

    for kw in node.keywords:
        value = _summarize_expr(_unparse_node(kw.value))
        if kw.arg == "wpp":
            extras.append(f"weighted by {value}")
        elif kw.arg == "such_that":
            extras.append(f"uniformly among values such that {value}")
        elif kw.arg == "to_be":
            extras.append(f"deterministically equal to {value}")
        elif kw.arg == "to_maximize":
            extras.append(f"to maximize {value}")
        elif kw.arg == "to_minimize":
            extras.append(f"to minimize {value}")

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


def _extract_returns(body: list[ast.stmt]) -> list[str]:
    """Extract the return expressions from a function body, in order."""
    return [_unparse_node(node.value) for node in body
            if isinstance(node, ast.Return) and node.value is not None]


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
    modified = _inject_memo_kwarg(code, "debug_trace")
    return _run_memo_subprocess(modified, timeout)


# ---------------------------------------------------------------------------
# Tool 11: critique_memo — validity + idiom + the explainability round-trip
# ---------------------------------------------------------------------------

_GENERIC_FRAME_NAMES = {
    "a", "b", "c", "x", "y", "z", "p", "q", "f", "g", "foo", "bar", "tmp",
    "agent", "agent1", "agent2", "person", "person1", "person2",
}

_CHOICE_STATEMENTS = {"chooses", "given", "draws", "assigned", "guesses"}

_DECISION_CONSTRUCTS = {"wants", "EU", "to_maximize", "to_minimize"}


def _frame_names(node: ast.FunctionDef) -> list[str]:
    """All agent frame names in a memo function: top-level statements are
    AnnAssign (`alice: chooses(...)`); inside thinks[...] they parse as Slice
    nodes (`alice: chooses(...)` becomes Slice(lower=Name, upper=Call))."""
    names = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name) \
                and sub.target.id not in ("cast", "forall"):
            names.append(sub.target.id)
        if isinstance(sub, ast.Slice) and isinstance(sub.lower, ast.Name) \
                and isinstance(sub.upper, ast.Call):
            names.append(sub.lower.id)
    return names


def _critique_function(node: ast.FunctionDef) -> list[str]:
    """Mechanical style checks for one @memo function. Returns finding lines."""
    findings: list[str] = []
    name = node.name

    if ast.get_docstring(node) is None:
        findings.append(
            f"WARN  {name}: no docstring — state the model's question in domain language"
        )

    n_lines = (node.end_lineno or node.lineno) - node.lineno + 1
    if n_lines > 30:
        findings.append(
            f"WARN  {name}: {n_lines} lines — one model, one question; "
            f"consider composing smaller memos (the RSA `L[u, r](t-1)` idiom)"
        )

    generic = sorted({f for f in _frame_names(node)
                      if f in _GENERIC_FRAME_NAMES or len(f) == 1})
    if generic:
        findings.append(
            f"WARN  {name}: generic frame name(s) {generic} — "
            f"frames are roles; name whose mind it is (observer, patient, world, ...)"
        )

    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) \
                and sub.func.id in _CHOICE_STATEMENTS:
            for kw in sub.keywords:
                if kw.arg == "wpp" and isinstance(kw.value, ast.Constant) \
                        and kw.value.value == 1:
                    findings.append(
                        f"HINT  {name} (line {kw.value.lineno}): `wpp=1` — write `uniformly` "
                        f"(same meaning, states the intent)"
                    )

    hatches = 0
    for sub in ast.walk(node):
        if isinstance(sub, ast.Set):
            if len(sub.elts) == 1 and isinstance(sub.elts[0], (ast.Attribute, ast.Constant)):
                continue  # {Enum.VALUE} / {literal} — the idiomatic reference form
            hatches += 1
    if hatches:
        findings.append(
            f"NOTE  {name}: {hatches} non-trivial {{...}} escape hatch(es) — "
            f"sanctioned, but each should carry a one-line justification comment"
        )

    decision = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            if isinstance(sub.func, ast.Name) and sub.func.id in _DECISION_CONSTRUCTS:
                decision.add(sub.func.id)
            for kw in sub.keywords:
                if kw.arg in ("to_maximize", "to_minimize"):
                    decision.add(kw.arg)
        if isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name) \
                and sub.value.id == "EU":
            decision.add("EU")
    if decision:
        findings.append(
            f"NOTE  {name}: decision-theoretic construct(s) {sorted(decision)} — "
            f"fine if the agent genuinely optimizes; forbidden in utility-free "
            f"(cybernetic) models of the mind"
        )

    return findings


@mcp.tool()
def critique_memo(code: str) -> str:
    """Critique memo code: compile verdict, idiom checks, and the round-trip readback.

    Three passes:
      1. VERDICT  — does it compile? (memo compilation is a semantic guarantee:
         the epistemic bookkeeping of who-knows-what is coherent.)
      2. STYLE    — mechanical idiom checks: docstrings, one-model-one-question
         size, generic frame names, `wpp=1` vs `uniformly`, escape-hatch
         quarantine, decision-theoretic constructs.
      3. READBACK — the plain-English description of the model, for the
         round-trip test: compare it with the spec you intended to write. If a
         naive reader would not recover your spec from the readback, the code
         is valid but not yet idiomatic — usually fixable by renaming frames
         and choosing more specific statements.

    Args:
        code: Complete Python script with imports, domains, and @memo functions.
    """
    sections = []

    verdict = validate_memo(code)
    sections.append("## VERDICT\n" + verdict)

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        sections.append(f"## STYLE\n(skipped: syntax error at line {exc.lineno})")
        return "\n\n".join(sections)

    findings: list[str] = []
    n_memo = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and _has_memo_decorator(node):
            n_memo += 1
            findings.extend(_critique_function(node))
    if n_memo == 0:
        findings.append("NOTE  no @memo functions found")
    sections.append(
        "## STYLE\n" + ("\n".join(findings) if findings
                        else f"clean — {n_memo} model(s), no findings")
    )

    sections.append(
        "## READBACK (round-trip test: does this match the spec you intended?)\n"
        + explain_memo(code)
    )
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run(transport="stdio")
