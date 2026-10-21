"""Regression checks for memo-mcp. Run: venv/bin/python test_server.py

Prints one [ pass name ] or [!fail name ] line per check, like memo's own
demo/test.py, and exits non-zero if any check fails. Covers: parity with the
vendored memo (version-specific syntax), every pattern template running as-is,
the example index covering every demo, every Handbook heading being reachable,
explain_memo knowing every statement form, and the stdio transport staying
clean when user code prints.
"""

import ast
import io
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout

import server

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"[ pass {name} ]")
    else:
        FAILURES.append(name)
        print(f"[!fail {name} ]" + (f"\n{detail}" if detail else ""))


# --- the vendored memo -----------------------------------------------------

def test_memo_version() -> None:
    from memo import __version__
    upstream = (server.MEMO_DIR / "memo" / "version.py").read_text()
    check("memo_version_is_vendored", __version__ in upstream, __version__)
    print(f"  (memo {__version__} from {server.MEMO_DIR})")


def test_is_operator_error() -> None:
    # memo >= 1.2.28 rejects `is` in expressions with a hint; 1.2.26 crashed with KeyError.
    out = server.validate_memo("""\
from memo import memo
import jax.numpy as np
X = np.arange(2)
@memo
def f():
    alice: chooses(x in X, uniformly)
    return alice.x is 0
""")
    check("is_operator_compile_error", "MemoError" in out and "`is` operator" in out, out)


def test_cost_metrics() -> None:
    # memo >= 1.3.0: time @ / space @ / size @, and cost @ as an alias of time @.
    out = server.validate_memo("""\
from memo import memo
import jax.numpy as np
X = np.arange(3)
@memo
def f[x: X](a):
    alice: chooses(x in X, wpp=exp(a * x))
    return Pr[alice.x == x]
@memo
def costs():
    return time @ f(1.0)
    return space @ f(1.0)
    return size @ f(1.0)
    return cost @ f(1.0)
""")
    check("time_space_size_cost", out.startswith("Success!"), out)


def test_var_redundant_known() -> None:
    # memo >= 1.2.29: Var of a known value is 0 (was a crash).
    out = server.run_memo("""\
from memo import memo
import jax.numpy as np
Bool = np.array([0, 1])
@memo
def f():
    alice: chooses(coffee in Bool, uniformly)
    bob: knows(alice.coffee)
    return E[bob[Var[alice.coffee == 1]]]
print("VALUE", float(f()))
""")
    check("var_redundant_known_is_zero", "VALUE 0.0" in out, out)


def test_snapshot_keeps_its_values() -> None:
    # memo >= 1.3.3: a snapshot is unaffected by its owner's later observation
    # (kach/memo issue 126); 1.3.2 gave [0. 1. 2.].
    out = server.run_memo("""\
from memo import memo
import jax.numpy as np
X = np.arange(3)
@memo
def f[x: X]():
    alice: thinks[ bob: chooses(x in X, wpp=1) ]
    alice: snapshots_self_as(past_alice)
    alice: observes [bob.x] is x
    return alice[past_alice[E[bob.x]]]
print("VALUE", np.broadcast_to(f(), (3,)).tolist())
""")
    check("snapshot_unaffected_by_later_observe", "VALUE [1.0, 1.0, 1.0]" in out, out)


# --- patterns --------------------------------------------------------------

def _run_template(item: tuple[str, dict]) -> tuple[str, int, str]:
    name, info = item
    env = dict(os.environ, PYTHONPATH=str(server.MEMO_DIR))
    r = subprocess.run([server.VENV_PYTHON, "-c", info["template"]],
                       capture_output=True, text=True, timeout=300, env=env)
    return name, r.returncode, r.stderr


def test_patterns_run() -> None:
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name, code, stderr in pool.map(_run_template, server.PATTERNS.items()):
            check(f"pattern_runs:{name}", code == 0, stderr[-2000:])


def test_patterns_style() -> None:
    for name, info in server.PATTERNS.items():
        tree = ast.parse(info["template"])
        findings = [f for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and server._has_memo_decorator(node)
                    for f in server._critique_function(node)
                    if not f.startswith("NOTE")]
        check(f"pattern_idiomatic:{name}", not findings, "\n".join(findings))


def test_pattern_docstring_lists_all() -> None:
    doc = server.list_patterns.__doc__ or ""
    missing = [name for name in server.PATTERNS if name not in doc]
    check("list_patterns_docstring_names_all", not missing, f"missing: {missing}")


# --- examples index --------------------------------------------------------

def test_examples_cover_demos() -> None:
    demo_dir = server.MEMO_DIR / "demo"
    demos = {f for f in os.listdir(demo_dir)
             if f.endswith((".py", ".ipynb")) and f != "test.py"}
    indexed = [ex["file"] for ex in json.loads(server.EXAMPLES_PATH.read_text())]
    check("examples_index_every_demo", demos <= set(indexed), f"unindexed: {sorted(demos - set(indexed))}")
    check("examples_index_no_stale", set(indexed) <= demos, f"stale: {sorted(set(indexed) - demos)}")
    check("examples_index_no_duplicates", len(indexed) == len(set(indexed)))


# --- handbook --------------------------------------------------------------

def test_handbook_headings() -> None:
    text = server.HANDBOOK_PATH.read_text()
    headings = server._handbook_headings(text)
    for level, title, body in headings:
        if level < 2:
            continue
        out = server.get_handbook(server._plain_heading(title))
        check(f"handbook_heading:{server._plain_heading(title)}",
              out.startswith("#") and title in out.splitlines()[0], out[:200])
    titles = {server._plain_heading(t) for _, t, _ in headings}
    bad = {k: v for k, v in server._HEADING_ALIASES.items() if v not in titles}
    check("handbook_aliases_name_real_headings", not bad, f"{bad}")
    for part in ("anatomy", "statements", "expressions", "running"):
        check(f"handbook_part:{part}", "not found" not in server.get_handbook(part))


# --- explain_memo ----------------------------------------------------------

EVERY_STATEMENT = """\
from memo import memo, domain
from enum import IntEnum
import jax.numpy as np

class Kid(IntEnum):
    BOY = 0
    GIRL = 1

Pair = domain(older=2, younger=2)
Score = np.linspace(0, 1, 5)

@memo(cache=True)
def everything[k: Kid](coin, arr: ..., depth=2):
    '''Every statement form memo parses.'''
    cast: [observer, family]
    forall: s in Score
    observer: thinks[
        family: given(p in Pair, uniformly),
        family: chooses(said in Kid, to_be={Kid.BOY}),
        family: draws(q in Kid, such_that=q == said),
        family: guesses(r in Kid, wpp=1),
        family: assigned(t in Kid, to_minimize=t)
    ]
    observer: observes_that[family.said == {Kid.BOY}]
    observer: observes_event(wpp=0.5)
    observer: snapshots_self_as(past)
    observer: has_theory(CHILD)
    observer: sees [family.q] is family.r
    observer: hears [family.r] is family.q
    observer: observes [family.t] is family.r
    observer: knows(k)
    observer: wants(win=k == family.q)
    observer: chooses(guess in Kid, to_maximize=EU[win])
    observer: forgets_about(guess)
    observer: thinks[family: chooses(u in Kid, uniformly)]
    observer: INSPECT()
    return observer[Pr[family.q == k]]
    return time @ everything(coin, arr, depth)
"""

EXPECTED_PHRASES = [
    "Options: cache=True",
    "Axis: k over Kid (BOY, GIRL)",
    "Axis: s over Score (np.linspace(0, 1, 5)) [forall]",
    "Parameters: coin, arr (array), depth=2",
    "Cast: observer, family",
    "given: p in Pair (uniformly)",
    "deterministically equal to {Kid.BOY}",
    "draws: q in Kid (uniformly among values such that q == said)",
    "guesses: r in Kid (weighted by 1)",
    "assigned: t in Kid (to minimize t)",
    "observes that family.said == {Kid.BOY} (conditions on it being true)",
    "observes soft evidence (an event with probability proportional to 0.5)",
    "snapshots its current beliefs as: past",
    "has an alternate theory of mind: CHILD",
    "sees family.q is family.r",
    "hears family.r is family.q",
    "observes family.t is family.r",
    "knows: k",
    "wants: win = k == family.q",
    "chooses: guess in Kid (to maximize EU[win])",
    "forgets about (marginalizes out): guess",
    "thinks about: [family: chooses: u in Kid (uniformly)]",
    "INSPECT (prints this frame's state at compile time)",
    "Returns 2 values",
    "[1] time @ everything(coin, arr, depth)",
    "Pair: domain(older=2, younger=2)",
]


def test_explain_every_statement() -> None:
    out = server.explain_memo(EVERY_STATEMENT)
    for phrase in EXPECTED_PHRASES:
        check(f"explain:{phrase[:50]}", phrase in out, out)
    check("explain_cast_forall_not_agents",
          "    cast:" not in out and "    forall:" not in out, out)


# --- decorator flag injection ------------------------------------------------

def test_inject_memo_kwarg() -> None:
    code = (
        "@memo\n"
        "def a(): ...\n"
        "@memo  # trailing comment\n"
        "def b(): ...\n"
        "@memo()\n"
        "def c(): ...\n"
        "@memo(cache=True)\n"
        "def d(): ...\n"
        "@memo(debug_trace=True)\n"
        "def e(): ...\n"
        "@memo\r\n"
        "def f(): ...\n"
    )
    out = server._inject_memo_kwarg(code, "debug_trace")
    try:
        tree = ast.parse(out)
        ok = all(any(isinstance(d, ast.Call) and any(k.arg == "debug_trace" for k in d.keywords)
                     for d in fn.decorator_list)
                 for fn in tree.body if isinstance(fn, ast.FunctionDef))
        check("inject_kwarg_every_decorator_form", ok and out.count("debug_trace") == 6, out)
    except SyntaxError as exc:
        check("inject_kwarg_every_decorator_form", False, f"{exc}\n{out}")
    check("inject_kwarg_keeps_line_numbers", out.count("\n") == code.count("\n"))


# --- stdout / stdio hygiene --------------------------------------------------

PRINTING_CODE = """\
from memo import memo
import jax.numpy as np
X = np.arange(3)
@memo
def f[x: X]():
    alice: chooses(x in X, uniformly)
    return Pr[alice.x == x]
print("HELLO FROM USER CODE")
"""


def test_validate_captures_stdout() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        out = server.validate_memo(PRINTING_CODE)
    check("validate_does_not_print", buf.getvalue() == "", buf.getvalue())
    check("validate_reports_success", out.startswith("Success! Defined memo functions: f"), out)


def test_validate_reports_warnings() -> None:
    out = server.validate_memo("""\
from memo import memo
import jax.numpy as np
X = np.arange(3)
@memo
def g():
    alice: chooses(x in X, uniformly)
    return E[alice[E[x]]]
""")
    check("validate_reports_warnings", "Warnings:" in out and "Redundant expectation" in out, out)


def test_stdio_stays_json() -> None:
    """A print in validated code must not reach the JSON-RPC stream."""
    import mcp.types as types
    proc = subprocess.Popen([server.VENV_PYTHON, str(server.BASE_DIR / "server.py")],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, cwd=server.BASE_DIR)
    assert proc.stdin and proc.stdout
    def send(msg: dict) -> None:
        proc.stdin.write(json.dumps(msg) + "\n")
        proc.stdin.flush()
    lines: list[str] = []
    try:
        send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": types.LATEST_PROTOCOL_VERSION, "capabilities": {},
            "clientInfo": {"name": "test_server", "version": "0"}}})
        lines.append(proc.stdout.readline())
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
            "name": "validate_memo", "arguments": {"code": PRINTING_CODE}}})
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            lines.append(line)
            try:
                if json.loads(line).get("id") == 2:
                    break
            except json.JSONDecodeError:
                pass
    finally:
        proc.kill()
        proc.wait()
    bad = []
    for line in lines:
        try:
            json.loads(line)
        except json.JSONDecodeError:
            bad.append(line)
    check("stdio_only_json_rpc", not bad and len(lines) >= 2, f"non-JSON lines: {bad}")
    check("stdio_validate_answered", any('"id":2' in l.replace(" ", "") for l in lines), "".join(lines))


if __name__ == "__main__":
    test_memo_version()
    test_is_operator_error()
    test_cost_metrics()
    test_var_redundant_known()
    test_snapshot_keeps_its_values()
    test_patterns_run()
    test_patterns_style()
    test_pattern_docstring_lists_all()
    test_examples_cover_demos()
    test_handbook_headings()
    test_explain_every_statement()
    test_inject_memo_kwarg()
    test_validate_captures_stdout()
    test_validate_reports_warnings()
    test_stdio_stays_json()
    print(f"\n{len(FAILURES)} failure(s)" + (f": {', '.join(FAILURES)}" if FAILURES else ""))
    sys.exit(1 if FAILURES else 0)
