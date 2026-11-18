#!/usr/bin/env python3
"""
pde_db.py -- notebooks -> SQLite database of PDE cases -> benchmark runs stored in the same database.
build     extract every PDESolver example/test from the notebooks into SQLite (idempotent)
list      list cases
run       execute cases (sweep of FFT workers), store times, errors, warnings, logs, failures
report    scaling / error table (times, speed-up vs the smallest tested worker count, errors)
failures  list runs that did not finish properly, with the stage and the exception
show      full detail of one run (per-time errors, warnings, traceback, logs, saved fields)
workers   show this machine's cores and suggest a ladder of FFT worker counts
profile   (re)compute the interior-error profile from fields stored in the database
sql       run any read-only SELECT on the database
Examples
python pde_db.py build  --notebooks pdesolver_*.ipynb --db pde.sqlite
python pde_db.py run    --solver-dir ~/mysolver --ids p1d-07 p2d-04 --workers 1 2 4 8 --repeats 3
python pde_db.py run    --solver-dir ~/mysolver --dim 2 --max-points 2e8 --timeout 900 --skip-done
python pde_db.py report --notebook p2d
python pde_db.py failures
python pde_db.py show 17 --logs
python pde_db.py run --solver-dir . --ids np1d-05 --margins 0 0.1 0.2 0.3      # error on nested interior boxes
python pde_db.py profile --margins 0 0.05 0.1 0.2 0.3                          # later, from stored fields
python pde_db.py sql "SELECT case_id, fft_workers, time_solve_s FROM v_latest_ok ORDER BY 3 DESC LIMIT 10"
Security: `build` EXECUTES the notebook cells (with a recording stub in place of PDESolver) and `run`
rebuilds callables with eval/exec from code strings stored in the database. Use your own files only.
Tables: cases, notebooks, runs, run_repeats, run_errors, run_error_profile, run_warnings, run_logs, run_blobs, meta
Views : v_latest_ok (latest solver version only), v_scaling, v_failures, v_dashboard, v_warmup, v_version_compare
"""
import argparse, ast, base64, builtins, csv, datetime as dt, hashlib, importlib, importlib.util, io, json, math
import os, platform, re, signal, sqlite3, statistics, subprocess, sys, tempfile, time, tokenize
import traceback, types, uuid, warnings, zlib
from collections import Counter

SCHEMA_VERSION = 6

# =============================================================================================
# 1. SQLite schema
# =============================================================================================
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS notebooks (
notebook TEXT PRIMARY KEY, sha256 TEXT, prefix TEXT, n_cases INTEGER, extracted_at TEXT);

CREATE TABLE IF NOT EXISTS cases (
case_id TEXT PRIMARY KEY, fingerprint TEXT UNIQUE, active INTEGER DEFAULT 1,
name TEXT, family TEXT, notebook TEXT, cell INTEGER,
dim INTEGER, type TEXT, temporal_order INTEGER, formulation TEXT, unknown TEXT,
equation TEXT, boundary_condition TEXT,
Lx REAL, Ly REAL, Nx INTEGER, Ny INTEGER, Lt REAL, Nt INTEGER, n_points REAL,
time_scheme TEXT, effective_time_scheme TEXT, dealiasing_ratio REAL, compute_energy INTEGER,
stationary_order INTEGER, seed INTEGER,
has_initial_velocity INTEGER, has_exact INTEGER, exact_kind TEXT,
test_norm TEXT, n_tests INTEGER, notes TEXT, comments TEXT,
source_code TEXT,           -- original notebook cell
definition TEXT,            -- full JSON description used by the runner
added_at TEXT, updated_at TEXT);

CREATE TABLE IF NOT EXISTS runs (
run_id INTEGER PRIMARY KEY AUTOINCREMENT,
case_id TEXT NOT NULL REFERENCES cases(case_id),
started_at TEXT, finished_at TEXT, wall_clock_total_s REAL,
status TEXT,                -- ok | failed | timeout | crashed
stage TEXT,                 -- last stage reached (import_solver, build_callables, build_equation, init, setup, solve, compute_errors, done)
fft_workers INTEGER, blas_threads TEXT, repeats INTEGER, warmup_excluded INTEGER,
time_init_s REAL, time_setup_s REAL, time_solve_s REAL, time_total_s REAL,
cpu_time_solve_s REAL, cpu_over_wall REAL, time_solve_min_s REAL,
error_max REAL, error_final REAL, passed INTEGER, non_finite INTEGER,
n_warnings INTEGER, peak_rss_mb REAL,
exception_type TEXT, exception_message TEXT, traceback TEXT,
returncode INTEGER, signal TEXT, timeout_s REAL,
fft_backend TEXT, solver_diagnostics TEXT,
solver_sha256 TEXT, psiop_sha256 TEXT, python TEXT, numpy TEXT, scipy TEXT, sympy TEXT,
platform TEXT, hostname TEXT, cpu_count INTEGER, environment TEXT, note TEXT, margins TEXT);
CREATE INDEX IF NOT EXISTS ix_runs_case ON runs(case_id, fft_workers);

CREATE TABLE IF NOT EXISTS run_repeats (          -- every repetition, warm-up included, partial if it failed
run_id INTEGER REFERENCES runs(run_id), rep INTEGER, is_warmup INTEGER,
t_init_s REAL, t_setup_s REAL, t_solve_s REAL, cpu_solve_s REAL, completed INTEGER);

CREATE TABLE IF NOT EXISTS run_errors (           -- one row per tested time
run_id INTEGER REFERENCES runs(run_id), t_eval REAL, actual_t REAL, component TEXT, norm TEXT,
error REAL, threshold REAL, passed INTEGER, ratio_to_threshold REAL,
rel_l2 REAL, abs_l2 REAL, linf REAL,
max_abs_u_num REAL, max_abs_u_exact REAL, n_nonfinite_u_num INTEGER, message TEXT);

CREATE TABLE IF NOT EXISTS run_error_profile (   -- error on the interior box |x| <= (1-margin)*L/2 (margin 0 = whole domain)
run_id INTEGER REFERENCES runs(run_id), t_eval REAL, actual_t REAL, component TEXT, margin REAL,
n_points_used INTEGER, ref_l2 REAL, abs_l2 REAL, rel_l2 REAL, linf REAL,
energy_share REAL);                           -- share of the squared error lying inside the box (1 = all of it)
CREATE INDEX IF NOT EXISTS ix_profile_run ON run_error_profile(run_id, margin);

CREATE TABLE IF NOT EXISTS run_warnings (
run_id INTEGER REFERENCES runs(run_id), category TEXT, message TEXT, filename TEXT, lineno INTEGER, count INTEGER);

CREATE TABLE IF NOT EXISTS run_logs (             -- stdout / stderr of the worker process (zlib)
run_id INTEGER REFERENCES runs(run_id), stream TEXT, nbytes INTEGER, truncated INTEGER, data BLOB);

CREATE TABLE IF NOT EXISTS run_blobs (            -- numerical fields / energy history (zlib-compressed .npy)
run_id INTEGER REFERENCES runs(run_id), name TEXT, t_eval REAL, dtype TEXT, shape TEXT, data BLOB);

DROP VIEW IF EXISTS v_scaling;
DROP VIEW IF EXISTS v_latest_ok;
DROP VIEW IF EXISTS v_failures;

-- latest ok run per (case, workers), restricted to the solver version (solver.py + psiop.py hashes) of the
-- most recent ok run of that case, so that speed-ups never mix timings of different code versions
CREATE VIEW v_latest_ok AS
WITH ok AS (
SELECT run_id, case_id, fft_workers, solver_sha256, psiop_sha256,
FIRST_VALUE(solver_sha256) OVER w AS cur_solver, FIRST_VALUE(psiop_sha256) OVER w AS cur_psiop
FROM runs WHERE status='ok' WINDOW w AS (PARTITION BY case_id ORDER BY run_id DESC)),
cur AS (SELECT * FROM ok WHERE solver_sha256 IS cur_solver AND psiop_sha256 IS cur_psiop),
pick AS (SELECT MAX(run_id) AS run_id FROM cur GROUP BY case_id, fft_workers)
SELECT r.* FROM runs r JOIN pick USING(run_id);

-- speedup / efficiency are relative to the SMALLEST worker count tested for the case (base_workers):
-- speedup = t(base)/t(w) ; efficiency = speedup * base/w  (1.0 = ideal scaling from the baseline)
CREATE VIEW v_scaling AS
SELECT l.case_id, c.name, c.dim, c.n_points, l.fft_workers, b.fft_workers AS base_workers,
l.time_init_s, l.time_setup_s, l.time_solve_s, l.cpu_over_wall,
b.time_solve_s / l.time_solve_s AS speedup,
b.time_solve_s / l.time_solve_s * b.fft_workers / l.fft_workers AS efficiency,
l.error_max, l.passed, l.n_warnings, l.run_id
FROM v_latest_ok l JOIN cases c ON c.case_id = l.case_id
LEFT JOIN v_latest_ok b ON b.case_id = l.case_id
AND b.fft_workers = (SELECT MIN(m.fft_workers) FROM v_latest_ok m WHERE m.case_id = l.case_id);

CREATE VIEW v_failures AS
SELECT r.run_id, r.case_id, c.name, r.fft_workers, r.status, r.stage, r.exception_type,
r.exception_message, r.returncode, r.signal, r.started_at
FROM runs r JOIN cases c ON c.case_id = r.case_id WHERE r.status <> 'ok';

DROP VIEW IF EXISTS v_error_profile;
DROP VIEW IF EXISTS v_dashboard;
DROP VIEW IF EXISTS v_warmup;
DROP VIEW IF EXISTS v_version_compare;

-- correctness dashboard: latest ok run of each case. margin = worst error/threshold (inf if the solution is non-finite)
CREATE VIEW v_dashboard AS
SELECT r.run_id, r.case_id, c.name, c.family, c.dim, c.type, c.boundary_condition, c.exact_kind, c.n_points,
r.fft_workers, r.started_at,
CASE WHEN r.passed IS NULL THEN 'no test' WHEN r.passed = 1 THEN 'PASS' ELSE 'FAIL' END AS status,
CASE WHEN r.non_finite = 1 THEN 9e999
ELSE (SELECT MAX(e.ratio_to_threshold) FROM run_errors e WHERE e.run_id = r.run_id) END AS margin,
r.passed, r.non_finite, r.error_max, r.error_final, r.time_solve_s, r.n_warnings
FROM runs r JOIN cases c ON c.case_id = r.case_id
WHERE r.status = 'ok' AND r.run_id IN (SELECT MAX(run_id) FROM runs WHERE status = 'ok' GROUP BY case_id);

-- interior-error profile of the latest ok run of each case that has interior information (margin > 0)
CREATE VIEW v_error_profile AS
SELECT p.*, r.case_id, c.name, c.family, c.dim, c.boundary_condition, c.exact_kind, r.fft_workers
FROM run_error_profile p JOIN runs r ON r.run_id = p.run_id JOIN cases c ON c.case_id = r.case_id
WHERE r.run_id IN (SELECT MAX(r2.run_id) FROM runs r2 WHERE r2.status = 'ok' AND EXISTS
(SELECT 1 FROM run_error_profile q WHERE q.run_id = r2.run_id AND q.margin > 0)
GROUP BY r2.case_id);

-- warm-up (first repetition) versus steady state; jitter_range = (max-min)/median of the steady repetitions
CREATE VIEW v_warmup AS
SELECT r.run_id, r.case_id, r.fft_workers, w.t_solve_s AS warm_s, r.time_solve_s AS steady_s,
w.t_solve_s / r.time_solve_s AS warm_over_steady,
CASE WHEN s.n > 1 THEN (s.mx - s.mn) / r.time_solve_s END AS jitter_range, s.n AS n_steady
FROM runs r
JOIN run_repeats w ON w.run_id = r.run_id AND w.is_warmup = 1 AND w.completed = 1
LEFT JOIN (SELECT run_id, COUNT(*) AS n, MIN(t_solve_s) AS mn, MAX(t_solve_s) AS mx
FROM run_repeats WHERE is_warmup = 0 AND completed = 1 GROUP BY run_id) s ON s.run_id = r.run_id
WHERE r.status = 'ok' AND r.time_solve_s > 0;

-- for each (case, workers): the two most recent solver versions (solver.py + psiop.py hashes) side by side
CREATE VIEW v_version_compare AS
WITH per AS (
SELECT * FROM runs WHERE status = 'ok' AND run_id IN
(SELECT MAX(run_id) FROM runs WHERE status = 'ok' GROUP BY case_id, fft_workers, solver_sha256, psiop_sha256)),
ranked AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY case_id, fft_workers ORDER BY run_id DESC) AS rk FROM per)
SELECT a.case_id, c.name, a.fft_workers,
b.solver_sha256 AS before_solver, a.solver_sha256 AS after_solver,
b.psiop_sha256 AS before_psiop, a.psiop_sha256 AS after_psiop,
b.time_solve_s AS before_s, a.time_solve_s AS after_s, b.time_solve_s / a.time_solve_s AS speedup,
b.error_max AS before_err, a.error_max AS after_err,
CASE WHEN b.error_max > 0 THEN a.error_max / b.error_max END AS err_ratio
FROM ranked a JOIN ranked b ON a.case_id = b.case_id AND a.fft_workers = b.fft_workers AND a.rk = 1 AND b.rk = 2
JOIN cases c ON c.case_id = a.case_id;
"""


def connect(path, create=False, readonly=False):
    if not create and not os.path.exists(path):
        sys.exit(f"database '{path}' not found (run `build` first)")
    if readonly:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    else:
        con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript(SCHEMA)
    if "margins" not in {r[1] for r in con.execute("PRAGMA table_info(runs)")}:
        con.execute("ALTER TABLE runs ADD COLUMN margins TEXT")       # migration of older databases
    con.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
    con.row_factory = sqlite3.Row
    return con


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def jdump(o):
    return json.dumps(o, ensure_ascii=False, allow_nan=False, default=str)


def jsonable(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    return o


# =============================================================================================
# 2. build : notebooks -> cases (cells are executed against a recording stub of PDESolver)
# =============================================================================================

# Manual annotations (read from the notebooks): (notebook file, cell) -> (exact_kind, note)
ANNOTATIONS = {
    ("pdesolver_tester_periodic_2d.ipynb", 20): ("approximate", "Author states this is a heuristic approximation valid for short times; no closed-form solution exists."),
    ("pdesolver_tester_nonperiodic_1d.ipynb", 6): ("approximate", "Author states the 'exact' solution is approximate (visual test only)."),
    ("pdesolver_tester_nonperiodic_1d.ipynb", 18): ("exact", "Ai(x) is stationary for this operator but does not vanish at the boundary; nonperiodic clipping creates a boundary layer (hence the large threshold)."),
    ("pdesolver_tester_nonperiodic_1d.ipynb", 22): ("exact", "P_n(+-1) != 0: nonperiodic clipping is technically violated (author comment); large threshold."),
    ("pdesolver_tester_periodic_1d.ipynb", 6): ("exact", "Stationary psiOp depending on x solved by order-1 asymptotic inverse: loose threshold in notebook."),
    ("pdesolver_tester_periodic_2d.ipynb", 6): ("exact", "Stationary psiOp depending on x,y solved by order-1 asymptotic inverse: loose threshold in notebook."),
}


def default_prefix(fname):
    b = os.path.basename(fname).lower()
    m = re.search(r"(non)?periodic_(\d)d", b)
    if m:
        return ("np" if m.group(1) else "p") + m.group(2) + "d"
    if "example" in b:
        return "ex"
    return re.sub(r"\W+", "", os.path.splitext(b)[0])[:6] or "nb"


def _extraction_tools():
    """Imports and stubs needed only by `build` (kept local so that `run` does not need them)."""
    import numpy as np
    import sympy as sp
    from unittest.mock import MagicMock
    from scipy.special import eval_hermite, airy, eval_legendre
    return np, sp, MagicMock, eval_hermite, airy, eval_legendre


def extract_notebook(path):
    """Return the list of case dicts found in one notebook."""
    np, sp, MagicMock, eval_hermite, airy, eval_legendre = _extraction_tools()
    from sympy.core.function import AppliedUndef

    class psiOp(sp.Function):
        pass

    class Op(sp.Function):
        pass

    records, cur = [], {}

    class StubSolver:
        def __init__(self, equation, time_scheme="default", dealiasing_ratio=2 / 3, compute_energy=True):
            self.__dict__["_rec"] = dict(equation=equation, time_scheme=time_scheme,
                                         dealiasing_ratio=dealiasing_ratio, compute_energy=compute_energy,
                                         setup=None, tests=[], stationary_order=None, extra_attrs={})
            records.append(self._rec)

        def setup(self, **kw):
            self._rec["setup"] = kw
            self._rec["setup_ns"] = dict(cur["ns"])

        def solve(self):
            return []

        def solve_stationary_psiOp(self, order=3):
            self._rec["stationary_order"] = order
            return np.zeros(1)

        def test(self, u_exact, t_eval=None, norm="relative", threshold=1e-2, component="real"):
            self._rec["tests"].append(dict(u_exact=u_exact, t_eval=t_eval, norm=norm, threshold=threshold,
                                           component=component, ns=dict(cur["ns"])))

        def __setattr__(self, k, v):
            self._rec["extra_attrs"][k] = v

        def __getattr__(self, k):
            if k in ("X", "Y", "psi_ops"):
                return MagicMock()
            return lambda *a, **kw: MagicMock()

    plt = MagicMock()
    plt.subplots.return_value = (MagicMock(), MagicMock())
    plt.subplots.return_value[1].plot.return_value = (MagicMock(),)

    ns = {k: v for k, v in sp.__dict__.items() if not k.startswith("_")}
    ns.update(dict(np=np, plt=plt, PDESolver=StubSolver, psiOp=psiOp, Op=Op, HTML=MagicMock(),
                   display=MagicMock(), FuncAnimation=MagicMock(), eval_hermite=eval_hermite, airy=airy,
                   eval_legendre=eval_legendre, warnings=warnings))
    cur["ns"] = ns

    def num(v):
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, (float, np.floating)):
            return float(v)
        if isinstance(v, (complex, np.complexfloating)):
            return {"re": float(v.real), "im": float(v.imag)}
        return None

    def pi_expr(v):
        if isinstance(v, (int, float)) and v and float(v) != int(v):
            r = v / math.pi
            if abs(r * 2 - round(r * 2)) < 1e-12:
                return f"{round(r * 2) / 2:g}*pi"
        return None

    def find_node(tree, lineno):
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.Lambda)) and n.lineno == lineno:
                return n

    def describe(f, src, tree, env, seen=None):
        """Code of a callable + numeric constants + helper callables + imports it needs."""
        if f is None:
            return None
        seen = seen if seen is not None else set()
        code = f.__code__
        node = find_node(tree, code.co_firstlineno)
        text = ast.get_source_segment(src, node) if node is not None else repr(f)
        out = {"code": text, "args": list(code.co_varnames[:code.co_argcount])}
        consts, helpers, others, imports = {}, {}, {}, {}
        names = set(code.co_names)
        for c in code.co_consts:
            if isinstance(c, types.CodeType):
                names |= set(c.co_names)
        for nm in sorted(names):
            if nm not in env:
                continue
            val = env[nm]
            if isinstance(val, types.FunctionType) and val.__code__.co_filename == "<cell>":
                if nm in seen:
                    continue
                seen.add(nm)
                h = describe(val, src, tree, env, seen)
                ht = f"{nm} = {h['code']}" if h["code"].startswith("lambda") else h["code"]
                helpers[nm] = {k: v for k, v in dict(code=ht, constants=h["constants"], helpers=h["helpers"],
                                                     imports=h.get("imports")).items() if v}
                continue
            n_ = num(val)
            if n_ is not None:
                consts[nm] = n_
            elif nm in ("np", "plt"):
                continue
            elif isinstance(val, (types.FunctionType, types.BuiltinFunctionType, np.ufunc)):
                mod = getattr(val, "__module__", None) or ("scipy.special" if isinstance(val, np.ufunc) else "?")
                imports[nm] = f"{mod}.{getattr(val, '__name__', nm)}"
            elif isinstance(val, (types.ModuleType, type, sp.Symbol)):
                continue
            elif isinstance(val, sp.Basic):
                try:
                    z = complex(val)
                    consts[nm] = num(z.real) if z.imag == 0 else num(z)
                except Exception:
                    others[nm] = str(val)
            else:
                others[nm] = type(val).__name__
        out.update(constants=consts, helpers=helpers)
        if imports:
            out["imports"] = imports
        if others:
            out["other_globals"] = others
        return out

    def comments_of(src):
        res = []
        try:
            for tok in tokenize.generate_tokens(io.StringIO(src).readline):
                if tok.type == tokenize.COMMENT and tok.string.lstrip("# ").strip():
                    res.append(tok.string.lstrip("# ").strip())
        except Exception:
            pass
        return res

    def analyse(eq):
        funcs = [f for f in eq.atoms(sp.Function)
                 if isinstance(f, AppliedUndef) and f.func.__name__ not in ("psiOp", "Op")]
        u = list({str(f): f for f in funcs}.values())[0]
        args = [str(a) for a in u.args]
        stationary = "t" not in args
        order = 0
        if not stationary:
            for d in eq.atoms(sp.Derivative):
                for v, c in d.variable_count:
                    if str(v) == "t":
                        order = max(order, int(c))
            if order == 0 and "omega" in str(eq):
                order = 1                                   # Op(-I*omega, u) encodes d/dt
        form = (["psiOp"] if eq.has(psiOp) else []) + (["Op"] if eq.has(Op) else []) or ["PDE"]
        spatial_der = any(str(v) != "t" for d in eq.atoms(sp.Derivative) for v, c in d.variable_count)
        if spatial_der and form != ["PDE"]:
            form = ["PDE"] + form
        return u, args, stationary, order, form

    nb = json.load(open(path, encoding="utf-8"))
    fname = os.path.basename(path)
    cases, headings = [], {}

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name in ("solver", "psiop", "imports"):
            return types.SimpleNamespace(PDESolver=StubSolver, psiOp=psiOp, Op=Op)
        return real_import(name, *a, **k)

    for ci, cell in enumerate(nb["cells"]):
        src = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            for line in src.splitlines():
                m = re.match(r"^(#+)\s+(.*)", line)
                if m:
                    lvl = len(m.group(1))
                    headings = {k: v for k, v in headings.items() if k < lvl}
                    headings[lvl] = m.group(2).strip()
            continue
        if src.lstrip().startswith("%"):
            continue
        start = len(records)
        try:
            tree = ast.parse(src)
            code = compile(src, "<cell>", "exec")
        except SyntaxError:
            continue
        builtins.__import__ = fake_import
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exec(code, ns)
        except Exception as e:                      # cells after setup/solve (plots...) may fail: harmless
            if len(records) == start:
                print(f"  ! {fname} cell {ci}: {type(e).__name__}: {e}", file=sys.stderr)
        finally:
            builtins.__import__ = real_import
        recs = records[start:]
        # numeric globals that differ between instances created in one cell (loops) -> name suffix
        varying = []
        if len(recs) > 1:
            keys = set.intersection(*[set(r.get("setup_ns", {})) for r in recs])
            for k in sorted(keys):
                vals = [num(r["setup_ns"][k]) for r in recs]
                if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals) \
                        and len(set(vals)) > 1 and not k.startswith("_"):
                    varying.append(k)
        for ri, rec in enumerate(recs):
            eq = rec["equation"]
            ufun, args, stationary, order, form = analyse(eq)
            su, sns = rec["setup"] or {}, rec.get("setup_ns", ns)
            ic, iv = su.get("initial_condition"), su.get("initial_velocity")
            dom = {}
            for k in ("Lx", "Ly", "Nx", "Ny", "Lt", "Nt"):
                if su.get(k) is not None:
                    dom[k] = num(su[k])
                    pe = pi_expr(dom[k])
                    if pe and k in ("Lx", "Ly", "Lt"):
                        dom[k + "_expr"] = pe
            extent = {}
            if "Lx" in dom:
                extent["x"] = [-dom["Lx"] / 2, dom["Lx"] / 2]
            if "Ly" in dom:
                extent["y"] = [-dom["Ly"] / 2, dom["Ly"] / 2]
            ts = rec["time_scheme"]
            eff = (None if stationary else
                   "psiOp-specific stepper (time_scheme not used)" if "psiOp" in form else
                   "ETD-RK4" if ts == "RK4" else "default (exponential Euler / spectral-exact linear step)")
            setup_extra = {k: (num(v) if num(v) is not None else v) for k, v in su.items() if k not in
                           ("Lx", "Ly", "Nx", "Ny", "Lt", "Nt", "boundary_condition", "initial_condition",
                            "initial_velocity", "plot")}
            tests, exact, test = rec["tests"], None, None
            if tests:
                exact = {"python": describe(tests[0]["u_exact"], src, tree, tests[0]["ns"]), "kind": "exact"}
                ths = sorted({t["threshold"] for t in tests})
                comps = sorted({t["component"] for t in tests})
                test = {"t_eval": [None] if stationary else [float(t["t_eval"]) if t["t_eval"] is not None else None for t in tests],
                        "threshold": ths[0] if len(ths) == 1 else [t["threshold"] for t in tests],
                        "component": comps[0] if len(comps) == 1 else [t["component"] for t in tests],
                        "norm": sorted({t["norm"] for t in tests})[0],
                        "threshold_note": "tolerance used in the notebook assert; not a measured error"}
            ic_s = describe(ic, src, tree, sns) if ic is not None else None
            iv_s = describe(iv, src, tree, sns) if iv is not None else None
            # seed: look in the code actually used; fall back to the cell if an RNG object is a global
            code_txt = json.dumps([ic_s, iv_s, exact])
            m = re.search(r"(?:default_rng|RandomState|seed)\(\s*(\d+)\s*\)", code_txt)
            if not m and "other_globals" in code_txt:
                m = re.search(r"(?:default_rng|RandomState|seed)\(\s*(\d+)\s*\)", src)
            parts = [headings[k] for k in sorted(headings) if k >= 2]
            name = " / ".join(parts) if parts else f"{fname} cell {ci}"
            if len(recs) > 1:
                name += " [" + (", ".join(f"{k}={num(sns[k])}" for k in varying) if varying else f"#{ri + 1}") + "]"
            notes = ""
            if not stationary and "psiOp" not in form and ts not in ("default", "RK4"):
                notes = f"time_scheme='{ts}' is not 'RK4': solve() falls back to the default scheme."
            case = {
                "name": name, "family": parts[0] if parts else None,
                "source": {"notebook": fname, "cell": ci},
                "dim": len([a for a in args if a != "t"]), "type": "stationary" if stationary else "evolution",
                "temporal_order": None if stationary else order, "formulation": form,
                "unknown": str(ufun.func), "variables": args, "equation": str(eq),
                "symbols": {str(s): {k: v for k, v in getattr(s, "_assumptions_orig", {}).items() if k != "commutative"}
                            for s in sorted(eq.free_symbols, key=str)},
                "unknown_assumptions": ({"real": True} if ufun.is_real else {"complex": True} if ufun.is_complex else {}),
                "boundary_condition": su.get("boundary_condition", "periodic"),
                "domain": dom, "domain_extent": extent,
                "solver": {"time_scheme": ts, "effective_time_scheme": eff,
                           "dealiasing_ratio": num(rec["dealiasing_ratio"]) if num(rec["dealiasing_ratio"]) is not None else None,
                           "compute_energy": rec["compute_energy"]},
                "setup_options": setup_extra, "stationary_order": rec["stationary_order"],
                "initial_condition": ic_s, "initial_velocity": iv_s,
                "exact_solution": exact, "test": test, "seed": int(m.group(1)) if m else None,
                "extra_attributes": {k: (num(v) if num(v) is not None else str(v)) for k, v in rec["extra_attrs"].items()},
                "notebook_comments": comments_of(src), "notes": notes or None,
            }
            ann = ANNOTATIONS.get((fname, ci))
            if ann and exact:
                case["exact_solution"]["kind"], case["notes"] = ann[0], ((notes + " ") if notes else "") + ann[1]
            case["source_code"] = src
            cases.append(case)
    return cases


def fingerprint(c):
    keys = ("equation", "symbols", "domain", "boundary_condition", "solver", "setup_options",
            "stationary_order", "initial_condition", "initial_velocity", "exact_solution", "extra_attributes", "seed")
    blob = json.dumps({k: c.get(k) for k in keys}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()


def case_row(c):
    d, s, t = c["domain"], c["solver"], c.get("test") or {}
    n_pts = (d.get("Nx", 1) * d.get("Ny", 1) * d.get("Nt", 1))
    return dict(
        name=c["name"], family=c.get("family"), notebook=c["source"]["notebook"], cell=c["source"]["cell"],
        dim=c["dim"], type=c["type"], temporal_order=c.get("temporal_order"), formulation="+".join(c["formulation"]),
        unknown=c["unknown"], equation=c["equation"], boundary_condition=c["boundary_condition"],
        Lx=d.get("Lx"), Ly=d.get("Ly"), Nx=d.get("Nx"), Ny=d.get("Ny"), Lt=d.get("Lt"), Nt=d.get("Nt"),
        n_points=float(n_pts), time_scheme=s["time_scheme"], effective_time_scheme=s["effective_time_scheme"],
        dealiasing_ratio=s["dealiasing_ratio"], compute_energy=int(bool(s["compute_energy"])),
        stationary_order=c.get("stationary_order"), seed=c.get("seed"),
        has_initial_velocity=int(bool(c.get("initial_velocity"))), has_exact=int(bool(c.get("exact_solution"))),
        exact_kind=(c.get("exact_solution") or {}).get("kind"), test_norm=t.get("norm"),
        n_tests=len(t.get("t_eval", [])), notes=c.get("notes"), comments=jdump(c.get("notebook_comments", [])),
        source_code=c["source_code"])


def cmd_build(a):
    con = connect(a.db, create=True)
    total_new = total_upd = 0
    used_prefix = {}
    for path in a.notebooks:
        fname = os.path.basename(path)
        prefix = default_prefix(path)
        if used_prefix.setdefault(prefix, fname) != fname:
            i = 2
            while f"{prefix}{i}" in used_prefix:
                i += 1
            prefix = f"{prefix}{i}"; used_prefix[prefix] = fname
        sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
        print(f"extracting {fname} (prefix '{prefix}') ...")
        cases = extract_notebook(path)
        existing = {r["fingerprint"]: r["case_id"] for r in con.execute("SELECT fingerprint, case_id FROM cases")}
        nums = [int(r[0].rsplit("-", 1)[1]) for r in con.execute("SELECT case_id FROM cases WHERE case_id LIKE ?", (prefix + "-%",))]
        nxt, seen_ids = (max(nums) + 1 if nums else 1), set()
        for c in cases:
            fp = fingerprint(c)
            cid = existing.get(fp)
            new = cid is None
            if new:
                cid = f"{prefix}-{nxt:02d}"; nxt += 1
            c["id"], c["fingerprint"] = cid, fp
            definition = {k: v for k, v in c.items() if k != "source_code"}
            row = case_row(c)
            row.update(definition=jdump(jsonable(definition)), fingerprint=fp, updated_at=now(), active=1)
            if new:
                row.update(case_id=cid, added_at=now())
            cols = ",".join(row); ph = ",".join(":" + k for k in row)
            con.execute(f"INSERT INTO cases ({cols}) VALUES ({ph})", row); total_new += 1
        else:
            sets = ",".join(f"{k}=:{k}" for k in row)
            row["case_id"] = cid
            con.execute(f"UPDATE cases SET {sets} WHERE case_id=:case_id", row); total_upd += 1
            seen_ids.add(cid)
        # cases of this notebook that disappeared stay in the DB (their runs are kept) but become inactive
        con.execute(f"UPDATE cases SET active=0 WHERE notebook=? AND case_id NOT IN ({','.join('?' * len(seen_ids)) or 'NULL'})",
                    (fname, *seen_ids))
        con.execute("INSERT OR REPLACE INTO notebooks VALUES (?,?,?,?,?)", (fname, sha, prefix, len(cases), now()))
    con.commit()
    n = con.execute("SELECT COUNT(*), SUM(active) FROM cases").fetchone()
    print(f"done: {total_new} new, {total_upd} updated; database holds {n[0]} cases ({n[1]} active) -> {a.db}")


# =============================================================================================
# 3. worker (runs in a subprocess): one case, one FFT worker count
# =============================================================================================

def _import_name(path):
    mod, _, attr = path.rpartition(".")
    return getattr(importlib.import_module(mod), attr)


def _collect_imports(spec, acc):
    if spec:
        acc.update(spec.get("imports", {}))
        for h in spec.get("helpers", {}).values():
            _collect_imports(h, acc)
    return acc


def _load_constants(spec, ns, seed):
    import numpy as np
    for k, v in spec.get("constants", {}).items():
        ns[k] = complex(v["re"], v["im"]) if isinstance(v, dict) else v
    for k, kind in spec.get("other_globals", {}).items():
        if kind == "Generator":
            ns[k] = np.random.default_rng(seed)
        elif kind == "RandomState":
            ns[k] = np.random.RandomState(seed)
        else:
            raise ValueError(f"cannot rebuild global '{k}' of type {kind}")
    for h in spec.get("helpers", {}).values():
        _load_constants(h, ns, seed)


def make_callable(spec, seed=None):
    import numpy as np
    import scipy.special as sps
    ns = {"np": np, "pi": math.pi}
    ns.update({k: getattr(sps, k) for k in ("eval_hermite", "eval_legendre", "airy") if hasattr(sps, k)})
    for name, path in _collect_imports(spec, {}).items():
        ns[name] = _import_name(path)
    _load_constants(spec, ns, seed)

    def run_helpers(s):
        for h in s.get("helpers", {}).values():
            run_helpers(h)
            exec(h["code"], ns)

    run_helpers(spec)
    code = spec["code"].strip()
    if code.startswith("lambda"):
        return eval(code, ns)
    exec(code, ns)
    return ns[re.match(r"def\s+(\w+)", code).group(1)]


def build_equation(case, S):
    import sympy as sp
    from sympy.parsing.sympy_parser import parse_expr
    loc = {n: sp.Symbol(n, **ass) for n, ass in case.get("symbols", {}).items()}
    loc[case["unknown"]] = sp.Function(case["unknown"], **case.get("unknown_assumptions", {}))
    loc["psiOp"], loc["Op"] = S.psiOp, S.Op
    eq = parse_expr(case["equation"], local_dict=loc)
    if not isinstance(eq, sp.Eq):
        raise ValueError("equation did not parse to sympy.Eq")
    return eq


def interior_mask(dom, shape, m):
    """Boolean mask of the box |x| <= (1-m)*Lx/2 (and |y| <= (1-m)*Ly/2 in 2D). Grid as in PDESolver:
    x_i = -L/2 + i*L/N (endpoint=False), arrays indexed 'ij'. m = fraction of the half-width removed on each side."""
    import numpy as np

    def axis(L, N):
        x = -L / 2 + np.arange(N) * (L / N)
        return np.abs(x) <= (1 - m) * L / 2 + 1e-12 * L

    mx = axis(dom["Lx"], dom["Nx"])
    if len(shape) == 1:
        if shape[0] != len(mx):
            raise ValueError(f"grid {shape} does not match domain Nx={dom['Nx']}")
        return mx
    my = axis(dom["Ly"], dom["Ny"])
    if shape != (len(mx), len(my)):
        raise ValueError(f"grid {shape} does not match domain ({dom['Nx']}, {dom['Ny']})")
    return mx[:, None] & my[None, :]


def profile_rows(u_num, u_ex, comp, dom, margins):
    """Error of the tested component on nested interior boxes (one dict per margin)."""
    import numpy as np
    f = {"real": np.real, "imag": np.imag, "abs": np.abs}[comp]
    diff, ref = f(u_num) - f(u_ex), f(u_ex)
    total = float(np.sum(np.abs(diff) ** 2))
    fin = lambda v: v if v is None or math.isfinite(v) else None
    rows = []
    for m in margins:
        mk = interior_mask(dom, np.shape(diff), m)
        d, r = diff[mk], ref[mk]
        a2, r2 = float(np.linalg.norm(d)), float(np.linalg.norm(r))
        rows.append({"margin": float(m), "n_points_used": int(mk.sum()), "ref_l2": fin(r2), "abs_l2": fin(a2),
                     "rel_l2": fin(a2 / r2) if r2 > 0 else None,
                     "linf": fin(float(np.max(np.abs(d)))) if d.size else None,
                     "energy_share": fin(float(np.sum(np.abs(d) ** 2)) / total) if total > 0 else None})
    return rows


def compute_errors(solver, case, u_exact, u_stat, want_fields, margins=(0.0,)):
    """Same selection/norm rule as PDESolver.test(); never asserts; adds diagnostics. Returns (rows, fields)."""
    import numpy as np
    t = case["test"]
    n = len(t["t_eval"])
    comps = t["component"] if isinstance(t["component"], list) else [t["component"]] * n
    ths = t["threshold"] if isinstance(t["threshold"], list) else [t["threshold"]] * n
    rows, fields = [], []
    for t_eval, comp, th in zip(t["t_eval"], comps, ths):
        row = {"t_eval": t_eval, "actual_t": None, "component": comp, "norm": t["norm"], "threshold": th,
               "error": None, "passed": 0, "message": None}
        try:
            if case["type"] == "stationary":
                u_num = u_stat
                u_ex = u_exact(solver.X) if case["dim"] == 1 else u_exact(solver.X, solver.Y)
            else:
                save_interval = max(1, solver.Nt // solver.n_frames)
                times = np.arange(0, solver.Lt + solver.dt, save_interval * solver.dt)
                idx = int(np.argmin(np.abs(times - t_eval)))
                row["actual_t"] = float(times[idx])
                u_num = solver.frames[idx]
                u_ex = (u_exact(solver.X, row["actual_t"]) if case["dim"] == 1
                        else u_exact(solver.X, solver.Y, row["actual_t"]))
            u_ex = np.broadcast_to(u_ex, np.shape(u_num))
            f = {"real": np.real, "imag": np.imag, "abs": np.abs}[comp]
            diff, ref = f(u_num) - f(u_ex), f(u_ex)
            abs_l2 = float(np.linalg.norm(diff))
            r = float(np.linalg.norm(ref))
            rel = abs_l2 / r if r > 0 else float("nan")
            err = rel if t["norm"] == "relative" else abs_l2
            row.update(error=err, abs_l2=abs_l2, rel_l2=rel, linf=float(np.max(np.abs(diff))) if diff.size else None,
                       max_abs_u_num=float(np.max(np.abs(u_num))), max_abs_u_exact=float(np.max(np.abs(u_ex))),
                       n_nonfinite_u_num=int(np.sum(~np.isfinite(u_num))))
            row["passed"] = int(math.isfinite(err) and err < th)
            row["ratio_to_threshold"] = err / th if math.isfinite(err) and th else None
            row["profile"] = profile_rows(u_num, u_ex, comp, case["domain"], margins)
            if want_fields:
                fields += [("u_num", row["t_eval"], np.asarray(u_num)), ("u_exact", row["t_eval"], np.asarray(u_ex))]
        except Exception as e:                       # a single time failing must not lose the others
            row["message"] = f"{type(e).__name__}: {e}"
        rows.append(row)
    return rows, fields


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(jsonable(obj), f, default=str)
    os.replace(tmp, path)


def worker(a):
    import faulthandler
    faulthandler.enable()                            # native crash -> traceback on stderr (stored in run_logs)
    os.environ.setdefault("MPLBACKEND", "Agg")
    import numpy as np
    sys.path.insert(0, os.path.abspath(a.solver_dir))
    case = json.loads(a.case_json)
    res = {"status": "failed", "stage": "start"}
    reps, wlist = [], []

    def snapshot():
        _write_json(a.progress, {"stage": res["stage"], "reps": reps})

    def stage(s):
        res["stage"] = s; snapshot()

    try:
        stage("import_solver")
        import solver as S
        for name in ("solver", "psiop", "imports"):
            if name in sys.modules:
                sys.modules[name].FFT_WORKERS = a.workers
        S.FFT_WORKERS = a.workers

        stage("build_callables")
        seed = case.get("seed")
        ic = make_callable(case["initial_condition"], seed) if case.get("initial_condition") else None
        iv = make_callable(case["initial_velocity"], seed) if case.get("initial_velocity") else None
        u_ex = make_callable(case["exact_solution"]["python"], seed) if case.get("exact_solution") else None

        # ── Override Lt / Nt if requested ──────────────────────────────────────────
        override_notes = []
        if case["type"] != "stationary":
            dom = case["domain"]
            if a.Lt is not None and "Lt" in dom:
                old_Lt = dom["Lt"]
                dom["Lt"] = a.Lt
                override_notes.append(f"Lt={a.Lt} (was {old_Lt})")
                # rescale t_eval so tests sample the same *fraction* of [0, Lt]
                if old_Lt and a.Lt != old_Lt and case.get("test"):
                    ratio = a.Lt / old_Lt
                    te = case["test"]["t_eval"]
                    case["test"]["t_eval"] = [
                        t * ratio if t is not None else t for t in te
                    ]
            if a.Nt is not None and "Nt" in dom:
                old_Nt = dom["Nt"]
                dom["Nt"] = a.Nt
                override_notes.append(f"Nt={a.Nt} (was {old_Nt})")
        # ── END override ───────────────────────────────────────────────────────────

        sv, dom = case["solver"], case["domain"]
        setup_kw = {k: dom[k] for k in ("Lx", "Ly", "Nx", "Ny", "Lt", "Nt") if k in dom}
        setup_kw.update(boundary_condition=case["boundary_condition"], initial_condition=ic, plot=False,
                        **case.get("setup_options", {}))
        if iv is not None:
            setup_kw["initial_velocity"] = iv
        n_rep = a.repeats + (0 if a.no_warmup else 1)
        solver = u_stat = None
        for rep in range(n_rep):
            r = {"rep": rep, "is_warmup": int(rep == 0 and not a.no_warmup), "t_init_s": None, "t_setup_s": None,
                 "t_solve_s": None, "cpu_solve_s": None, "completed": 0}
            reps.append(r)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    stage("build_equation"); eq = build_equation(case, S)
                    stage("init"); t0 = time.perf_counter()
                    solver = S.PDESolver(eq, time_scheme=sv["time_scheme"], dealiasing_ratio=sv["dealiasing_ratio"],
                                         compute_energy=sv["compute_energy"])
                    t1 = time.perf_counter(); r["t_init_s"] = t1 - t0
                    stage("setup"); solver.setup(**setup_kw)
                    for k, v in case.get("extra_attributes", {}).items():
                        setattr(solver, k, v)
                    t2 = time.perf_counter(); r["t_setup_s"] = t2 - t1
                    stage("solve"); c2 = time.process_time()
                    if case["type"] == "stationary":
                        u_stat = solver.solve_stationary_psiOp(order=case["stationary_order"])
                    else:
                        solver.solve()
                    r["t_solve_s"] = time.perf_counter() - t2
                    r["cpu_solve_s"] = time.process_time() - c2
                    r["completed"] = 1
                finally:
                    wlist.extend(w)
                sys.stdout.flush()
                snapshot()
        good = [r for r in reps if r["completed"] and not r["is_warmup"]]
        med = {k: statistics.median(r[k] for r in good) for k in ("t_init_s", "t_setup_s", "t_solve_s", "cpu_solve_s")}
        res.update(status="ok", reps=reps, time_init_s=med["t_init_s"], time_setup_s=med["t_setup_s"],
                   time_solve_s=med["t_solve_s"], cpu_time_solve_s=med["cpu_solve_s"],
                   time_total_s=med["t_init_s"] + med["t_setup_s"] + med["t_solve_s"],
                   cpu_over_wall=med["cpu_solve_s"] / med["t_solve_s"] if med["t_solve_s"] > 0 else None,
                   time_solve_min_s=min(r["t_solve_s"] for r in good))
        diag = {}
        for k in ("_bc_max_correction", "temporal_order", "has_psi", "is_stationary", "dt"):
            if hasattr(solver, k):
                v = getattr(solver, k)
                diag[k] = v if isinstance(v, (bool, int, float, str)) else (float(v) if np.isscalar(v) else str(v))
        res["solver_diagnostics"] = diag
        blobs = []
        try:
            eh = np.asarray(getattr(solver, "energy_history", []), dtype=float)
            if eh.size:
                blobs.append(("energy_history", None, eh))
        except Exception:
            pass
        if u_ex is not None and case.get("test"):
            stage("compute_errors")
            want = a.store_fields != "none"
            margins = json.loads(a.margins) if a.margins else [0.0]
            rows, fields = compute_errors(solver, case, u_ex, u_stat, want, margins)
            res["margins"] = margins
            if a.store_fields == "final":
                fields = fields[-2:]
            for name, te, arr in fields:
                if arr.nbytes <= a.max_field_mb * 1e6:
                    blobs.append((name, te, arr))
            errs = [x["error"] for x in rows if x["error"] is not None]
            res.update(errors=rows, error_max=max(errs) if errs else None,
                       error_final=rows[-1]["error"], passed=int(all(x["passed"] for x in rows)),
                       non_finite=int(any(x["error"] is None or x.get("n_nonfinite_u_num") for x in rows)))
        else:
            res["note"] = "no exact solution: time only"
        res["blobs"] = [(n, t, str(x.dtype), list(x.shape), base64.b64encode(_pack(x)).decode("ascii")) for n, t, x in blobs]
        res["fft_backend"] = "pyfftw" if (case["dim"] == 2 and importlib.util.find_spec("pyfftw")) else "scipy.fft"

        # ── Append override info and user note to the run's note ──
        note_parts = []
        if override_notes:
            note_parts.append("override: " + ", ".join(override_notes))
        if a.note:
            note_parts.append(a.note)
        existing_note = res.get("note") or ""
        if note_parts:
            res["note"] = (existing_note + " | " if existing_note else "") + " | ".join(note_parts)
        # ── END note ──

        res["stage"] = "done"
    except BaseException as e:                       # includes SystemExit from the solver
        res.update(status="failed", exception_type=type(e).__name__, exception_message=str(e)[:2000],
                   traceback=traceback.format_exc()[-6000:], reps=reps)
    res["warnings"] = [dict(category=k[0], message=k[1], filename=k[2], lineno=k[3], count=v) for k, v in Counter(
        (x.category.__name__, str(x.message)[:400], x.filename, x.lineno) for x in wlist).items()]
    res["n_warnings"] = len(wlist)
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        res["peak_rss_mb"] = rss / 1048576 if sys.platform == "darwin" else rss / 1024
    except Exception:
        pass
    _write_json(a.out, res)


def _pack(arr):
    bio = io.BytesIO()
    np_save(bio, arr)
    return zlib.compress(bio.getvalue(), 6)


def np_save(f, arr):
    import numpy as np
    np.save(f, arr, allow_pickle=False)


def unpack(blob):
    import numpy as np
    return np.load(io.BytesIO(zlib.decompress(blob)), allow_pickle=False)


# =============================================================================================
# 4. run : parent side
# =============================================================================================

def cpu_topology():
    """Logical CPUs, CPUs this process may use (affinity/cgroup), physical cores if detectable."""
    info = {"logical": os.cpu_count(), "available": None, "physical": None}
    try:
        info["available"] = len(os.sched_getaffinity(0))
    except AttributeError:
        info["available"] = info["logical"]
    try:
        import psutil
        info["physical"] = psutil.cpu_count(logical=False)
    except Exception:
        pass
    if not info["physical"]:
        try:
            if sys.platform.startswith("linux"):
                pairs, pid = set(), None
                for line in open("/proc/cpuinfo"):
                    if line.startswith("physical id"):
                        pid = line.split(":")[1].strip()
                    elif line.startswith("core id"):
                        pairs.add((pid, line.split(":")[1].strip()))
                info["physical"] = len(pairs) or None
            elif sys.platform == "darwin":
                info["physical"] = int(subprocess.run(["sysctl", "-n", "hw.physicalcpu"], capture_output=True, text=True).stdout)
        except Exception:
            pass
    return info


def suggest_ladder(base, top):
    """base, doubling steps, midpoints, and top: e.g. base=4, top=8 -> [4, 6, 8]."""
    pts = {base, top}
    k = base
    while k * 2 <= top:
        k *= 2; pts.add(k)
    pts = sorted(pts)
    for lo, hi in zip(pts, pts[1:]):
        if hi - lo >= 2:
            pts.append((lo + hi) // 2)
    return sorted(set(pts))


def cmd_workers(a):
    t = cpu_topology()
    top = a.max or t["physical"] or t["available"]
    print(f"logical CPUs: {t['logical']} | usable by this process: {t['available']} | physical cores: {t['physical'] or 'unknown'}")
    print(f"suggested ladder (base {a.base}, top {top}): {' '.join(map(str, suggest_ladder(a.base, top)))}")
    print("rules of thumb: stay <= physical cores (hyper-threads rarely help FFTs; above that you measure oversubscription);\n"
          "add your production setting; add one odd value (e.g. 5) only to probe load imbalance;\n"
          "use --repeats >= 3 on an otherwise idle machine; speed-ups are relative to the smallest value you test.")


def parse_workers(vals):
    out = []
    for v in vals:
        if v == "auto":
            t = cpu_topology()
            out += suggest_ladder(4 if (t["physical"] or t["available"]) >= 8 else 1, t["physical"] or t["available"])
        else:
            out.append(int(v))
    return sorted(set(out))


def file_sha(path):
    try:
        return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    except OSError:
        return None


def environment(solver_dir):
    import numpy, scipy, sympy
    return {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
            "sympy": sympy.__version__, "platform": platform.platform(), "hostname": platform.node(),
            "cpu_count": os.cpu_count(), "cpus_available": cpu_topology()["available"],
            "physical_cores": cpu_topology()["physical"], "machine": platform.machine(),
            "solver_sha256": file_sha(os.path.join(solver_dir, "solver.py")),
            "psiop_sha256": file_sha(os.path.join(solver_dir, "psiop.py")),
            "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}


def _read_log(path, cap):
    try:
        data = open(path, "rb").read()
    except OSError:
        return b"", 0, 0
    n = len(data)
    return (data[-cap:], n, 1) if n > cap else (data, n, 0)


def execute(case, workers, a, env_info):
    """Run one (case, workers) in a subprocess; always return a complete record, whatever happens."""
    tmp = tempfile.mkdtemp(prefix="pdedb_")
    out, prog, fo, fe = (os.path.join(tmp, n) for n in ("out.json", "progress.json", "stdout.txt", "stderr.txt"))
    cmd = [sys.executable, os.path.abspath(__file__), "_worker", "--solver-dir", a.solver_dir,
           "--workers", str(workers), "--repeats", str(a.repeats), "--out", out, "--progress", prog,
           "--store-fields", a.store_fields, "--max-field-mb", str(a.max_field_mb), "--margins", json.dumps(a.margins),
           "--case-json", json.dumps(case)] \
          + (["--no-warmup"] if a.no_warmup else []) \
          + (["--Lt", str(a.Lt)] if a.Lt is not None else []) \
          + (["--Nt", str(a.Nt)] if a.Nt is not None else []) \
          + (["--note", a.note] if a.note else [])
    env = dict(os.environ)
    if a.blas_threads != "default":
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[v] = str(workers if a.blas_threads == "match" else 1)
    rec = {"started_at": now(), "fft_workers": workers, "blas_threads": a.blas_threads, "repeats": a.repeats,
           "warmup_excluded": int(not a.no_warmup), "environment": env_info}
    t0 = time.perf_counter()
    rc, timed_out = None, False
    with open(fo, "wb") as so, open(fe, "wb") as se:
        try:
            rc = subprocess.run(cmd, env=env, stdout=so, stderr=se, timeout=a.timeout).returncode
        except subprocess.TimeoutExpired:
            timed_out = True
    rec["wall_clock_total_s"] = time.perf_counter() - t0
    rec["finished_at"] = now()
    rec["returncode"] = rc
    result = None
    for p in (out,):
        try:
            result = json.load(open(p))
        except (OSError, json.JSONDecodeError):
            pass
    if result is not None and not timed_out:
        rec.update(result)
        if rc and rc < 0:
            rec["signal"] = signal.Signals(-rc).name
    else:                                            # no final result: timeout or hard crash (segfault, OOM kill...)
        partial = {}
        try:
            partial = json.load(open(prog))
        except (OSError, json.JSONDecodeError):
            pass
        rec.update(stage=partial.get("stage"), reps=partial.get("reps", []), warnings=[], n_warnings=0)
        if timed_out:
            rec.update(status="timeout", timeout_s=a.timeout,
                       note=f"killed after {a.timeout}s while in stage '{partial.get('stage')}'")
        else:
            sig = signal.Signals(-rc).name if rc is not None and rc < 0 else None
            rec.update(status="crashed", signal=sig,
                       note=f"worker died without result (exit code {rc}{', ' + sig if sig else ''}) in stage '{partial.get('stage')}'")
    rec["logs"] = {"stdout": _read_log(fo, a.max_log_kb * 1024), "stderr": _read_log(fe, a.max_log_kb * 1024)}
    for p in (out, prog, fo, fe, out + ".tmp", prog + ".tmp"):
        try: os.remove(p)
        except OSError: pass
    try: os.rmdir(tmp)
    except OSError: pass
    return rec


def store_run(con, case_id, r):
    env = r.get("environment", {})
    flat = dict(case_id=case_id, started_at=r.get("started_at"), finished_at=r.get("finished_at"),
                wall_clock_total_s=r.get("wall_clock_total_s"), status=r.get("status"), stage=r.get("stage"),
                fft_workers=r.get("fft_workers"), blas_threads=r.get("blas_threads"), repeats=r.get("repeats"),
                warmup_excluded=r.get("warmup_excluded"),
                margins=jdump(r["margins"]) if r.get("margins") else None)
    for k in ("time_init_s", "time_setup_s", "time_solve_s", "time_total_s", "cpu_time_solve_s", "cpu_over_wall",
              "time_solve_min_s", "error_max", "error_final", "passed", "non_finite", "n_warnings", "peak_rss_mb",
              "exception_type", "exception_message", "traceback", "returncode", "signal", "timeout_s", "fft_backend", "note"):
        flat[k] = r.get(k)
    flat.update(solver_diagnostics=jdump(jsonable(r["solver_diagnostics"])) if r.get("solver_diagnostics") else None,
                environment=jdump(env), **{k: env.get(k) for k in ("solver_sha256", "psiop_sha256", "python", "numpy",
                                                                    "scipy", "sympy", "platform", "hostname", "cpu_count")})
    flat = jsonable(flat)
    cur = con.execute(f"INSERT INTO runs ({','.join(flat)}) VALUES ({','.join('?' * len(flat))})", list(flat.values()))
    rid = cur.lastrowid
    for x in r.get("reps", []):
        con.execute("INSERT INTO run_repeats VALUES (?,?,?,?,?,?,?,?)",
                    (rid, x["rep"], x["is_warmup"], x["t_init_s"], x["t_setup_s"], x["t_solve_s"], x["cpu_solve_s"], x["completed"]))
    for e in jsonable(r.get("errors", [])):
        con.execute("INSERT INTO run_errors VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (rid, e["t_eval"], e["actual_t"], e["component"], e["norm"], e["error"], e["threshold"], e["passed"],
                     e.get("ratio_to_threshold"), e.get("rel_l2"), e.get("abs_l2"), e.get("linf"), e.get("max_abs_u_num"),
                     e.get("max_abs_u_exact"), e.get("n_nonfinite_u_num"), e.get("message")))
    for e in jsonable(r.get("errors", [])):
        for pr in e.get("profile", []):
            con.execute("INSERT INTO run_error_profile VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (rid, e["t_eval"], e["actual_t"], e["component"], pr["margin"], pr["n_points_used"],
                         pr["ref_l2"], pr["abs_l2"], pr["rel_l2"], pr["linf"], pr["energy_share"]))
    for w in r.get("warnings", []):
        con.execute("INSERT INTO run_warnings VALUES (?,?,?,?,?,?)",
                    (rid, w["category"], w["message"], w["filename"], w["lineno"], w["count"]))
    for stream, (data, nbytes, trunc) in r.get("logs", {}).items():
        con.execute("INSERT INTO run_logs VALUES (?,?,?,?,?)", (rid, stream, nbytes, trunc, zlib.compress(data, 6)))
    for name, te, dtype, shape, blob in r.get("blobs", []):
        con.execute("INSERT INTO run_blobs VALUES (?,?,?,?,?,?)", (rid, name, te, dtype, json.dumps(shape), base64.b64decode(blob)))
    con.commit()
    return rid


def case_filter(a):
    cl, p = [], []
    if not getattr(a, "all_cases", False):
        cl.append("active=1")
    if getattr(a, "ids", None):
        cl.append(f"case_id IN ({','.join('?' * len(a.ids))})"); p += a.ids
    if getattr(a, "notebook", None):
        cl.append("(" + " OR ".join("case_id LIKE ?" for _ in a.notebook) + ")"); p += [x + "-%" for x in a.notebook]
    if getattr(a, "match", None):
        cl.append("name LIKE ?"); p.append(f"%{a.match}%")
    for col, attr in (("dim", "dim"), ("type", "type"), ("boundary_condition", "bc")):
        if getattr(a, attr, None):
            cl.append(f"{col}=?"); p.append(getattr(a, attr))
    if getattr(a, "formulation", None):
        cl.append("('+' || formulation || '+') LIKE ?"); p.append(f"%+{a.formulation}+%")
    if getattr(a, "with_exact", False):
        cl.append("has_exact=1")
    if getattr(a, "max_points", None):
        cl.append("n_points <= ?"); p.append(a.max_points)
    return (" WHERE " + " AND ".join(cl)) if cl else "", p


def select_cases(con, a):
    w, p = case_filter(a)
    return con.execute(f"SELECT case_id, name, definition, n_points FROM cases{w} ORDER BY case_id", p).fetchall()


def cmd_run(a):
    con = connect(a.db)
    cases = select_cases(con, a)
    a.workers = parse_workers(a.workers)
    a.margins = sorted({0.0, *map(float, a.margins)})
    if not all(0 <= m < 1 for m in a.margins):
        sys.exit("--margins must be fractions in [0, 1)")
    env_info = environment(a.solver_dir)
    for lim, label in ((env_info["cpus_available"], "usable CPUs"), (env_info["physical_cores"], "physical cores")):
        if lim and max(a.workers) > lim:
            print(f"WARNING: workers up to {max(a.workers)} exceed the {lim} {label}: oversubscribed timings are not a scaling measure")
    if min(a.workers) > 1:
        print(f"note: smallest worker count is {min(a.workers)}: speed-ups/efficiencies will be relative to it")

    # ── Expand Lt / Nt sweep values ──
    lt_values = a.Lt if a.Lt is not None else [None]
    nt_values = a.Nt if a.Nt is not None else [None]
    # ── END ──

    todo = []
    for c in cases:
        for w in a.workers:
            last = con.execute("SELECT status, solver_sha256 FROM runs WHERE case_id=? AND fft_workers=? ORDER BY run_id DESC LIMIT 1",
                               (c["case_id"], w)).fetchone()
            if a.skip_done and con.execute("SELECT 1 FROM runs WHERE case_id=? AND fft_workers=? AND status='ok' AND solver_sha256 IS ?",
                                           (c["case_id"], w, env_info["solver_sha256"])).fetchone():
                continue
            if a.only_failed and (last is None or last["status"] == "ok"):
                continue
            # ── One entry per (Lt, Nt) combination ──
            for lt_val in lt_values:
                for nt_val in nt_values:
                    todo.append((c, w, lt_val, nt_val))
            # ── END ──
    print(f"{len(todo)} runs ({len(cases)} cases, workers {a.workers}, "
          f"Lt={a.Lt or 'notebook'}, Nt={a.Nt or 'notebook'}); solver sha {env_info['solver_sha256']}")
    if a.dry_run:
        for c, w, lt_val, nt_val in todo:
            ov = ""
            if lt_val is not None: ov += f" Lt={lt_val}"
            if nt_val is not None: ov += f" Nt={nt_val}"
            print(f"  {c['case_id']:8s} workers={w:<3d}{ov}  {c['name'][:60]}")
        return
    for i, (c, w, lt_val, nt_val) in enumerate(todo, 1):
        # ── Set single values for this run ──
        a.Lt = lt_val
        a.Nt = nt_val
        # ── END ──
        rec = execute(json.loads(c["definition"]), w, a, env_info)
        rid = store_run(con, c["case_id"], rec)
        if rec["status"] == "ok":
            em = rec.get("error_max")
            msg = (f"solve={rec['time_solve_s']:.3f}s cpu/wall={rec['cpu_over_wall'] or 0:.2f} "
                   f"err_max={'-' if em is None else format(em, '.2e')} passed={rec.get('passed')} warn={rec['n_warnings']}")
        else:
            msg = f"stage={rec.get('stage')} " + (rec.get("exception_type") and f"{rec['exception_type']}: {rec['exception_message'][:90]}" or rec.get("note", ""))
        print(f"[{i}/{len(todo)}] run {rid:<5d} {c['case_id']:8s} w={w:<3d} {rec['status']:8s} {msg}")


# =============================================================================================
# 5. reports
# =============================================================================================

def table(rows, cols=None):
    if not rows:
        print("(no rows)"); return
    cols = cols or list(rows[0].keys())

    def fmt(v):
        if v is None: return "-"
        if isinstance(v, float):
            return f"{v:.2e}" if v and (abs(v) < 1e-3 or abs(v) >= 1e5) else f"{v:.3f}"
        return str(v).replace("\n", " ")[:70]

    wid = {k: max(len(k), *(len(fmt(r[k])) for r in rows)) for k in cols}
    print("  ".join(k.ljust(wid[k]) for k in cols))
    for r in rows:
        print("  ".join(fmt(r[k]).ljust(wid[k]) for k in cols))


def cmd_list(a):
    con = connect(a.db)
    w, p = case_filter(a)
    rows = con.execute(f"""SELECT case_id, dim, substr(type,1,4) AS type, temporal_order AS ord, formulation, boundary_condition AS bc,
COALESCE(Nx,'') || COALESCE('x'||Ny,'') AS grid, Nt, exact_kind AS exact,
(SELECT COUNT(*) FROM runs r WHERE r.case_id=cases.case_id) AS runs, substr(name,1,48) AS name
FROM cases{w} ORDER BY case_id""", p).fetchall()
    table(rows)


def cmd_report(a):
    con = connect(a.db)
    w, p = case_filter(a)
    ids = [r[0] for r in con.execute(f"SELECT case_id FROM cases{w}", p)]
    rows = [r for r in con.execute("SELECT * FROM v_scaling ORDER BY case_id, fft_workers") if r["case_id"] in ids]
    cols = ["case_id", "name", "dim", "n_points", "fft_workers", "base_workers", "time_init_s", "time_setup_s", "time_solve_s",
            "cpu_over_wall", "speedup", "efficiency", "error_max", "passed", "n_warnings"]
    rows = [{k: (r[k][:40] if k == "name" else r[k]) for k in cols} for r in rows]
    table(rows)
    bad = con.execute("SELECT COUNT(*) FROM v_failures").fetchone()[0]
    if bad:
        print(f"\n{bad} non-ok run(s) stored: see `failures`")
    if a.csv and rows:
        with open(a.csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=cols); wr.writeheader(); wr.writerows(rows)
        print(f"CSV written: {a.csv}")


def cmd_failures(a):
    con = connect(a.db)
    q = "SELECT * FROM v_failures" + (" WHERE run_id IN (SELECT MAX(run_id) FROM runs GROUP BY case_id, fft_workers)" if a.last else "")
    rows = con.execute(q + " ORDER BY run_id").fetchall()
    table([{k: r[k] for k in ("run_id", "case_id", "fft_workers", "status", "stage", "exception_type", "exception_message", "signal")}
           for r in rows])
    print("\nuse `show RUN_ID --logs` for traceback and logs")


def cmd_show(a):
    con = connect(a.db)
    r = con.execute("SELECT r.*, c.name, c.equation FROM runs r JOIN cases c USING(case_id) WHERE run_id=?", (a.run_id,)).fetchone()
    if not r:
        sys.exit("no such run")
    skip = {"environment", "traceback", "name", "equation", "solver_diagnostics"}
    print(f"RUN {r['run_id']}  {r['case_id']}  {r['name']}\n{r['equation']}")
    for k in r.keys():
        if k not in skip and r[k] is not None:
            print(f"  {k:22s} {r[k]}")
    if r["solver_diagnostics"]:
        print(f"  solver_diagnostics     {r['solver_diagnostics']}")
    print("\nrepetitions:")
    table([dict(x) for x in con.execute("SELECT rep, is_warmup, t_init_s, t_setup_s, t_solve_s, cpu_solve_s, completed FROM run_repeats WHERE run_id=?", (a.run_id,))])
    print("\nerrors per time:")
    table([dict(x) for x in con.execute("SELECT t_eval, actual_t, component, error, threshold, passed, ratio_to_threshold AS ratio, linf, max_abs_u_num, n_nonfinite_u_num AS n_nan, message FROM run_errors WHERE run_id=?", (a.run_id,))])
    pr = con.execute("SELECT t_eval, margin, n_points_used AS n_used, rel_l2, linf, energy_share FROM run_error_profile WHERE run_id=? ORDER BY t_eval, margin", (a.run_id,)).fetchall()
    if pr:
        print("\ninterior profile (margin 0 = whole domain):"); table([dict(x) for x in pr])
    ws = con.execute("SELECT category, count, message FROM run_warnings WHERE run_id=?", (a.run_id,)).fetchall()
    if ws:
        print("\nwarnings:"); table([dict(x) for x in ws])
    if r["traceback"]:
        print("\ntraceback:\n" + r["traceback"])
    if a.logs:
        for x in con.execute("SELECT stream, nbytes, truncated, data FROM run_logs WHERE run_id=?", (a.run_id,)):
            print(f"\n----- {x['stream']} ({x['nbytes']} bytes{', truncated to tail' if x['truncated'] else ''}) -----")
            print(zlib.decompress(x["data"]).decode("utf-8", "replace"))
    if a.save_fields:
        import numpy as np
        arrs = {}
        for x in con.execute("SELECT name, t_eval, data FROM run_blobs WHERE run_id=?", (a.run_id,)):
            arrs[x["name"] + ("" if x["t_eval"] is None else f"_t{x['t_eval']:g}")] = unpack(x["data"])
        np.savez_compressed(a.save_fields, **arrs)
        print(f"\nsaved {len(arrs)} array(s) -> {a.save_fields}: {', '.join(arrs)}")


def cmd_profile(a):
    """Compute the interior profile from the fields already stored (run --store-fields final|all)."""
    con = connect(a.db)
    margins = sorted({0.0, *map(float, a.margins)})
    if not all(0 <= m < 1 for m in margins):
        sys.exit("--margins must be fractions in [0, 1)")
    w, p = case_filter(a)
    cases = {r["case_id"]: json.loads(r["definition"]) for r in con.execute(f"SELECT case_id, definition FROM cases{w}", p)}
    runs = con.execute("""SELECT run_id, case_id FROM runs r WHERE status = 'ok' AND EXISTS
(SELECT 1 FROM run_blobs b WHERE b.run_id = r.run_id AND b.name = 'u_num') ORDER BY run_id""").fetchall()
    runs = [r for r in runs if r["case_id"] in cases]
    if not a.all_runs:
        runs = list({r["case_id"]: r for r in runs}.values())          # latest run with stored fields per case
    n_t = 0
    for r in runs:
        dom = cases[r["case_id"]]["domain"]
        blobs = {}
        for b in con.execute("SELECT name, t_eval, data FROM run_blobs WHERE run_id=? AND name IN ('u_num','u_exact')", (r["run_id"],)):
            blobs[(b["name"], b["t_eval"])] = b["data"]
        for t in sorted({te for (nm, te) in blobs if nm == "u_num" and ("u_exact", te) in blobs}, key=lambda x: (x is None, x)):
            meta = con.execute("SELECT component, actual_t FROM run_errors WHERE run_id=? AND t_eval IS ?", (r["run_id"], t)).fetchone()
            if not meta:
                continue
            rows = profile_rows(unpack(blobs[("u_num", t)]), unpack(blobs[("u_exact", t)]), meta["component"], dom, margins)
            con.execute(f"DELETE FROM run_error_profile WHERE run_id=? AND t_eval IS ? AND margin IN ({','.join('?' * len(margins))})",
                        (r["run_id"], t, *margins))
            for pr in rows:
                con.execute("INSERT INTO run_error_profile VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                            (r["run_id"], t, meta["actual_t"], meta["component"], pr["margin"], pr["n_points_used"],
                             pr["ref_l2"], pr["abs_l2"], pr["rel_l2"], pr["linf"], pr["energy_share"]))
            n_t += 1
    con.commit()
    missing = con.execute(f"""SELECT COUNT(*) FROM cases{w + (' AND' if w else ' WHERE')} has_exact = 1 AND case_id IN
(SELECT case_id FROM runs WHERE status='ok') AND case_id NOT IN
(SELECT case_id FROM runs r WHERE EXISTS (SELECT 1 FROM run_blobs b WHERE b.run_id = r.run_id AND b.name = 'u_num'))""", p).fetchone()[0]
    print(f"profile computed for {len(runs)} run(s), {n_t} tested time(s), margins {margins}")
    if missing:
        print(f"{missing} case(s) with an exact solution have no stored fields: run them with --store-fields final|all "
              "(and --margins ... to compute the profile directly)")


def cmd_sql(a):
    con = connect(a.db)
    con.execute("PRAGMA query_only=ON")
    if not re.match(r"\s*(select|with|pragma)", a.query, re.I):
        sys.exit("read-only: only SELECT/WITH/PRAGMA")
    table([dict(r) for r in con.execute(a.query)])


# =============================================================================================
# 6. CLI
# =============================================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="pde.sqlite")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def filters(p):
        p.add_argument("--ids", nargs="+"); p.add_argument("--notebook", nargs="+", help="case prefixes, e.g. p1d p2d np1d np2d ex")
        p.add_argument("--match", help="substring of the case name"); p.add_argument("--dim", type=int, choices=(1, 2))
        p.add_argument("--type", choices=("stationary", "evolution")); p.add_argument("--bc", choices=("periodic", "nonperiodic"))
        p.add_argument("--formulation", choices=("PDE", "Op", "psiOp")); p.add_argument("--with-exact", action="store_true")
        p.add_argument("--max-points", type=float, help="skip cases with Nx*Ny*Nt above this"); p.add_argument("--all-cases", action="store_true", help="include inactive cases")

    p = sub.add_parser("build"); p.add_argument("--notebooks", nargs="+", required=True); p.set_defaults(f=cmd_build)
    p = sub.add_parser("list"); filters(p); p.set_defaults(f=cmd_list)

    p = sub.add_parser("run"); filters(p)
    p.add_argument("--solver-dir", default=".", help="folder with solver.py, psiop.py, imports.py")
    p.add_argument("--workers", nargs="+", default=["1"], help="e.g. 4 5 8, or 'auto' (ladder from this machine's cores)"); p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--no-warmup", action="store_true"); p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--blas-threads", choices=("match", "1", "default"), default="match")
    p.add_argument("--store-fields", choices=("none", "final", "all"), default="none", help="keep u_num/u_exact arrays in the DB")
    p.add_argument("--margins", nargs="+", default=["0"], help="interior boxes: fraction of the half-width removed on each side; 0 (whole domain, default) is always computed")
    p.add_argument("--max-field-mb", type=float, default=20); p.add_argument("--max-log-kb", type=int, default=2048)
    p.add_argument("--skip-done", action="store_true", help="skip (case, workers) already ok with the same solver.py")
    p.add_argument("--only-failed", action="store_true", help="re-run only (case, workers) whose last run is not ok")
    p.add_argument("--dry-run", action="store_true")
    # ── NEW: Lt / Nt override and note ──
    p.add_argument("--Lt", nargs="+", type=float, default=None,
                   help="override Lt (one or more values, rescales t_eval proportionally)")
    p.add_argument("--Nt", nargs="+", type=int, default=None,
                   help="override Nt (one or more values)")
    p.add_argument("--note", default=None, help="free-text note stored with each run")
    # ── END NEW ──
    p.set_defaults(f=cmd_run)

    p = sub.add_parser("report"); filters(p); p.add_argument("--csv"); p.set_defaults(f=cmd_report)
    p = sub.add_parser("failures"); p.add_argument("--last", action="store_true", help="only the latest run of each (case, workers)"); p.set_defaults(f=cmd_failures)
    p = sub.add_parser("show"); p.add_argument("run_id", type=int); p.add_argument("--logs", action="store_true")
    p.add_argument("--save-fields", metavar="FILE.npz"); p.set_defaults(f=cmd_show)
    p = sub.add_parser("workers"); p.add_argument("--base", type=int, default=4); p.add_argument("--max", type=int); p.set_defaults(f=cmd_workers)
    p = sub.add_parser("profile"); filters(p); p.add_argument("--margins", nargs="+", required=True, type=float)
    p.add_argument("--all-runs", action="store_true", help="every run with stored fields, not only the latest per case"); p.set_defaults(f=cmd_profile)
    p = sub.add_parser("sql"); p.add_argument("query"); p.set_defaults(f=cmd_sql)

    w = sub.add_parser("_worker")
    for n in ("--solver-dir", "--out", "--progress", "--case-json", "--store-fields"):
        w.add_argument(n)
    w.add_argument("--workers", type=int); w.add_argument("--repeats", type=int); w.add_argument("--max-field-mb", type=float)
    w.add_argument("--margins"); w.add_argument("--no-warmup", action="store_true")
    # ── NEW: Lt / Nt override and note (received from execute()) ──
    w.add_argument("--Lt", type=float, default=None)
    w.add_argument("--Nt", type=int, default=None)
    w.add_argument("--note", default=None)
    # ── END NEW ──
    w.set_defaults(f=worker)

    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()