"""make_pde_db_report.py — build pde_db_report.ipynb from a pde_db.py database."""
import argparse
from pathlib import Path
import nbformat as nbf

cells = []
def md(s):   cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# PDE benchmark report — `pde_db.py` database

Read-only exploration of `cases / runs / run_repeats / run_errors / run_warnings / run_blobs`.
Data produced by `pde_db.py build` + `pde_db.py run`. This notebook never writes to the DB.
""")

code("""
import io, json, sqlite3, zlib
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display, Markdown

DB = "pde.sqlite"
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)

C  = pd.read_sql("SELECT * FROM cases", con)
R  = pd.read_sql("SELECT * FROM runs", con)
E  = pd.read_sql("SELECT * FROM run_errors", con)
W  = pd.read_sql("SELECT * FROM run_warnings", con)
RR = pd.read_sql("SELECT * FROM run_repeats", con)
S  = pd.read_sql("SELECT * FROM v_scaling", con)
LO = pd.read_sql("SELECT * FROM v_latest_ok", con)

# runs / v_latest_ok carry run-level columns only: attach case metadata once, here
CC = C[["case_id", "name", "family", "dim", "type", "n_points", "boundary_condition"]]
R  = R.merge(CC, on="case_id", how="left")
LO = LO.merge(CC, on="case_id", how="left")
LAST = R[R.status == "ok"].sort_values("run_id").groupby("case_id").tail(1)
C["defn"] = C.definition.map(json.loads)

def blobs(run_id):
    \"\"\"Decompress numerical fields / energy history stored by the worker.\"\"\"
    out = {}
    for name, te, data in con.execute(
            "SELECT name, t_eval, data FROM run_blobs WHERE run_id=?", (run_id,)):
        out[name if te is None else f"{name}@{te:g}"] = np.load(
            io.BytesIO(zlib.decompress(data)), allow_pickle=False)
    return out

plt.rcParams.update({"figure.figsize": (9, 4), "axes.grid": True, "grid.alpha": .3})
print(f"{len(C)} cases ({int(C.active.sum())} active) | {len(R)} runs | {R.status.value_counts().to_dict()}")
print("workers:", sorted(R.fft_workers.dropna().unique()),
      "| backends:", R.fft_backend.dropna().unique().tolist(),
      "| solver sha:", R.solver_sha256.dropna().unique().tolist())
""")

md("## 1 — Catalogue")
code("""
disp = C.merge(R.groupby("case_id").size().rename("n_runs"), on="case_id", how="left")
display(disp.groupby(["notebook", "dim", "type"]).size().rename("n").to_frame())
display(disp[["case_id", "name", "dim", "type", "formulation", "boundary_condition",
              "Nx", "Ny", "Nt", "exact_kind", "n_runs"]]
        .assign(name=lambda d: d.name.str.slice(0, 48)).fillna("-").head(25))
""")

md("## 2 — Correctness dashboard (latest ok run per case)")
code("""
marg = E.groupby("run_id").ratio_to_threshold.max().rename("margin")
dash = (LAST.merge(C[["case_id", "name", "dim", "type", "boundary_condition", "exact_kind"]], on="case_id")
            .merge(marg, left_on="run_id", right_index=True, how="left"))
dash["status"] = np.where(dash.passed.isna(), "no test", np.where(dash.passed == 1, "PASS", "FAIL"))
display(dash[["case_id", "status", "margin", "error_max", "time_solve_s", "n_warnings", "exact_kind"]]
        .sort_values("margin", ascending=False).head(25).round(4))
print(f"PASS {int((dash.passed==1).sum())} | FAIL {int((dash.passed==0).sum())} "
      f"| no exact {int(dash.passed.isna().sum())}")
approx = C[C.exact_kind == "approximate"][["case_id", "name", "notes"]]
if len(approx):
    display(Markdown("**Heuristic 'exact' solutions (smoke tests):**")); display(approx)
""")

md("## 3 — Error anatomy")
code("""
fig, ax = plt.subplots()
for cid, g in (E[E.run_id.isin(LAST.run_id)].merge(LAST[["run_id", "case_id"]], on="run_id")
               .groupby("case_id")):
    g = g.sort_values("actual_t")
    ax.plot(g.actual_t, g.error.clip(lower=1e-16), ".-", lw=.8, ms=3, label=cid)
ax.set_yscale("log"); ax.set_xlabel("t"); ax.set_ylabel("error")
ax.set_title("error vs time (latest ok run)"); ax.legend(fontsize=6, ncol=3); plt.show()

e = E[E.run_id.isin(LAST.run_id)]
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].scatter(e.rel_l2, e.linf, c=e.run_id.map(LAST.set_index("run_id").dim), s=14, cmap="viridis")
axs[0].set_xscale("log"); axs[0].set_yscale("log")
axs[0].set_xlabel("relative L2"); axs[0].set_ylabel("Linf"); axs[0].set_title("norm anatomy")
axs[1].hist(np.log10(e.ratio_to_threshold.dropna().clip(lower=1e-16)), bins=30)
axs[1].axvline(0, color="r", ls="--", label="threshold")
axs[1].set_xlabel("log10(error / threshold)"); axs[1].legend()
plt.tight_layout(); plt.show()
nn = e[e.n_nonfinite_u_num > 0]
print(f"runs with non-finite fields: {nn.run_id.nunique()}")
""")

md("""
## 4 — Scaling and parallel efficiency

`speedup`/`efficiency` come from view `v_scaling` (latest ok per (case, workers) vs 1 worker).
`cpu_over_wall` ≈ effective number of threads actually working.
""")
code("""
Sp = S.pivot_table(index="case_id", columns="fft_workers", values="speedup")
Ef = S.pivot_table(index="case_id", columns="fft_workers", values="efficiency")
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for _, row in Sp.dropna(how="all").iterrows():
    axs[0].plot(row.index, row.values, ".-", alpha=.4)
dim_of = C.set_index("case_id").dim
for d in (1, 2):
    sel = Sp.index[dim_of.reindex(Sp.index).fillna(0).values == d]
    if len(sel):
        axs[0].plot(Sp.columns, Sp.loc[sel].mean(), lw=3, marker="o", label=f"{d}D mean")
axs[0].plot(Sp.columns, list(Sp.columns), "k--", label="ideal")
axs[0].set_xlabel("fft_workers"); axs[0].set_ylabel("speedup"); axs[0].legend()
for _, row in Ef.dropna(how="all").iterrows():
    axs[1].plot(row.index, row.values, ".-", alpha=.4)
axs[1].axhline(1, color="k", ls="--"); axs[1].set_ylim(0, 1.3)
axs[1].set_xlabel("fft_workers"); axs[1].set_ylabel("efficiency")
LO.groupby("fft_workers").cpu_over_wall.median().plot(ax=axs[2], marker="o")
axs[2].set_xlabel("fft_workers"); axs[2].set_ylabel("cpu_solve / wall_solve")
plt.tight_layout(); plt.show()
display(S[["case_id", "fft_workers", "time_solve_s", "speedup", "efficiency", "cpu_over_wall"]]
        .assign(name=S.name.str.slice(0, 30)).round(3).head(30))
""")

md("## 5 — Warm-up cost and repeat jitter (from `run_repeats`)")
code("""
g = RR[RR.completed == 1]
if len(g):
    warm = g[g.is_warmup == 1].groupby("run_id").t_solve_s.median()
    cold = g[g.is_warmup == 0].groupby("run_id").t_solve_s.median()
    both = pd.DataFrame({"warm": warm, "cold": cold}).dropna()
    cv = g[g.is_warmup == 0].groupby("run_id").t_solve_s.agg(
        lambda s: s.std() / s.median() if len(s) > 1 else np.nan).dropna()
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    (both.warm / both.cold).hist(bins=30, ax=axs[0])
    axs[0].set_xlabel("warm-up / steady solve time  (>1: FFT planning, symbol tables)")
    cv.hist(bins=30, ax=axs[1]); axs[1].set_xlabel("std/median of repeats (jitter)")
    plt.tight_layout(); plt.show()
    print(f"median warm-up penalty x{(both.warm / both.cold).median():.2f} | median jitter {cv.median():.3f}")
else:
    print("no repeat data")
""")

md("## 6 — Where the time goes, cost per point, memory")
code("""
A = LO[LO.fft_workers == LO.fft_workers.min()].set_index("case_id")
A = A[["time_init_s", "time_setup_s", "time_solve_s"]].fillna(0)
A["serial"] = (A.time_init_s + A.time_setup_s) / A.sum(axis=1)
A.serial.sort_values(ascending=False).head(12).plot.barh(
    title="serial fraction at 1 worker: (init+setup)/total"); plt.show()

cost = LO.dropna(subset=["n_points"]).assign(ns=1e9 * LO.time_solve_s / LO.n_points)
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for (d, b), gg in cost.groupby(["dim", "fft_backend"]):
    axs[0].scatter(gg.n_points, gg.ns, s=14, alpha=.6, label=f"{d}D {b}")
axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[0].legend(fontsize=7)
axs[0].set_xlabel("Nx*Ny*Nt"); axs[0].set_ylabel("ns per point per step")
for (d, b), gg in LO.dropna(subset=["peak_rss_mb"]).groupby(["dim", "fft_backend"]):
    axs[1].scatter(gg.n_points, gg.peak_rss_mb, s=14, alpha=.6, label=f"{d}D {b}")
axs[1].set_xscale("log"); axs[1].set_yscale("log"); axs[1].legend(fontsize=7)
axs[1].set_xlabel("Nx*Ny*Nt"); axs[1].set_ylabel("peak RSS (MB)")
plt.tight_layout(); plt.show()
""")

md("## 7 — Code versions (solver.py + psiop.py hashes)")
code("""
R2 = R[R.status == "ok"].copy()
R2["ver"] = R2.solver_sha256.fillna("?") + "|" + R2.psiop_sha256.fillna("?")
V = R2.groupby("ver").agg(runs=("run_id", "size"), first=("started_at", "min"),
                          last=("started_at", "max")).sort_values("last")
display(V)
if len(V) >= 2:
    a, b = V.index[-2], V.index[-1]
    sub = R2[R2.ver.isin([a, b])]
    t = sub.pivot_table(index="case_id", columns="ver", values="time_solve_s", aggfunc="median").dropna()
    e = sub.pivot_table(index="case_id", columns="ver", values="error_max", aggfunc="max")
    cmp = pd.DataFrame({"before_s": t[a], "after_s": t[b], "speedup": t[a] / t[b],
                        "err_ratio": e[b] / e[a]}).sort_values("speedup", ascending=False)
    display(cmp.round(4))
    print(f"median speedup {cmp.speedup.median():.2f}x | slower: {(cmp.speedup < .95).sum()} "
          f"| error degraded >2x: {(cmp.err_ratio > 2).sum()}")
""")

md("## 8 — Failure forensics")
code("""
F = pd.read_sql("SELECT * FROM v_failures", con)
if len(F):
    display(F.groupby(["status", "stage"]).size().rename("n").to_frame())
    display(F.sort_values("run_id").tail(15)[["run_id", "case_id", "fft_workers", "status",
                                              "stage", "exception_type", "exception_message"]])
    def autopsy(run_id, tail=25):
        r = con.execute("SELECT status, stage, exception_type, exception_message, signal, note "
                        "FROM runs WHERE run_id=?", (run_id,)).fetchone()
        print(f"run {run_id}: {r[0]} at stage '{r[1]}' | {r[2]}: {r[3]} | signal={r[4]} | {r[5]}")
        for stream, data in con.execute("SELECT stream, data FROM run_logs WHERE run_id=?", (run_id,)):
            txt = zlib.decompress(data).decode("utf-8", "replace").strip().splitlines()
            print(f"--- {stream} (last {tail} lines) ---"); print("\\n".join(txt[-tail:]))
    # autopsy(int(F.run_id.iloc[-1]))
else:
    print("no failed runs")
""")

md("## 9 — Warnings and boundary-clipping health")
code(r"""
if len(W):
    display(W.groupby(["category", "message"]).agg(n=("run_id", "size"), runs=("run_id", "nunique"))
             .sort_values("runs", ascending=False).head(10))
diag = pd.DataFrame([json.loads(s) if s else {} for s in LO.solver_diagnostics])
diag.index = LO.index
diag["case_id"] = LO.case_id.values
diag["boundary_condition"] = LO.boundary_condition.values
if "_bc_max_correction" in diag.columns:
    np_ = diag[diag.boundary_condition != "periodic"].groupby("case_id")._bc_max_correction.max()
    np_ = np_[np_ > 0]                          # zeros cannot be drawn on a log axis
    if len(np_):
        fig, ax = plt.subplots()
        np_.sort_values(ascending=False).plot.barh(ax=ax)
        ax.set_xscale("log"); ax.axvline(1e-8, color="r", ls="--", label="boundary_tol")
        ax.set_xlabel("max relative clipping at boundary"); ax.legend(); plt.show()
    else:
        print("no nonperiodic run with non-zero clipping recorded")
else:
    print("no solver diagnostics stored")
""")

md("## 10 — Energy conservation (from stored `energy_history` blobs)")
code("""
rows = []
for run_id, case_id, data in con.execute(
        "SELECT b.run_id, r.case_id, b.data FROM run_blobs b JOIN runs r USING(run_id) "
        "WHERE b.name='energy_history' AND r.status='ok' AND b.run_id IN "
        "(SELECT MAX(run_id) FROM runs WHERE status='ok' GROUP BY case_id)"):
    eh = np.load(io.BytesIO(zlib.decompress(data)), allow_pickle=False)
    if eh.size > 1:
        rows.append((case_id, run_id, float(np.max(np.abs(eh - eh[0])) / abs(eh[0]))))
EN = pd.DataFrame(rows, columns=["case_id", "run_id", "drift"]).sort_values("drift", ascending=False)
if len(EN):
    display(EN.head(15))
    fig, ax = plt.subplots()
    for _, r in EN.head(6).iterrows():
        eh = blobs(r.run_id)["energy_history"]
        ax.plot(np.linspace(0, 1, eh.size), (eh - eh[0]) / abs(eh[0]), label=r.case_id)
    ax.set_xlabel("t / Lt"); ax.set_ylabel("relative energy drift"); ax.legend(); plt.show()
else:
    print("no energy history stored (no second-order case run yet)")
""")

md("## 11 — Case card")
code(r"""
def card(case_id):
    sel = C[C.case_id == case_id]
    if sel.empty:
        print("no such case"); return
    c = sel.iloc[0]; d = c.defn
    display(Markdown(f"### {c.case_id} — {c.name}"))
    display(pd.Series({
        "type": c.type, "dim": c.dim, "temporal_order": c.temporal_order,
        "formulation": c.formulation, "bc": c.boundary_condition,
        "grid": f"{c.Nx}" + (f"x{int(c.Ny)}" if pd.notna(c.Ny) else ""),
        "Lt / Nt": "-" if pd.isna(c.Nt) else f"{c.Lt} / {int(c.Nt)}",
        "scheme": c.effective_time_scheme or c.time_scheme,
        "dealiasing": c.dealiasing_ratio, "stationary_order": c.stationary_order,
        "exact": c.exact_kind, "source": f"{c.notebook} cell {c.cell}"}).to_frame("value"))
    display(Markdown(f"**equation**\n```\n{c.equation}\n```"))
    for key, lab in (("initial_condition", "initial condition"), ("exact_solution", "exact solution")):
        spec = d.get(key)
        src = (spec.get("python", spec) or {}).get("code") if isinstance(spec, dict) else None
        if src:
            display(Markdown(f"**{lab}**\n```python\n{src}\n```"))
    r = LO[LO.case_id == case_id]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
    if len(r):
        g = E[E.run_id.isin(r.sort_values("run_id").tail(1).run_id)].sort_values("actual_t")
        g = g.dropna(subset=["actual_t"])          # stationary runs have actual_t = NULL
        if len(g):
            axs[0].plot(g.actual_t, g.error.clip(lower=1e-16), "o-")
            axs[0].axhline(g.threshold.iloc[0], color="r", ls="--"); axs[0].set_yscale("log")
        else:
            axs[0].text(.5, .5, "stationary: single test", ha="center", va="center",
                        transform=axs[0].transAxes)
        tt = r.groupby("fft_workers").time_solve_s.median()
        axs[1].plot(tt.index, tt.values, "o-", color="C0", label="median, latest ok run")
        rr = (RR[(RR.is_warmup == 0) & (RR.completed == 1) & RR.run_id.isin(r.run_id)]
              .merge(r[["run_id", "fft_workers"]], on="run_id"))
        for w, gg in rr.groupby("fft_workers"):    # individual repeats as a grey cloud
            axs[1].scatter([w] * len(gg), gg.t_solve_s, s=8, color="gray", alpha=.6)
        axs[1].set_xlabel("fft_workers"); axs[1].set_ylabel("solve time (s)")
        axs[1].legend(fontsize=7)
    axs[0].set_title("error vs t"); axs[1].set_title("solve time vs workers")
    plt.tight_layout(); plt.show()
    if len(r):
        eh = blobs(int(r.run_id.max())).get("energy_history")
        if eh is not None and eh.size > 1:
            plt.figure(); plt.plot((eh - eh[0]) / abs(eh[0]))
            plt.title(f"relative energy drift (run {int(r.run_id.max())})"); plt.show()
        w = W[W.run_id.isin(r.run_id)]
        if len(w):
            display(w[["category", "message", "count"]].head(5))
    if c.notes:
        display(Markdown(f"**note:** {c.notes}"))

def fields(run_id):
    '''u_num / u_exact arrays, only present if runs used --store-fields final|all.'''
    return {k: v for k, v in blobs(run_id).items() if k.startswith(("u_num", "u_exact"))}

card("p1d-01")
""")

code("""
dash = (LAST.merge(C[["case_id", "exact_kind"]], on="case_id", how="left")
            .merge(marg, left_on="run_id", right_index=True, how="left"))
S.pivot_table(index="case_id", columns="fft_workers", values="time_solve_s").to_csv("pde_db_times_by_workers.csv")
print("wrote pde_db_summary.csv, pde_db_times_by_workers.csv")
""")

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--db", default="pde.sqlite")
ap.add_argument("--out", default="pde_db_report.ipynb")
a = ap.parse_args()
cells[1].source = cells[1].source.replace('"pde.sqlite"', repr(a.db), 1)
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
Path(a.out).write_text(nbf.writes(nb), encoding="utf-8")
print(f"wrote {a.out} — {len(cells)} cells (db={a.db})")