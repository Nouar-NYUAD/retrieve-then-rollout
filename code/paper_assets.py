"""Generate every table and in-text number of the paper from the result files (nothing by hand).

Writes paper/tables/*.tex and paper/numbers.tex (\newcommand macros). Rules (see analyze_full.py):
only complete task files (all rollout starts present), GP rows only from the fixed GP code
(CODE_VERSION 2026-11-14g, GP-only reruns), unit of analysis = seed (geometric mean over starts),
paired Wilcoxon signed-rank tests over seeds.
Usage: python paper_assets.py   (run from pilot/, results in pilot/results/)
"""
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

PAPER = Path(__file__).resolve().parent.parent / "paper"
VALID_GP = {"2026-11-14g", "2026-11-15h"}   # h = g + new system pendulum_box (GP code unchanged)


def load(patterns, n_starts):
    rows = []
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            R = json.load(open(f))["results"]
            if not R or len({r["t0"] for r in R}) < n_starts:
                continue                                         # incomplete task file: never used
            rows += [r for r in R if not (r["model"] == "gp" and r.get("code_version") not in VALID_GP)]
    return rows


def setting(r):
    return r["system"] + ("_noisy" if r.get("obs_noise") else "")


def per_seed(rows, h):
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        acc[(setting(r), r["strategy"], r["H"], r["model"])][r["seed"]].append(r["rollout_nrmse"][h])
    out = {}
    for k, d in acc.items():
        out[k] = {s: float(np.exp(np.mean(np.log(np.maximum(np.where(np.isfinite(v), v, 1e3), 1e-12)))))
                  for s, v in d.items()}
    return out


def fmt(x):
    if not np.isfinite(x):
        return "--"
    if x >= 100:
        return f"{x:.0f}"
    if x >= 0.01:
        return f"{x:.3f}" if x < 1 else (f"{x:.2f}" if x < 10 else f"{x:.1f}")   # 0.240, 2.45, 24.5
    m, e = f"{x:.1e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


def paired(T, s, a, b):
    A, B = T.get((s,) + a, {}), T.get((s,) + b, {})
    ks = sorted(set(A) & set(B))
    x = np.array([A[k] for k in ks]); y = np.array([B[k] for k in ks])
    p = wilcoxon(x, y).pvalue if len(ks) >= 3 and np.any(x != y) else np.nan
    # factor = median over seeds of err(B) / err(A): how many times A is more accurate than B
    return float(np.median(y / x)) if ks else np.nan, int((x < y).sum()), len(ks), float(p)


ARMS = [  # (label, strategy, H, model, ours?)
    ("Local GP (GP + retrieval)", "retrieval", 100, "gp", False),
    ("TabPFN + retrieval", "retrieval", 100, "tabpfn", True),
    ("DynaPFN + retrieval", "retrieval", 100, "dynapfn", True),
    ("TabPFN, full pool", "full", 0, "tabpfn", False),
    ("TabPFN, random context", "random", 100, "tabpfn", False),
    ("Trained MLP", "trained", 0, "mlp", False),
    ("DynaPFN, recent window", "recent", 100, "dynapfn", False),
    ("TabPFN, recent window", "recent", 100, "tabpfn", False),
    ("TabPFN, sliding window", "sliding", 100, "tabpfn", False),
    ("GP, recent window", "recent", 100, "gp", False),
]


def synthetic(macros):
    rows = load(["results/sync/pilot/full_lorenz_*.json", "results/sync/pilot/full_pendulum_*.json",
                 "results/sync/gp/gpg_full_lorenz_*.json", "results/sync/gp/gpg_full_pendulum_*.json",
                 "results/sync/sliding/sliding_full_lorenz_*.json", "results/sync/sliding/sliding_full_pendulum_*.json"], 3)
    T = per_seed(rows, "300")
    cols = [("lorenz", "Lorenz-63"), ("pendulum", "Pendulum"), ("lorenz_noisy", "Lorenz-63, 5\\% noise"),
            ("pendulum_noisy", "Pendulum, 5\\% noise")]
    nseeds = {c: len(T.get((c, "retrieval", 100, "tabpfn"), {})) for c, _ in cols}
    L = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
         "Model and context & " + " & ".join(n for _, n in cols) + r" \\", r"\midrule"]
    for label, st, H, m, ours in ARMS:
        if not any(k[1:] == (st, H, m) for k in T):
            continue                                   # no results for this arm (e.g. sliding not run yet)
        vals = []
        for c, _ in cols:
            v = list(T.get((c, st, H, m), {}).values())
            vals.append(fmt(np.median(v)) if v else "--")
        name = (r"\textbf{" + label + " (ours)}") if ours else label
        L.append(name + " & " + " & ".join(vals) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "tables").mkdir(parents=True, exist_ok=True)
    (PAPER / "tables" / "synthetic.tex").write_text("\n".join(L) + "\n")
    macros["NSeedsSynthetic"] = str(min(nseeds.values()))
    # key paired comparisons used in the text
    for key, c in [("Lorenz", "lorenz"), ("Pendulum", "pendulum"), ("PendulumNoisy", "pendulum_noisy"),
                   ("LorenzNoisy", "lorenz_noisy")]:
        ratio, better, n, p = paired(T, c, ("retrieval", 100, "tabpfn"), ("recent", 100, "tabpfn"))
        macros[f"RetVsRecent{key}Factor"] = f"{ratio:.0f}" if ratio >= 10 else f"{ratio:.2f}"
        macros[f"RetVsRecent{key}Better"] = f"{better}/{n}"
        macros[f"RetVsRecent{key}P"] = f"{p:.3f}"
        if T.get((c, "sliding", 100, "tabpfn")):
            ratio, better, n, p = paired(T, c, ("retrieval", 100, "tabpfn"), ("sliding", 100, "tabpfn"))
            macros[f"RetVsSliding{key}Factor"] = f"{ratio:.0f}" if ratio >= 10 else f"{ratio:.1f}"
            macros[f"RetVsSliding{key}Better"] = f"{better}/{n}"
            ratio, better, n, p = paired(T, c, ("recent", 100, "tabpfn"), ("sliding", 100, "tabpfn"))
            macros[f"RecentVsSliding{key}Factor"] = f"{ratio:.0f}" if ratio >= 10 else f"{ratio:.1f}"
            macros[f"RecentVsSliding{key}Better"] = f"{better}/{n}"
        ratio, better, n, p = paired(T, c, ("retrieval", 100, "tabpfn"), ("random", 100, "tabpfn"))
        macros[f"RetVsRandom{key}Factor"] = f"{ratio:.0f}" if ratio >= 10 else f"{ratio:.1f}"
        macros[f"RetVsRandom{key}Better"] = f"{better}/{n}"
        ratio, better, n, p = paired(T, c, ("retrieval", 100, "gp"), ("retrieval", 100, "tabpfn"))
        macros[f"GPOverTabPFN{key}Factor"] = f"{ratio:.0f}" if ratio >= 10 else f"{ratio:.2f}"
    ratio, better, n, p = paired(T, "lorenz", ("retrieval", 100, "tabpfn"), ("full", 0, "tabpfn"))
    macros["RetVsFullLorenzFactor"] = f"{ratio:.1f}"


def main():
    PAPER.mkdir(parents=True, exist_ok=True)
    macros = {}
    synthetic(macros)
    import paper_tables_full
    paper_tables_full.all_tables(macros)
    import paper_figures
    paper_figures.all_figures(macros)
    lines = ["% generated by pilot/paper_assets.py - do not edit by hand"]
    lines += [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    (PAPER / "numbers.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
