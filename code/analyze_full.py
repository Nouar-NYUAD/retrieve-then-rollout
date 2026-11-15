"""Analysis of the full study (study.py outputs).

Unit of analysis = SEED (each seed = new training data and new test trajectory / episode /
basin); the 2-3 rollout starts of a seed are averaged first (geometric mean of errors, mean NSE),
so the 10 (or 18) units are independent. Paired comparisons over seeds: median ratio of errors
(A / B), number of seeds where A is better, two-sided Wilcoxon signed-rank p-value.

Usage: python analyze_full.py <out.md> <results json files...>
"""
import json
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon


def setting(r):
    s = r["system"]
    if r.get("obs_noise"):
        s += f"+noise{r['obs_noise']}"
    if r.get("l96_dim"):
        s += f"(d={r['l96_dim']})"
    # every config that changes the experiment gets its own row (never pooled together)
    if r.get("wall") is not None:
        s += f"+wall{r['wall']:g}"
    if r.get("lags", 1) != 1:
        s += f"+lags{r['lags']}"
    if r.get("target", "delta") != "delta":
        s += f"+target={r['target']}"
    if r.get("retrieval_metric") not in (None, "std"):
        s += f"+metric={r['retrieval_metric']}"
    if r.get("camels_features") not in (None, "basic"):
        s += f"+feat={r['camels_features']}"
    if r.get("retrieve_every", 1) != 1:
        s += f"+every{r['retrieve_every']}"
    if "_n_train" in r:
        s += f"+nt{r['_n_train']}"
    return s


def arm(r):
    v = r.get("tabpfn_version", "v3.5")
    tag = f"{r['model']}" + (f"[{v}]" if r["model"] == "tabpfn" and v != "v3.5" else "")
    return (r["strategy"], r["H"], tag)


# Rows produced by code with a KNOWN BUG / weakness are excluded here, never silently used:
#   GP rows before CODE_VERSION "2026-11-14g": c fixed the kernel-scaling bug (2026-11-13 20:13);
#   g adds optimiser restarts (a single start collapsed to an all-noise fit on lagged noisy inputs).
#   ALL GP results come from the dedicated GP-only reruns (gpg_*.json, version g).
VALID_GP = {"2026-11-14g", "2026-11-15h"}   # h = g + new system pendulum_box (GP code unchanged)


def load(paths):
    rows, dropped = [], 0
    incomplete = 0
    for p in paths:
        d = json.load(open(p))
        R, a = d["results"], d.get("args", {})
        if not R or len({r["t0"] for r in R}) < a.get("n_starts", 1):
            incomplete += 1                      # task file not finished: never used
            continue
        for r in R:
            r["_n_train"] = a.get("n_train")
            if r["model"] == "gp" and r.get("code_version") not in VALID_GP:
                dropped += 1
                continue
            rows.append(r)
    print(f"loaded {len(rows)} rows; dropped {dropped} GP rows from code before the GP fix; "
          f"skipped {incomplete} incomplete files")
    return rows


def per_seed(rows, metric):
    """{(setting, arm): {seed: value}}; errors geometric-mean over starts, NSE / VPT arithmetic."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if metric == "nse":
            v = r.get("nse"); v = np.nan if v is None else max(v, -10.0)   # cap diverged rollouts
        elif metric == "vpt":
            v = r["vpt_steps"]
        elif metric == "one":
            v = r["one_step_nrmse"]
        else:
            v = r["rollout_nrmse"].get(metric, np.nan)
        acc[(setting(r), arm(r))][r["seed"]].append(v)
    out = {}
    for k, d in acc.items():
        out[k] = {}
        for s, vs in d.items():
            vs = np.array(vs, float)
            if metric in ("nse", "vpt"):
                out[k][s] = np.nanmean(vs) if np.isfinite(vs).any() else np.nan
            else:
                vs = np.where(np.isfinite(vs), vs, 1e3)                     # diverged -> large
                out[k][s] = float(np.exp(np.mean(np.log(np.maximum(vs, 1e-12)))))
    return out


def paired(tab, setting_, a, b, lower_better=True):
    A, B = tab.get((setting_, a), {}), tab.get((setting_, b), {})
    ks = sorted(set(A) & set(B))
    x = np.array([A[k] for k in ks]); y = np.array([B[k] for k in ks])
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    if len(x) < 3:
        return None
    p = wilcoxon(x, y).pvalue if np.any(x != y) else 1.0
    better = (x < y) if lower_better else (x > y)
    stat = np.median(x / y) if lower_better else np.median(x - y)
    return len(x), stat, int(better.sum()), p


def main():
    out, paths = sys.argv[1], sys.argv[2:]
    rows = load(paths)
    settings = sorted({setting(r) for r in rows})
    L = ["# Full study tables\n", f"{len(rows)} rollouts from {len(paths)} files.\n"]
    for st in settings:
        rs = [r for r in rows if setting(r) == st]
        hs = sorted({int(h) for r in rs for h in r["rollout_nrmse"]})
        hmax = str(max(hs))
        T = {m: per_seed(rs, m) for m in ["one", hmax, "nse", "vpt"]}
        arms = sorted({arm(r) for r in rs}, key=lambda a: (a[0], a[1], a[2]))
        L += [f"\n## {st}\n",
              f"| context | H | model | seeds | one-step nRMSE | rollout nRMSE @{hmax} | NSE (window) | VPT |",
              "|---|---|---|---|---|---|---|---|"]
        for a in arms:
            v = {m: np.array(list(T[m].get((st, a), {}).values()), float) for m in T}
            med = lambda z: np.nanmedian(z) if np.isfinite(z).any() else np.nan
            L.append(f"| {a[0]} | {a[1]} | {a[2]} | {len(v['one'])} | {med(v['one']):.4g} | "
                     f"{med(v[hmax]):.4g} | {med(v['nse']):.3f} | {med(v['vpt']):.0f} |")
        # key paired comparisons
        H0 = 100
        ref = ("retrieval", H0, "tabpfn")
        comps = [("recent", H0, "tabpfn"), ("recent", 500, "tabpfn"), ("random", H0, "tabpfn"),
                 ("full", 0, "tabpfn"), ("retrieval", H0, "locallin"), ("retrieval", H0, "gp"),
                 ("retrieval", H0, "parrot"), ("trained", 0, "mlp"), ("recent", H0, "dynapfn"),
                 ("retrieval", H0, "dynapfn"), ("retrieval", H0, "tabpfn[v2]"),
                 ("recent", H0, "tabpfn[v2]"), ("retrieval", H0, "tabpfn[v3.5-fast]")]
        L += [f"\nPaired over seeds, A = TabPFN v3.5 + retrieval (H={H0}):\n",
              f"| B | metric | seeds | median A/B (error) or A-B (NSE) | seeds A better | Wilcoxon p |",
              "|---|---|---|---|---|---|"]
        for b in comps:
            for m, lb in [(hmax, True), ("nse", False)]:
                r = paired(T[m], st, ref, b, lb)
                if r:
                    L.append(f"| {b[0]}/{b[1]}/{b[2]} | {'nRMSE@' + m if lb else 'NSE'} | {r[0]} | "
                             f"{r[1]:.3g} | {r[2]}/{r[0]} | {r[3]:.3g} |")
        # retrieval helps the other models too?
        L += ["\nRetrieval vs recent window for each model (A = retrieval, B = recent, H=100):\n",
              "| model | seeds | median error ratio @" + hmax + " | seeds better | p |", "|---|---|---|---|---|"]
        for mname in sorted({a[2] for a in arms}):
            r = paired(T[hmax], st, ("retrieval", H0, mname), ("recent", H0, mname))
            if r:
                L.append(f"| {mname} | {r[0]} | {r[1]:.3g} | {r[2]}/{r[0]} | {r[3]:.3g} |")
    open(out, "w", newline="\n").write("\n".join(L) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
