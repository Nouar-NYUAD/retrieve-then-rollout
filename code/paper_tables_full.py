"""Tables and in-text numbers for the full study (D4RL, Lorenz-96, CAMELS, noise, ablations, planning,
diagnostics). Called from paper_assets.main(). Same loading / grouping / per-seed rules as
analyze_full.py (complete task files only, GP rows only from CODE_VERSION 2026-11-14g, every
configuration its own setting, seed = unit of analysis, geometric mean over the starts of a seed).
Result files: results/sync/<jubail folder>/*.json (copy of the Jubail results)."""
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

import analyze_full as A

PAPER = Path(__file__).resolve().parent.parent / "paper"
SYNC = "results/sync"


def fmt(x):
    if not np.isfinite(x):
        return "--"
    if x >= 100:
        return f"{x:.0f}"
    if x >= 0.01:
        return f"{x:.3f}".rstrip("0").rstrip(".") if x < 1 else f"{x:.2f}"
    m, e = f"{x:.1e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


def fmt_nse(x):
    return f"{x:.2f}" if x > -10 + 1e-9 else r"$<\!-10$"


def full_rows():
    files = [f for f in sorted(glob.glob(f"{SYNC}/*/*.json"))
             if "diag_" not in Path(f).name and not Path(f).name.startswith("plan")
             and Path(f).parent.name not in ("cheetah", "demo", "scaling")]   # own formats (counterfactual, control, cost)
    return A.load(files)


def med(T, setting_, arm, seeds=None):
    d = T.get((setting_, arm), {})
    v = [x for s, x in d.items() if (seeds is None or s in seeds) and np.isfinite(x)]
    return (float(np.median(v)) if v else np.nan), len(v)


def table(path, cols, arms, T_by_col, fmt_by_col, best_by_col, seeds=None):
    """cols: [(setting key, header)], arms: [(label, arm tuple, ours?)]; best value per column underlined"""
    vals = {(c, l): med(T_by_col[c], c, arm, seeds) for c, _ in cols for l, arm, _ in arms}
    best = {}
    for c, _ in cols:
        xs = [(vals[(c, l)][0], l) for l, _, _ in arms if np.isfinite(vals[(c, l)][0])]
        if xs:
            best[c] = (max if best_by_col[c] == "max" else min)(xs)[1]
    L = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
         "Model and context & " + " & ".join(h for _, h in cols) + r" \\", r"\midrule"]
    for label, arm, ours in arms:
        cells = []
        for c, _ in cols:
            x = vals[(c, label)][0]
            t = fmt_by_col[c](x) if np.isfinite(x) else "--"
            cells.append(r"\underline{" + t + "}" if best.get(c) == label else t)
        L.append(((r"\textbf{" + label + " (ours)}") if ours else label) + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "tables").mkdir(parents=True, exist_ok=True)
    (PAPER / "tables" / path).write_text("\n".join(L) + "\n")
    return vals


def paired_macros(macros, key, T, setting_, a, b):
    """how many times A beats B (median paired error ratio B/A), seeds A better, Wilcoxon p"""
    Ad, Bd = T.get((setting_, a), {}), T.get((setting_, b), {})
    ks = sorted(set(Ad) & set(Bd))
    x = np.array([Ad[k] for k in ks]); y = np.array([Bd[k] for k in ks])
    p = wilcoxon(x, y).pvalue if len(ks) >= 5 and np.any(x != y) else np.nan
    r = np.median(y / x)
    macros[key + "Factor"] = f"{r:.1f}" if r >= 1 else f"{r:.2f}"
    macros[key + "Better"] = f"{int((x < y).sum())}/{len(ks)}"
    macros[key + "P"] = f"{p:.3f}" if np.isfinite(p) else "--"


RET, REC, FULL = ("retrieval", 100, "tabpfn"), ("recent", 100, "tabpfn"), ("full", 0, "tabpfn")


def robots(macros, rows):
    T = A.per_seed(rows, "100"); N = A.per_seed(rows, "nse")
    cam = "camels+lags7+feat=ext+nt4"
    cols = [("hopper+nt10", "Hopper"), ("halfcheetah+nt10", "HalfCheetah"), ("walker2d+nt10", "Walker2d"),
            ("lorenz96(d=10)+nt4", "L96 $d{=}10$"), ("lorenz96(d=20)+nt4", "$d{=}20$"),
            ("lorenz96(d=40)+nt4", "$d{=}40$"), (cam, "CAMELS NSE")]
    arms = [("Local GP (GP + retrieval)", ("retrieval", 100, "gp"), False),
            ("TabPFN + retrieval", RET, True),
            ("DynaPFN + retrieval", ("retrieval", 100, "dynapfn"), True),
            ("Local linear (retrieval)", ("retrieval", 100, "locallin"), False),
            ("TabPFN, full pool", FULL, False),
            ("Trained MLP", ("trained", 0, "mlp"), False),
            ("DynaPFN, recent window", ("recent", 100, "dynapfn"), False),
            ("TabPFN, recent window", REC, False),
            ("GP, recent window", ("recent", 100, "gp"), False)]
    table("robots.tex", cols, arms, {c: (N if c == cam else T) for c, _ in cols},
          {c: (fmt_nse if c == cam else fmt) for c, _ in cols}, {c: ("max" if c == cam else "min") for c, _ in cols})
    macros["NSeedsDFourRL"] = str(min(med(T, c, RET)[1] for c, _ in cols[:3]))
    macros["NSeedsLNinetySix"] = str(min(med(T, c, RET)[1] for c, _ in cols[3:6]))
    macros["NBasins"] = str(med(N, cam, RET)[1])
    for key, c in [("Hopper", "hopper+nt10"), ("Walker", "walker2d+nt10"), ("Cheetah", "halfcheetah+nt10")]:
        paired_macros(macros, f"RetVsRecent{key}", T, c, RET, REC)
        paired_macros(macros, f"DynaRetVsRecent{key}", T, c, ("retrieval", 100, "dynapfn"), ("recent", 100, "dynapfn"))
        paired_macros(macros, f"DynaRetVsTabRet{key}", T, c, ("retrieval", 100, "dynapfn"), RET)
    for lab, arm in [("DynaRet", ("retrieval", 100, "dynapfn")), ("TabRet", RET), ("TabFull", FULL),
                     ("TabRecent", REC), ("DynaRecent", ("recent", 100, "dynapfn")), ("GPRet", ("retrieval", 100, "gp")),
                     ("MLP", ("trained", 0, "mlp"))]:
        macros["CamelsNSE" + lab] = f"{med(N, cam, arm)[0]:.2f}"
    for d, w in ((10, "Ten"), (20, "Twenty"), (40, "Forty")):
        c = f"lorenz96(d={d})+nt4"
        macros[f"LNSix{w}GPRet"] = fmt(med(T, c, ("retrieval", 100, "gp"))[0])
        macros[f"LNSix{w}TabRet"] = fmt(med(T, c, RET)[0])
        macros[f"LNSix{w}TabFull"] = fmt(med(T, c, FULL)[0])
        macros[f"LNSix{w}TabRecent"] = fmt(med(T, c, REC)[0])
        macros[f"LNSix{w}TabRandom"] = fmt(med(T, c, ("random", 100, "tabpfn"))[0])
        macros[f"LNSix{w}GPRandom"] = fmt(med(T, c, ("random", 100, "gp"))[0])


def noise(macros, rows):
    T = A.per_seed(rows, "300"); seeds = set(range(5))    # lagged runs use seeds 0-4: like with like
    cols = [("lorenz+noise0.05+nt4", "Lorenz, 1 lag"), ("lorenz+noise0.05+lags4+nt4", "4 lags"),
            ("pendulum+noise0.05+nt4", "Pendulum, 1 lag"), ("pendulum+noise0.05+lags4+nt4", "4 lags")]
    arms = [("Local GP (GP + retrieval)", ("retrieval", 500, "gp"), False),
            ("TabPFN + retrieval", ("retrieval", 500, "tabpfn"), True),
            ("DynaPFN + retrieval", ("retrieval", 500, "dynapfn"), True),
            ("Local linear (retrieval)", ("retrieval", 500, "locallin"), False),
            ("TabPFN, full pool", FULL, False),
            ("DynaPFN, full pool", ("full", 0, "dynapfn"), False),
            ("Trained MLP", ("trained", 0, "mlp"), False),
            ("GP, recent window", ("recent", 500, "gp"), False)]
    table("noise.tex", cols, arms, {c: T for c, _ in cols}, {c: fmt for c, _ in cols},
          {c: "min" for c, _ in cols}, seeds)
    for key, sys_ in [("Lorenz", "lorenz"), ("Pendulum", "pendulum")]:
        r = []
        for _, arm, _ in arms:
            a1 = T.get((f"{sys_}+noise0.05+nt4", arm), {}); a4 = T.get((f"{sys_}+noise0.05+lags4+nt4", arm), {})
            r += [a1[k] / a4[k] for k in sorted(set(a1) & set(a4) & seeds)]
        macros[f"LagGain{key}"] = f"{np.median(r):.1f}"
        macros[f"LagGain{key}Frac"] = f"{int(np.sum(np.array(r) > 1))}/{len(r)}"
    macros["NSeedsNoise"] = str(len(seeds))


def ablations(macros, rows):
    T3, T1 = A.per_seed(rows, "300"), A.per_seed(rows, "100"); seeds = set(range(5))
    d4rl = ("hopper", "halfcheetah", "walker2d")
    g = lambda c, arm: med(T1 if c.startswith(d4rl) else T3, c, arm, seeds)[0]
    L = [r"\begin{tabular}{llccc}", r"\toprule",
         r"Ablation & Setting & \textbf{TabPFN + retrieval (ours)} & Recent window & Full pool \\", r"\midrule"]
    for sys_, name in [("lorenz", "Lorenz-63"), ("pendulum", "Pendulum")]:
        for nt in (1, 4, 16):
            L.append(rf"Pool size, {name} & {nt} training traj. & " + " & ".join(fmt(g(f"{sys_}+nt{nt}", a)) for a in (RET, REC, FULL)) + r" \\")
    L.append(r"\midrule")
    for nz in ("0.01", "0.05", "0.1"):
        L.append(rf"Noise, Lorenz-63 & {float(nz) * 100:.0f}\% & " + " & ".join(fmt(g(f"lorenz+noise{nz}+nt4", a)) for a in (RET, REC, FULL)) + r" \\")
    L.append(r"\midrule")
    for name, base in [("Hopper", "hopper+nt10"), ("Pendulum", "pendulum+nt4")]:
        for met, lab in [("std", "standardised (default)"), ("raw", "raw inputs"), ("state_only", "state only")]:
            c = base if met == "std" else base.replace("+nt", f"+metric={met}+nt")
            v = g(c, RET)
            if np.isfinite(v):
                L.append(rf"Distance, {name} & {lab} & {fmt(v)} & & \\")
    L.append(r"\midrule")
    for name, c in [("Lorenz-63", "lorenz+nt4"), ("Pendulum", "pendulum+nt4"), ("Hopper", "hopper+nt10"),
                    ("HalfCheetah", "halfcheetah+nt10")]:
        for ver in ("v2", "v3.5-fast", "v3.5"):
            tag = "tabpfn" if ver == "v3.5" else f"tabpfn[{ver}]"
            vs = [g(c, (st, H, tag)) for st, H, _ in (RET, REC)]
            if np.all(np.isfinite(vs)):
                L.append(rf"Version, {name} & TabPFN {ver} & {fmt(vs[0])} & {fmt(vs[1])} & \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "tables" / "ablation.tex").write_text("\n".join(L) + "\n")
    for sys_, k in [("lorenz", "Lorenz"), ("pendulum", "Pendulum")]:
        macros[f"PoolOne{k}"] = fmt(g(f"{sys_}+nt1", RET)); macros[f"PoolSixteen{k}"] = fmt(g(f"{sys_}+nt16", RET))
    macros["DistHopperStd"] = fmt(g("hopper+nt10", RET)); macros["DistHopperRaw"] = fmt(g("hopper+metric=raw+nt10", RET))


def planning(macros):
    acc = defaultdict(list)
    for f in glob.glob(f"{SYNC}/plan/plan*.json"):
        for r in json.load(open(f))["results"]:
            acc[(r["task"], r["model"], r["strategy"])].append((r["seed"], r["frac_upright"]))
    arms = [("Oracle (true dynamics)", "oracle", "-"),
            (r"\textbf{TabPFN, trajectory retrieval (ours)}", "tabpfn", "trajretrieval"),
            (r"\textbf{TabPFN, retrieval (ours)}", "tabpfn", "retrieval"),
            (r"\textbf{TabPFN, hybrid (ours)}", "tabpfn", "hybrid"), ("TabPFN, recent window", "tabpfn", "recent"),
            ("TabPFN, random", "tabpfn", "random"),
            ("Local linear, trajectory retrieval", "locallin", "trajretrieval"),
            ("Local linear, retrieval", "locallin", "retrieval"), ("Local linear, recent window", "locallin", "recent"),
            ("Local linear, random", "locallin", "random")]
    L = [r"\begin{tabular}{lcc}", r"\toprule", r"Planner model and context & Cart-pole & Acrobot \\", r"\midrule"]
    for lab, m, st in arms:
        cells = []
        for task in ("cartpole", "acrobot"):
            v = acc.get((task, m, st), [])
            cells.append(f"{np.mean([u for _, u in v]):.2f} ({len({s for s, _ in v})})" if v else "--")
        L.append(lab + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "tables" / "planning.tex").write_text("\n".join(L) + "\n")
    for task, k in (("cartpole", "Cartpole"), ("acrobot", "Acrobot")):
        for lab, m, st in [("Oracle", "oracle", "-"), ("TabRet", "tabpfn", "retrieval"), ("TabTraj", "tabpfn", "trajretrieval"),
                           ("TabRecent", "tabpfn", "recent"), ("LinTraj", "locallin", "trajretrieval"),
                           ("LinRecent", "locallin", "recent")]:
            v = acc.get((task, m, st), [])
            macros[f"Plan{k}{lab}"] = f"{np.mean([u for _, u in v]):.2f}" if v else "--"


def diagnostics(macros):
    d1 = json.load(open(f"{SYNC}/gp/diag_d1.json")); d2 = json.load(open(f"{SYNC}/gp/diag_d2.json"))
    d3 = json.load(open(f"{SYNC}/gp/diag_d3.json"))
    g1 = {(r["func"], r["n"]): r for r in d1}
    for func, k in (("linear", "Linear"), ("lorenz_quadratic", "Quad"), ("sine", "Sine")):
        macros[f"DOneTab{k}"] = fmt(g1[(func, 2000)]["tabpfn"]); macros[f"DOneGP{k}"] = fmt(g1[(func, 2000)]["gp"])
    g2 = {(r["dim"], r["H"], r["strategy"]): r for r in d2}
    for H, w in ((50, "Small"), (2000, "Large")):
        r_, q_ = g2[(10, H, "retrieval")], g2[(10, H, "random")]
        macros[f"DTwoSpread{w}"] = f"{r_['context_spread']:.2f}"
        for m, mk in (("tabpfn", "Tab"), ("gp", "GP")):
            macros[f"DTwo{mk}Ret{w}"] = fmt(r_[m]); macros[f"DTwo{mk}Rand{w}"] = fmt(q_[m])
    for r in d3:
        k = "Hopper" if r["system"] == "hopper" else "Walker"
        if r["diag"] == "D3" and r["bin"].startswith("most abrupt"):
            macros[f"DThreeAbrupt{k}Tab"] = fmt(r["tabpfn"]); macros[f"DThreeAbrupt{k}GP"] = fmt(r["gp"])
        if r["diag"] == "D3" and r["bin"].startswith("smooth"):
            macros[f"DThreeSmooth{k}Tab"] = fmt(r["tabpfn"]); macros[f"DThreeSmooth{k}GP"] = fmt(r["gp"])
        if r["diag"] == "D4":
            macros[f"DFourNoiseBound{k}"] = f"{100 * r['frac_dims_noise_at_bound']:.0f}"
            macros[f"DFourLengthBound{k}"] = f"{100 * r['frac_length_scales_at_bound']:.0f}"
    L = [r"\begin{tabular}{lcccc}", r"\toprule",
         r"Function & \multicolumn{2}{c}{$n=50$} & \multicolumn{2}{c}{$n=2000$} \\",
         r" & TabPFN & GP & TabPFN & GP \\", r"\midrule"]
    for func, lab in [("linear", "Linear"), ("sine", "Sine"), ("lorenz_quadratic", "Lorenz quadratic")]:
        L.append(lab + " & " + " & ".join(fmt(g1[(func, n)][m]) for n in (50, 2000) for m in ("tabpfn", "gp")) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "tables" / "precision.tex").write_text("\n".join(L) + "\n")


def all_tables(macros):
    rows = full_rows()
    robots(macros, rows); noise(macros, rows); ablations(macros, rows); planning(macros); diagnostics(macros)
