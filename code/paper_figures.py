"""Paper figures (vector PDF) from the result files - same loading / per-seed rules as the tables
(paper_tables_full.py, analyze_full.py). Every method keeps one colour + marker in every figure
(validated categorical palette, fixed order); in-context foundation models with retrieved context
(ours) are solid, baselines dashed. Ratios are paired per seed (median of per-seed ratios).
Main text: regime (where retrieval / the foundation model wins), robots, noise, planning,
counterfactual, diagnostics. Appendix: synthetic curves, precision floor.
Usage: python paper_figures.py   (from pilot/; writes paper/figures/*.pdf, *.png and macros)"""
import glob
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import analyze_full as A
import paper_tables_full as PT

FIG = PT.PAPER / "figures"
INK, MUTED, GRID, SHADE = "#1f1f1e", "#6b6a63", "#e4e3de", "#f3f2ee"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.2,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "lines.linewidth": 1.7, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                     "#e87ba4", "#008300", "#6250d6")
STY = {  # model / context -> (label, colour, marker, ours)
    "tab_ret": ("TabPFN + retrieval (ours)", BLUE, "o", True),
    "dyna_ret": ("DynaPFN + retrieval (ours)", MAGENTA, "v", True),
    "tab_full": ("TabPFN, full pool", AQUA, "^", False),
    "tab_recent": ("TabPFN, recent window", ORANGE, "s", False),
    "gp_ret": ("Local GP (GP + retrieval)", YELLOW, "D", False),
    "mlp": ("Trained MLP", GREEN, "P", False),
    "lin_ret": ("Local linear (retrieval)", VIOLET, "X", False),
}
RIGHT, LEFT = "→", "←"


def arm(key, H=100):
    return {"tab_ret": ("retrieval", H, "tabpfn"), "dyna_ret": ("retrieval", H, "dynapfn"),
            "tab_full": ("full", 0, "tabpfn"), "tab_recent": ("recent", H, "tabpfn"),
            "gp_ret": ("retrieval", H, "gp"), "mlp": ("trained", 0, "mlp"),
            "lin_ret": ("retrieval", H, "locallin")}[key]


def style(key, line=True):
    lab, c, m, ours = STY[key]
    s = dict(color=c, marker=m, label=lab, ms=4.5, mec=c, mfc=c if ours else "white", mew=1.1)
    if line:
        s.update(ls="-" if ours else "--", lw=2.2 if ours else 1.5)
    return s


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIG / f"{name}.png", dpi=160, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def paired_ratio(T, st, a, b, seeds=None):
    """median over seeds of err(b) / err(a): > 1 means a is better"""
    A_, B_ = T.get((st, a), {}), T.get((st, b), {})
    ks = sorted(k for k in set(A_) & set(B_) if seeds is None or k in seeds)
    if not ks:
        return np.nan, 0, 0
    r = np.array([B_[k] / A_[k] for k in ks])
    return float(np.median(r)), int((r > 1).sum()), len(ks)


def nice_log_y(ax):
    """log y axis with plain-number ticks at 1-2-5 steps (no 2x10^-2 style labels)"""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator
    lo, hi = ax.get_ylim()
    ticks = [m * 10.0 ** e for e in range(-8, 5) for m in (1, 2, 5) if lo <= m * 10.0 ** e <= hi]
    if len(ticks) > 7:
        ticks = [t for t in ticks if f"{t:g}".lstrip("0.").startswith("1")] or ticks
    ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax.yaxis.set_minor_formatter(NullFormatter())


def curves(ax, Th, hs, st, keys, H=100, seeds=None):
    for k in keys:
        y = [PT.med(Th[h], st, arm(k, H), seeds)[0] for h in hs]
        if np.all(np.isnan(y)):
            continue
        ax.plot([int(h) for h in hs], y, **style(k))
    ax.set_xscale("log"); ax.set_yscale("log"); nice_log_y(ax)


# ----------------------------------------------------------------------------- figure 1: regime map
MACRO = {"Hopper": "Hopper", "Walker2d": "Walker", "HalfCheetah": "Cheetah", "CAMELS river flow": "Camels",
         "Lorenz-63, 5% noise": "LorenzNoisy", "Pendulum, 5% noise": "PendulumNoisy", "Lorenz-63": "Lorenz",
         "Pendulum": "Pendulum", "Lorenz-96, d=10": "LNSixTen", "Lorenz-96, d=20": "LNSixTwenty",
         "Lorenz-96, d=40": "LNSixForty"}
REGIME_ROWS = [  # (label, horizon, setting, H, seeds)
    ("Hopper", "100", "hopper+nt10", 100, None),
    ("Walker2d", "100", "walker2d+nt10", 100, None),
    ("HalfCheetah", "100", "halfcheetah+nt10", 100, None),
    ("CAMELS river flow", "window", "camels+lags7+feat=ext+nt4", 100, None),
    ("Lorenz-63, 5% noise", "300", "lorenz+noise0.05+lags4+nt4", 500, set(range(5))),
    ("Pendulum, 5% noise", "300", "pendulum+noise0.05+lags4+nt4", 500, set(range(5))),
    ("Lorenz-63", "300", "lorenz+nt4", 100, None),
    ("Pendulum", "300", "pendulum+nt4", 100, None),
    ("Lorenz-96, d=10", "100", "lorenz96(d=10)+nt4", 100, None),
    ("Lorenz-96, d=20", "100", "lorenz96(d=20)+nt4", 100, None),
    ("Lorenz-96, d=40", "100", "lorenz96(d=40)+nt4", 100, None),
]


def fig_regime(rows, macros):
    T = {h: A.per_seed(rows, h) for h in ("100", "300")}
    # CAMELS: error over the whole simulated period, sqrt(1 - NSE) = RMSE / std of the observed flow
    N = A.per_seed(rows, "nse")
    T["window"] = {k: {s_: float(np.sqrt(max(1 - v, 1e-12))) for s_, v in d.items() if np.isfinite(v)}
                   for k, d in N.items()}
    n = len(REGIME_ROWS)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.5, 2.95), sharey=True)
    w = 0.36
    for i, (lab, h, st, H, sd) in enumerate(REGIME_ROWS):
        key = "Regime" + MACRO[lab]
        # (a) TabPFN with retrieval vs the sliding window of Lee et al. and vs the fixed recent window
        # (same model, same H = 100); noisy rows use one observation as input (the lagged runs have
        # no recent-window arm)
        st_a = st.replace("+lags4", "") if "noise" in st else st
        for j, (ctx, c) in enumerate([("sliding", VIOLET), ("recent", ORANGE)]):
            r, b, nn = paired_ratio(T[h], st_a, ("retrieval", 100, "tabpfn"), (ctx, 100, "tabpfn"))
            if np.isfinite(r):
                y = i + (j - 0.5) * w
                ax.barh(y, np.log10(r), height=w * 0.92, color=c)
                ax.text(np.log10(r) + (0.06 if r > 1 else -0.06), y, f"{b}/{nn}", fontsize=5.6, color=MUTED,
                        va="center", ha="left" if r > 1 else "right")
                tag = "Recent" if ctx == "recent" else "Sliding"
                macros[f"{key}Ret{tag}Tab"] = f"{r:.1f}" if r < 100 else f"{r:.0f}"
                macros[f"{key}Ret{tag}TabBetter"] = f"{b}/{nn}"
        # (b) local GP / ours, both with retrieved context
        for j, (mdl, c) in enumerate([("tabpfn", BLUE), ("dynapfn", MAGENTA)]):
            r, b, nn = paired_ratio(T[h], st, ("retrieval", H, mdl), ("retrieval", H, "gp"), sd)
            if np.isfinite(r):
                y = i + (j - 0.5) * w
                bx.barh(y, np.log10(r), height=w * 0.92, color=c)
                bx.text(np.log10(r) + (0.05 if r > 1 else -0.05), y, f"{b}/{nn}", fontsize=5.6, color=MUTED,
                        va="center", ha="left" if r > 1 else "right")
                tag = "Tab" if j == 0 else "Dyna"
                macros[f"{key}GPOver{tag}"] = f"{r:.1f}"
                macros[f"{key}GPOver{tag}Better"] = f"{b}/{nn}"
                macros[f"{key}GPBetterThan{tag}"] = f"{1 / r:.0f}" if r < 0.1 else f"{1 / r:.1f}"
    for a_, xl, ttl, ticks in [(ax, "error ratio: window / retrieval", "(a) TabPFN: retrieval vs recent windows",
                                [0, 1, 2, 3, 4]),
                               (bx, "error ratio: local GP / ours", "(b) Foundation model vs local GP",
                                [-3, -2, -1, 0, 1])]:
        a_.axvline(0, color=INK, lw=0.9)
        a_.set_title(ttl, loc="left"); a_.set_xlabel(xl); a_.grid(axis="y", visible=False)
        a_.set_xticks(ticks)
        a_.set_xticklabels([(f"{10.0 ** t:g}" if t < 4 else "$10^4$") if t >= 0 else f"1/{10 ** -t}" for t in ticks])
        for g0, g1 in [(-0.5, 3.5), (5.5, n - 0.5)]:
            a_.axhspan(g0, g1, color=SHADE, zorder=0, lw=0)
    ax.set_xlim(-0.6, 5.0); bx.set_xlim(-3.6, 1.5)
    ax.text(5.0, -0.95, "retrieval better " + RIGHT, ha="right", fontsize=6.5)
    bx.text(1.5, -0.95, "ours better " + RIGHT, ha="right", fontsize=6.5)
    bx.text(-3.6, -0.95, LEFT + " local GP better", ha="left", fontsize=6.5)
    ax.set_yticks(range(n)); ax.set_yticklabels([r[0] for r in REGIME_ROWS]); ax.invert_yaxis()
    for (y0, y1), g in [((-0.4, 3.4), "real systems"), ((3.6, 5.4), "noisy"), ((5.6, n - 0.6), "noise-free ODEs")]:
        ax.annotate("", xy=(-0.62, y0), xytext=(-0.62, y1), xycoords=("axes fraction", "data"),
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9), annotation_clip=False)
        ax.text(-0.66, (y0 + y1) / 2, g, transform=ax.get_yaxis_transform(), rotation=90, fontsize=6.5,
                color=MUTED, ha="center", va="center", style="italic")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    ax.legend([Line2D([], [], color=VIOLET, lw=6), Line2D([], [], color=ORANGE, lw=6)],
              ["vs sliding window (Lee et al.)", "vs fixed window"], loc="upper center", ncol=2,
              frameon=False, bbox_to_anchor=(0.45, -0.17), handlelength=1.2, columnspacing=0.8)
    bx.legend([Line2D([], [], color=BLUE, lw=6), Line2D([], [], color=MAGENTA, lw=6)],
              ["TabPFN", "DynaPFN"], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.17))
    save(fig, "regime")


# ----------------------------------------------------------------------------- figure 2: robots + CAMELS
def fig_robots(rows):
    hs = ["1", "10", "25", "50", "100"]
    Th = {h: A.per_seed(rows, h) for h in hs}
    N = A.per_seed(rows, "nse")
    keys = ["dyna_ret", "tab_ret", "tab_full", "gp_ret", "tab_recent", "mlp"]
    fig, axs = plt.subplots(1, 4, figsize=(6.5, 2.35), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.15]))
    for ax, (st, name) in zip(axs[:3], [("hopper+nt10", "Hopper"), ("walker2d+nt10", "Walker2d"),
                                         ("halfcheetah+nt10", "HalfCheetah")]):
        curves(ax, Th, hs, st, keys)
        ax.set_title(name, loc="left"); ax.set_xlabel("rollout step")
        ax.set_xticks([1, 10, 100]); ax.set_xticklabels(["1", "10", "100"])
    axs[0].set_ylabel("normalised RMSE")
    ax = axs[3]
    cam = "camels+lags7+feat=ext+nt4"
    for k in keys:
        v = np.sort([x for x in N.get((cam, arm(k)), {}).values() if np.isfinite(x)])
        if len(v):
            s = style(k); s.pop("label"); s.update(marker=None)
            ax.step(np.maximum(v, -1.5), np.arange(1, len(v) + 1) / len(v), where="post", **s)
    ax.set_xlim(-1.5, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("NSE (right is better)"); ax.set_ylabel("fraction of basins")
    ax.set_title("CAMELS, 18 basins", loc="left")
    h, l = axs[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=0.6)
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    save(fig, "robots")


# ----------------------------------------------------------------------------- figure 3: noise
def fig_noise(rows):
    hs = ["1", "10", "50", "100", "200", "300"]
    Th = {h: A.per_seed(rows, h) for h in hs}
    keys = ["dyna_ret", "tab_ret", "tab_full", "gp_ret", "mlp"]
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.3))
    for ax, (st, name) in zip(axs, [("lorenz+noise0.05+lags4+nt4", "Lorenz-63, 5% noise, 4 lagged inputs"),
                                    ("pendulum+noise0.05+lags4+nt4", "Pendulum, 5% noise, 4 lagged inputs")]):
        curves(ax, Th, hs, st, keys, H=500, seeds=set(range(5)))
        ax.set_title(name, loc="left"); ax.set_xlabel("rollout step")
        ax.set_xticks([1, 10, 100, 300]); ax.set_xticklabels(["1", "10", "100", "300"])
    axs[0].set_ylabel("normalised RMSE")
    h, l = axs[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    save(fig, "noise")


# ----------------------------------------------------------------------------- figure 4: planning
def fig_planning():
    acc = defaultdict(list)
    for f in glob.glob(f"{PT.SYNC}/plan/plan*.json"):
        for r in json.load(open(f))["results"]:
            acc[(r["task"], r["model"], r["strategy"])].append(r["frac_upright"])
    bars = [("oracle", "-", "Planner with\ntrue dynamics", "#b9b8b0", None),
            ("tabpfn", "trajretrieval", "TabPFN, traj.\nretrieval (ours)", BLUE, None),
            ("tabpfn", "retrieval", "TabPFN,\nretrieval (ours)", BLUE, None),
            ("tabpfn", "random", "TabPFN,\nrandom context", BLUE, "///"),
            ("tabpfn", "recent", "TabPFN,\nrecent window", ORANGE, "///"),
            ("locallin", "trajretrieval", "Local linear,\ntraj. retrieval", VIOLET, None),
            ("locallin", "recent", "Local linear,\nrecent window", VIOLET, "///")]
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.35), sharey=True)
    for ax, (task, name) in zip(axs, [("cartpole", "Cart-pole swing-up"), ("acrobot", "Acrobot swing-up")]):
        for k, (m, st, lab, c, hatch) in enumerate(bars):
            v = acc.get((task, m, st), [])
            if not v:
                ax.text(k, 0.02, "running", rotation=90, fontsize=6.3, color=MUTED, ha="center", va="bottom")
                continue
            ax.bar(k, np.mean(v), width=0.72, color=c if hatch is None else "white", edgecolor=c, hatch=hatch, lw=1.2)
            ax.scatter(k + np.linspace(-0.15, 0.15, len(v)), v, s=7, color=INK, zorder=3, lw=0)
        ax.set_xticks(range(len(bars)))
        ax.set_xticklabels([b[2].replace("\n", " ") for b in bars], fontsize=6.3, rotation=32, ha="right",
                           rotation_mode="anchor")
        ax.set_title(name, loc="left"); ax.set_ylim(0, 1); ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("fraction of time upright")
    fig.tight_layout(w_pad=0.6)
    save(fig, "planning")


# ----------------------------------------------------------------------------- figure 5: counterfactual
def cf_table():
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for f in glob.glob(f"{PT.SYNC}/cheetah/cf_pendulum_*.json"):
        d = json.load(open(f)); a = d["args"]
        st = f"g{a['gain']:g}_s{a['sigma']:g}" + ("_ex" if a["explore_frac"] > 0 else "")
        for r in d["results"]:
            for lev in ("0.3", "uniform"):
                acc[(st, r["strategy"], r["model"])][lev][r["seed"]].append(r[lev]["effect_rel"])
    out = {}
    for k, levs in acc.items():
        out[k] = {lev: float(np.median([np.median(v) for v in seeds.values()])) for lev, seeds in levs.items()}
        out[k]["n_seeds"] = len(levs["uniform"])
    return out


def fig_counterfactual(macros):
    C = cf_table()
    xs = [("g1_s0", "0\n(policy only)"), ("g1_s0.1", "0.1"), ("g1_s0.3", "0.3"), ("g1_s1", "1"), ("g0_s1", "random\nactions")]
    arms_ = [("retrieval", "tabpfn", "tab_ret"), ("retrieval", "dynapfn", "dyna_ret"), ("full", "tabpfn", "tab_full"),
             ("recent", "tabpfn", "tab_recent"), ("trained", "mlp", "mlp")]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.5, 2.35), gridspec_kw=dict(width_ratios=[2.2, 1]))
    for st, m, k in arms_:
        y = np.maximum([C.get((x, st, m), {}).get("uniform", np.nan) for x, _ in xs], 3e-3)
        ax.plot(range(len(xs)), y, **style(k))
    ax.axhline(1, color=INK, lw=0.9, ls=":"); ax.text(len(xs) - 1, 1.12, "ignores the action", fontsize=6.5, ha="right")
    ax.set_xticks(range(len(xs))); ax.set_xticklabels([l for _, l in xs]); ax.set_yscale("log"); ax.set_ylim(2e-3, 5)
    ax.set_xlabel("exploration noise of the data-collecting controller")
    ax.set_ylabel("error of predicted action effect"); ax.set_title("(a) Effect of unseen actions vs exploration", loc="left")
    for i, (st, m, k) in enumerate(arms_):
        c = STY[k][1]
        v0 = C.get(("g1_s0", st, m), {}).get("uniform", np.nan); v1 = C.get(("g1_s0_ex", st, m), {}).get("uniform", np.nan)
        bx.bar(i - 0.19, v0, width=0.36, color="white", edgecolor=c, lw=1.2)
        bx.bar(i + 0.19, max(v1, 3e-3), width=0.36, color=c, edgecolor=c)
    bx.set_yscale("log"); bx.set_ylim(2e-3, 5); bx.axhline(1, color=INK, lw=0.9, ls=":")
    bx.set_xticks(range(len(arms_)))
    bx.set_xticklabels(["TabPFN retrieval", "DynaPFN retrieval", "TabPFN full", "TabPFN recent", "MLP"],
                       fontsize=6.3, rotation=32, ha="right", rotation_mode="anchor")
    nice_log_y(ax); nice_log_y(bx)
    bx.set_title("(b) No exploration + 5% random actions", loc="left", fontsize=7.5); bx.grid(axis="x", visible=False)
    h, l = ax.get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    save(fig, "counterfactual")
    f2 = lambda v: f"{v:.2f}" if v >= 0.095 else f"{v:.1g}"      # 0.003, not 0.00
    for key, st, m in [("Ret", "retrieval", "tabpfn"), ("Recent", "recent", "tabpfn"), ("Full", "full", "tabpfn"),
                       ("Dyna", "retrieval", "dynapfn"), ("MLP", "trained", "mlp")]:
        for x, xk in [("g1_s0", "Zero"), ("g1_s0.1", "Small"), ("g1_s1", "One"), ("g1_s0_ex", "Explore")]:
            macros[f"CF{key}{xk}"] = f2(C.get((x, st, m), {}).get("uniform", np.nan))
    macros["CFNSeeds"] = str(C[("g1_s0", "retrieval", "tabpfn")]["n_seeds"])


def control_table(macros):
    """Closed-loop MPC on D4RL robots (results/sync/demo): distance and steps before falling, mean over
    seeds; appendix table + macros. Partial checkpoint files are ignored."""
    acc = defaultdict(list); sec = []
    for f in glob.glob(f"{PT.SYNC}/demo/*.json"):
        if "partial" in f:
            continue
        for r in json.load(open(f))["results"]:
            env = r.get("env", "halfcheetah")
            acc[(env, r["controller"], r["strategy"])].append((r["distance"], r.get("steps_alive", r["T"]), r["T"]))
            if env == "halfcheetah" and r["controller"] == "tabpfn":
                sec.append(r["secs_per_step"])
    arms_ = [("Planner with the true dynamics", "oracle", "-"), (r"\textbf{TabPFN, trajectory retrieval (ours)}", "tabpfn", "trajretrieval"),
             ("TabPFN, recent window", "tabpfn", "recent"), ("TabPFN, random context", "tabpfn", "random"),
             ("Trained MLP", "mlp", "-"), ("Random actions", "random", "-")]
    L = [r"\begin{tabular}{lccc}", r"\toprule",
         r" & HalfCheetah & \multicolumn{2}{c}{Hopper} \\",
         r"Planner model and context & distance (m) & distance (m) & steps before falling \\", r"\midrule"]
    for lab, c, st in arms_:
        hc = acc.get(("halfcheetah", c, st), []); hp = acc.get(("hopper", c, st), [])
        f = lambda v, i, d: (PT.minus(f"{np.mean([x[i] for x in v]):.{d}f}") + f" ({len(v)})" if v else "--")
        L.append(f"{lab} & {f(hc, 0, 2)} & {f(hp, 0, 2)} & {f(hp, 1, 0)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (PT.PAPER / "tables" / "control.tex").write_text("\n".join(L) + "\n")
    for env, ek in (("halfcheetah", "Cheetah"), ("hopper", "Hopper")):
        for c, st, k in (("oracle", "-", "Oracle"), ("tabpfn", "trajretrieval", "Traj"), ("tabpfn", "recent", "Recent"),
                         ("mlp", "-", "MLP")):
            v = acc.get((env, c, st), [])
            if v:
                macros[f"Ctl{ek}{k}Dist"] = f"{np.mean([x[0] for x in v]):.1f}"
                macros[f"Ctl{ek}{k}Steps"] = f"{np.mean([x[1] for x in v]):.0f}"
    macros["CtlHopperT"] = "200"
    macros["CtlCheetahTabSecPerStep"] = f"{round(float(np.median(sec)), -1):.0f}"   # TabPFN planners, all contexts


def cf_robots(macros):
    """D4RL counterfactual (MuJoCo ground truth): median over seeds of the per-seed median over query
    states of the effect error, for actions drawn uniformly over the data range; appendix table + macros."""
    acc = defaultdict(list)
    for f in glob.glob(f"{PT.SYNC}/cheetah/cf2_*.json"):
        d = json.load(open(f)); a = d["args"]
        per = defaultdict(list)
        for r in d["results"]:
            per[(r["strategy"], r["model"])].append((r["uniform"]["effect_rel"], r["0.3"]["effect_rel"]))
        for k, v in per.items():
            acc[(a["system"], a["explore_frac"] > 0) + k].append(np.median(np.array(v), 0))
    med = {k: np.median(np.array(v), 0) for k, v in acc.items()}
    nseeds = {k: len(v) for k, v in acc.items()}
    arms_ = [("TabPFN + retrieval (ours)", "retrieval", "tabpfn", True), ("DynaPFN + retrieval (ours)", "retrieval", "dynapfn", True),
             ("TabPFN, full pool", "full", "tabpfn", False), ("TabPFN, recent window", "recent", "tabpfn", False),
             ("Trained MLP", "trained", "mlp", False)]
    systems = [("hopper", "Hopper"), ("halfcheetah", "HalfCheetah"), ("walker2d", "Walker2d")]
    L = [r"\begin{tabular}{l" + "cc" * len(systems) + "}", r"\toprule",
         " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{n}}}" for _, n in systems) + r" \\",
         "Model and context" + " & policy & +5\\%" * len(systems) + r" \\", r"\midrule"]
    for lab, st, m, ours in arms_:
        cells = []
        for s, _ in systems:
            for ex in (False, True):
                v = med.get((s, ex, st, m))
                cells.append("--" if v is None else f"{v[0]:.2f}")
        L.append((r"\textbf{" + lab + "}" if ours else lab) + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (PT.PAPER / "tables" / "counterfactual_robots.tex").write_text("\n".join(L) + "\n")
    for s, key in [("hopper", "Hopper"), ("halfcheetah", "Cheetah"), ("walker2d", "Walker")]:
        for st, m, k in [("retrieval", "tabpfn", "Ret"), ("recent", "tabpfn", "Recent"), ("full", "tabpfn", "Full"),
                         ("trained", "mlp", "MLP")]:
            for ex, ek in [(False, ""), (True, "Explore")]:
                v = med.get((s, ex, st, m))
                if v is not None:
                    macros[f"CFR{key}{k}{ek}"] = f"{v[0]:.2f}"           # uniform actions
                    macros[f"CFR{key}{k}{ek}Mod"] = f"{v[1]:.2f}"        # moderate perturbation (0.3 sd)
    for s, key in [("halfcheetah", "Cheetah"), ("walker2d", "Walker"), ("hopper", "Hopper")]:
        a, b = med.get((s, False, "retrieval", "tabpfn")), med.get((s, True, "retrieval", "tabpfn"))
        if a is not None and b is not None:          # % reduction of the effect error by 5% exploration
            macros[f"CFR{key}RetCut"] = f"{100 * (1 - b[0] / a[0]):.0f}"
    macros["CFRNSeeds"] = str(min(nseeds.values()))


# ----------------------------------------------------------------------------- figure 6: where the GP fails
def fig_diagnostics(macros):
    d1 = json.load(open(f"{PT.SYNC}/gp/diag_d1.json")); d2 = json.load(open(f"{PT.SYNC}/gp/diag_d2.json"))
    d3 = json.load(open(f"{PT.SYNC}/gp/diag_d3.json"))
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.25))
    ax = axs[0]
    bins = ["smooth (lower 50%)", "50-90%", "most abrupt 10%"]
    for sysn, ls_, nm in [("hopper", "-", "Hopper"), ("walker2d", "--", "Walker2d")]:
        for model, k in [("tabpfn", "tab_ret"), ("gp", "gp_ret")]:
            r = {x["bin"]: x[model] for x in d3 if x["diag"] == "D3" and x["system"] == sysn}
            s = style(k); s.update(ls=ls_, label=f"{'TabPFN' if model == 'tabpfn' else 'GP'}, {nm}")
            ax.plot(range(3), [r[b] for b in bins], **s)
    ax.set_xticks(range(3)); ax.set_xticklabels(["smooth\n50%", "middle\n40%", "abrupt\n10%"])
    ax.set_ylabel("one-step error"); ax.set_title("(a) contact transitions", loc="left")
    ax.legend(fontsize=5.8, frameon=False, loc="upper left")
    ax = axs[1]
    ns = [50, 100, 500, 2000]
    tt = [float(np.median([x["t_tabpfn"] for x in d1 if x["n"] == n])) for n in ns]
    tg = [float(np.median([x["t_gp"] for x in d1 if x["n"] == n])) for n in ns]
    ax.plot(ns, tt, **dict(style("tab_ret"), label="TabPFN"))
    ax.plot(ns, tg, **dict(style("gp_ret"), label="GP (marginal likelihood)"))
    ax.set_xscale("log"); ax.set_yscale("log"); nice_log_y(ax); ax.set_xlabel("training samples n")
    ax.set_ylabel("fit + predict time (s, CPU)")
    ax.set_title("(b) cost of fitting", loc="left"); ax.legend(fontsize=5.8, frameon=False, loc="upper left")
    macros["DOneTimeTabLarge"] = f"{tt[-1]:.0f}"; macros["DOneTimeGPLarge"] = f"{tg[-1]:.0f}"
    macros["DOneTimeGPSmall"] = f"{tg[0]:.0f}"; macros["DOneTimeTabSmall"] = f"{tt[0]:.0f}"
    macros["DOneTimeGPMid"] = f"{tg[2]:.0f}"; macros["DOneTimeTabMid"] = f"{tt[2]:.0f}"   # n = 500
    ax = axs[2]
    for model, k, nm in [("tabpfn", "tab_ret", "TabPFN"), ("gp", "gp_ret", "GP")]:
        for strat, fill in [("retrieval", True), ("random", False)]:
            r = sorted([x for x in d2 if x["dim"] == 10 and x["strategy"] == strat], key=lambda x: x["H"])
            s = style(k); s.update(ls="-" if fill else "--", mfc=s["color"] if fill else "white",
                                   label=f"{nm}, {'retrieved' if fill else 'random'}")
            ax.plot([x["H"] for x in r], [x[model] for x in r], **s)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("context size H")
    ax.set_ylabel("one-step error"); ax.set_title("(c) Lorenz-96: narrow contexts", loc="left")
    ax.legend(fontsize=5.6, frameon=False, loc="lower left")
    fig.tight_layout(w_pad=0.7)
    save(fig, "diagnostics")


# ----------------------------------------------------------------------------- appendix figures
def fig_synthetic(rows):
    hs = ["1", "10", "50", "100", "200", "300"]
    Th = {h: A.per_seed(rows, h) for h in hs}
    keys = ["tab_ret", "dyna_ret", "gp_ret", "tab_full", "mlp", "tab_recent"]
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.35))
    for ax, (st, name) in zip(axs, [("lorenz+nt4", "Lorenz-63 (noise-free)"), ("pendulum+nt4", "Forced pendulum (noise-free)")]):
        curves(ax, Th, hs, st, keys)
        ax.set_title(name, loc="left"); ax.set_xlabel("rollout step")
        ax.set_xticks([1, 10, 100, 300]); ax.set_xticklabels(["1", "10", "100", "300"])
    axs[0].set_ylabel("normalised RMSE")
    h, l = axs[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    save(fig, "synthetic")


def fig_precision():
    d1 = json.load(open(f"{PT.SYNC}/gp/diag_d1.json"))
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for func, ls_, lab in [("linear", "-", "linear"), ("sine", "--", "sine"), ("lorenz_quadratic", ":", "Lorenz quadratic")]:
        r = sorted([x for x in d1 if x["func"] == func], key=lambda x: x["n"])
        for model, k in [("tabpfn", "tab_ret"), ("gp", "gp_ret")]:
            s = style(k); s.update(ls=ls_, label=f"{'TabPFN' if model == 'tabpfn' else 'GP'}, {lab}")
            ax.plot([x["n"] for x in r], [x[model] for x in r], **s)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("training samples n"); ax.set_ylabel("relative error")
    ax.legend(fontsize=5.6, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "precision")


def fig_scaling(macros):
    """Retrieval vs full-pool context as the D4RL Hopper pool grows (scaling.py, results/sync/scaling)."""
    import scaling_summary as SS
    folder = f"{PT.SYNC}/scaling"
    if not glob.glob(f"{folder}/scale_hopper_nt*.json"):
        return
    S = SS.summary(folder)
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.15))
    for st, k, lab in [("retrieval", "tab_ret", "TabPFN + retrieval (ours), 100 rows"), ("full", "tab_full", "TabPFN, full pool")]:
        pts = sorted((r["pool_rows"], r) for (n, s), r in S.items() if s == st and "pool_rows" in r)
        if not pts:
            continue
        x = [p for p, _ in pts]
        for ax, key in zip(axs, ("secs_per_step", "gpu_gb", "err50")):
            s = style(k); s["label"] = lab
            ax.plot(x, [r[key] for _, r in pts], **s)
    for ax, yl, ttl in zip(axs, ("seconds per rollout step", "peak GPU memory (GB)", "rollout error after 50 steps"),
                           ("(a) time", "(b) memory", "(c) accuracy")):
        from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator
        ax.set_xscale("log"); ax.set_xlabel("transitions in the pool"); ax.set_ylabel(yl); ax.set_title(ttl, loc="left")
        ax.xaxis.set_major_locator(FixedLocator([2000, 5000, 10000, 20000, 50000]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
        ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
    axs[0].set_yscale("log"); nice_log_y(axs[0]); axs[2].set_yscale("log"); nice_log_y(axs[2])
    h, l = axs[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=0.6)
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    save(fig, "scaling")
    full = sorted((r["pool_rows"], r) for (n, s), r in S.items() if s == "full" and "pool_rows" in r)
    ret = sorted((r["pool_rows"], r) for (n, s), r in S.items() if s == "retrieval" and "pool_rows" in r)
    macros["ScaleFullSmallRows"] = f"{full[0][0] / 1000:.1f}"; macros["ScaleFullSmallSec"] = f"{full[0][1]['secs_per_step']:.1f}"
    macros["ScaleFullLargeRows"] = f"{full[-1][0] / 1000:.0f}"; macros["ScaleFullLargeSec"] = f"{full[-1][1]['secs_per_step']:.0f}"
    macros["ScaleRetMinSec"] = f"{min(r['secs_per_step'] for _, r in ret):.1f}"
    macros["ScaleRetMaxSec"] = f"{max(r['secs_per_step'] for _, r in ret):.1f}"
    macros["ScaleRetGPU"] = f"{np.median([r['gpu_gb'] for _, r in ret]):.1f}"
    macros["ScaleFullGPU"] = f"{min(r['gpu_gb'] for _, r in full):.0f}--{max(r['gpu_gb'] for _, r in full):.0f}"
    macros["ScaleRetLargeRows"] = f"{ret[-1][0] / 1000:.0f}"
    macros["ScaleRetSmallErr"] = f"{ret[0][1]['err50']:.2f}"; macros["ScaleRetLargeErr"] = f"{ret[-1][1]['err50']:.2f}"
    macros["ScaleFullSmallErr"] = f"{full[0][1]['err50']:.2f}"; macros["ScaleFullLargeErr"] = f"{full[-1][1]['err50']:.2f}"
    macros["ScaleNSeeds"] = str(min(r["n_seeds"] for _, r in full + ret))
    macros["ScaleLargeSpeedup"] = f"{full[-1][1]['secs_per_step'] / ret[-1][1]['secs_per_step']:.0f}"
    macros["ScaleRetLargeSec"] = f"{ret[-1][1]['secs_per_step']:.1f}"
    macros["ScaleFullLargeSeeds"] = str(full[-1][1]["n_seeds"]); macros["ScaleRetLargeSeeds"] = str(ret[-1][1]["n_seeds"])
    f80, r80 = S.get((80, "full"), {}), S.get((80, "retrieval"), {})          # largest pool with all seeds
    if "pool_rows" in f80 and "pool_rows" in r80:
        macros["ScaleMidRows"] = f"{f80['pool_rows'] / 1000:.0f}"; macros["ScaleMidSeeds"] = str(min(f80["n_seeds"], r80["n_seeds"]))
        macros["ScaleMidFullErr"] = f"{f80['err50']:.2f}"; macros["ScaleMidRetErr"] = f"{r80['err50']:.2f}"
        macros["ScaleMidFullSec"] = f"{f80['secs_per_step']:.0f}"; macros["ScaleMidRetSec"] = f"{r80['secs_per_step']:.1f}"
        macros["ScaleMidSpeedup"] = f"{f80['secs_per_step'] / r80['secs_per_step']:.0f}"


def fig_wall(macros):
    """Pendulum with / without walls (impacts) x observation noise (study.py pendulum_box)."""
    files = sorted(glob.glob(f"{PT.SYNC}/wall/*.json"))
    if not files:
        return
    rows = A.load(files)
    T = A.per_seed(rows, "100")
    noises = ["0", "0.005", "0.01", "0.02", "0.05"]
    keys = ["tab_ret", "dyna_ret", "tab_full", "gp_ret", "mlp", "lin_ret"]
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.35), sharey=True)
    for ax, (w, name) in zip(axs, [("0", "Smooth pendulum"), ("0.15", "Pendulum with walls (impacts)")]):
        for k in keys:
            st = lambda n: "pendulum_box" + (f"+noise{n}" if n != "0" else "") + f"+wall{float(w):g}+lags4+nt4"
            y = [PT.med(T, st(n), arm(k))[0] for n in noises]
            ax.plot(range(len(noises)), y, **style(k))
        ax.set_xticks(range(len(noises))); ax.set_xticklabels(["0", "0.5%", "1%", "2%", "5%"])
        ax.set_yscale("log"); nice_log_y(ax); ax.set_xlabel("observation noise"); ax.set_title(name, loc="left")
    axs[0].set_ylabel("rollout error after 100 steps")
    h, l = axs[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    save(fig, "wall")
    for w, wk in (("0", "Smooth"), ("0.15", "Wall")):
        for n, nk in zip(noises, ("Zero", "Half", "One", "Two", "Five")):
            st = "pendulum_box" + (f"+noise{n}" if n != "0" else "") + f"+wall{float(w):g}+lags4+nt4"
            for mdl, mk in (("tabpfn", "Tab"), ("dynapfn", "Dyna")):
                r, b, nn = paired_ratio(T, st, ("retrieval", 100, mdl), ("retrieval", 100, "gp"))
                if np.isfinite(r):
                    macros[f"Wall{wk}{nk}GPOver{mk}"] = f"{r:.2f}" if r < 1 else f"{r:.1f}" if r < 10 else f"{r:.0f}"
                    macros[f"Wall{wk}{nk}GPOver{mk}Better"] = f"{b}/{nn}"
                    macros[f"Wall{wk}{nk}GPBetterThan{mk}"] = f"{1 / r:.0f}" if r < 0.1 else f"{1 / r:.1f}"
    macros["WallNSeeds"] = str(min(len(d) for (s_, a_), d in T.items() if s_.startswith("pendulum_box") and a_ == arm("tab_ret")))


def all_figures(macros):
    rows = PT.full_rows()
    fig_regime(rows, macros); fig_robots(rows); fig_noise(rows); fig_planning(); fig_counterfactual(macros)
    cf_robots(macros); control_table(macros); fig_scaling(macros); fig_wall(macros)
    fig_diagnostics(macros); fig_synthetic(rows); fig_precision()
