"""Checks for counterfactual.py metrics and ground truth (no TabPFN; CPU, seconds).
1) oracle predictor (the truth itself)  -> effect_rel 0, effect_cos 1, rank_rho 1, cf_err 0
2) action-blind predictor (factual change for every action) -> effect_rel exactly 1
3) MuJoCo truth reproduces logged D4RL transitions; action changes really change the outcome
4) action-diverse retrieval returns H distinct rows whose action spread exceeds plain state retrieval"""
import sys, numpy as np, argparse
sys.path.insert(0, ".")
import counterfactual as C, pilot as P

rng = np.random.default_rng(0)
for system in sys.argv[1:]:
    tr = C.Truth(system)
    a = argparse.Namespace(system=system, n_train=4 if system == "pendulum" else 10, T_train=1000, T_test=1000,
                           gain=1.0, sigma=0.1, t0_min=100, horizon=100, lags=1, data_dir="/scratch/naa9497/icwm/data/d4rl")
    Xtr, Ytr, S_te, U_te, sd = C.seed_data(a, np.random.default_rng(7))
    d = S_te.shape[1]; a_sd = Xtr[:, d:].std(0); lo, hi = Xtr[:, d:].min(0), Xtr[:, d:].max(0)
    t = next(t for t in range(120, 900) if tr.ok(S_te[t])); s, u = S_te[t], U_te[t]
    acts = C.counterfactual_actions(u, a_sd, lo, hi, 6, rng)
    A = np.vstack([u[None]] + [acts[l] for l in C.LEVELS]); D = tr.step(s, A) - s
    m_or = C.metrics(D[0], D[0], D[1:], D[1:], sd, C.VX[system])
    m_bl = C.metrics(D[0], D[0], np.repeat(D[:1], len(D) - 1, 0), D[1:], sd, C.VX[system])
    print(system, "oracle", {k: round(v, 6) for k, v in m_or.items()})
    print(system, "blind ", {k: round(v, 6) for k, v in m_bl.items()})
    assert m_or["effect_rel"] < 1e-9 and m_or["cf_err"] < 1e-12 and abs(m_or["rank_rho"] - 1) < 1e-9
    assert abs(m_bl["effect_rel"] - 1) < 1e-9 and m_bl["rank_rho"] == 0.0
    eff = np.sqrt(np.mean(((D[1:] - D[0]) / sd) ** 2)); print(system, "mean |true effect| (sd units):", round(eff, 4))
    assert eff > 1e-3, "actions have no effect in the simulator - wrong mapping?"
    ok = [i for i in rng.choice(len(Xtr), 300, replace=False) if tr.ok(Xtr[i, :d])]
    cons = np.array([np.abs(tr.step(Xtr[i, :d], Xtr[i, d:][None])[0] - Xtr[i, :d] - Ytr[i]) / sd for i in ok])
    print(system, "sim vs data: median %.2e  95%% %.2e  (n=%d)" % (np.median(cons), np.quantile(cons, .95), len(ok)))
    pool = P.Pool(Xtr, Ytr); ctx = C.Contexts(pool, d, 100, np.arange(len(Xtr)), rng, 5000)
    i_s, i_d = ctx.get("retrieval_state", s, u), ctx.get("retrieval_div", s, u)
    assert len(set(i_d.tolist())) == len(i_d) == 100 and ctx.get("retrieval_div", s, u) is i_d
    sp = lambda I: pool.Z[I, d:].std(0).mean()
    print(system, "action spread in context: state-retrieval %.3f  diverse %.3f" % (sp(i_s), sp(i_d)))
    assert sp(i_d) > sp(i_s)
    print(system, "action-from-state R2:", round(C.confounding_r2(Xtr, d, rng), 3))
print("ALL CHECKS PASSED")
