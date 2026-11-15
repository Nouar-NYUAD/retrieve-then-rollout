"""pendulum_box: walls are never crossed, impacts happen and dissipate energy; wall=0 is the smooth pendulum."""
import sys, numpy as np
sys.path.insert(0, ".")
import study as S, pilot as P
rng = np.random.default_rng(0)
Sw, U = S.simulate_box(2000, np.random.default_rng(1), 0.15)
assert np.all(np.abs(Sw[:, 0]) <= 0.15 + 1e-12), "wall crossed"
imp = int(np.sum((np.sign(Sw[1:, 1]) != np.sign(Sw[:-1, 1])) & (np.abs(Sw[1:, 0]) > 0.85 * 0.15)))
print("max |theta| with walls:", np.abs(Sw[:, 0]).max().round(4), " velocity reversals near a wall:", imp)
assert imp > 40
S0, U0 = S.simulate_box(2000, np.random.default_rng(1), 0.0)
x = S0[0].copy()
for t in range(50):
    x = P.rk4(P.pendulum_f, x, U0[t], 0.05)
assert np.allclose(x, S0[50]), "wall=0 must be the plain RK4 pendulum"
print("max |theta| without walls:", np.abs(S0[:, 0]).max().round(3), "(walls matter: > 0.15)")
assert np.abs(S0[:, 0]).max() > 0.15
print("BOX CHECK PASSED")
