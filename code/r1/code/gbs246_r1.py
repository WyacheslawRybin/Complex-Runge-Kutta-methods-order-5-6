"""GBS(2,4,6): extrapolated Gragg midpoint method without smoothing, written as a real RK tableau.
Check of the reviewer's numbers: 10 stages, depth 6, order 6, P = T_6, E_7 = 6.2e-4."""
import sys
from fractions import Fraction as F
import mpmath as mp
import r1common  # noqa: F401  (puts the original code on sys.path)
import crk

seq = (2, 4, 6)
stages = [dict()]                      # stage 0: y0 (coefficient dict over stage indices)
out = {}
for n in seq:
    d = F(1, n)
    z_prev = {}                         # z_0 = y0 (no increments)
    z = {0: d}                          # z_1 = y0 + d k0
    idx_prev = 0                        # stage index of f(z_0) = k0 (shared)
    for i in range(1, n):
        stages.append(dict(z))          # new stage at z_i
        ki = len(stages) - 1
        z_new = dict(z_prev)
        z_new[ki] = z_new.get(ki, 0) + 2 * d
        z_prev, z = z, z_new
    out[n] = z
w = {}
for j, nj in enumerate(seq):
    wj = F(1)
    for i, ni in enumerate(seq):
        if i != j:
            wj *= F(nj * nj, nj * nj - ni * ni)
    w[nj] = wj
print("weights", w, sum(w.values()))
s = len(stages)
A = [[F(0)] * s for _ in range(s)]
for i, st in enumerate(stages):
    for k, v in st.items():
        A[i][k] = v
b = [F(0)] * s
for n in seq:
    for k, v in out[n].items():
        b[k] += w[n] * v
# embedded T32: extrapolation of n = 2, 4 only
w2 = {2: F(4, 4 - 16), 4: F(16, 16 - 4)}
bh = [F(0)] * s
for n in (2, 4):
    for k, v in out[n].items():
        bh[k] += w2[n] * v
print("s =", s, " c =", [str(sum(r)) for r in A])
mp.mp.dps = 40
T = crk.Tableau("GBS246", [[mp.mpf(x.numerator) / x.denominator for x in r] for r in A],
                [(s, 6, [mp.mpf(x.numerator) / x.denominator for x in b]),
                 (s, 4, [mp.mpf(x.numerator) / x.denominator for x in bh])], projected=False)
res = T.residuals()
print("max residual order 6:", mp.nstr(max(abs(v) for v in res[0]), 3), " embedded order 4:",
      mp.nstr(max(abs(v) for v in res[1]), 3))
E7 = T.E(0)
print("E7 =", mp.nstr(E7, 6), " depth*E7^(1/6) =", mp.nstr(6 * E7 ** (mp.mpf(1) / 6), 5),
      " 10*E7^(1/6) =", mp.nstr(10 * E7 ** (mp.mpf(1) / 6), 5))
print("stab coeffs * k!:", [mp.nstr(c * mp.factorial(k), 6) for k, c in enumerate(T.stab_coeffs(0))])
print("max|coef| =", mp.nstr(T.maxcoef(), 4), " max|b| =", max(abs(x) for x in b), " max|a| =",
      max(abs(x) for r in A for x in r), " max|bh| =", max(abs(x) for x in bh))
# depth: longest dependency chain of stages
dep = [0] * s
for i in range(s):
    dep[i] = 1 + max([dep[k] for k in stages[i]] + [0])
print("depth (sequential evaluations) =", max(dep))
