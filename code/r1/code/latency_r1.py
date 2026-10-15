"""Latency and resource model of a fully parallel datapath (review R1b, items C1 and C2).

The cost model kappa = 1 counts a complex evaluation of f as one real evaluation.  This script replaces
it by an explicit model of a pipelined datapath in which every arithmetic operation has its own unit,
so the latency of a step is the length of the critical path through the operations of the step.
A real multiplication takes L_m, a real addition L_a (a negation or a multiplication by a power of
two is free).  The complex operations are built from real units with the textbook formulas:
    complex + complex            : 2 adders,                    latency L_a
    complex + real constant      : 1 adder,                     latency L_a
    real constant * complex      : 2 multipliers,               latency L_m
    complex * complex            : 4 multipliers and 2 adders,  latency L_m + L_a
    complex square               : 3 multipliers and 1 adder,   latency L_m + L_a
(three-multiplier variants of the product need more adders and a longer path).  A stage computes
Y_i = y_n + h sum_j a_ij k_j and k_i = f(Y_i).  The partial sums over j < i - 1 are formed while k_{i-1}
is computed, so the increment adds one product and one addition to the critical path: L_m + L_a for a
real method, L_m + 2 L_a for a complex one (complex constant times complex k: L_m + L_a, then one complex
addition), except for the second stage, since k_1 = f(y_n) is real.  The projected output needs only
the real part, Re(b_s k_s) = Re b_s Re k_s - Im b_s Im k_s, i.e., L_m + 2 L_a.  Hence, with D_R and D_C
the critical paths of the real and the complex evaluation of f,
    real method, per evaluation      : l_R = D_R + L_m + L_a                  (FSAL included)
    complex method with s stages     : T_C = D_R + (L_m + L_a + D_C) + (s - 2)(L_m + 2 L_a + D_C) + L_m + 2 L_a
and the latency ratio kappa_lat = T_C / (s l_R) replaces kappa = 1.  It depends on the problem and on
nu = L_a / L_m (r in the code), and kappa_lat -> 1 as nu -> 0.  The controller of a variable-step code adds the same
latency per step to both methods and is not included.

The right-hand sides are the polynomial problems of Appendix B and BRUSS-2D (per grid point); sums of
more than two terms are evaluated as trees that combine the two operands available first.  The
non-polynomial problems are excluded: complex exponentials, trigonometric functions and square roots
need longer pipelines than their real counterparts, and their latencies depend on the implementation.

Writes tables/latency.tex and data/latency_r1.json.
"""
import json
import math
import os

import r1common as R
import r1patch

MT = r1patch.apply()
import figures_cost as FC  # noqa: E402
from figures_cost import adapt_curve, cost_at, fixed_curve  # noqa: E402

RS = (0.25, 0.5, 1.0)                 # r = nu = L_a / L_m
R_MAIN = (0.25, 1.0)                  # columns of the table
OUT = os.path.join(R.DATA, "latency_r1.json")


class Ctx:
    def __init__(self, r):
        self.Lm, self.La = 1.0, r
        self.n = dict(rm=0, ra=0, cm=0, ca=0)         # real mults/adds, complex-datapath mults/adds


class V:
    """A value of the datapath: ready times of the real and of the complex implementation."""
    __slots__ = ("x", "tr", "tc", "path_r", "path_c")

    def __init__(self, x, tr=0.0, tc=0.0, pr=(0, 0), pc=(0, 0)):
        self.x, self.tr, self.tc, self.path_r, self.path_c = x, tr, tc, pr, pc


def _late(a, b, which):
    if which == "r":
        return (a.tr, a.path_r) if a.tr >= b.tr else (b.tr, b.path_r)
    return (a.tc, a.path_c) if a.tc >= b.tc else (b.tc, b.path_c)


def add(X, a, b):
    """a + b, either may be a Python number (constant)."""
    if not isinstance(a, V):
        a, b = b, a
    if not isinstance(b, V):                            # var + constant
        X.n["ra"] += 1; X.n["ca"] += 1
        return V(None, a.tr + X.La, a.tc + X.La, (a.path_r[0], a.path_r[1] + 1), (a.path_c[0], a.path_c[1] + 1))
    tr, pr = _late(a, b, "r")
    tc, pc = _late(a, b, "c")
    X.n["ra"] += 1; X.n["ca"] += 2
    return V(None, tr + X.La, tc + X.La, (pr[0], pr[1] + 1), (pc[0], pc[1] + 1))


def mul(X, a, b):
    if not isinstance(a, V):
        a, b = b, a
    if not isinstance(b, V):                            # constant * var
        X.n["rm"] += 1; X.n["cm"] += 2
        return V(None, a.tr + X.Lm, a.tc + X.Lm, (a.path_r[0] + 1, a.path_r[1]), (a.path_c[0] + 1, a.path_c[1]))
    tr, pr = _late(a, b, "r")
    tc, pc = _late(a, b, "c")
    X.n["rm"] += 1
    if a is b:
        X.n["cm"] += 3; X.n["ca"] += 1
    else:
        X.n["cm"] += 4; X.n["ca"] += 2
    return V(None, tr + X.Lm, tc + X.Lm + X.La, (pr[0] + 1, pr[1]), (pc[0] + 1, pc[1] + 1))


def S(X, *terms):
    """Sum of several terms: repeatedly add the two operands that are ready first (separately for the
    real and the complex datapath the order may differ; the real order is used for both, which is
    optimal for the real one and nearly so for the complex one)."""
    terms = list(terms)
    consts = [t for t in terms if not isinstance(t, V)]
    vs = sorted([t for t in terms if isinstance(t, V)], key=lambda v: (v.tr, v.tc))
    if consts:                                          # the constant goes to the operand ready first
        vs[0] = add(X, vs[0], sum(consts))
    while len(vs) > 1:
        vs.sort(key=lambda v: (v.tr, v.tc))
        a, b = vs.pop(0), vs.pop(0)
        vs.append(add(X, a, b))
    return vs[0]


def neg(v):
    return v


# ------------------------------------------------------------- right-hand sides (Appendix B)
def f_vdp(X, y):
    y1, y2 = y
    return [y2, S(X, mul(X, S(X, 1.0, neg(mul(X, y1, y1))), y2), neg(y1))]


def f_rossler(X, y):
    y1, y2, y3 = y
    return [S(X, neg(y2), neg(y3)), S(X, y1, mul(X, 0.2, y2)), S(X, 0.2, mul(X, y3, S(X, y1, -5.7)))]


def f_bouali(X, y):
    y1, y2, y3 = y
    return [S(X, mul(X, mul(X, 3.0, y1), S(X, 1.0, neg(y2))), mul(X, -2.2, y3)),
            neg(mul(X, y2, S(X, 1.0, neg(mul(X, y1, y1))))),
            mul(X, 0.001, y1)]


def f_lorenz(X, y):
    y1, y2, y3 = y
    return [mul(X, 10.0, S(X, y2, neg(y1))), S(X, mul(X, y1, S(X, 28.0, neg(y3))), neg(y2)),
            S(X, mul(X, y1, y2), mul(X, -8.0 / 3.0, y3))]


def f_rigid(X, y):
    y1, y2, y3 = y
    return [mul(X, y2, y3), neg(mul(X, y1, y3)), mul(X, -0.51, mul(X, y1, y2))]


def f_bruss(X, y):
    y1, y2 = y
    u2v = mul(X, mul(X, y1, y1), y2)
    return [S(X, 1.0, u2v, mul(X, -4.0, y1)), S(X, mul(X, 3.0, y1), neg(u2v))]


def f_bruss2d(X, y):
    """One grid point: u, v and their four neighbours; alpha / dx^2 = 0.8, -4 alpha / dx^2 = -3.2."""
    u, v, uE, uW, uN, uS, vE, vW, vN, vS = y
    u2v = mul(X, mul(X, u, u), v)
    lu = mul(X, 0.8, S(X, uE, uW, uN, uS))
    lv = mul(X, 0.8, S(X, vE, vW, vN, vS))
    return [S(X, 1.0, u2v, mul(X, -4.4 - 3.2, u), lu), S(X, mul(X, 3.4, u), neg(u2v), mul(X, -3.2, v), lv)]


PROBS = [("VanderPol", f_vdp, 2), ("Rossler", f_rossler, 3), ("Bouali", f_bouali, 3), ("Lorenz5", f_lorenz, 3),
         ("RigidBody", f_rigid, 3), ("Brusselator", f_bruss, 2), ("Bruss2D", f_bruss2d, 10)]


def evaluate(fun, dim, r):
    X = Ctx(r)
    out = fun(X, [V(None) for _ in range(dim)])
    vs = [o for o in out if isinstance(o, V)]
    o_r = max(vs, key=lambda v: v.tr)
    o_c = max(vs, key=lambda v: v.tc)
    return dict(DR=o_r.tr, DC=o_c.tc, path_r=o_r.path_r, path_c=o_c.path_c, **X.n)


def kappa_lat(DR, DC, s, r):
    Lm, La = 1.0, r
    lR = DR + Lm + La
    TC = DR + (Lm + La + DC) + (s - 2) * (Lm + 2 * La + DC) + Lm + 2 * La
    return TC / (s * lR)


def nu_star(name, v, s=6):
    """Break-even nu = L_a / L_m: the largest nu at which v * kappa_lat(nu) <= 1 (kappa_lat grows with nu);
    None if the complex pair is not cheaper even at kappa = 1."""
    if v is None or v >= 1.0:
        return None
    fun, dim = {q: (f, n) for q, f, n in PROBS}[name]
    g = lambda r: v * kappa_lat(*(lambda e: (e["DR"], e["DC"]))(evaluate(fun, dim, r)), s, r) - 1.0
    lo, hi = 0.0, 4.0
    if g(hi) <= 0:
        return math.inf
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if g(mid) <= 0 else (lo, mid)
    return lo


def cost_ratios(eps_list=(1e-6, 1e-8, 1e-10)):
    """Best complex / best real cost at kappa = 1, within each order, fixed and variable step."""
    ad = json.load(open(os.path.join(r1patch.MERGED, "adaptive.json")))
    meas = json.load(open(os.path.join(r1patch.MERGED, "measured.json")))
    skip = ("GJM5", "CRK5-s", "CRK5-1", "CRK5-2", "CRK6-s", "CRK6-0", "CRK6-lam", "CCK6(5)-s")
    out = {}
    for p in [q for q, _, _ in PROBS if q != "Bruss2D"]:
        for eps in eps_list:
            for mode, groups in (("fixed", (FC.FIXED5, FC.FIXED6)), ("adaptive", (FC.ADAPT5, FC.ADAPT6))):
                for order, group in zip((5, 6), groups):
                    vals = {}
                    for m in group:
                        if m in skip:
                            continue
                        if mode == "fixed":
                            N, e = fixed_curve(meas, m, p)
                            vals[m] = cost_at(FC.EVALS[m] * N, e, eps)
                        else:
                            n, e, _ = adapt_curve(ad, m, p)
                            vals[m] = cost_at(n, e, eps)
                    cx = min(v for m, v in vals.items() if m.startswith(("CRK", "CCK")))
                    rl = min(v for m, v in vals.items() if not m.startswith(("CRK", "CCK")))
                    out[f"{mode}|{p}|{order}|{eps:g}"] = cx / rl if math.isfinite(cx) and math.isfinite(rl) else None
    # BRUSS-2D (data/bruss2d_costs.json, accuracies 1e-6, 1e-8, 1e-10)
    bc = json.load(open(os.path.join(R.DATA, "bruss2d_costs.json")))
    groups = {5: (["CRK5-opt"], ["DP5", "CK5", "Tsit5(4)", "BS5(4)"]),
              6: (["CRK6-opt", "CCK6(5)", "CCK6-casc"], ["Butcher6", "RK6-opt", "DP6(5)", "Verner6(5)", "Vern6(5)"])}
    for k, eps in enumerate(eps_list):
        for mode in ("fixed", "adaptive"):
            for order, (cg, rg) in groups.items():
                c = [bc[m][mode][k] for m in cg if m in bc and bc[m][mode][k] is not None]
                rr = [bc[m][mode][k] for m in rg if m in bc and bc[m][mode][k] is not None]
                c = [v for v in c if math.isfinite(v)]
                rr = [v for v in rr if math.isfinite(v)]
                out[f"{mode}|Bruss2D|{order}|{eps:g}"] = min(c) / min(rr) if c and rr else None
    return out


def main():
    res = {}
    for name, fun, dim in PROBS:
        ev = {r: evaluate(fun, dim, r) for r in RS}
        paths = {(v["path_r"], v["path_c"]) for v in ev.values()}
        if len(paths) > 1:
            print(f"note: critical path of {name} depends on r: {paths}")
        e = ev[0.25]                                   # critical paths for r = 1/4 (table)
        res[name] = dict(real_mult=e["rm"], real_add=e["ra"], cplx_mult=e["cm"], cplx_add=e["ca"],
                         path_real=e["path_r"], path_cplx=e["path_c"],
                         kappa_lat={f"{s}|{r:g}": kappa_lat(ev[r]["DR"], ev[r]["DC"], s, r) for s in (5, 6) for r in RS})
    cr = cost_ratios()
    for name in res:
        res[name]["cost_ratio_kappa1"] = {k.split("|", 2)[0] + "|" + k.split("|")[2] + "|" + k.split("|")[3]: v
                                          for k, v in cr.items() if k.split("|")[1] == name}
    json.dump(res, open(OUT, "w"), indent=1)
    # printout
    print(f"{'problem':12s} real f (m, a, path)   complex f (m, a, path)   kappa_lat s=6 for r = " +
          ", ".join(f"{r:g}" for r in RS) + "   | s=5")
    for name, d in res.items():
        print(f"{name:12s} {d['real_mult']:3d} {d['real_add']:3d} {str(d['path_real']):8s}   "
              f"{d['cplx_mult']:3d} {d['cplx_add']:3d} {str(d['path_cplx']):8s}   "
              + "  ".join(f"{d['kappa_lat'][f'6|{r:g}']:.3f}" for r in RS) + "   | "
              + "  ".join(f"{d['kappa_lat'][f'5|{r:g}']:.3f}" for r in RS))
    print("\nbest complex / best real at kappa = 1, and times kappa_lat (r = 0.25, 1):")
    for mode in ("fixed", "adaptive"):
        for order in (5, 6):
            for eps in (1e-6, 1e-8, 1e-10):
                row = []
                for name, d in res.items():
                    v = d["cost_ratio_kappa1"].get(f"{mode}|{order}|{eps:g}")
                    if v is None:
                        row.append(f"{name[:6]:6s}   ---")
                        continue
                    k1, k4 = d["kappa_lat"][f"{order}|0.25"], d["kappa_lat"][f"{order}|1"]
                    row.append(f"{name[:6]:6s} {v:.2f}/{v * k1:.2f}/{v * k4:.2f}")
                print(f"  {mode:8s} order {order} {eps:g}: " + "  ".join(row))
    # table: resources, critical paths and kappa_lat of the polynomial problems; order-6 variable-step
    # cost ratio (best complex / best real) at 1e-8 for kappa = 1 and for kappa_lat at r = 1/4 and r = 1
    lines = []
    fmt = lambda p: "$" + ("" if p[0] == 0 else ("L_m" if p[0] == 1 else f"{p[0]}L_m")) +         ("+" if p[0] and p[1] else "") + ("" if p[1] == 0 else ("L_a" if p[1] == 1 else f"{p[1]}L_a")) + "$"
    for name, d in res.items():
        cells = [MT.PNAME.get(name, "BRUSS-2D").replace(", $T=5$", ""),
                 f"{d['real_mult']}/{d['real_add']}", f"{d['cplx_mult']}/{d['cplx_add']}",
                 fmt(d["path_real"]), fmt(d["path_cplx"])]
        cells += [f"${d['kappa_lat'][f'6|{r:g}']:.2f}$" for r in R_MAIN]
        v = d["cost_ratio_kappa1"].get("adaptive|6|1e-08")
        for r in (0.0,) + R_MAIN:
            w = v * (1.0 if r == 0.0 else d["kappa_lat"][f"6|{r:g}"])
            cells.append(MT.best(f"${w:.2f}$") if w < 0.995 else f"${w:.2f}$")
        nus = nu_star(name, v)
        d["nu_star_1e-08"] = nus
        cells.append("---" if nus is None else f"${nus:.2f}$")
        lines.append(" & ".join(cells) + r"\\")
    json.dump(res, open(OUT, "w"), indent=1)
    open(os.path.join(R.TAB, "latency.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("wrote tables/latency.tex and", OUT)


if __name__ == "__main__":
    main()
