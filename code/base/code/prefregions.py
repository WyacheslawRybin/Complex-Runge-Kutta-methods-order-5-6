"""
Stability and preference regions in the sense of Fedoseev, Karimov, Legat and
Butusov (Mathematics 10:4327, 2022), as implemented in U-CUDA:
a point z = h*lambda of the complex plane is PREFERRED when the step is stable,
|P(z)| <= 1, and its relative one-step error is below a tolerance,
    |e^z - P(z)| / |e^z| <= tol,     tol = 1 as in the original paper.
For a projected method on the real test system with eigenvalues sigma +- i*omega
the step matrix is P(hA) with the real polynomial P of stability.py; A is normal,
so the spectral-norm error of U-CUDA reduces to the scalar formula above.
Methods of order five and of order six are compared separately.
Writes ../figures/fig_prefregions.pdf and ../data/prefregions.json.
"""
import json, os
from math import factorial
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from figures import ST, lbl, save, DATA

TOL = 1.0
STAB_FILL, PREF_FILL, PREF_EDGE = "#d9d9d9", "#6cc76c", "#1e7b1e"

def Tm(m):
    return [1.0 / factorial(k) for k in range(m + 1)]

def grid(xlim=(-5.5, 2.0), ylim=(-4.5, 4.5), n=(1100, 1300)):
    x = np.linspace(*xlim, n[0]); y = np.linspace(*ylim, n[1])
    X, Y = np.meshgrid(x, y)
    return X, Y, X + 1j * Y, (x[1] - x[0]) * (y[1] - y[0])

def masks(co, Z):
    P = np.polynomial.Polynomial(co)(Z)
    rho = np.abs(P)
    err = np.abs(np.exp(Z) - P) / np.abs(np.exp(Z))
    return rho, err

def areas(co, Z, dA):
    rho, err = masks(co, Z)
    st = rho <= 1
    return {"stab": float(st.sum() * dA), "pref": float((st & (err <= TOL)).sum() * dA)}

ORDER5 = [("$T_5$: CRK5, CCK6 emb.", Tm(5), 5, "CRK5-opt"), ("DP5(4)", None, 6, "DP5"), ("CK5(4)", None, 6, "CK5")]
ORDER6 = [("$T_6$: CRK6, CCK6", Tm(6), 6, "CCK6(5)"), ("Butcher6", None, 7, "Butcher6"),
          ("RK6-opt", None, 7, "RK6-opt"), ("DP6(5)", None, 8, "DP6(5)"), ("Verner6(5)", None, 8, "Verner6(5)")]

def main():
    st = json.load(open(os.path.join(DATA, "stability.json")))
    for lst in (ORDER5, ORDER6):
        for i, (lab, co, ev, key) in enumerate(lst):
            if co is None:
                lst[i] = (lab, st[key]["coeffs"], ev, key)
    X, Y, Z, dA = grid()
    res = {}
    for lab, co, ev, key in ORDER5 + ORDER6:
        a = areas(co, Z, dA)
        a.update(evals=ev, stab_per_eval=a["stab"] / ev ** 2, pref_per_eval=a["pref"] / ev ** 2)
        res[lab] = a
        print(f"{lab:26s} stab {a['stab']:6.2f}  pref {a['pref']:6.2f}   per eval: "
              f"{a['stab_per_eval']:.4f} {a['pref_per_eval']:.4f}")
    json.dump(res, open(os.path.join(DATA, "prefregions.json"), "w"), indent=1)

    gs = dict(width_ratios=[1, 1, 1, 1.0])
    fig, axs = plt.subplots(2, 4, figsize=(7.4, 4.55), gridspec_kw=gs)
    def panel(a, lab, co):
        rho, err = masks(co, Z)
        stab = rho <= 1
        a.contourf(X, Y, stab.astype(float), levels=[0.5, 1.5], colors=[STAB_FILL])
        a.contourf(X, Y, (stab & (err <= TOL)).astype(float), levels=[0.5, 1.5], colors=[PREF_FILL])
        a.contour(X, Y, rho, [1], colors=["k"], linewidths=0.6)
        a.contour(X, Y, np.where(stab, err, 10.0), [TOL], colors=[PREF_EDGE], linewidths=0.8)
        a.set_title(lab, fontsize=8)
        a.set_aspect("equal"); a.grid(False)
        a.set_xlim(-5.5, 2.0); a.set_ylim(-4.5, 4.5)
        a.axhline(0, color="#999999", lw=0.3); a.axvline(0, color="#999999", lw=0.3)
    for r, lst in enumerate([ORDER5, ORDER6]):
        shown = lst[:3] if r == 0 else [lst[0], lst[2], lst[3]]
        for c, (lab, co, ev, key) in enumerate(shown):
            panel(axs[r, c], f"({'abcdefgh'[4 * r + c]}) {lab}", co)
        # per-evaluation comparison: boundaries of the preferred regions in z/s for every method of this order
        a = axs[r, 3]
        for lab, co, ev, key in lst:
            rho, err = masks(co, Z * ev)
            col, _, ls = ST[key]
            a.contour(X, Y, np.where(rho <= 1, err, 10.0), [TOL], colors=[col], linestyles=[ls], linewidths=1.2)
        a.set_xlim(-1.0, 0.12); a.set_ylim(-0.75, 0.75); a.set_aspect("equal")
        a.axhline(0, color="#999999", lw=0.3); a.axvline(0, color="#999999", lw=0.3)
        a.set_title(f"({'abcdefgh'[4 * r + 3]}) per evaluation", fontsize=8)
        handles = [Line2D([], [], color=ST[key][0], ls=ST[key][2], lw=1.2, label=lab.split(":")[0])
                   for lab, co, ev, key in lst]
        a.legend(handles=handles, loc="upper left", fontsize=5.8, handlelength=2.0, borderaxespad=0.3,
                 labelspacing=0.25)
        a.set_xlabel(r"Re $z/s$")
        a.set_ylabel(r"Im $z/s$", labelpad=1)
    for a in axs[:, 0]:
        a.set_ylabel(r"Im $z$")
    for a in axs[1, :3]:
        a.set_xlabel(r"Re $z$")
    leg = [Patch(facecolor=STAB_FILL, edgecolor="k", lw=0.6, label=r"stable, $|P(z)|\leq1$"),
           Patch(facecolor=PREF_FILL, edgecolor=PREF_EDGE, lw=0.8, label=r"preferred, tol $=1$")]
    fig.legend(handles=leg, loc="lower center", ncol=2, fontsize=7)
    fig.tight_layout(rect=(0, 0.045, 1, 1), w_pad=0.3, h_pad=0.6)
    save(fig, "fig_prefregions")

if __name__ == "__main__":
    main()
