"""Figure fig_measured: measured error constants of all methods relative to CRK5-opt (order five) and
CRK6-opt (order six) on the eleven nonlinear problems.  It replaces Tables measured5/measured6 in the main
text (they move to the appendix).  The constants are those of the tables (code/tables_r1.py,
classify() of the original code): degenerate constants are omitted, the bound for Vern6(5) on the
non-autonomous problem is plotted with an open marker.
Writes paper/figures/fig_measured.pdf/.png.
"""
import json
import os

import numpy as np

import r1common as R
import r1patch

MT = r1patch.apply()
import figures as F  # noqa: E402
from results import classify  # noqa: E402

F.FIG = R.FIG
plt = F.plt
F.ST.update(r1patch.STYLE)

M5 = ["CRK5-opt", "CRK5-s", "CRK5-1", "CRK5-2", "GJM5", "CCK6(5)/5", "DP5", "CK5", "Tsit5(4)", "BS5(4)"]
M6 = ["CRK6-opt", "CRK6-s", "CRK6-0", "CRK6-lam", "CCK6(5)", "CCK6(5)-s", "CCK6-casc", "RK6-opt", "DP6(5)",
      "Verner6(5)", "Vern6(5)"]
REAL = {"DP5", "CK5", "Tsit5(4)", "BS5(4)", "Butcher6", "RK6-opt", "DP6(5)", "Verner6(5)", "Vern6(5)"}
LBL = dict(F.LBL, **{"CCK6(5)/5": "CCK6(5), level 5"})
PROBS = [p for p in MT.PROBS if p not in ("Dahlquist", "pendulum")]
SHORT = {"quadratic": r"$y'=-y^2$", "exponential": r"$y'=-e^{y}$", "nonautonomous": "non-aut.",
         "VanderPol": "Van der Pol", "Rossler": "Rössler", "Bouali": "Bouali", "Lorenz05": "Lorenz $T{=}0.5$",
         "Lorenz5": "Lorenz $T{=}5$", "Kepler": "Kepler", "RigidBody": "rigid body", "Brusselator": "Brusselator"}


def constants(meas, m):
    out = {}
    for p in PROBS:
        v = classify(meas[m][p]["rows"], MT.ORDER_OF(m))
        if m == "Vern6(5)" and p == "nonautonomous":
            rows = sorted(meas[m][p]["rows"], key=lambda r: r["N"])
            out[p] = (max(abs(r["Ch"]) for r in rows[-5:]), "bound")
        elif v["cls"] != "deg":
            out[p] = (v["C"], v["cls"])
    return out


def panel(ax, meas, methods, ref, title):
    cref = constants(meas, ref)
    n = len(methods)
    for k, m in enumerate(methods):
        c = constants(meas, m)
        col, mk, _ = F.ST.get(m.split("/")[0], ("#444444", "o", "-"))
        xs, ys, open_ = [], [], []
        for i, p in enumerate(PROBS):
            if p in c and p in cref:
                xs.append(i + (k - (n - 1) / 2) * 0.06)
                ys.append(c[p][0] / cref[p][0])
                open_.append(c[p][1] == "bound")
        xs, ys, open_ = np.array(xs), np.array(ys), np.array(open_, bool)
        if mk in ("x", "+", "."):                       # unfilled markers take the face colour
            ax.scatter(xs[~open_], ys[~open_], s=22, marker=mk, c=col, linewidths=1.0, label=LBL.get(m, m), zorder=3)
        else:
            face = "none" if m in REAL else col
            ax.scatter(xs[~open_], ys[~open_], s=16, marker=mk, facecolors=face, edgecolors=col, linewidths=0.8,
                       label=LBL.get(m, m), zorder=3)
        if open_.any():
            ax.scatter(xs[open_], ys[open_], s=16, marker=mk, facecolors="none", edgecolors=col, linewidths=0.5,
                       linestyle=":", zorder=3)
    ax.axhline(1.0, color="#999999", lw=0.6, zorder=1)
    ax.set_yscale("log")
    ax.set_xticks(range(len(PROBS)))
    ax.set_xticklabels([SHORT.get(p, p) for p in PROBS], rotation=40, ha="right")
    ax.set_xlim(-0.6, len(PROBS) - 0.4)
    ax.grid(axis="x", visible=False)
    ax.set_title(title)
    ax.legend(ncol=1, loc="center left", bbox_to_anchor=(1.0, 0.5), handletextpad=0.2, borderaxespad=0.3)


def main():
    meas = json.load(open(os.path.join(r1patch.MERGED, "measured.json")))
    fig, ax = plt.subplots(2, 1, figsize=(7.0, 5.2))
    panel(ax[0], meas, M5, "CRK5-opt", "(a) order five: constant relative to CRK5-opt")
    panel(ax[1], meas, M6, "CRK6-opt", "(b) order six: constant relative to CRK6-opt")
    for a in ax:
        a.set_ylabel(r"$K/K_{\mathrm{ref}}$")
    fig.tight_layout()
    F.save(fig, "fig_measured")


if __name__ == "__main__":
    main()
