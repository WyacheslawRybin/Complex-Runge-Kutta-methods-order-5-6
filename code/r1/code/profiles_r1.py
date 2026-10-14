"""Performance profiles of the variable-step pairs (review, Section 5, item 1; Dolan and More 2002).

For a set of pairs M and the ten problems P of the variable-step experiments, let c(m, q) be the number
of right-hand side evaluations (rejected steps included, kappa = 1) that pair m needs to reach the
accuracy eps on problem q, read off the work--precision curve as in Tables adaptive6/8/10, and
c(m, q) = inf if the accuracy was not reached for tol >= 1e-15.  The profile of m is
    phi_m(omega) = #{q : c(m, q) <= omega * min_m' c(m', q)} / |P|,
i.e., the fraction of problems on which m is within a factor omega of the cheapest pair of its group.
Groups: the 5(4) pairs and the order-six pairs, including the real pairs Tsit5(4), BS5(4) and Vern6(5)
added in R1 (code/r1patch.py); accuracies 1e-6, 1e-8, 1e-10.

Writes figures/fig_profiles.pdf/.png and data/profiles.json; prints phi(1), phi(1.1), phi(1.25), phi(1.5).
"""
import json
import math
import os

import numpy as np

import r1common as R
import r1patch
r1patch.apply()
import figures as F
import figures_cost as FC

F.FIG = R.FIG
plt = F.plt
EPS = (1e-6, 1e-8, 1e-10)
TAUS = (1.0, 1.1, 1.25, 1.5)
GROUPS = [("5(4) pairs", FC.ADAPT5), ("order-six pairs", FC.ADAPT6)]
NAME = lambda m: FC.PAIRNAME.get(m, F.lbl(m))


def costs(ad, methods, eps):
    return {m: {q: FC.cost_at(*FC.adapt_curve(ad, m, q)[:2], eps) for q in FC.PREF_ADAPT} for m in methods}


def profile(c, m, tau):
    probs = FC.PREF_ADAPT
    best = {q: min(c[k][q] for k in c) for q in probs}
    return sum(1 for q in probs if math.isfinite(c[m][q]) and c[m][q] <= tau * best[q] * (1 + 1e-12)) / len(probs)


def main():
    ad = json.load(open(os.path.join(r1patch.MERGED, "adaptive.json")))
    out = {}
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.6), sharey=True, layout="constrained")
    tau = np.linspace(1.0, 3.0, 801)
    for row, (title, methods) in enumerate(GROUPS):
        for col, eps in enumerate(EPS):
            c = costs(ad, methods, eps)
            ax = axs[row, col]
            for m in methods:
                rho = [profile(c, m, t) for t in tau]
                col_, mk, ls = F.ST[m]
                ax.step(tau, rho, where="post", color=col_, ls=ls, lw=1.1, label=NAME(m))
                out[f"{eps:g}|{m}"] = {f"{t:g}": profile(c, m, t) for t in TAUS}
                out[f"{eps:g}|{m}"]["failed"] = [q for q in FC.PREF_ADAPT if not math.isfinite(c[m][q])]
            ax.set_xlim(1, 3); ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"{title}, accuracy $10^{{{int(round(math.log10(eps)))}}}$", fontsize=7.5)
            if row == 1:
                ax.set_xlabel(r"factor $\omega$ relative to the cheapest pair", fontsize=7)
            if col == 0:
                ax.set_ylabel(r"fraction of problems $\varphi(\omega)$")
        h, l = axs[row, 0].get_legend_handles_labels()
        axs[row, 2].legend(h, l, loc="lower right", fontsize=6.3, handlelength=2.2)
    F.save(fig, "fig_profiles")
    json.dump(out, open(os.path.join(R.DATA, "profiles.json"), "w"), indent=1)
    for title, methods in GROUPS:
        print(f"\n{title}: phi(omega) for omega = {', '.join(map(str, TAUS))}")
        for eps in EPS:
            for m in methods:
                v = out[f"{eps:g}|{m}"]
                print(f"  {eps:g}  {NAME(m):12s} " + "  ".join(f"{v[f'{t:g}']:.1f}" for t in TAUS)
                      + (f"   failed: {', '.join(v['failed'])}" if v["failed"] else ""))


if __name__ == "__main__":
    main()
