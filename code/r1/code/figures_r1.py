"""Figures of the R1 revision that differ from the original submission.

fig_spectrum: singular values from data/spectra_mp.json (90-digit SVD of the Jacobian of all
              projected order conditions; review, Section 4, item 2).
fig_pareto:   the front E_min(B) now includes the minima of the multistart runs with B = 100
              (data/Emin_<FAM>_100.json of the original submission), so that for CRK5 it drops to
              1.38953e-3 at max|coef| = 25.65 (review, Section 4, item 1).
fig_wp_fixed, fig_wp_adaptive, fig_tolerance: the work--precision and tolerance figures of
              code/figures_cost.py with the real pairs Tsit5(4), BS5(4) and Vern6(5) added (code/r1patch.py;
              review, Section 5, item 4); the legends are set in one row per order.
All other figures are those of the original submission.  Output goes to R1/figures/.
Usage:  python figures_r1.py [fig_spectrum] [fig_pareto] [fig_wp_fixed] [fig_adaptive]
"""
import json, os, sys
import numpy as np
import r1common as R
import figures as F

F.FIG = R.FIG
plt = F.plt


FONT_SCALE = 1.3        # figures 4 and 5 of the MDPI version (review of 15.10.2026, M7): larger lettering


def enlarge(fig, k=FONT_SCALE):
    """Scale every text of a figure (titles, labels, tick labels, legends) by k."""
    import matplotlib.text
    for t in fig.findobj(matplotlib.text.Text):
        t.set_fontsize(t.get_fontsize() * k)


def front_E(fam, bmax=100.0):
    """E_min(B) over the bound/continuation chains and the multistart runs with B = 100."""
    d = json.load(open(os.path.join(R.ORIG_DATA, "front_E.json")))
    pts = []
    for ch in d:
        if ch["family"] == fam:
            pts += [(p["B"], p["E"]) for p in ch["points"]
                    if p["E"] is not None and p["maxcoef"] <= p["B"] * 1.001 and p["B"] <= bmax]
    fn = os.path.join(R.ORIG_DATA, "Emin_%s_100.json" % fam)
    if os.path.exists(fn):
        # a multistart minimum with coefficient size c bounds E_min(B) for every B >= c
        pts += [(float(np.ceil(r["maxcoef"] * 100) / 100), r["E"]) for r in json.load(open(fn))["results"]
                if r.get("E") and r["maxcoef"] <= bmax]
    best = {}
    for B, E in pts:
        best[B] = min(E, best.get(B, np.inf))
    B = np.array(sorted(best))
    return B, np.minimum.accumulate([best[b] for b in B])


F.front_E = front_E


def fig_spectrum():
    d = json.load(open(os.path.join(R.DATA, "spectra_mp.json")))
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.45))
    panels = [
        (ax[0], "(a) CRK5: $17$ conditions, $29$ unknowns", [("CRK5-opt", "o"), ("CRK5-s", "d")]),
        (ax[1], "(b) order six: $37$ conditions, $41$ unknowns",
         [("CRK6-opt", "^"), ("CRK6-0", "v"), ("CCK6(5)[6]", "s")]),
        (ax[2], "(c) CCK6 pair and cascade", [("CCK6(5)", "s"), ("CCK6-casc", "D")]),
    ]
    lab = {"CCK6(5)[6]": "CCK6(5), level 6", "CCK6-casc": "CCK6-cascade"}
    col = {"CCK6(5)[6]": "#7a3fb0"}
    for a, title, items in panels:
        for name, mk in items:
            v = d[name]
            sv = np.array([float(x) for x in v["sv"]])
            rel = np.maximum(sv / sv[0], 1e-80)
            c = col.get(name, F.ST.get(name, ("#444444",))[0])
            a.semilogy(np.arange(1, len(rel) + 1), rel, mk, ms=3, color=c, mfc="none",
                       label=f"{lab.get(name, name)}: rank {v['rank']}/{v['n']}")
        a.set_title(title)
        a.set_xlabel("index")
        a.set_ylim(1e-68, 3)
        a.set_yticks([1, 1e-10, 1e-20, 1e-30, 1e-40, 1e-50, 1e-60])
        a.legend(loc="center left", fontsize=6.2)
    ax[0].set_ylabel(r"$\varsigma_k/\varsigma_1$")
    fig.tight_layout(w_pad=0.6)
    F.save(fig, "fig_spectrum")


def fig_pareto():
    F.fig_pareto()


def fig_wp_fixed():
    import r1patch
    r1patch.apply()
    import figures_cost as FC
    meas = json.load(open(os.path.join(r1patch.MERGED, "measured.json")))
    probs = ["VanderPol", "Bouali", "Lorenz5", "Kepler"]

    def curves(m, p):
        N, e = FC.fixed_curve(meas, m, p)
        ok = e < 1
        return FC.EVALS[m] * N[ok], e[ok]
    fig = plt.figure(figsize=(7.4, 8.6), layout="constrained")
    sfs = fig.subfigures(2, 1, hspace=0.04)
    FC._wp_block(sfs[0], curves, FC.FIG5, FC.REF[5], probs, "order 5", ylim=(1e-16, 1), ncol=len(FC.FIG5))
    FC._wp_block(sfs[1], curves, FC.FIG6, FC.REF[6], probs, "order 6", ylim=(1e-16, 1), ncol=len(FC.FIG6))
    F.save(fig, "fig_wp_fixed")


def fig_adaptive():
    """fig_wp_adaptive and fig_tolerance (code/figures_cost.py, fig_adaptive) with the new pairs."""
    import r1patch
    r1patch.apply()
    import figures_cost as FC
    ad = json.load(open(os.path.join(r1patch.MERGED, "adaptive.json")))
    probs = ["Arenstorf", "Kepler", "Lorenz5", "Brusselator"]
    curves = lambda m, p: FC.adapt_curve(ad, m, p)[:2]
    name = lambda m: FC.PAIRNAME.get(m, F.lbl(m))
    fig = plt.figure(figsize=(7.4, 8.6), layout="constrained")
    sfs = fig.subfigures(2, 1, hspace=0.04)
    FC._wp_block(sfs[0], curves, FC.ADAPT5, FC.REF[5], probs, "5(4) pairs", name, "RHS evaluations incl. rejections",
                 ncol=len(FC.ADAPT5))
    FC._wp_block(sfs[1], curves, FC.ADAPT6, FC.REF[6], probs, "order-6 pairs", name, "RHS evaluations incl. rejections",
                 ncol=len(FC.ADAPT6))
    enlarge(fig)
    F.save(fig, "fig_wp_adaptive")
    fig = plt.figure(figsize=(7.2, 5.0), layout="constrained")
    sfs = fig.subfigures(2, 1, hspace=0.04)
    for sf, methods, title in [(sfs[0], FC.ADAPT5, "5(4) pairs"), (sfs[1], FC.ADAPT6, "order-6 pairs")]:
        axs = sf.subplots(1, 3)
        for a, p in zip(axs, ["Arenstorf", "Kepler", "VanderPol"]):
            a.loglog([1e-15, 1e-3], [1e-15, 1e-3], color="#999999", lw=0.6, label="error = tol")
            for m in methods:
                _, e, tol = FC.adapt_curve(ad, m, p)
                col, mk, ls = F.ST[m]
                a.loglog(tol, e, ls=ls, marker=mk, ms=2.2, color=col, label=name(m), mfc="none", lw=1.0)
            a.set_title(f"{title}: {FC.PTITLE[p]}", fontsize=7.5)
            a.set_xlabel("tolerance", fontsize=7, labelpad=1)
        axs[0].set_ylabel(r"error at $T$")
        h, l = axs[0].get_legend_handles_labels()
        sf.legend(h[1:] + h[:1], l[1:] + l[:1], loc="outside lower center", ncol=len(methods) + 1, fontsize=6.6,
                  columnspacing=1.0, handlelength=2.4)
    F.save(fig, "fig_tolerance")


if __name__ == "__main__":
    for name in sys.argv[1:] or ["fig_spectrum", "fig_pareto"]:
        globals()[name]()
    B, E = front_E("CRK5")
    print("CRK5 front:", [(float(b), float("%.6g" % e)) for b, e in zip(B, E) if b >= 1.9])
