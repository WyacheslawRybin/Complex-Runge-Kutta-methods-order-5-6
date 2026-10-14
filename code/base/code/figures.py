"""
All figures of the paper (PDF), in one style and with the final notation.
Usage:  python figures.py [name ...]      (no argument: all figures)
"""
import json, os, sys
from math import factorial
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from matplotlib.ticker import NullFormatter, LogLocator, FormatStrFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(HERE, "..", "data")
FIG = os.path.join(HERE, "..", "figures")
OLD5 = os.path.join(ROOT, "old", "crk5_paper")
OLD6 = os.path.join(ROOT, "old", "crk6_figures")
CCK = os.path.join(ROOT, "report_figs")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5, "legend.fontsize": 7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.5, "lines.linewidth": 1.2,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42})

# fixed colour / marker / line style for every method
ST = {
    "CRK5-opt": ("#1f5fbf", "o", "-"), "CRK5-1": ("#5b8fd9", "o", "--"), "CRK5-2": ("#8fb3e8", "o", ":"),
    "GJM5": ("#9aa7b8", "o", "-."), "CRK5(4)": ("#1f5fbf", "o", "-"), "CRK5-s": ("#2bb3c9", "d", "--"),
    "CRK6-opt": ("#1a8a4a", "^", "-"), "CRK6-0": ("#55b57d", "^", "--"), "CRK6-lam": ("#8fd1a8", "^", ":"),
    "CRK6(4)": ("#1a8a4a", "^", "-"), "CRK6-s": ("#a3b81a", "d", "--"),
    "CRK5-s(4)": ("#2bb3c9", "d", "--"), "CRK6-s(4)": ("#a3b81a", "d", "--"),
    "CCK6(5)": ("#d1495b", "s", "-"), "CCK6(5)-s": ("#e8888f", "s", "--"), "CCK6-casc": ("#e07b14", "D", "-"),
    "DP5": ("#222222", "x", "--"), "CK5": ("#777777", "+", "--"), "RK4": ("#aaaaaa", ".", ":"),
    "Butcher6": ("#8e5bb5", "v", ":"), "RK6-opt": ("#6a2c91", "P", "-."), "DP6(5)": ("#8c5a2b", "*", "--"),
    "Verner6(5)": ("#b59a2b", "h", ":"),
}
LBL = {"CRK6-lam": r"CRK6-$\lambda$", "CCK6-casc": "CCK6-cascade", "DP5": "DP5(4)", "CK5": "CK5(4)"}
lbl = lambda k: LBL.get(k, k)

def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=200)
    plt.close(fig)
    print("wrote", name)

def read_csv(fn):
    rows = [l for l in open(fn, encoding="utf-8") if not l.startswith("#")]
    head = rows[0].strip().split(",")
    out = {h: [] for h in head}
    for l in rows[1:]:
        for h, v in zip(head, l.strip().split(",")):
            try:
                out[h].append(float(v))
            except ValueError:
                out[h].append(v)
    return out

# ---------------------------------------------------------------- 1. singular-value spectra
def fig_spectrum():
    tab = json.load(open(os.path.join(DATA, "tableaux.json")))
    s5 = read_csv(os.path.join(OLD5, "fig1_spectrum.csv"))
    s6 = read_csv(os.path.join(OLD6, "fig1_singular_values.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.35))
    v = np.array(s5["singular_value"])
    ax[0].semilogy(np.arange(1, len(v) + 1), v, "o", color=ST["CRK5-opt"][0], ms=3.5)
    ax[0].axvline(8.5, color="#999999", lw=0.7, ls="--")
    ax[0].set_title(r"(a) CRK5-s: $D\rho$, $20$ unknowns")
    for key, lab, mk in [("sv_M_37x12", r"$M(A)$, $37\times12$", "s"), ("sv_Dg_26x30", r"$Dg$, $26\times30$", "o"),
                         ("sv_fullJacobian_37x42", r"full $J$, $37\times42$", "^")]:
        w = np.array([x for x in s6[key] if x != "" and x > 0])
        ax[1].semilogy(np.arange(1, len(w) + 1), w, mk, ms=3, label=lab, mfc="none")
    ax[1].set_title("(b) CRK6-0")
    ax[1].legend(loc="center right")
    for name, mk, col in [("CCK6(5)", "s", ST["CCK6(5)"][0]), ("CCK6-casc", "D", ST["CCK6-casc"][0]),
                          ("CRK6-0", "^", ST["CRK6-0"][0])]:
        w = np.array(tab[name]["jac_sv"]); w = np.maximum(w, 1e-30)
        ax[2].semilogy(np.arange(1, len(w) + 1), w, mk, ms=3, color=col, mfc="none",
                       label=f"{lbl(name)}: rank {tab[name]['jac_rank']}/{tab[name]['n_free']}")
    ax[2].set_title("(c) Jacobian of all conditions")
    ax[2].legend(loc="center right")
    for a in ax:
        a.set_xlabel("index")
    ax[0].set_ylabel("singular value")
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_spectrum")

# ---------------------------------------------------------------- 2. continuation
def fig_continuation():
    c5 = read_csv(os.path.join(OLD5, "fig2_continuation.csv"))
    c6 = read_csv(os.path.join(OLD6, "fig2_continuation.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.3))
    for a, d, name, key in [(ax[0], c5, "(a) CRK5, from CRK5-s", "C2"), (ax[1], c6, "(b) CRK6, from CRK6-0", "C2")]:
        s = np.array(d["arclength"])
        a.semilogy(s, d["residual_order_cond"] if "residual_order_cond" in d else d["residual"], "-", color="#444444",
                   label="order-condition residual")
        a.semilogy(s, d[key], "-", color=ST["CCK6(5)"][0], label=r"$E_{p+1}$")
        a.semilogy(s, d["max_abs_coef"], "--", color=ST["CRK5-opt"][0], label=r"$\max|$coef$|$")
        a.set_xlabel("arclength")
        a.set_title(name)
    ax[0].legend(loc="center right")
    fig.tight_layout()
    save(fig, "fig_continuation")

# ---------------------------------------------------------------- 3. Pareto fronts
def _front(x, y):
    o = np.argsort(x); best = np.inf; fx, fy = [], []
    for i in o:
        if y[i] < best:
            best = y[i]; fx.append(x[i]); fy.append(y[i])
    return np.array(fx), np.array(fy)

def _cck_modulus(x, levels):
    S = 6
    aidx = [(i, j) for i in range(S) for j in range(i)]
    woff, off = [], len(aidx)
    for m, _ in levels:
        woff.append(off); off += m
    nz = off
    drop = {nz + o for o in woff}
    keep = [q for q in range(2 * nz) if q not in drop]
    xx = np.zeros(2 * nz); xx[keep] = np.array(x, float)
    z = xx[:nz] + 1j * xx[nz:]
    return float(np.abs(z).max())

def front_E(fam, bmax=100.0):
    """Constrained minimum E_min(B) from data/front_E.json (best over all chains, monotone in B)."""
    d = json.load(open(os.path.join(DATA, "front_E.json")))
    best = {}
    for ch in d:
        if ch["family"] != fam:
            continue
        for p in ch["points"]:
            if p["E"] is None or p["maxcoef"] > p["B"] * 1.001 or p["B"] > bmax:
                continue
            if p["B"] not in best or p["E"] < best[p["B"]]["E"]:
                best[p["B"]] = p
    B = np.array(sorted(best))
    return B, np.minimum.accumulate([best[b]["E"] for b in B])

def fig_pareto():
    tab = json.load(open(os.path.join(DATA, "tableaux.json")))
    fig, ax = plt.subplots(1, 3, figsize=(7.3, 3.25))
    fkw = dict(color="k", lw=1.1, drawstyle="steps-post", label=r"$\min E_{p+1}$ for $\max|$coef$|\leq B$")
    # (a) CRK5
    cl = read_csv(os.path.join(OLD5, "fig3_pareto_cloud.csv"))
    ax[0].scatter(cl["max_abs_coef"], cl["C2"], s=3, color="#bbbbbb", lw=0, label="random walks")
    ax[0].plot(*front_E("CRK5"), **fkw)
    for k in ["CRK5-opt", "CRK5-s", "CRK5-1", "CRK5-2", "GJM5"]:
        ax[0].plot(tab[k]["maxcoef"], tab[k]["E"][0], ST[k][1], color=ST[k][0], ms=5, mec="k", mew=0.4, label=lbl(k))
    ax[0].set_ylabel(r"$E_{p+1}=\|e\|_2$")
    ax[0].set_title("(a) CRK5")
    ax[0].set_ylim(1.2e-3, 4e-2)
    # (b) CRK6
    c6 = read_csv(os.path.join(OLD6, "fig3_pareto_cloud.csv"))
    dims = 42 - np.array(c6["rank_J"])
    for dval, col in [(11, "#1a8a4a"), (8, "#e0a020"), (6, "#7799cc"), (5, "#cc6677")]:
        msk = dims == dval
        if msk.any():
            ax[1].scatter(np.array(c6["max_abs_coef"])[msk], np.array(c6["C2"])[msk], s=7, color=col, lw=0,
                          alpha=0.8, label=f"dim {dval - 1}")
    ax[1].plot(*front_E("CRK6"), **fkw)
    for k in ["CRK6-opt", "CRK6-s", "CRK6-0", "CRK6-lam", "RK6-opt", "Butcher6"]:
        ax[1].plot(tab[k]["maxcoef"], tab[k]["E"][0], ST[k][1], color=ST[k][0], ms=5, mec="k", mew=0.4, label=lbl(k))
    ax[1].set_title("(b) CRK6")
    # (c) CCK6
    for fn, key, levels, lab in [("pareto_65.json", "CCK6(5)", [(6, 6), (5, 5)], "CCK6(5)"),
                                 ("pareto_full.json", "CCK6-casc", [(6, 6), (5, 5), (4, 4), (3, 3), (2, 2), (1, 1)], "CCK6-cascade")]:
        pts = json.load(open(os.path.join(CCK, fn)))
        mx = np.array([_cck_modulus(p["x"], levels) for p in pts]); E = np.array([p["C"][0] for p in pts])
        ok = np.isfinite(E) & (E > 0)
        ax[2].scatter(mx[ok], E[ok], s=4, color=ST[key][0], alpha=0.3, lw=0, label=f"{lab}: descents")
    ax[2].plot(*front_E("CCK65"), **dict(fkw, label=r"pair: $\min E_7$ for $\max|$coef$|\leq B$"))
    for k in ["CCK6(5)", "CCK6(5)-s", "CCK6-casc", "CRK6-opt", "RK6-opt"]:
        ax[2].plot(tab[k]["maxcoef"], tab[k]["E"][0], ST[k][1], color=ST[k][0], ms=5, mec="k", mew=0.4, label=lbl(k))
    ax[2].set_title("(c) CCK6")
    for a in ax:
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel(r"$\max|$coefficient$|$")
        a.set_xlim(0.3, 120)
        a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.0, handletextpad=0.3,
                 columnspacing=0.8, handlelength=1.6)
    ax[1].set_ylim(2.0e-4, 5e-2); ax[2].set_ylim(2.0e-4, 5e-2)
    for a in ax:
        a.xaxis.set_minor_formatter(NullFormatter())
        a.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.5, 1, 2, 5, 10, 20, 50, 100]))
        a.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    fig.tight_layout(w_pad=0.5)
    save(fig, "fig_pareto")

# ---------------------------------------------------------------- 4. degenerate branch
def fig_degenerate():
    d = read_csv(os.path.join(OLD5, "fig4_degenerate.csv"))
    gap = np.array(d["min_abscissa_gap"]); res = np.array(d["residual"]); mb = np.array(d["max_abs_b"])
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.4))
    ax[0].loglog(gap, res, "o", ms=2.5, color=ST["CRK5-opt"][0], mfc="none")
    ax[0].set_xlabel(r"$\min_{i\neq j}|c_i-c_j|$"); ax[0].set_ylabel("residual of the full complex conditions")
    ax[0].set_title("(a)")
    ax[1].loglog(1 / gap, mb, "o", ms=2.5, color=ST["CCK6(5)"][0], mfc="none")
    k = np.median(gap * mb)
    xx = np.geomspace((1 / gap).min(), (1 / gap).max(), 50)
    ax[1].loglog(xx, k * xx, "-", color="k", lw=0.8, label=rf"$\max|b|={k:.3f}/\mathrm{{gap}}$")
    ax[1].set_xlabel(r"$1/\mathrm{gap}$"); ax[1].set_ylabel(r"$\max_i|b_i|$"); ax[1].legend(loc="upper left")
    ax[1].set_title("(b)")
    fig.tight_layout()
    save(fig, "fig_degenerate")

# ---------------------------------------------------------------- 5. stability regions
def fig_stability():
    st = json.load(open(os.path.join(DATA, "stability.json")))
    X, Y = np.meshgrid(np.linspace(-5.2, 1.6, 900), np.linspace(-4.4, 4.4, 1000)); Z = X + 1j * Y
    P = lambda co: np.abs(np.polynomial.Polynomial(co)(Z))
    Tm = lambda m: [1 / factorial(k) for k in range(m + 1)]
    fig, ax = plt.subplots(1, 4, figsize=(7.4, 3.25), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.15]))
    # RK6-opt and DP6(5) have almost the same polynomial: DP6(5) is drawn as a wide light band under RK6-opt
    band = {"DP6(5)": dict(lw=3.2, alpha=0.4, ls="-")}
    panels = [
        ("(a) order 5", [(r"$T_5$: CRK5, CCK6 emb.", Tm(5), "CRK5-opt"), ("DP5(4)", st["DP5"]["coeffs"], "DP5"),
                         ("CK5(4)", st["CK5"]["coeffs"], "CK5")]),
        ("(b) order 6", [(r"$T_6$: CRK6, CCK6", Tm(6), "CCK6(5)"), ("Butcher6", st["Butcher6"]["coeffs"], "Butcher6"),
                         ("RK6-opt", st["RK6-opt"]["coeffs"], "RK6-opt"), ("DP6(5)", st["DP6(5)"]["coeffs"], "DP6(5)")]),
        ("(c) cascade levels", [(rf"$T_{m}$", Tm(m), None) for m in range(1, 5)]),
    ]
    greys = ["#c8c8c8", "#a0a0a0", "#707070", "#404040"]
    for a, (title, curves) in zip(ax[:3], panels):
        a.contourf(X, Y, P(curves[0][1]), levels=[0, 1], colors=["#f0e6e8" if "6" in title else "#e6edf7"])
        for q, (lab, co, key) in sorted(enumerate(curves), key=lambda t: t[1][2] not in band):
            col, _, ls = ST[key] if key else (greys[q], None, "-")
            kw = band.get(key, dict(lw=1.1, alpha=1.0, ls=ls))
            a.contour(X, Y, P(co), [1], colors=[col], linestyles=[kw["ls"]], linewidths=kw["lw"], alpha=kw["alpha"])
        for q, (lab, co, key) in enumerate(curves):
            col, _, ls = ST[key] if key else (greys[q], None, "-")
            kw = band.get(key, dict(lw=1.1, alpha=1.0, ls=ls))
            a.plot([], [], color=col, ls=kw["ls"], lw=kw["lw"], alpha=kw["alpha"], label=lab)
        a.set_aspect("equal"); a.set_title(title); a.set_xlabel(r"Re $z$")
        a.axhline(0, color="#999999", lw=0.4); a.axvline(0, color="#999999", lw=0.4)
        a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1 if len(curves) < 4 else 2,
                 fontsize=6.2, handlelength=2.0, columnspacing=0.8)
        a.set_xlim(-5.2, 1.6); a.set_ylim(-4.4, 4.4)
    ax[0].set_ylabel(r"Im $z$")
    x = np.linspace(-5, 0.5, 2000)
    for lab, co, key in [(r"$T_5$", Tm(5), "CRK5-opt"), (r"$T_6$", Tm(6), "CCK6(5)"), ("DP5(4)", st["DP5"]["coeffs"], "DP5"),
                         ("CK5(4)", st["CK5"]["coeffs"], "CK5"), ("RK6-opt", st["RK6-opt"]["coeffs"], "RK6-opt"),
                         ("Butcher6", st["Butcher6"]["coeffs"], "Butcher6")]:
        col, _, ls = ST[key]
        ax[3].plot(x, np.abs(np.polynomial.Polynomial(co)(x)), color=col, ls=ls, label=lab)
    ax[3].axhline(1, color="k", lw=0.6)
    ax[3].set_ylim(0, 1.6); ax[3].set_xlim(-5, 0.3)
    ax[3].set_xlabel(r"$z\in\mathbb{R}$"); ax[3].set_ylabel(r"$|P(z)|$"); ax[3].set_title("(d) real axis")
    ax[3].legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=6.2, ncol=2, handlelength=2.0,
                 columnspacing=0.8)
    fig.tight_layout(w_pad=0.4)
    save(fig, "fig_stability")

ALL = {"spectrum": fig_spectrum, "continuation": fig_continuation, "pareto": fig_pareto,
       "degenerate": fig_degenerate, "stability": fig_stability}

if __name__ == "__main__":
    names = sys.argv[1:] or list(ALL)
    for n in names:
        ALL[n]()
