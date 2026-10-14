"""Rank of the Jacobian of all projected order conditions at every CRK5 and CRK6
tableau visited by the optimization runs (data/front_E.json, data/opt_search.json,
data/Emin_<FAM>_100.json).  Double precision; the rank is read off
of the singular values (relative threshold TOL).  Writes ../data/rank_survey.json and prints a summary.

CRK5: 17 conditions, 29 free real unknowns; rank 17 <=> rank Dg = 8, local dimension 12.
CRK6: 37 conditions, 41 free real unknowns; rank 31 on the ten-dimensional component.
"""
import json, os, sys, collections
import numpy as np
from optimize_E import load_module

TOL = 1e-9          # relative threshold on the singular values (double precision)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def points(fam):
    out = []
    for ch in json.load(open(os.path.join(DATA, "front_E.json"))):
        if ch["family"] == fam:
            out += [("front:" + ch["chain"], p) for p in ch["points"]]
    out += [("opt:" + p["start"], p) for p in json.load(open(os.path.join(DATA, "opt_search.json")))[fam]]
    fn = os.path.join(DATA, "Emin_%s_100.json" % fam)
    if os.path.exists(fn):
        out += [("Emin100:" + str(p.get("start", "")), p) for p in json.load(open(fn))["results"]]
    return out


def survey(fam):
    TC = load_module(fam)
    rows = []
    for label, p in points(fam):
        x = np.array(p["x"], float)
        r, J = TC.residual_and_jacobian(x)
        sv = np.linalg.svd(J, compute_uv=False)
        rel = sv / sv[0]
        k = int(np.sum(rel > TOL))                         # numerical rank
        rows.append(dict(label=label, E=TC.error_constant(x), maxcoef=float(TC.max_coef(x)),
                         resid=float(np.abs(r).max()), rank=k,
                         smallest_kept=float(rel[k - 1]), largest_dropped=float(rel[k]) if k < len(rel) else 0.0,
                         n=int(J.shape[1]), m=int(J.shape[0])))
        rows[-1].update(embedded_kernel(TC, x))
    return rows


def embedded_kernel(TC, x):
    """Kernel of M_q(A), q = p - 1, without the inert directions (Im b_1, and Im b_2 if c_2 is real)."""
    A, W = TC.unpack(x)
    s, q = TC.S, TC.LEV[0][1] - 1
    ph = TC.phi_all(A, q, deriv=False)
    V = np.array([ph[t][0][:s] for k in range(1, q + 1) for t in TC.TREES[k]])
    M = np.hstack([V.real, -V.imag])
    drop = [s] + ([s + 1] if abs((A.sum(axis=1))[1].imag) < 1e-12 else [])
    M = np.delete(M, drop, axis=1)
    sv = np.linalg.svd(M, compute_uv=False)
    rk = int(np.sum(sv > TOL * sv[0]))
    return dict(emb_q=q, emb_unknowns=int(M.shape[1]), emb_rank=rk, emb_kernel=int(M.shape[1] - rk),
                emb_smallest_kept=float(sv[rk - 1] / sv[0]))


if __name__ == "__main__":
    res = {}
    for fam in ("CRK5", "CRK6"):
        rows = survey(fam)
        res[fam] = rows
        cnt = collections.Counter(r["rank"] for r in rows)
        small = [r for r in rows if r["maxcoef"] <= 5.0]
        print("%s: %d tableaux, ranks %s; with max|coef| <= 5: %d tableaux, ranks %s; max residual %.1e"
              % (fam, len(rows), dict(cnt), len(small), dict(collections.Counter(r["rank"] for r in small)),
                 max(r["resid"] for r in rows)))
        for k in sorted(cnt):
            g = [r for r in rows if r["rank"] == k]
            print("   rank %d: smallest kept sv/sv0 >= %.1e, largest dropped <= %.1e, max|coef| %.2f-%.1f"
                  % (k, min(r["smallest_kept"] for r in g), max(r["largest_dropped"] for r in g),
                     min(r["maxcoef"] for r in g), max(r["maxcoef"] for r in g)))
        print("   embedded order %d: kernel dimensions %s, smallest kept sv/sv0 %.1e"
              % (rows[0]["emb_q"], dict(collections.Counter(r["emb_kernel"] for r in rows)),
                 min(r["emb_smallest_kept"] for r in rows)))
    json.dump(res, open(os.path.join(DATA, "rank_survey.json"), "w"), indent=1)
