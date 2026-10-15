"""Integration of the R1 data with the table and figure code of the original submission.

R1 adds three real FSAL pairs to every comparison (review, Section 5, item 4): Tsit5(4) [Tsitouras
2011], BS5(4) [Bogacki and Shampine 1996] and Vern6(5) [Verner 2010] (code/real_pairs_r1.py).  Their
data were computed by the R1 scripts in the format of the original data files:
  data/measured_r1.json   fixed-step constants, 34 digits     (measure_r1.py)
  data/adaptive_r1.json   variable steps, all pairs           (adaptive_new_r1.py)
  data/tableaux_r1.json   coefficients, 80 digits             (real_pairs_r1.py)
build() merges them with the original data into R1/data/merged/ (the original files are not modified),
and apply() points the modules of the original code (figures, figures_cost, make_tables, results) to the
merged data and adds the new pairs to their method lists, styles and evaluation counts.  Every R1 script
that regenerates a table or figure with the new pairs calls apply() first.
"""
import json
import os
import shutil

import r1common as R

NEW5 = ["Tsit5(4)", "BS5(4)"]
NEW6 = ["Vern6(5)"]
NEW = NEW5 + NEW6
EVALS_NEW = {"Tsit5(4)": 6, "BS5(4)": 7, "Vern6(5)": 8}
CITE = {"Tsit5(4)": r"\cite{Tsitouras11}", "BS5(4)": r"\cite{BS96}", "Vern6(5)": r"\cite{Verner10}"}
STYLE = {"Tsit5(4)": ("#009688", "<", "-."), "BS5(4)": ("#c51b7d", ">", "-."), "Vern6(5)": ("#111111", "X", "-")}
MERGED = os.path.join(R.DATA, "merged")
INFO_FN = os.path.join(R.DATA, "new_pairs_info.json")


def order_of(lab):
    """Order of the propagated solution; results.ORDER_OF of the original code knows only the original names."""
    if lab in ("Tsit5(4)", "BS5(4)"):
        return 5
    if lab in ("Vern6(5)", "RK6-opt(4)"):
        return 6
    return _ORDER_OF0(lab)


def new_pairs_info(recompute=False):
    """Invariants of the new pairs, computed as in code/check_tableaux.py (60 digits) and code/embedded.py."""
    if os.path.exists(INFO_FN) and not recompute:
        return json.load(open(INFO_FN))
    import mpmath as mp
    from tableaux import load_refined
    from check_tableaux import jac_rank
    from embedded import real_pair_measures
    mp.mp.dps = 60
    src = json.load(open(os.path.join(R.DATA, "tableaux_r1.json")))
    TT = load_refined(os.path.join(R.DATA, "tableaux_r1.json"))
    out = {}
    for k in NEW:
        T, s = TT[k], src[k]
        r = float(max(abs(v) for lev in T.residuals() for v in lev))
        rk, n, _ = jac_rank(T)
        tab = dict(s=s["s"], cost=s["cost"], projected=False, levels=s["levels"], resid_printed=r,
                   resid_refined=r, refine_shift=0.0, E=s["E"], maxcoef=s["maxcoef"], full_complex_resid=0.0,
                   jac_rank=rk, n_free=n, local_dim=n - rk, M_rank=None, c=[[mp.nstr(mp.re(v), 25), "0"] for v in T.c],
                   A=s["A"], b=s["b"])
        emb = {a: float(v) for a, v in real_pair_measures(T).items()}
        out[k] = dict(tableau=tab, embedded=emb)
        print(f"{k:9s} resid {r:.1e}  rank J = {rk}/{n}, dim {n - rk}  Ahat {emb['Ahat']:.3e} B {emb['B']:.2f} "
              f"C {emb['C']:.2f} E {emb['E']:.2f} max|bh| {emb['maxbh']:.3f}", flush=True)
    json.dump(out, open(INFO_FN, "w"), indent=1)
    return out


def build():
    """Write R1/data/merged/: the original data files with the new pairs added."""
    os.makedirs(MERGED, exist_ok=True)
    o = lambda f: json.load(open(os.path.join(R.ORIG_DATA, f)))
    meas = o("measured.json")
    meas.update(json.load(open(os.path.join(R.DATA, "measured_r1.json"))))
    ad = json.load(open(os.path.join(R.DATA, "adaptive_r1.json")))
    # tolerances 10^-13.5 ... 1e-15 (adaptive_tol15_r1.py), appended to every curve
    ext = os.path.join(R.DATA, "adaptive_tol15_r1.json")
    if os.path.exists(ext):
        for name, probs in json.load(open(ext)).items():
            for prob, rows in probs.items():
                ad[name][prob] = [r for r in ad[name][prob] if r["tol"] > 10 ** -13.25] + rows
    # Vern6(5) with split double-precision coefficients (vern6_split_r1.py) replaces the runs with plainly
    # rounded coefficients, whose defect in the condition of order two caused a floor near 1e-10
    split = os.path.join(R.DATA, "adaptive_vern6split_r1.json")
    if os.path.exists(split):
        ad.update(json.load(open(split)))
    # the real FSAL pair 6(4) on the seven stages of RK6-opt (real64_r1.py), the counterpart of CRK6(4)
    r64 = os.path.join(R.DATA, "adaptive_real64_r1.json")
    if os.path.exists(r64):
        ad.update({k: {q: rows for q, rows in v.items() if q != "Bruss2D"} for k, v in json.load(open(r64)).items()})
    info = new_pairs_info()
    tab = o("tableaux.json")
    emb = o("embedded.json")
    for k in NEW:
        tab[k] = info[k]["tableau"]
        emb[k] = info[k]["embedded"]
    for name, d in (("measured.json", meas), ("adaptive.json", ad), ("tableaux.json", tab), ("embedded.json", emb)):
        json.dump(d, open(os.path.join(MERGED, name), "w"), indent=0)
    for f in ("stability.json", "prefregions.json", "tuned.json"):
        shutil.copy(os.path.join(R.ORIG_DATA, f), MERGED)
    # CPU times: the R1 runs (all methods in the same runs) once they exist, otherwise the original ones
    b1 = os.path.join(R.DATA, "bench.csv")
    shutil.copy(b1 if os.path.exists(b1) else os.path.join(R.ORIG_DATA, "bench.csv"), os.path.join(MERGED, "bench.csv"))
    return MERGED


_ORDER_OF0 = None
_APPLIED = False


def apply():
    """Point the original modules to the merged data and add the new pairs to their lists (idempotent)."""
    global _ORDER_OF0, _APPLIED
    import figures as F
    import figures_cost as FC
    import make_tables as MT
    import results as RS
    if _APPLIED:
        return MT
    build()
    _ORDER_OF0 = RS.ORDER_OF
    RS.ORDER_OF = MT.ORDER_OF = order_of
    RS.COST.update(EVALS_NEW)
    F.ST.update(STYLE)
    F.FIG = R.FIG
    F.DATA = FC.DATA = MT.DATA = MERGED
    MT.TAB = R.TAB
    FC.EVALS.update(EVALS_NEW)
    FC.BENCH_KEY.update({k: k for k in NEW})
    MT.PAIR_OF.update({k: k for k in NEW})
    # RK6-opt(4): column of the order-six pairs, CPU time per step of RK6-opt (same seven evaluations)
    MT.PAIR_OF["RK6-opt"] = "RK6-opt(4)"
    MT.MNAME["RK6-opt(4)"] = "RK6-opt(4)"
    FC.BENCH_KEY["RK6-opt(4)"] = "RK6-opt"
    F.ST["RK6-opt(4)"] = F.ST["RK6-opt"]
    RS.COST["RK6-opt(4)"] = 7
    for lst in (FC.ADAPT6, FC.ADAPT):
        if "RK6-opt(4)" not in lst:
            lst.insert(lst.index("DP6(5)"), "RK6-opt(4)")
    # method lists: in place, since make_tables imported the list objects by name
    for lst, after, items in ((FC.FIXED5, "CK5", NEW5), (FC.FIXED6, "Verner6(5)", NEW6),
                              (FC.FIG5, "CK5", NEW5), (FC.FIG6, "Verner6(5)", NEW6),
                              (FC.ADAPT5, "CK5", NEW5), (FC.ADAPT6, "Verner6(5)", NEW6),
                              (FC.ADAPT, "CK5", NEW5), (FC.ADAPT, "Verner6(5)", NEW6),
                              (FC.FIXED, "DP6(5)", ["Verner6(5)"] + NEW6), (FC.FIXED, "CK5", NEW5)):
        i = lst.index(after) + 1
        lst[i:i] = [m for m in items if m not in lst]
    _APPLIED = True
    return MT


if __name__ == "__main__":
    import sys
    new_pairs_info(recompute="recompute" in sys.argv)
    apply()
    import figures_cost as FC
    print("merged data in", MERGED)
    print("FIXED5", FC.FIXED5, "\nFIXED6", FC.FIXED6, "\nADAPT5", FC.ADAPT5, "\nADAPT6", FC.ADAPT6, "\nADAPT", FC.ADAPT)
