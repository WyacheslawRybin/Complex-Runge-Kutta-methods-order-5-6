"""
All tableaux of the paper, read from their primary sources so that nothing is
retyped by hand:

  CRK5-s, CRK5-1, CRK5-2, GJM5     appendix of crk5_paper_U1152.tex (CRK5-s is "CRK5-opt" there)
  CRK6-0, CRK6-s, CRK6-lam         appendix of crk6_paper_U1152.tex (CRK6-s is "CRK6-opt" there)
  CRK5-opt, CRK6-opt               minimum of E_{p+1} under max|coef| <= 5 (optimize_E.py,
                                   front_E.py, polish_opt.py -> data/tableaux_opt.json, 45 digits)
  RK6-opt (real, 7 stages)         appendix of crk6_paper_U1152.tex
  CCK6(5), CCK6(5)-s, CCK6-casc    complex_cash_karp_{65,full}.json (50 digits)
  DP5, CK5, Butcher6, DP6(5)8M,    rk_tableaux.json (exact rationals)
  Verner6(5)8, RK4
"""
import json, os, re
from fractions import Fraction
import mpmath as mp
from crk import Tableau, to_mpc

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CRK5_TEX = os.path.join(ROOT, "_CRK5__The_structure_of_the_five_stage_fifth_order_family_of_complex_coefficient_Runge_Kutta_methods_with_real_projection", "crk5_paper_U1152.tex")
CRK6_TEX = os.path.join(ROOT, "_CRK6__Sixth_order_at_six_stages__the_six_stage_family_of_complex_coefficient_Runge_Kutta_methods_with_real_projection", "crk6_paper_U1152.tex")
CCK_DIR = os.path.join(ROOT, "report_figs")
RK_JSON = os.path.join(ROOT, "rk_tableaux.json")

NUM = r"[+-]?\s*\d[\d.]*(?:e[+-]?\d+)?"
LINE = re.compile(r"([ab])_\{(\d)(\d)?\}\s*&=\s*(" + NUM + r")(?:\s*([+-])\s*(\d[\d.]*(?:e[+-]?\d+)?)\\,\\mathrm\{i\})?")

def _parse_block(text):
    A, b = {}, {}
    for m in LINE.finditer(text):
        kind, i, j, re_, sgn, im = m.groups()
        re_ = re_.replace(" ", "")
        im = (sgn + im) if im else "0"
        if kind == "a":
            A[(int(i), int(j))] = (re_, im)
        else:
            b[int(i)] = (re_, im)
    return A, b

def _section(tex, start_marker, end_marker):
    i = tex.index(start_marker)
    j = tex.index(end_marker, i + len(start_marker))
    return tex[i:j]

def _build(name, Ad, bd, s, projected=True):
    A = [[Ad.get((i + 1, j + 1), "0") if j < i else "0" for j in range(s)] for i in range(s)]
    A = [[to_mpc(v) if isinstance(v, tuple) else mp.mpc(v) for v in row] for row in A]
    b = [to_mpc(bd[i + 1]) for i in range(s)]
    return Tableau(name, A, [(s, None, b)], projected)

def _set_order(T, p):
    m, _, b = T.levels[0]
    T.levels = [(m, p, b)]
    return T

def crk5():
    tex = open(CRK5_TEX, encoding="utf-8").read()
    marks = [r"\subsection*{$\Topt$", r"\subsection*{$\Tfirst$", r"\subsection*{$\Talt$",
             r"\subsection*{$\Tref$", r"\subsection*{Classical RK4"]
    names = ["CRK5-s", "CRK5-1", "CRK5-2", "GJM5"]
    out = {}
    for k, nm in enumerate(names):
        Ad, bd = _parse_block(_section(tex, marks[k], marks[k + 1]))
        assert len(Ad) == 10 and len(bd) == 5, (nm, len(Ad), len(bd))
        out[nm] = _set_order(_build(nm, Ad, bd, 5), 5)
    return out

def crk6():
    tex = open(CRK6_TEX, encoding="utf-8").read()
    marks = [r"\subsection*{$\Tgiven$}", r"\subsection*{$\Topt$}", r"\subsection*{$\Tpen$}",
             r"\subsection*{$\Treal$", r"\subsection*{$\Tropt$", r"\section{Test problems}"]
    out = {}
    for k, nm in [(0, "CRK6-0"), (1, "CRK6-s"), (2, "CRK6-lam")]:
        Ad, bd = _parse_block(_section(tex, marks[k], marks[k + 1]))
        assert len(Ad) == 15 and len(bd) == 6, (nm, len(Ad), len(bd))
        out[nm] = _set_order(_build(nm, Ad, bd, 6), 6)
    Ad, bd = _parse_block(_section(tex, marks[4], marks[5]))
    assert len(Ad) == 21 and len(bd) == 7, ("RK6-opt", len(Ad), len(bd))
    out["RK6-opt"] = _set_order(_build("RK6-opt", Ad, bd, 7, projected=False), 6)
    return out

def _cck(fn, tag, name):
    import sys
    sys.path.insert(0, CCK_DIR)
    d = json.load(open(os.path.join(ROOT, fn)))
    levels = [tuple(l) for l in d["levels"]]
    x = d["picks"][tag]["x"]
    # layout of crk_cascade.Problem: complex unknowns = 15 A entries, then the
    # weights of every level; real vector = Re(all), Im(all) minus Im w1 of each level
    S = 6
    aidx = [(i, j) for i in range(S) for j in range(i)]
    woff, off = [], len(aidx)
    for m, _ in levels:
        woff.append(off); off += m
    nz = off
    drop = {nz + o for o in woff}
    keep = [q for q in range(2 * nz) if q not in drop]
    xx = ["0"] * (2 * nz)
    for q, v in zip(keep, x):
        xx[q] = v
    z = [mp.mpc(mp.mpf(xx[k]), mp.mpf(xx[nz + k])) for k in range(nz)]
    A = [[mp.mpc(0)] * S for _ in range(S)]
    for k, (i, j) in enumerate(aidx):
        A[i][j] = z[k]
    lv = [(m, p, z[o:o + m]) for o, (m, p) in zip(woff, levels)]
    return Tableau(name, A, lv, True)

def cck6():
    return {"CCK6(5)": _cck("complex_cash_karp_65.json", "Cmin", "CCK6(5)"),
            "CCK6(5)-s": _cck("complex_cash_karp_65.json", "coefmin", "CCK6(5)-s"),
            "CCK6-casc": _cck("complex_cash_karp_full.json", "Cmin", "CCK6-casc")}

def _rk_json(key, name, cost=None, embedded=False):
    d = json.load(open(RK_JSON, encoding="utf-8"))[key]
    s = d["stages"]
    A = [[Fraction(d["A_exact"][i][j]) if j < len(d["A_exact"][i]) else 0 for j in range(s)] for i in range(s)]
    A = [[mp.mpc(mp.mpf(v.numerator) / v.denominator) if isinstance(v, Fraction) else mp.mpc(0) for v in row] for row in A]
    fr = lambda v: mp.mpc(mp.mpf(Fraction(v).numerator) / Fraction(v).denominator)
    levels = [(s, d["order"], [fr(v) for v in d["b_exact"]])]
    if embedded and d.get("bhat_exact"):
        levels.append((s, d["embedded_order"], [fr(v) for v in d["bhat_exact"]]))
    return Tableau(name, A, levels, projected=False, cost=cost or s)

def _butcher6():
    """The seven-stage sixth-order method with rational coefficients used in the
    CRK6 draft and the CCK6 report (c = 0, 1/3, 2/3, 1/3, 5/6, 1/6, 1)."""
    F = Fraction
    a = {(2, 1): F(1, 3), (3, 2): F(2, 3), (4, 1): F(1, 12), (4, 2): F(1, 3), (4, 3): F(-1, 12),
         (5, 1): F(25, 48), (5, 2): F(-55, 24), (5, 3): F(35, 48), (5, 4): F(15, 8),
         (6, 1): F(3, 20), (6, 2): F(-11, 24), (6, 3): F(-1, 8), (6, 4): F(1, 2), (6, 5): F(1, 10),
         (7, 1): F(-261, 260), (7, 2): F(33, 13), (7, 3): F(43, 156), (7, 4): F(-118, 39),
         (7, 5): F(32, 195), (7, 6): F(80, 39)}
    b = [F(13, 200), F(0), F(11, 40), F(11, 40), F(4, 25), F(4, 25), F(13, 200)]
    q = lambda v: mp.mpc(mp.mpf(v.numerator) / v.denominator)
    A = [[q(a.get((i + 1, j + 1), F(0))) for j in range(7)] for i in range(7)]
    return Tableau("Butcher6", A, [(7, 6, [q(v) for v in b])], projected=False)

def real_methods():
    out = {
        "DP5": _rk_json("DOPRI5", "DP5", cost=6, embedded=True),
        "CK5": _rk_json("CashKarp5", "CK5", embedded=True),
        "Butcher6": _butcher6(),
        "DP6(5)": _rk_json("DormandPrince6", "DP6(5)", embedded=True),
        "Verner6(5)": _rk_json("Verner6_1978", "Verner6(5)", embedded=True),
    }
    h = mp.mpf(1) / 2
    out["RK4"] = Tableau("RK4", [[0, 0, 0, 0], [h, 0, 0, 0], [0, h, 0, 0], [0, 0, 1, 0]],
                         [(4, 4, [mp.mpf(1) / 6, mp.mpf(1) / 3, mp.mpf(1) / 3, mp.mpf(1) / 6])], projected=False)
    return out

def load_refined(fn=None):
    """Tableaux after Newton refinement (written by check_tableaux.py, 45 digits)."""
    fn = fn or os.path.join(os.path.dirname(__file__), "..", "data", "tableaux.json")
    d = json.load(open(fn))
    out = {}
    for name, info in d.items():
        s = info["s"]
        A = [[mp.mpc(0)] * s for _ in range(s)]
        for i in range(s):
            for j in range(i):
                A[i][j] = mp.mpc(mp.mpf(info["A"][i][j][0]), mp.mpf(info["A"][i][j][1]))
        levels = [(m, p, [mp.mpc(mp.mpf(v[0]), mp.mpf(v[1])) for v in b])
                  for (m, p), b in zip(info["levels"], info["b"])]
        out[name] = Tableau(name, A, levels, info["projected"], cost=info["cost"])
    return out

def emin():
    """CRK5-opt and CRK6-opt: minimum of E_{p+1} under max|coef| <= 5 (polish_opt.py, 45 digits)."""
    return load_refined(os.path.join(os.path.dirname(__file__), "..", "data", "tableaux_opt.json"))

def all_tableaux():
    T = {}
    T.update(emin()); T.update(crk5()); T.update(crk6()); T.update(cck6()); T.update(real_methods())
    return T

if __name__ == "__main__":
    mp.mp.dps = 50
    for k, T in all_tableaux().items():
        r = max(abs(v) for lev in T.residuals() for v in lev)
        print(f"{k:12s} s={T.s} levels={[(m, p) for m, p, _ in T.levels]} resid={mp.nstr(r, 3):>9s} "
              f"E={mp.nstr(T.E(), 6):>12s} maxcoef={mp.nstr(T.maxcoef(), 6)}")
