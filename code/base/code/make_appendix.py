"""Appendix with all coefficients (refined, 30 significant digits) -> ../tables/appendix_tableaux.tex"""
import json, os
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
TAB = os.path.join(HERE, "..", "tables")
mp.mp.dps = 50
D = 30

def num(re_s, im_s, real_only=False):
    re_v, im_v = mp.mpf(re_s), mp.mpf(im_s)
    def f(x):
        s = mp.nstr(x, D, min_fixed=-6, max_fixed=6, strip_zeros=False)
        return s
    r = ("+" if re_v >= 0 else "-") + f(abs(re_v))
    if real_only or abs(im_v) < mp.mpf(10) ** -40:
        return r
    return r + (" + " if im_v >= 0 else " - ") + f(abs(im_v)) + r"\,\mathrm{i}"

def tex(x, d=2):
    x = mp.mpf(x)
    if x == 0:
        return "0"
    e = int(mp.floor(mp.log10(abs(x))))
    m = x / mp.mpf(10) ** e
    return rf"{mp.nstr(m, d)}\cdot10^{{{e}}}"

def block(name, t, title, note, wlabels=None, extra_b=None):
    real = not t["projected"]
    L = [rf"\subsection*{{{title}}}", note, r"{\small", r"\begin{align*}"]
    s = t["s"]
    for i in range(s):
        for j in range(i):
            re_s, im_s = t["A"][i][j]
            if mp.mpf(re_s) == 0 and mp.mpf(im_s) == 0:
                continue
            L.append(rf"a_{{{i + 1}{j + 1}}} &= {num(re_s, im_s, real)}\\")
    L[-1] = L[-1] + r"[1mm]"
    for lev, ((m, p), b) in enumerate(zip(t["levels"], t["b"])):
        sup = "" if (wlabels is None) else f"^{{({wlabels[lev]})}}"
        for i, (re_s, im_s) in enumerate(b):
            if mp.mpf(re_s) == 0 and mp.mpf(im_s) == 0:
                continue
            L.append(rf"b{sup}_{{{i + 1}}} &= {num(re_s, im_s, real)}\\")
        L[-1] = L[-1] + r"[1mm]"
    if extra_b:
        lab, bh = extra_b
        for i, (re_s, im_s) in enumerate(bh):
            L.append(rf"{lab}_{{{i + 1}}} &= {num(re_s, im_s, real)}\\")
        L[-1] = L[-1] + r"[1mm]"
    for i in range(1, s):
        cre = mp.fsum(mp.mpf(t["A"][i][j][0]) for j in range(i))
        cim = mp.fsum(mp.mpf(t["A"][i][j][1]) for j in range(i))
        L.append(rf"c_{{{i + 1}}} &= {num(cre, cim, real)}\\")
    L[-1] = L[-1].rstrip("\\")
    L += [r"\end{align*}", "}", ""]
    return "\n".join(L)

def main():
    T = json.load(open(os.path.join(DATA, "tableaux.json")))
    E = json.load(open(os.path.join(DATA, "embedded.json")))
    out = []
    def note(k, extra=""):
        t = T[k]
        Es = ", ".join(rf"E_{{{p + 1}}}={tex(e, 6)}" for (m, p), e in zip(t["levels"], t["E"]))
        return (rf"${Es}$, $\max|\text{{coef}}|={t['maxcoef']:.6f}$, residual of the projected conditions "
                rf"${tex(t['resid_refined'])}$; shift of the printed coefficients by the refinement "
                rf"${tex(t['refine_shift'])}$. {extra}")
    out.append(block("CRK5-opt", T["CRK5-opt"], r"CRK5-opt --- minimum of $E_6$ under $\max|\text{coef}|\le5$ (this work)",
                     note("CRK5-opt"), extra_b=(r"\hat b", E["CRK5(4)"]["bh"])))
    out.append(r"The weights $\hat b$ give the embedded fourth-order formula of the pair CRK5(4) (Section~\ref{sec:embedded}).")
    out.append(block("CRK5-s", T["CRK5-s"], "CRK5-s --- five-stage tableau with small coefficients (this work)",
                     note("CRK5-s")))
    out.append(block("CRK5-1", T["CRK5-1"], "CRK5-1 --- first tableau found (this work)", note("CRK5-1")))
    out.append(block("CRK5-2", T["CRK5-2"], "CRK5-2 --- tableau of an independent search", note("CRK5-2",
                     r"As found, this tableau had $\Imag b_1=0.5730233156598692$, which is inert (Proposition~\ref{prop:inert}).")))
    out.append(block("GJM5", T["GJM5"], r"GJM5 --- tableau of \cite[Table~4]{GJM}, refined", note("GJM5",
                     r"As published: $\Imag b_1=0.047034144968353016$ (inert), 16 significant digits.")))
    out.append(block("CRK6-0", T["CRK6-0"], "CRK6-0 --- tableau found by direct search (this work)", note("CRK6-0",
                     r"Printed with 16 digits in the draft, with $\Imag b_1=-0.2404222140778199$ (inert, set to zero here).")))
    out.append(block("CRK6-opt", T["CRK6-opt"], r"CRK6-opt --- minimum of $E_7$ under $\max|\text{coef}|\le5$ (this work)",
                     note("CRK6-opt"), extra_b=(r"\hat b", E["CRK6(4)"]["bh"])))
    out.append(r"The weights $\hat b$ give the embedded fourth-order formula of the pair CRK6(4) (Section~\ref{sec:embedded}).")
    out.append(block("CRK6-s", T["CRK6-s"], "CRK6-s --- six-stage tableau with small coefficients (this work)",
                     note("CRK6-s")))
    out.append(block("CRK6-lam", T["CRK6-lam"], r"CRK6-$\lambda$ --- tableau of the penalized descent (this work)", note("CRK6-lam",
                     r"Here $c_2$ is real, so $\Imag b_2$ is inert as well (Proposition~\ref{prop:inert}).")))
    out.append(block("CCK6(5)", T["CCK6(5)"], "CCK6(5) --- complex Cash--Karp pair, error-optimized in the earlier search (this work)", note("CCK6(5)"),
                     wlabels=["6", "5"]))
    out.append(block("CCK6(5)-s", T["CCK6(5)-s"], "CCK6(5)-s --- complex Cash--Karp pair, small coefficients (this work)",
                     note("CCK6(5)-s"), wlabels=["6", "5"]))
    out.append(block("CCK6-casc", T["CCK6-casc"], "CCK6-cascade --- complex Cash--Karp cascade 6(5)(4)(3)(2)(1) (this work)",
                     note("CCK6-casc"), wlabels=["6", "5", "4", "3", "2", "1"]))
    out.append(block("RK6-opt", T["RK6-opt"], "RK6-opt --- error-optimized real seven-stage tableau of order six (this work)",
                     note("RK6-opt", "All coefficients are real.")))
    open(os.path.join(TAB, "appendix_tableaux.tex"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("ok", sum(len(x) for x in out))

if __name__ == "__main__":
    main()
