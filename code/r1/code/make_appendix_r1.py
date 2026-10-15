"""Appendix with all coefficients for the R1 revision -> R1/tables/appendix_tableaux.tex.

Changes against code/make_appendix.py (review, Section 4, item 6, and Section 6):
  * parts with modulus below 1e-40 are printed as 0 (RK6-opt had b_2 = 7.8e-61);
  * CRK6-lambda is normalized by Im b_2 = 0 as well, since c_2 is real and Im b_2 is inert
    (Proposition "Inert directions"); its stored value 6.0e-7 had fewer than 30 significant digits;
  * titles and notes describe the role of a tableau instead of the history of the search;
  * each coefficient block may break across pages (\allowdisplaybreaks inside its group).
"""
import copy, json, os
import mpmath as mp
import r1common as R
import make_appendix as MA

TINY = mp.mpf(10) ** -40


def num(re_s, im_s, real_only=False):
    re_v, im_v = mp.mpf(re_s), mp.mpf(im_s)
    f = lambda x: mp.nstr(x, MA.D, min_fixed=-6, max_fixed=6, strip_zeros=False)
    r = "0" if abs(re_v) < TINY else ("+" if re_v >= 0 else "-") + f(abs(re_v))
    if real_only or abs(im_v) < TINY:
        return r
    return r + (" + " if im_v >= 0 else " - ") + f(abs(im_v)) + r"\,\mathrm{i}"


MA.num = num
MA.D = 20          # significant digits in the appendix; 45-digit values are in data/tableaux.json


def main():
    T = json.load(open(os.path.join(R.ORIG_DATA, "tableaux.json")))
    E = json.load(open(os.path.join(R.ORIG_DATA, "embedded.json")))
    lam = copy.deepcopy(T["CRK6-lam"])
    lam["b"][0][1][1] = "0"                    # Im b_2 = 0 (inert because c_2 is real)
    T["CRK6-lam"] = lam
    block, note = MA.block, None

    def note(k, extra=""):
        t = T[k]
        Es = ", ".join(rf"E_{{{p + 1}}}={MA.tex(e, 6)}" for (m, p), e in zip(t["levels"], t["E"]))
        return (rf"${Es}$, $\max|\text{{coef}}|={t['maxcoef']:.6f}$, residual of the projected conditions "
                rf"${MA.tex(t['resid_refined'])}$; shift of the original coefficients by the refinement "
                rf"${MA.tex(t['refine_shift'])}$. {extra}")

    out = []
    # Appendix A keeps the tableaux of the comparisons and the verified pair; CRK5-s, CRK5-1, CRK5-2, GJM5
    # (refined), CRK6-0, CRK6-s, CRK6-lambda and CCK6(5)-s are in data/tableaux.json of the repository
    # (internal review 5)
    V = json.load(open(os.path.join(R.DATA, "cck6v_r1.json")))
    T.update(V)
    r64 = json.load(open(os.path.join(R.DATA, "real64_r1.json")))["RK6-opt(4)"]
    out.append(block("CRK5-opt", T["CRK5-opt"], r"CRK5-opt --- minimum of $E_6$ under $\max|\text{coef}|\le5$ (this work)",
                     note("CRK5-opt"), extra_b=(r"\hat b", E["CRK5(4)"]["bh"])))
    out.append(r"The weights $\hat b$ give the embedded fourth-order formula of the pair CRK5(4) (Section~\ref{sec:embedded}).")
    out.append(block("CRK6-opt", T["CRK6-opt"], r"CRK6-opt --- minimum of $E_7$ under $\max|\text{coef}|\le5$ (this work)",
                     note("CRK6-opt"), extra_b=(r"\hat b", E["CRK6(4)"]["bh"])))
    out.append(r"The weights $\hat b$ give the embedded fourth-order formula of the pair CRK6(4) (Section~\ref{sec:embedded}).")
    out.append(block("CCK6(5)", T["CCK6(5)"], "CCK6(5) --- complex Cash--Karp pair (this work)", note("CCK6(5)"),
                     wlabels=["6", "5"]))
    out.append(block("CCK6(5)-v", T["CCK6(5)-v"], "CCK6(5)-v --- complex Cash--Karp pair with a proof of existence (this work)",
                     note("CCK6(5)-v", r"An exact pair exists within $10^{-50}$ of these coefficients (Proposition~\ref{prop:pairexist}); $c_6$ is real."),
                     wlabels=["6", "5"]))
    out.append(block("CCK6-casc", T["CCK6-casc"], "CCK6-cascade --- complex Cash--Karp cascade 6(5)(4)(3)(2)(1) (this work)",
                     note("CCK6-casc"), wlabels=["6", "5", "4", "3", "2", "1"]))
    out.append(block("RK6-opt", T["RK6-opt"], "RK6-opt --- error-optimized real seven-stage tableau of order six (this work)",
                     note("RK6-opt", "All coefficients are real."), extra_b=(r"\hat b", [[v, "0"] for v in r64["bh"]])))
    out.append(r"The weights $\hat b$ give the embedded fourth-order formula of the real pair RK6-opt(4); $\hat b_8$ is the "
               r"weight of the FSAL stage $k_8=f(t_{n+1},y_{n+1})$ (Section~\ref{sec:embedded}).")
    # a long align* cannot break across pages, so the end of a block ran into the footer (review, Section 6);
    # \allowdisplaybreaks is local to the {\small ...} group of each block
    text = "\n".join(out) + "\n"
    key = "{\\small\n\\begin{align*}"
    assert text.count(key) == 6, text.count(key)
    text = text.replace(key, "{\\small\\allowdisplaybreaks\n\\begin{align*}")
    open(os.path.join(R.TAB, "appendix_tableaux.tex"), "w", encoding="utf-8").write(text)
    print("ok", sum(len(x) for x in out))


if __name__ == "__main__":
    main()
