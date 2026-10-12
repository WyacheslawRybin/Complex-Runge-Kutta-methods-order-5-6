"""
tune_cck6.py -- tune a complex Cash-Karp tableau (CCK6, real projection) for
                ONE specific problem, keeping the order of EVERY level exact.

Why this exists.  The CCK6 tableaux are not unique: the order conditions of all
levels leave a manifold of dimension four (pair 6(5)) or six (cascade
6(5)(4)(3)(2)(1)).  The leading error of a member on a given problem is the
projection of its error coefficients onto that problem, transported by the
variational flow, so minimizing the tableau norm E_7 = ||e||_2 does not minimize
the error on YOUR problem.  This script walks along the manifold, keeps all
order conditions of all levels satisfied to machine precision, and minimizes
the global error actually committed on the problem typed in below.

Requires numpy only.  Double precision: if the error at the working step size is
below ~1e-13, use a larger step or rewrite the integrator in mpmath.

Usage:  python tune_cck6.py      (edit the block USER INPUT first)
"""
import numpy as np

# =============================================================================
# USER INPUT
# =============================================================================
# f(t, y) must be real-analytic with real coefficients and must accept COMPLEX
# t and y: use numpy functions (np.sin, np.exp, ...), never math.*, abs(), or
# branches on the sign of a component.

def f(t, y):
    """Roessler system (example).  Replace with your own right-hand side."""
    a, b, c = 0.2, 0.2, 5.7
    return np.array([-y[1] - y[2],
                     y[0] + a * y[1],
                     b + y[2] * (y[0] - c)])

y0 = np.array([1.0, 1.0, 1.0])  # initial state (real)
T = 5.0                         # end of the integration interval
N = 100                         # number of steps you intend to use
N_LIST = (N // 2, N, 2 * N)     # the objective averages over these step counts, so
                                # that the descent cannot exploit a cancellation of
                                # the leading and the next error term at ONE step size

LEVELS = "CCK65"     # "CCK65": pair CCK6(5);  "CCK6-cascade": CCK6 6(5)(4)(3)(2)(1);
                     # the same descent also works for the single-level families:
                     # "CRK5" (five stages, order 5) and "CRK6" (six stages, order 6)
MAX_COEF = 5.0       # bound on max |a_ij|, |b_i| (all levels) kept during the descent
EMB_WEIGHT = 0.0     # > 0 adds EMB_WEIGHT * (error of the embedded fifth-order
                     # solution integrated on its own) to the objective
ROUNDS = 25          # descent rounds
# =============================================================================

import sys as _sys
if len(_sys.argv) > 1:                     # python tune_cck6.py LEVELS [N]
    LEVELS = _sys.argv[1]
    if len(_sys.argv) > 2:
        N = int(_sys.argv[2]); N_LIST = (N // 2, N, 2 * N)


# --- starting tableaux (Appendix of the paper, 20 digits) ---------------------
S = 6
A_CCK65 = {
    (2, 1): (0.23870322673748683925, 0.039084231162576517928),
    (3, 1): (-0.21217417744767087168, -0.42404302597406126565),
    (3, 2): (0.66454279065283384549, 0.49811190091494678828),
    (4, 1): (0.33213007094843194003, 0.79857073534266959837),
    (4, 2): (-0.036803508540519795578, -0.96751950580954984359),
    (4, 3): (0.37756912782195351002, 0.32434812601797809521),
    (5, 1): (-0.026047083025394350820, -0.66823791923018130304),
    (5, 2): (0.46916236399958357029, 0.95281285604124192202),
    (5, 3): (0.13423790825837125613, -0.16374444406912610737),
    (5, 4): (0.26982829349864122165, -0.15306343190435138092),
    (6, 1): (0.81055393809653871824, -0.023649747379388614517),
    (6, 2): (-0.64016012899890017844, 0.033600781489092883583),
    (6, 3): (0.38986355155999184312, 0.0043913190907800280638),
    (6, 4): (0.17399051804746795072, 0.12190487359164247090),
    (6, 5): (0.23283166502628901384, -0.13624722679212676803)}
W_CCK65 = [
    [(0.082239269481761387996, 0.0), (0.34060022369683037432, -0.071692327694748855010),
     (0.24923946474373638724, -0.13778319216325026409), (0.12401684697854992015, -0.17846029776977053733),
     (0.051460001551057071230, -0.28736466153057659100), (0.15244419354806485906, -0.066379005476304026618)],
    [(0.16990227070182876608, 0.0), (0.43559838931184825464, -1.6244258915681450177),
     (0.18847877990476728832, -0.30339523049419140090), (-0.12864424062906956445, -0.20076003306414425324),
     (0.33466480071062525541, -0.10417456954857497355)]]

A_CASC = {
    (2, 1): (0.35152025982078338426, 0.13203777575636354485),
    (3, 1): (-0.19723449260091803510, -0.021723739215052623876),
    (3, 2): (0.54608061421539097848, 0.15275705718850955120),
    (4, 1): (0.13642011947991197393, -0.035173389997844118072),
    (4, 2): (0.11740002659345116586, -0.0043732903830174683592),
    (4, 3): (0.56378819680503414185, 0.34622547635061851258),
    (5, 1): (0.12991500694869878111, 0.060005539176938526230),
    (5, 2): (0.43614756037431440360, 0.45180409413931473619),
    (5, 3): (0.22947169149553910907, -0.44701539257004380368),
    (5, 4): (0.12534074653228604950, -0.11292111279839938884),
    (6, 1): (0.19089852900346155338, 0.12838653896677867910),
    (6, 2): (0.32554715394005767000, 0.25494030060655273488),
    (6, 3): (0.22392473568345144947, -0.28976348990749245288),
    (6, 4): (0.11401400034747011169, -0.046605084010003725495),
    (6, 5): (0.043367418984041018017, -0.046958265655835235601)}
W_CASC = [
    [(0.10715647081688042691, 0.0), (0.25932385081541374528, 0.28003225991344674889),
     (0.11411852475677875233, 0.039608681385309010872), (0.012633312389080510930, 0.10431096497585186953),
     (-0.28366164537307691143, -0.32277693244820799512), (0.79042948659492347598, 0.48776022495594868302)],
    [(0.12238907534750782478, 0.0), (0.33904731854617513501, 0.23368933300671289365),
     (0.11863563360127683852, -0.10452912831583090048), (0.076744678021963561350, 0.057333769286080826583),
     (0.34318329448307664033, -0.095612213515906783438)],
    [(0.22284757158506792850, 0.0), (0.75665726653708871966, -0.13728401674537942604),
     (-0.0045095370905615299612, -0.77968789495366511413), (0.025004698968404881801, -0.30930393061822017258)],
    [(0.87902852459007284276, 0.0), (0.28495441847803204323, -1.9244186379343632495),
     (-0.16398294306810488599, -1.5487786544787225070)],
    [(-0.20398891914426439282, 0.0), (1.2039889191442643928, -0.58145104218300900013)],
    [(1.0, 0.0)]]

A_CRK5 = {
    (2, 1): (-0.0039500971323927656573, 0.28251949346552868159),
    (3, 1): (0.11650177979179908438, 0.22570113256671114899),
    (3, 2): (0.12389715727356388764, -0.0092352852878275822762),
    (4, 1): (0.084715632838748250455, 0.15531018826566125868),
    (4, 2): (0.025496499410525768486, 0.05281485892909871388),
    (4, 3): (0.37607918913780574721, -0.053597414144053980023),
    (5, 1): (0.095072429934319413469, 0.066619064317132409281),
    (5, 2): (0.0035117977052656753831, 0.016724417831258085038),
    (5, 3): (0.17004482831636094642, 0.15908563345633139341),
    (5, 4): (0.45340305659690688378, -0.14970262063575207057),}
W_CRK5 = [[(0.19668703872144407963, 0.0), (0.026254046007790105, -0.015689564817854103479), (0.37261796429101567973, 0.073204133804100804395), (1.1042571749538911963, -1.3790052245491737359), (-0.69981622397414100512, -1.9104035260738947422)]]
A_CRK6 = {
    (2, 1): (0.19955289081975391419, 0.061716592008779602718),
    (3, 1): (0.047582869422575370388, -0.011946647630386879324),
    (3, 2): (0.30165154254846365811, 0.051242797521783574333),
    (4, 1): (0.09381235042735877605, 0.084164302411942296112),
    (4, 2): (0.35370878547466877428, -0.046129341327005156381),
    (4, 3): (-0.05192056251252136212, 0.12618507801334707952),
    (5, 1): (0.11448171099449194754, 0.025870219653212640082),
    (5, 2): (0.29488036389276989224, -0.073379553783714746018),
    (5, 3): (0.048838633033381380344, 0.53645468065684498438),
    (5, 4): (0.16081607696511998618, -0.39122425087886875117),
    (6, 1): (0.5009770809575445627, -0.21232333987578153045),
    (6, 2): (-1.2524994634580495223, 0.51389591955818070268),
    (6, 3): (0.39388099495277745232, -0.54194644391209845224),
    (6, 4): (0.64275617097540904687, 0.79012706064493287705),
    (6, 5): (0.71488521657231851592, -0.54975319641523356928),}
W_CRK6 = [[(0.059884848825516437398, 0.0), (0.34668149936546244216, 0.055467766758519859716), (0.81281831002163706934, -0.75037160704943739464), (0.079625921207964606152, -0.6503780352875240256), (-0.38336341012551045093, -1.3870243058535982605), (0.084352830704929840366, -0.62191389387249329523)]]

if LEVELS == "CCK65":
    LEV = [(6, 6), (5, 5)]
    A_START, W_START = A_CCK65, W_CCK65
elif LEVELS == "CCK6-cascade":
    LEV = [(6, 6), (5, 5), (4, 4), (3, 3), (2, 2), (1, 1)]
    A_START, W_START = A_CASC, W_CASC
elif LEVELS == "CRK5":
    S = 5
    LEV = [(5, 5)]
    A_START, W_START = A_CRK5, W_CRK5
elif LEVELS == "CRK6":
    LEV = [(6, 6)]
    A_START, W_START = A_CRK6, W_CRK6
else:
    raise ValueError(LEVELS)


# --- rooted trees ---------------------------------------------------------------
def order(t):
    return 1 + sum(order(u) for u in t)

def trees_by_order(pmax):
    """Rooted trees as sorted nested tuples; () is the single node."""
    by = {1: [()]}
    for n in range(2, pmax + 1):
        pool = [t for k in range(1, n) for t in by[k]]
        out, seen = [], set()
        def multisets(total, start):
            if total == 0:
                yield []
                return
            for i in range(start, len(pool)):
                w = order(pool[i])
                if w <= total:
                    for rest in multisets(total - w, i):
                        yield [pool[i]] + rest
        for ms in multisets(n - 1, 0):
            key = tuple(sorted(ms, key=repr))
            if key not in seen:
                seen.add(key)
                out.append(key)
        by[n] = out
    return by

def gamma(t):
    g = order(t)
    for u in t:
        g *= gamma(u)
    return g

def sigma(t):
    out = 1
    for u in set(t):
        k = t.count(u)
        out *= sigma(u) ** k
        for q in range(2, k + 1):
            out *= q
    return out

PMAX = max(p for _, p in LEV)
TREES = trees_by_order(PMAX + 1)
ROWS = [(L, t) for L, (m, p) in enumerate(LEV) for k in range(1, p + 1) for t in TREES[k]]
RHS = np.array([1.0 / gamma(t) for _, t in ROWS])


# --- unknowns: Re/Im of A and of the weights of all levels, minus the inert Im w1 ---
AIDX = [(i, j) for i in range(S) for j in range(i)]
NA = len(AIDX)
WOFF = []
_off = NA
for m, _ in LEV:
    WOFF.append(_off)
    _off += m
NZ = _off
_DROP = {NZ + o for o in WOFF}           # Im w_1 of every level: k_1 is real
KEEP = np.array([q for q in range(2 * NZ) if q not in _DROP])

def unpack(x):
    xx = np.zeros(2 * NZ)
    xx[KEEP] = x
    z = xx[:NZ] + 1j * xx[NZ:]
    A = np.zeros((S, S), complex)
    for k, (i, j) in enumerate(AIDX):
        A[i, j] = z[k]
    W = [z[o:o + m] for o, (m, _) in zip(WOFF, LEV)]
    return A, W

def pack(A, W):
    z = np.zeros(NZ, complex)
    for k, (i, j) in enumerate(AIDX):
        z[k] = A[i, j]
    for o, w in zip(WOFF, W):
        z[o:o + len(w)] = w
    return np.concatenate([z.real, z.imag])[KEEP]


# --- elementary weights with exact derivatives with respect to the a_ij --------
def phi_all(A, pmax, deriv=True):
    cache = {}
    def Phi(t):
        if t in cache:
            return cache[t]
        v = np.ones(S, complex)
        dv = np.zeros((S, NA), complex) if deriv else None
        for u in t:
            pu, dpu = Phi(u)
            w = A @ pu
            if deriv:
                dw = A @ dpu
                for k, (i, j) in enumerate(AIDX):
                    dw[i, k] += pu[j]
                dv = dv * w[:, None] + v[:, None] * dw
            v = v * w
        cache[t] = (v, dv)
        return v, dv
    for k in range(1, pmax + 1):
        for t in TREES[k]:
            Phi(t)
    return cache

def residual_and_jacobian(x):
    """Projected order conditions of all levels, Re(w^T Phi_t) - 1/gamma(t), and their Jacobian."""
    A, W = unpack(x)
    ph = phi_all(A, PMAX)
    r = np.empty(len(ROWS))
    Jc = np.zeros((len(ROWS), NZ), complex)
    for q, (L, t) in enumerate(ROWS):
        m = LEV[L][0]
        v, dv = ph[t]
        r[q] = (W[L] @ v[:m]).real - RHS[q]
        Jc[q, :NA] = W[L] @ dv[:m]
        Jc[q, WOFF[L]:WOFF[L] + m] = v[:m]
    J = np.hstack([Jc.real, -Jc.imag])[:, KEEP]
    return r, J

def residual(x):
    return residual_and_jacobian(x)[0]

def error_constant(x, L=0):
    """E_{p+1} = ||e||_2 over the trees of order p+1 of level L (for reporting)."""
    A, W = unpack(x)
    m, p = LEV[L]
    ph = phi_all(A, p + 1, deriv=False)
    e = [((W[L] @ ph[t][0][:m]).real - 1.0 / gamma(t)) / sigma(t) for t in TREES[p + 1]]
    return float(np.linalg.norm(e))

def max_coef(x):
    A, W = unpack(x)
    return max(np.abs(A).max(), max(np.abs(w).max() for w in W))

def newton_project(x, iters=6):
    """Minimum-norm Newton back onto the manifold of exact order conditions."""
    for _ in range(iters):
        r, J = residual_and_jacobian(x)
        if np.linalg.norm(r) < 1e-14:
            break
        x = x - np.linalg.lstsq(J, r, rcond=1e-12)[0]
    return x

def tangent_basis(x, tol=1e-10):
    _, J = residual_and_jacobian(x)
    _, sv, Vt = np.linalg.svd(J)
    rank = int((sv > sv[0] * tol).sum())
    return Vt[rank:].T


# --- the projected integrator and the objective ----------------------------------
def integrate(x, n_steps, level=0):
    A, W = unpack(x)
    m = LEV[level][0]
    w = W[level]
    c = A.sum(axis=1)
    h = T / n_steps
    y = np.array(y0, float)
    for step in range(n_steps):
        t = step * h
        K = []
        for i in range(m):
            Y = y + h * sum(A[i, j] * K[j] for j in range(i)) if i else y.astype(complex)
            K.append(np.asarray(f(t + c[i] * h, Y), complex))
        y = np.real(y + h * sum(w[i] * K[i] for i in range(m)))
    return y

def errors(x, reference, level=0):
    return np.array([np.linalg.norm(integrate(x, n, level) - reference) for n in N_LIST])

def objective(x, reference, scale):
    """Geometric mean over N_LIST of the error relative to the starting tableau."""
    val = float(np.exp(np.mean(np.log(errors(x, reference) / scale[0]))))
    if EMB_WEIGHT > 0:
        val += EMB_WEIGHT * float(np.exp(np.mean(np.log(errors(x, reference, 1) / scale[1]))))
    return val


# --- descent inside the tangent space of the manifold ------------------------------
def tune(x, reference, scale, rounds=ROUNDS, step0=2e-3, verbose=True):
    best = objective(x, reference, scale)
    step = step0
    for it in range(rounds):
        Tn = tangent_basis(x)
        g = np.zeros(Tn.shape[1])
        eps = 1e-3       # large enough to dominate the round-off noise of the error itself
        for k in range(Tn.shape[1]):                     # central directional derivatives
            xp = newton_project(x + eps * Tn[:, k], iters=4)
            xm = newton_project(x - eps * Tn[:, k], iters=4)
            g[k] = (objective(xp, reference, scale) - objective(xm, reference, scale)) / (2 * eps)
        ng = np.linalg.norm(g)
        if ng < 1e-14:
            break
        d = -Tn @ (g / ng)
        improved = False
        for _ in range(14):                              # backtracking line search
            xc = newton_project(x + step * d, iters=4)
            if np.linalg.norm(residual(xc)) < 1e-12 and max_coef(xc) <= MAX_COEF:
                val = objective(xc, reference, scale)
                if val < best:
                    x, best, improved = xc, val, True
                    step *= 1.6
                    break
            step /= 2.0
        if verbose:
            print("  round %2d  relative error = %.4f  dim tangent = %d  max|coef| = %.3f  step = %.1e"
                  % (it + 1, best, Tn.shape[1], max_coef(x), step))
        if not improved:
            break
    return x, best


if __name__ == "__main__":
    A0 = np.zeros((S, S), complex)
    for (i, j), (re, im) in A_START.items():
        A0[i - 1, j - 1] = re + 1j * im
    W0 = [np.array([re + 1j * im for re, im in w]) for w in W_START]
    x0 = pack(A0, W0)
    print("levels (stages, order):", LEV, "  real unknowns:", len(x0), "  conditions:", len(ROWS))
    print("start: residual of the order conditions = %.2e" % np.linalg.norm(residual(x0)))
    x0 = newton_project(x0)
    print("after Newton                            = %.2e" % np.linalg.norm(residual(x0)))

    # reference solution: the starting tableau at a sixteen times finer step
    ref = integrate(x0, 16 * max(N_LIST))
    scale = (errors(x0, ref), errors(x0, ref, 1) if EMB_WEIGHT > 0 else None)
    print("errors of the starting tableau at N =", N_LIST, ":", " ".join("%.3e" % v for v in scale[0]),
          "  E_{p+1} = %.4e   max|coef| = %.3f" % (error_constant(x0), max_coef(x0)))

    x1, _ = tune(x0, ref, scale)
    x1 = newton_project(x1, iters=10)
    e1 = errors(x1, ref)
    print("\ntuned tableau, errors at N =", N_LIST, ":", " ".join("%.3e" % v for v in e1))
    print("improvement factors:", " ".join("%.2f" % v for v in scale[0] / e1))
    print("E_{p+1} = %.4e (start %.4e),  max|coef| = %.3f" % (error_constant(x1), error_constant(x0), max_coef(x1)))
    r = residual(x1)
    off = 0
    for (m, p) in LEV:
        n = sum(len(TREES[k]) for k in range(1, p + 1))
        print("  level (%d stages, order %d): max residual %.1e" % (m, p, np.abs(r[off:off + n]).max()))
        off += n

    A, W = unpack(x1)
    print("\ntuned coefficients:")
    for i in range(1, S):
        for j in range(i):
            print("  a%d%d = %+.18f %+.18fj" % (i + 1, j + 1, A[i, j].real, A[i, j].imag))
    for L, (m, p) in enumerate(LEV):
        for i in range(m):
            print("  w%d^(%d) = %+.18f %+.18fj" % (i + 1, m, W[L][i].real, W[L][i].imag))
    np.save("tuned_%s.npy" % LEVELS, x1)
    print("\nsaved to tuned_%s.npy" % LEVELS)
