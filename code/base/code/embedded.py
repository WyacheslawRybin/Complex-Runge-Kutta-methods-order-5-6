"""
Embedded formulas for a FIXED coefficient matrix A.

For order q the projected conditions on the embedded weights bh are linear:
    Re(bh^T Phi_t) = 1/gamma(t),   rho(t) <= q.
The propagated weights b already solve them (order p > q), so the embedded
family is  bh = b + V t  with V a basis of the kernel of the real matrix
M_q(A) = [P | -Q] (Im bh_1 is inert and dropped).  Optionally a FSAL stage
k_{s+1} = f(t+h, y_{n+1}) is appended: y_{n+1} is real, so k_{s+1} is real and
its elementary weights are the exact values prod_k 1/gamma(u_k).

The script reports the kernel dimensions (which decide existence) and, for the
chosen pairs, the Dormand-Prince quality measures.
"""
import json, os, sys
import numpy as np
import mpmath as mp
import crk
from tableaux import load_refined

mp.mp.dps = 50

def stage_phi(T, pmax, fsal=False):
    """Elementary weights Phi_t (complex, s stages) plus the FSAL stage (real, exact)."""
    ph = T.phi(pmax)
    out = {}
    for t in crk.trees_upto(pmax):
        v = list(ph[t])
        if fsal:
            # Phi_t(s+1) = prod over children u of sum_j a_{s+1,j} Phi_u(j) = prod Re(b^T Phi_u) = prod 1/gamma(u)
            val = mp.mpf(1)
            for u in t:
                val /= crk.gamma(u)
            v.append(mp.mpc(val))
        out[t] = v
    return out

def columns(T, fsal):
    """Real unknowns of bh: Re bh_i (all), Im bh_i (i >= 2, complex stages only)."""
    s = T.s
    cols = [("re", i) for i in range(s)] + [("im", i) for i in range(1, s)]
    if fsal:
        cols.append(("re", s))
    return cols

def weight_matrix(T, q, fsal=False):
    ph = stage_phi(T, q, fsal)
    cols = columns(T, fsal)
    M = []
    for t in crk.trees_upto(q):
        row = []
        for kind, i in cols:
            row.append(float(mp.re(ph[t][i])) if kind == "re" else float(-mp.im(ph[t][i])))
        M.append(row)
    return np.array(M), cols

def bvec(T, fsal):
    b = T.levels[0][2]
    x = [float(mp.re(v)) for v in b] + [float(mp.im(v)) for v in b[1:]]
    if fsal:
        x.append(0.0)
    return np.array(x)

def to_complex(x, T, fsal):
    s = T.s
    bh = [complex(x[i], 0.0 if i == 0 else x[s + i - 1]) for i in range(s)]
    if fsal:
        bh.append(complex(x[2 * s - 1], 0.0))
    return bh

def err_vectors(T, bh, q, fsal, extra):
    """tau_hat(t) = (Re(bh^T Phi_t) - 1/gamma(t))/sigma(t) for the trees of order q+extra."""
    ph = stage_phi(T, q + extra, fsal)
    return np.array([float((mp.re(mp.fsum(mp.mpc(bh[i]) * ph[t][i] for i in range(len(bh))))
                            - mp.mpf(1) / crk.gamma(t)) / crk.sigma(t)) for t in crk.TREES[q + extra]])

def family(T, q, fsal=False):
    M, cols = weight_matrix(T, q, fsal)
    sv = np.linalg.svd(M, compute_uv=False)
    _, s_, Vt = np.linalg.svd(M)
    rank = int(np.sum(sv > 1e-12 * sv[0]))
    V = Vt[rank:].T
    b = bvec(T, fsal)
    return dict(M=M, rank=rank, n=M.shape[1], kernel=V.shape[1], V=V, b=b, sv=sv)

def dp_measures(T, bh, q, fsal):
    """Dormand-Prince measures for a pair with propagated order p and embedded order q = p-1."""
    p = T.levels[0][1]
    tau_hat_q1 = err_vectors(T, bh, q, fsal, 1)
    tau_hat_q2 = err_vectors(T, bh, q, fsal, 2)
    b = to_complex(bvec(T, fsal), T, fsal)
    tau_q1 = err_vectors(T, b, q, fsal, 1)          # zero if p >= q+1
    tau_q2 = err_vectors(T, b, q, fsal, 2)
    A1 = np.linalg.norm(tau_hat_q1)
    return dict(Ahat=A1, B=np.linalg.norm(tau_hat_q2) / A1,
                C=np.linalg.norm(tau_hat_q2 - tau_q2) / A1,
                E=np.linalg.norm(tau_q2) / A1, main_q1=np.linalg.norm(tau_q1),
                maxbh=max(abs(v) for v in bh))

if __name__ == "__main__":
    TT = load_refined()
    print("kernel dimension of the embedded family for fixed A  (rank / unknowns -> dim of {bh} = kernel)")
    for name, q in [("CRK5-opt", 4), ("CRK5-s", 4), ("CRK5-1", 4), ("CRK5-2", 4), ("GJM5", 4),
                    ("CRK6-opt", 5), ("CRK6-s", 5), ("CRK6-0", 5), ("CRK6-lam", 5), ("CRK6-opt", 4), ("CRK6-s", 4), ("CRK6-0", 4),
                    ("CCK6(5)", 5), ("CCK6-casc", 5)]:
        for fsal in (False, True):
            F = family(TT[name], q, fsal)
            print(f"  {name:10s} q={q} fsal={fsal!s:5s}: rank {F['rank']:2d} of {F['M'].shape}  kernel {F['kernel']}"
                  f"   smallest sv {F['sv'][-1]:.2e}")

def real_pair_measures(T):
    """DP measures of a real (or projected) pair given as a tableau with two levels."""
    p = T.levels[0][1]; q = T.levels[1][1]
    m, _, bh = T.levels[1]
    bh = [complex(v) for v in bh] + [0j] * (T.s - m)
    return dp_measures(T, bh, q, False)

AHAT_DOPRI = None   # set from DP5 at run time

def mp_nullspace(M_mp, k):
    """Orthonormal basis (columns) of the k-dimensional kernel of an mp matrix via SVD."""
    U, S, V = mp.svd_r(M_mp, full_matrices=True)
    n = M_mp.cols
    return mp.matrix([[V[n - k + c, r] for c in range(k)] for r in range(n)])

def make_pair(T, q, Ahat, name):
    """Embedded weights of order q for the fixed A of T: direction minimizing C, scale giving Ahat."""
    import scipy.linalg as sl
    F = family(T, q, False)
    k = F["kernel"]
    ph = stage_phi(T, q, False)
    cols = columns(T, False)
    with mp.workdps(60):
        Mm = mp.matrix([[mp.re(ph[t][i]) if kind == "re" else -mp.im(ph[t][i]) for kind, i in cols]
                        for t in crk.trees_upto(q)])
        Vm = mp_nullspace(Mm, k)
        b = T.levels[0][2]
        bx = [mp.re(v) for v in b] + [mp.im(v) for v in b[1:]]
        def bh_of(tvec):
            x = [bx[r] + mp.fsum(Vm[r, c] * tvec[c] for c in range(k)) for r in range(len(bx))]
            s = T.s
            return [mp.mpc(x[i], 0 if i == 0 else x[s + i - 1]) for i in range(s)]
        def errs(bh, extra):
            php = stage_phi(T, q + extra, False)
            return np.array([float((mp.re(mp.fsum(bh[i] * php[t][i] for i in range(T.s))) - mp.mpf(1) / crk.gamma(t))
                                   / crk.sigma(t)) for t in crk.TREES[q + extra]])
        base2 = errs(bh_of([0] * k), 2)
        W1 = np.array([errs(bh_of([1 if c == j else 0 for c in range(k)]), 1) for j in range(k)]).T
        W2 = np.array([errs(bh_of([1 if c == j else 0 for c in range(k)]), 2) - base2 for j in range(k)]).T
        if k == 1:
            u = np.array([1.0])
        else:
            w, U = sl.eigh(W2.T @ W2, W1.T @ W1)
            u = U[:, 0]
        u = u / np.linalg.norm(W1 @ u) * Ahat
        bh = bh_of([mp.mpf(float(v)) for v in u])
        # residual check of the embedded conditions
        res = max(abs(mp.re(mp.fsum(bh[i] * ph[t][i] for i in range(T.s))) - mp.mpf(1) / crk.gamma(t))
                  for t in crk.trees_upto(q))
    d = dp_measures(T, [complex(v) for v in bh], q, False)
    d.update(name=name, base=T.name, q=q, kernel=k, resid=float(res),
             bh=[[mp.nstr(mp.re(v), 40), mp.nstr(mp.im(v), 40)] for v in bh])
    return d

def write_pairs():
    TT = load_refined()
    ahat = real_pair_measures(TT["DP5"])["Ahat"]
    out = {"Ahat_DOPRI5": ahat}
    for base, q, name in [("CRK5-opt", 4, "CRK5(4)"), ("CRK6-opt", 4, "CRK6(4)"),
                         ("CRK5-s", 4, "CRK5-s(4)"), ("CRK6-s", 4, "CRK6-s(4)")]:
        d = make_pair(TT[base], q, ahat, name)
        out[name] = d
        print(f"{name}: kernel {d['kernel']} resid {d['resid']:.1e}  Ahat={d['Ahat']:.4e} B={d['B']:.3f} "
              f"C={d['C']:.3f} E={d['E']:.3f} max|bh|={d['maxbh']:.3f}")
    for nm in ["DP5", "CK5", "DP6(5)", "Verner6(5)", "CCK6(5)", "CCK6(5)-s", "CCK6-casc"]:
        d = real_pair_measures(TT[nm])
        out[nm] = {k: float(v) for k, v in d.items()}
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "data", "embedded.json"), "w"), indent=1)

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "pairs":
    write_pairs()
