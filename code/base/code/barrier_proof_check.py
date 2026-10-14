"""Computer check of every step of the proof of Theorem 3 (Butcher's barrier over C, s = p >= 5).

Notation: explicit s = p stages, c = A 1, beta^T = b^T A^k with k = p - 5 (so beta_j = 0 for j > 5).
  u = ( beta_5 a54 a43,  beta_4 (c4 - c5) a43,  beta_3 (c3 - c5)(c3 - c4) ),
  v = ( a32 c2,  c3 (c3 - c2) ).
Step 1 (collapse): each product u_r v_q equals a sum over all stages,
  u_1 v_q = beta^T A^2 x_q,  u_2 v_q = (beta o (c - c5))^T A x_q,  u_3 v_q = (beta o (c - c5) o (c - c4))^T x_q,
  x_1 = A c, x_2 = c o (c - c2);  checked as polynomial identities in generic symbolic A, b.
Step 2 (values): by the order conditions of order p every such sum is a combination of b^T Phi_t with
  rho(t) <= p, i.e. a known polynomial E_rq(c2, c4, c5) with rational coefficients (tree algebra).
Step 3: the 3x2 matrix (u_r v_q) has rank <= 1, so its 2x2 minors vanish; Groebner bases show that over C
  this forces c2 = 0 or c4 = c5 = 1.
Step 4: u_1 v_1 = 1/p! != 0 gives c2 != 0, hence c4 = c5 = 1.
Step 5: then (beta o (1 - c))^T A^2 c = 0 by the explicit structure, but the order conditions give (p-4)/p!.
Usage:  python barrier_proof_check.py [PMAX]     (checks p = 5..PMAX; the symbolic collapse up to p = 8)
"""
import sys
from math import factorial
import sympy as sp

# ---------------- tree algebra: a stage vector is {node (sorted tuple of child trees): coefficient} -------------
def order(t):
    return 1 + sum(order(u) for u in t)

def gamma(t):
    g = order(t)
    for u in t:
        g *= gamma(u)
    return g

def node(*children):
    return tuple(sorted(children, key=repr))

ONE = {(): sp.Integer(1)}                    # the vector 1 = Phi of the single node

def vadd(*vs):
    out = {}
    for v in vs:
        for k, c in v.items():
            out[k] = out.get(k, 0) + c
    return {k: c for k, c in out.items() if c != 0}

def vscale(v, a):
    return {k: a * c for k, c in v.items()}

def vmul(v, w):                              # componentwise product: union of the children
    out = {}
    for k1, c1 in v.items():
        for k2, c2 in w.items():
            k = node(*(k1 + k2))
            out[k] = out.get(k, 0) + c1 * c2
    return out

def vA(v):                                   # A v: a new node whose single child is the tree of v
    return {node(k): c for k, c in v.items()}

def value(v, k):
    """b^T A^k v from the order conditions: b^T Phi_t = 1/gamma(t) for t = k-fold wrap of each node."""
    tot = 0
    for t, c in v.items():
        for _ in range(k):
            t = node(t)
        tot += c / gamma(t)
    return sp.nsimplify(tot)

c2, c4, c5, y = sp.symbols("c2 c4 c5 y")
C = vA(ONE)                                  # c = A 1


def matrix_E(p):
    k = p - 5
    x = [vA(C), vadd(vmul(C, C), vscale(C, -c2))]                        # A c,  c o (c - c2)
    lam = [lambda w: vA(vA(w)),                                          # beta^T A^2 w
           lambda w: vadd(vmul(C, vA(w)), vscale(vA(w), -c5)),           # (beta o (c - c5))^T A w
           lambda w: vadd(vmul(vmul(C, C), w), vscale(vmul(C, w), -(c4 + c5)), vscale(w, c4 * c5))]
    return sp.Matrix(3, 2, lambda r, q: sp.expand(value(lam[r](x[q]), k)))


def check_collapse(p):
    """Step 1 as polynomial identities in generic symbolic A, b (s = p)."""
    s, k = p, p - 5
    A = sp.zeros(s, s)
    for i in range(s):
        for j in range(i):
            A[i, j] = sp.Symbol("a%d_%d" % (i + 1, j + 1))
    b = sp.Matrix([[sp.Symbol("b%d" % (i + 1)) for i in range(s)]])
    c = A * sp.ones(s, 1)
    beta = b * A ** k
    assert all(sp.expand(beta[j]) == 0 for j in range(5, s))
    a = lambda i, j: A[i - 1, j - 1]
    cc = lambda i: c[i - 1]
    bt = lambda i: beta[i - 1]
    u = [bt(5) * a(5, 4) * a(4, 3), bt(4) * (cc(4) - cc(5)) * a(4, 3), bt(3) * (cc(3) - cc(5)) * (cc(3) - cc(4))]
    v = [a(3, 2) * cc(2), cc(3) * (cc(3) - cc(2))]
    X = [A * c, sp.Matrix([c[i] * (c[i] - c[1]) for i in range(s)])]
    D = lambda w: sp.Matrix([w[i] for i in range(s)])
    sums = [lambda w: (beta * A * A * w)[0],
            lambda w: (sp.Matrix([[beta[i] * (c[i] - c[4]) for i in range(s)]]) * A * w)[0],
            lambda w: (sp.Matrix([[beta[i] * (c[i] - c[4]) * (c[i] - c[3]) for i in range(s)]]) * w)[0]]
    for r in range(3):
        for q in range(2):
            assert sp.expand(sums[r](X[q]) - u[r] * v[q]) == 0, (p, r, q)
    # step 5 structure: (beta o (1 - c))^T A^2 c vanishes when c4 = c5 = 1
    z = (sp.Matrix([[beta[i] * (1 - c[i]) for i in range(s)]]) * A * A * c)[0]
    z = sp.expand(z)
    # every term of z contains a factor (1 - c4) or (1 - c5): substitute c4 = c5 = 1 through a41, a51
    sub = {A[3, 0]: 1 - sum(A[3, j] for j in range(1, 3)), A[4, 0]: 1 - sum(A[4, j] for j in range(1, 4))}
    assert sp.expand(z.subs(sub)) == 0
    return True


def check_minors(p):
    E = matrix_E(p) * factorial(p)
    minors = [sp.expand(E[i, 0] * E[j, 1] - E[j, 0] * E[i, 1]) for i in range(3) for j in range(i + 1, 3)]
    ok = True
    for target in (c2 * (c4 - 1), c2 * (c5 - 1)):              # target vanishes on V(minors)?
        G = sp.groebner(minors + [1 - y * target], c2, c4, c5, y, order="grevlex", domain=sp.QQ)
        ok &= list(G.exprs) == [1]
    final = value(vadd(vA(vA(C)), vscale(vmul(C, vA(vA(C))), -1)), p - 5)   # (beta o (1-c))^T A^2 c
    return E, minors, ok, final


if __name__ == "__main__":
    pmax = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    for p in range(5, pmax + 1):
        if p <= 8:
            check_collapse(p)
        E, minors, ok, final = check_minors(p)
        print("p = %2d: E_11 = 1/p! %s;  V(minors) in {c2=0} u {c4=c5=1}: %s;  final value %s = (p-4)/p! %s%s"
              % (p, "ok" if E[0, 0] == 1 else "FAIL", ok, final, final == sp.Rational(p - 4, factorial(p)),
                 ";  collapse identities verified" if p <= 8 else ""))
    # symbolic p: entries of p! * E are polynomials in p
    P = sp.symbols("p")
    print("\np! * E as polynomials in p (fitted from p = 5..13, checked at p = 14):")
    rows = []
    for r in range(3):
        row = []
        for q in range(2):
            pts = [(pp, matrix_E(pp)[r, q] * factorial(pp)) for pp in range(5, 14)]
            poly = sp.expand(sp.interpolate(pts, P))
            assert sp.expand(poly.subs(P, 14) - matrix_E(14)[r, q] * factorial(14)) == 0
            row.append(sp.factor(poly))
        rows.append(row)
        print("  ", row)
    # the entries as printed in the proof of Theorem 3, and the two minors used there
    printed = sp.Matrix([[1, 2 - P * c2],
                         [4 - P * c5, 8 - 3 * P * c2 - 2 * P * c5 + P * (P - 1) * c2 * c5],
                         [12 - 3 * P * (c4 + c5) + P * (P - 1) * c4 * c5,
                          24 - 6 * P * (c2 + c4 + c5) + 2 * P * (P - 1) * (c2 * c4 + c2 * c5 + c4 * c5)
                          - P * (P - 1) * (P - 2) * c2 * c4 * c5]])
    assert all(sp.expand(printed[r, q] - rows[r][q]) == 0 for r in range(3) for q in range(2))
    m12 = sp.expand(printed[0, 0] * printed[1, 1] - printed[1, 0] * printed[0, 1])
    m13 = sp.expand((printed[0, 0] * printed[2, 1] - printed[2, 0] * printed[0, 1]).subs(c5, 1))
    assert sp.expand(m12 - P * c2 * (1 - c5)) == 0 and sp.expand(m13 - P * (P - 4) * c2 * (c4 - 1)) == 0
    print("printed entries and minors p c2 (1 - c5), p (p - 4) c2 (c4 - 1) [at c5 = 1]: verified")
