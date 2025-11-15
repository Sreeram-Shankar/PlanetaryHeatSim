import mpmath as mp
mp.dps = 100

#lobatto roots and weights generator
def lobatto_roots_and_weights(s):
    if s < 2:
        raise ValueError("Lobatto quadrature requires at least 2 stages")
    n = s - 1

    xs = [mp.mpf(-1)]
    Pn = lambda x: mp.legendre(n, x)
    Pn_prime = lambda x: n*(x*Pn(x) - mp.legendre(n-1, x)) / (x**2 - 1)

    for k in range(1, n):
        x0 = mp.cos(mp.pi * (k) / (n))
        x = x0
        for _ in range(100):
            fx = Pn_prime(x)
            dfx = mp.diff(Pn_prime, x)
            if dfx == 0:
                break
            x_new = x - fx/dfx
            if mp.almosteq(x_new, x, rel_eps=mp.eps*100, abs_eps=mp.eps*100):
                break
            x = x_new
        xs.append(x)

    xs.append(mp.mpf(1))

    ws = []
    for i, x in enumerate(xs):
        if i == 0 or i == s-1:
            w = mp.mpf(2) / (n*(n+1))
        else:
            w = mp.mpf(2) / (n*(n+1) * (Pn(x)**2))
        ws.append(w)

    return xs, ws

#lagrange basis polynomial
def lagrange_basis(c, j):
    xj = c[j]
    others = [c[k] for k in range(len(c)) if k != j]
    denom = mp.mpf(1)
    for xk in others:
        denom *= (xj - xk)
    def Lj(x):
        num = mp.mpf(1)
        for xk in others:
            num *= (x - xk)
        return num / denom
    return Lj

#build lobatto iic irk tableau
def build_lobatto_IIIC_irk(s):
    xs, ws = lobatto_roots_and_weights(s)
    c = [(x + 1)/2 for x in xs]
    b = [w/2 for w in ws]

    A = [[mp.mpf(0) for _ in range(s)] for _ in range(s)]
    for j in range(s):
        Lj = lagrange_basis(c, j)
        for i in range(s):
            A[i][j] = mp.quad(Lj, [0, c[i]])

    for i in range(s):
        rs = mp.fsum(A[i])
        diff = rs - c[i]

    for k in range(min(10, 2*s-2)):
        lhs = mp.fsum([b[j]*c[j]**k for j in range(s)])
        rhs = mp.mpf(1)/(k+1)
        err = lhs - rhs

    return A, b, c

#format number string
def nstr_fixed(x, digits=80): return mp.nstr(x, n=digits)

#write tableau to files
def write_A_b_c_triplets(A, b, c, basename, digits=80):
    s = len(b)
    with open(f"{basename}_A.txt", "w") as fa:
        for i in range(s):
            for j in range(s):
                fa.write(f"{i+1} {j+1} {nstr_fixed(A[i][j],digits)}\n")
    with open(f"{basename}_b.txt", "w") as fb:
        for j in range(s):
            fb.write(f"{j+1} {nstr_fixed(b[j],digits)}\n")
    with open(f"{basename}_c.txt", "w") as fc:
        for i in range(s):
            fc.write(f"{i+1} {nstr_fixed(c[i],digits)}\n")
    with open(f"{basename}_triplets.txt", "w") as ft:
        for i in range(s):
            for j in range(s):
                ft.write(f"{i+1} {j+1} {nstr_fixed(A[i][j],digits)}\n")
        for i in range(s):
            ft.write(f"{i+1} 0 {nstr_fixed(c[i],digits)}\n")
        for j in range(s):
            ft.write(f"0 {j+1} {nstr_fixed(b[j],digits)}\n")


