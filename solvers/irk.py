import numpy as np
from generation.gauss_legendre import build_gauss_legendre_irk
from generation.radau import build_radau_irk
from generation.lobatto import build_lobatto_IIIC_irk

#loads the Butcher tableau from MethodForge generator
def get_tableau(family, s):
    family = family.lower()
    if family == "gauss": A, b, c = build_gauss_legendre_irk(s)
    elif family == "radau": A, b, c = build_radau_irk(s)
    elif family == "lobatto": A, b, c = build_lobatto_IIIC_irk(s)
    else: raise ValueError(f"Unknown family '{family}', must be 'gauss', 'radau', or 'lobatto'.")
    
    #converts the tableau to numpy arrays
    A = np.array([[float(A[i][j]) for j in range(s)] for i in range(s)])
    b = np.array([float(b[i]) for i in range(s)])
    c = np.array([float(c[i]) for i in range(s)])
    return A, b, c

#defines the irk step with Picard
def step_collocation(f, t, y, h, A, b, c, sweeps=12, tol=1e-10):
    s = len(b)
    n = len(y)
    Y = np.tile(y, (s, 1)) 

    #implements Gauss-Seidel relaxation
    for _ in range(sweeps):
        Y_old = Y.copy()
        for i in range(s):
            rhs = np.zeros(n)
            for j in range(s):
                rhs += A[i, j] * f(t + c[j]*h, Y[j])
            Y[i] = y + h * rhs
        if np.linalg.norm(Y - Y_old) < tol:
            break

    #computes the final state update
    K = np.zeros((s, n))
    for i in range(s):
        K[i] = f(t + c[i]*h, Y[i])
    y_next = y + h * np.sum(b[:, None] * K, axis=0)
    return y_next

#main solver for any collocation irk method using Gauss-Seidel relaxation
def solve_collocation(f, t_span, y0, h, family="gauss", s=3, sweeps=12, tol=1e-10):
    A, b, c = get_tableau(family, s)
    t0, tf = t_span
    N = int(np.ceil((tf - t0)/h))
    t_grid = np.linspace(t0, tf, N+1)
    Y = np.zeros((N+1, len(y0)))
    Y[0] = y0
    for n in range(N):
        Y[n+1] = step_collocation(f, t_grid[n], Y[n], h, A, b, c, sweeps=sweeps, tol=tol)
    return t_grid, Y