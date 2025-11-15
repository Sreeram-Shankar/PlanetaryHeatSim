import numpy as np

#definesthe SDIRK step with Gauss-Seidel relaxation
def step_sdirk(f, t, y, h, A, b, c, sweeps=12, tol=1e-10):
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


#solves the nonlinear system of equations with a Gauss-Seidel relaxation SDIRK2
def solve_sdirk2(f, t_span, y0, h, sweeps=12, tol=1e-10):
    gamma = 1.0 - 1.0/np.sqrt(2.0)
    A = np.array([[gamma, 0.0], [1.0 - gamma, gamma]])
    b = np.array([1.0 - gamma, gamma])
    c = np.array([gamma, 1.0])
    t0, tf = t_span
    N = int(np.ceil((tf - t0)/h))
    t_grid = np.linspace(t0, tf, N+1)
    Y = np.zeros((N+1, len(y0)))
    Y[0] = y0
    for n in range(N): Y[n+1] = step_sdirk(f, t_grid[n], Y[n], h, A, b, c, sweeps=sweeps, tol=tol)
    return t_grid, Y


#solves the nonlinear system of equations with a Gauss-Seidel relaxation SDIRK3
def solve_sdirk3(f, t_span, y0, h, sweeps=12, tol=1e-10):
    gamma = 0.435866521508459
    A = np.array([
        [gamma, 0.0, 0.0],
        [0.2820667395, gamma, 0.0],
        [1.208496649, -0.644363171, gamma]
    ])
    b = np.array([1.208496649, -0.644363171, gamma])
    c = np.array([gamma, 0.7179332605, 1.0])
    t0, tf = t_span
    N = int(np.ceil((tf - t0)/h))
    t_grid = np.linspace(t0, tf, N+1)
    Y = np.zeros((N+1, len(y0)))
    Y[0] = y0
    for n in range(N): Y[n+1] = step_sdirk(f, t_grid[n], Y[n], h, A, b, c, sweeps=sweeps, tol=tol)
    return t_grid, Y


#solves the nonlinear system of equations with a Gauss-Seidel relaxation SDIRK4
def solve_sdirk4(f, t_span, y0, h, sweeps=12, tol=1e-10):
    gamma = 0.572816062482135
    A = np.array([
        [gamma, 0.0, 0.0, 0.0],
        [-0.6557110092, gamma, 0.0, 0.0],
        [0.757184241, 0.237758128, gamma, 0.0],
        [0.155416858, 0.701913790, 0.142669351, gamma]
    ])
    b = np.array([0.155416858, 0.701913790, 0.142669351, gamma])
    c = np.array([gamma, 0.344, 0.995, 1.0])
    t0, tf = t_span
    N = int(np.ceil((tf - t0)/h))
    t_grid = np.linspace(t0, tf, N+1)
    Y = np.zeros((N+1, len(y0)))
    Y[0] = y0
    for n in range(N): Y[n+1] = step_sdirk(f, t_grid[n], Y[n], h, A, b, c, sweeps=sweeps, tol=tol)
    return t_grid, Y
