import numpy as np

#solves the nonlinear system of equations with a Gauss-Seidel relaxation AM2
def solve_am2(f, t_span, y0, h, sweeps=12, tol=1e-10):
    t0, tf = t_span
    N = int(np.ceil((tf - t0)/h))
    t_grid = np.linspace(t0, tf, N+1)
    Y = np.zeros((N+1, len(y0)))
    F = np.zeros((N+1, len(y0)))
    Y[0] = y0
    F[0] = f(t_grid[0],Y[0])

    #uses BE bootstrap
    Y[1] = Y[0] + h*f(t_grid[1],Y[0])
    F[1] = f(t_grid[1],Y[1])

    for n in range(1,N):
        t_next = t_grid[n+1]
        f_n    = F[n]
        y      = Y[n].copy()

        #implements Gauss-Seidel relaxation
        for _ in range(sweeps):
            y_old = y.copy()
            f_next = f(t_next,y)
            y = Y[n] + h*(0.5*f_next + 0.5*f_n)
            if np.linalg.norm(y - y_old) < tol:
                break

        Y[n+1] = y
        F[n+1] = f_next

    return t_grid,Y

#solves the nonlinear system of equations with a Gauss-Seidel relaxation AM3
def solve_am3(f, t_span, y0, h, sweeps=12, tol=1e-10):
    #uses AM2 to bootstrap 1 step
    t_grid,Y = solve_am2(f,(t_span[0],t_span[0]+2*h),y0,h,sweeps,tol)
    t0,tf = t_span
    N = int(np.ceil((tf-t0)/h))
    t_grid=np.linspace(t0,tf,N+1)

    #compute initial F
    F = np.zeros((N+1,len(y0)))
    for i in range(3):
        F[i]=f(t_grid[i],Y[i])

    #defines the main AM3 solver
    for n in range(2,N):
        t_next = t_grid[n+1]
        f_n    = F[n]
        f_nm1  = F[n-1]
        y      = Y[n].copy()

        #implements Gauss-Seidel relaxation
        for _ in range(sweeps):
            y_old = y.copy()
            f_next = f(t_next,y)
            y = Y[n] + h*( (5/12)*f_next + (2/3)*f_n - (1/12)*f_nm1 )
            if np.linalg.norm(y - y_old) < tol:
                break

        Y[n+1]=y
        F[n+1]=f_next

    return t_grid,Y

#solves the nonlinear system of equations with a Gauss-Seidel relaxation AM4
def solve_am4(f, t_span, y0, h, sweeps=12, tol=1e-10):
    #uses AM3 to bootstrap 2 steps
    t_grid,Y = solve_am3(f,(t_span[0],t_span[0]+3*h),y0,h,sweeps,tol)
    t0,tf = t_span
    N = int(np.ceil((tf-t0)/h))
    t_grid=np.linspace(t0,tf,N+1)

    #compute initial F
    F = np.zeros((N+1,len(y0)))
    for i in range(4):
        F[i]=f(t_grid[i],Y[i])

    #defines the main AM4 solver
    for n in range(3,N):
        t_next = t_grid[n+1]
        f_n    = F[n]
        f_nm1  = F[n-1]
        f_nm2  = F[n-2]
        y      = Y[n].copy()

        #implements Gauss-Seidel relaxation
        for _ in range(sweeps):
            y_old = y.copy()
            f_next = f(t_next,y)
            y = Y[n] + h*( (3/8)*f_next + (19/24)*f_n - (5/24)*f_nm1 + (1/24)*f_nm2 )
            if np.linalg.norm(y - y_old) < tol:
                break

        Y[n+1]=y
        F[n+1]=f_next

    return t_grid,Y

#solves the nonlinear system of equations with a Gauss-Seidel relaxation AM5
def solve_am5(f, t_span, y0, h, sweeps=12, tol=1e-10):
    #uses AM4 to bootstrap 3 steps of history
    t_grid, Y = solve_am4(f, (t_span[0], t_span[0] + 4*h), y0, h, sweeps, tol)
    t0, tf = t_span
    N = int(np.ceil((tf - t0) / h))
    t_grid = np.linspace(t0, tf, N+1)

    #computes initial F
    F = np.zeros((N+1, len(y0)))
    for i in range(5):
        F[i] = f(t_grid[i], Y[i])

    #defines the main AM5 solver
    for n in range(4, N):
        t_next = t_grid[n+1]
        f_n    = F[n]
        f_nm1  = F[n-1]
        f_nm2  = F[n-2]
        f_nm3  = F[n-3]
        y      = Y[n].copy()

        #implements Gauss-Seidel relaxation
        for _ in range(sweeps):
            y_old = y.copy()
            f_next = f(t_next, y)
            y = Y[n] + h * (
                (251/720)*f_next +
                (646/720)*f_n   -
                (264/720)*f_nm1 +
                (106/720)*f_nm2 -
                (19/720)*f_nm3
            )
            if np.linalg.norm(y - y_old) < tol:
                break

        Y[n+1] = y
        F[n+1] = f_next

    return t_grid, Y