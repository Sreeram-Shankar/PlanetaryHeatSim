import numpy as np
from solvers import solver_map


#builds the radial grid across the planet from center to surface
def build_radial_grid(planet):
    r_surface = float(planet.R_p)
    N = int(planet.N)
    r = np.linspace(0.0, r_surface, N)
    dr = np.diff(r)
    return r, dr

#computes the hydrostatic pressure profile
def hydrostatic_pressure(r, rho, g):
    P = np.zeros_like(r)
    for i in range(len(r)-2, -1, -1):
        dr = r[i+1] - r[i]
        P[i] = P[i+1] + rho[i] * g * dr
    return P

#initializes t(r,0) using adiabatic/isothermal definitions and falls back to a linear gradient
def initialize_temperature(planet, r):
    init_cfg = planet.initial
    if "T0" in init_cfg and isinstance(init_cfg["T0"], dict):
        Tcfg = init_cfg["T0"]
        if Tcfg.get("type") == "adiabatic":
            #applies the adiabatic profile: t(r) = tc * exp(alpha*g*r/cp)
            Tc = Tcfg.get("Tc", 3500)
            cp = Tcfg.get("cp", 1200)
            alpha = Tcfg.get("alpha", 3e-5)

            #computes the gravitational acceleration from planet parameters
            G = planet.constants.get("G", 6.6743e-11)
            M_p = planet.M_p
            R_p = planet.R_p
            g = G * M_p / (R_p**2)
            return Tc * np.exp(-alpha * g * r / cp)
        elif Tcfg.get("type") == "isothermal":
            #applies a uniform initial temperature across the whole radius
            return np.ones_like(r) * Tcfg.get("T", 300)

    #falls back to a simple linear gradient from interior hot to cool surface
    return np.linspace(3500, 288, len(r))


#dynamic evaluation of rho,k,cp,eta,tm at the current t profile each timestep
def evaluate_material_arrays(planet, r, T, P):
    #initializes the arrays for the material properties
    rho_arr = np.zeros_like(r)
    k_arr   = np.zeros_like(r)
    cp_arr  = np.zeros_like(r)
    eta_arr = np.zeros_like(r)
    Tm_arr  = np.zeros_like(r)

    #iterates through the layers and evaluates the material properties
    prev_outer = 0.0
    for L in planet.layers:
        if prev_outer == 0.0: mask = (r >= prev_outer) & (r <= L.r_outer)
        else: mask = (r > prev_outer) & (r <= L.r_outer)

        #create arrays for the material properties
        if hasattr(L,"rho"):
            if callable(L.rho): rho_arr[mask] = L.rho(P[mask], T[mask])
            else: rho_arr[mask] = L.rho

        if hasattr(L,"k"):
            if callable(L.k): k_arr[mask] = L.k(P[mask], T[mask])
            else: k_arr[mask] = L.k

        if hasattr(L,"cp"):
            if callable(L.cp): cp_arr[mask] = L.cp(P[mask], T[mask])
            else: cp_arr[mask] = L.cp

        if hasattr(L,"eta"):
            if callable(L.eta): eta_arr[mask] = L.eta(P[mask], T[mask])
            else: eta_arr[mask] = L.eta

        if hasattr(L,"Tm"):
            if callable(L.Tm): Tm_arr[mask] = L.Tm(P[mask], T[mask])
            else: Tm_arr[mask] = L.Tm

        prev_outer = L.r_outer

    #ensures no zero values are present
    default_cp = 1200.0
    default_rho = 3000.0
    default_k = 3.0
    default_eta = 1e23

    cp_arr = np.where(cp_arr > 0, cp_arr, default_cp)
    rho_arr = np.where(rho_arr > 0, rho_arr, default_rho)
    k_arr = np.where(k_arr > 0, k_arr, default_k)
    eta_arr = np.where(eta_arr > 0, eta_arr, default_eta)
    Tm_arr = np.where(Tm_arr > 0, Tm_arr, 2000.0)

    #applies melt weakening: mush reduces viscosity strongly
    mush_mask = T > Tm_arr
    eta_arr[mush_mask] *= 1e-10

    return rho_arr, k_arr, cp_arr, eta_arr

#computes the heat rhs for spherical diffusion: (1/r^2) d/dr( r^2 k_eff dt/dr ) + h/(rho*cp)
def heat_rhs(T, t, r, rho, cp, k, eta, H_func, boundary, planet):
    N = len(r)
    dr = np.diff(r)
    dTdt = np.zeros_like(T)

    #assumes gravity is constant at surface value
    G = planet.constants.get("G", 6.6743e-11)
    g = G * planet.M_p / (planet.R_p**2)

    #defines the pressure field
    P = hydrostatic_pressure(r, rho, g)

    #computes the convection scaling (parameterized nusselt) - mantle only
    r_core = planet.layers[0].r_outer
    mantle_mask = r >= r_core

    #defines the convection parameters
    Ra_c = planet.convection.get("Ra_c",1000)
    A_c = planet.convection.get("A",0.1)
    n_c = planet.convection.get("n",0.333)

    #defines the index of the CMB and the surface
    i_cmb = np.argmax(mantle_mask)
    T_cmb = T[i_cmb]
    T_surf = T[-1]
    DeltaT = T_cmb - T_surf
    D = planet.R_p - r_core

    #computes the characteristic mantle viscosity, conductivity, density, cp - mid mantle sample
    i_mid = int(0.5*(i_cmb + (N-1)))
    kappa = k[i_mid]/(rho[i_mid]*cp[i_mid])
    alpha_therm = 3e-5

    #computes the rayleigh number: ra = rho*alpha*g*deltat*d^3 / (kappa*eta)
    Ra = rho[i_mid]*alpha_therm*g*np.abs(DeltaT)*D*D*D/(kappa*eta[i_mid] + 1e-30)
    Nu = 1.0 + A_c*(Ra/Ra_c)**n_c
    if A_c == 0.0: Nu = 1.0

    #applies effective conductivity with convection enhancement
    k_eff = k.copy()
    k_eff[mantle_mask] *= Nu

    #computes the interface conductivities with arithmetic mean
    k_mid = 0.5 * (k_eff[:-1] + k_eff[1:])

    #iterates through the layers and computes the heat rhs
    for i in range(1, N-1):
        r_i = r[i]
        drL, drR = dr[i-1], dr[i]
        term_L = k_mid[i-1] * (T[i-1] - T[i]) / drL
        term_R = k_mid[i]   * (T[i+1] - T[i]) / drR
        div_q = (r_i**-2) * ((r_i+drR/2)**2 * term_R - (r_i-drL/2)**2 * term_L) / ((drL+drR)/2)
        dTdt[i] = div_q/(rho[i]*cp[i]) + H_func(t)/(rho[i]*cp[i])

    #defines the center boundary condition
    if r[0] < 1e-10:
        dr0 = dr[0]
        dTdt[0] = (3.0*k_eff[0]*(T[1]-T[0])/(dr0**2))/(rho[0]*cp[0]) + H_func(t)/(rho[0]*cp[0])
    else: dTdt[0] = 0

    #defines the surface boundary condition
    bc = boundary.get("surface_BC",{"type":"neumann"})
    if bc["type"].lower()=="dirichlet":

        #fixes t[-1] at surface temperature
        Tsurf = bc.get("value",288.0)
        T[-1] = Tsurf

        #approximates outward flux using last interior node
        dr_surf = dr[-1]
        if len(k_mid) > 0: k_surface = k_mid[-1]
        else: k_surface = k_eff[-1]
        q_out = k_surface * (T[-2] - Tsurf) / dr_surf
        dTdt[-1] = -q_out / (rho[-1]*cp[-1]) + H_func(t)/(rho[-1]*cp[-1])
        dTdt[-2] += q_out/(rho[-2]*cp[-2])

    else:
        #defines flux-based neumann boundary condition
        mode = bc.get("mode", "insulated").lower()
        if mode == "radiative":
            sigma = 5.670374419e-8
            eps = bc.get("emissivity", 0.9)
            T_space = bc.get("T_space", 3.0)
            q_surf = eps * sigma * (T[-1]**4 - T_space**4)
        elif mode == "constant": q_surf = bc.get("q_flux", 0.0)
        else: q_surf = 0.0
        dr_surf = dr[-1]
        denom = np.maximum(rho[-1] * cp[-1] * dr_surf, 1e-10)
        dTdt[-1] = -q_surf / denom
    return dTdt

#runs the simulation for planetary heat diffusion
def run_simulation(planet, progress_tracker=None):

    #prepares grid, initial state, heating source
    r, dr = build_radial_grid(planet)
    T0 = initialize_temperature(planet, r)
    H_func = planet.radiogenic

    #reads simulation parameters and selects solver
    t_max = planet.sim.get("t_max", 4.5e9)
    dt = planet.sim.get("dt", 1e5)
    solver_type = planet.integrator.type.upper()

    #raises an error if the solver type is not supported
    if solver_type not in solver_map: raise ValueError(f"Unknown integrator '{solver_type}'")
    solver = solver_map[solver_type]

    #calculates total steps for progress tracking
    total_steps = int(round(t_max / dt))
    if progress_tracker is not None:
        progress_tracker.total_steps = total_steps
        progress_tracker.current_step = 0

    #initializes the current step
    current_step = [-1]

    #defines the function for the solver
    def f(t, T):
        #defines the gravitational acceleration
        G = planet.constants.get("G", 6.6743e-11)
        g = G * planet.M_p / (planet.R_p**2)

        #uses constant density for pressure estimate
        rho_guess = np.full_like(T, 4000.0)
        P = hydrostatic_pressure(r, rho_guess, g)

        #computes material properties with pressure estimate
        rho, k, cp, eta = evaluate_material_arrays(planet, r, T, P)

        #recomputes pressure with actual density
        P = hydrostatic_pressure(r, rho, g)

        #revaluates material properties with correct pressure
        rho, k, cp, eta = evaluate_material_arrays(planet, r, T, P)
        step_num = int(round(t/dt))
        if step_num != current_step[0]:
            current_step[0] = step_num

            #updates the progress tracker only when step increases
            if progress_tracker is not None: 
                if step_num > progress_tracker.current_step: progress_tracker.current_step = step_num
        return heat_rhs(T,t,r,rho,cp,k,eta,H_func,planet.boundary,planet)
    t_grid, T_hist = solver(f,(0,t_max),T0,dt)

    #computes final material properties with proper pressure
    G = planet.constants.get("G", 6.6743e-11)
    g = G * planet.M_p / (planet.R_p**2)

    #computes pressure iteratively: rho from t, then p, then final rho
    rho_final_guess = np.full_like(T_hist[-1], 4000.0)
    P_final = hydrostatic_pressure(r, rho_final_guess, g)
    rho_last, k_last, cp_last, eta_last = evaluate_material_arrays(planet, r, T_hist[-1], P_final)
    P_final = hydrostatic_pressure(r, rho_last, g)
    rho_last, k_last, cp_last, eta_last = evaluate_material_arrays(planet, r, T_hist[-1], P_final)

    #returns the results
    results={
        "t":t_grid,
        "r":r,
        "T":T_hist,
        "rho":rho_last,
        "k":k_last,
        "cp":cp_last,
        "eta":eta_last
    }
    return results