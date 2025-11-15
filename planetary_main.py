import numpy as np
import yaml
from types import SimpleNamespace
import os
import math
from planetary_backend import run_simulation

#defines the types of functions for the planetary parameters and variables
def constant(value): return lambda *a, **kw: value
def linear_eos(P, T, rho0, alpha, K, T0): return rho0 * (1 + P/K - alpha*(T-T0))
def power_law_T(P, T, k0, T0, exp): return k0*(T0/T)**exp
def arrhenius(P, T, eta0, E_ast, V_ast, R_g=8.314): return eta0*np.exp((E_ast+P*V_ast)/(R_g*T))
def quadratic_Tm(P, T, Tm0, a, b): return Tm0+a*P+b*P**2
def multi_isotope(t, isotopes): return sum(float(iso["H0"]) * np.exp(-np.log(2) * t / float(iso["half_life"])) for iso in isotopes)
def constant_H(v): return lambda t: v

#coerces numeric-like strings to numbers
def _to_number(x):
    if isinstance(x, (int, float)): return x
    if isinstance(x, str):
        try:
            f = float(x)
            i = int(f)
            return i if f.is_integer() else f
        except Exception: return x
    return x

#recursively coerces numbers inside dicts/lists/tuples
def _coerce_numbers_in(obj):
    if isinstance(obj, dict): return {k: _coerce_numbers_in(v) for k, v in obj.items()}
    if isinstance(obj, list): return [_coerce_numbers_in(v) for v in obj]
    if isinstance(obj, tuple): return tuple(_coerce_numbers_in(v) for v in obj)
    return _to_number(obj)

#loads the yaml file and returns a dictionary
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

#builds the functions for the planetary parameters and variables
def build_function(defn, func_map):
    if isinstance(defn, (int, float, str)): return constant(_to_number(defn))
    if isinstance(defn, dict):
        ftype = defn.get("type", "constant")
        if ftype == "constant": return constant(_to_number(defn.get("value", 0.0)))
        if ftype in func_map:
            func = func_map[ftype]
            params = {}
            for k, v in defn.items():
                if k != "type":
                    key = str(k).strip() if k else None
                    if key: params[key] = _to_number(v)
            return (lambda p: lambda *a, **kw: func(*a, **p))(params.copy())
    raise ValueError(f"Unrecognized function definition {defn}")

#loads the yaml file and returns a dictionary
def load_planet_config(path):
    cfg = load_yaml(path)
    fns = {"linear_eos":linear_eos,"power_law_T":power_law_T,
           "arrhenius":arrhenius,"quadratic":quadratic_Tm,
           "quadratic_Tm":quadratic_Tm}

    planet_cfg = cfg["planet"]
    R_p = float(_to_number(planet_cfg["R_p"]))
    layers = []
    prev_outer = 0.0
    for idx, L in enumerate(planet_cfg["layers"]):
        r_outer = float(_to_number(L["r_outer"]))
        if r_outer <= prev_outer:
            raise ValueError(f"Layer '{L['name']}' has non-increasing outer radius (index {idx})")
        if r_outer - R_p > 5e-3 * R_p:
            raise ValueError(f"Layer '{L['name']}' outer radius {r_outer} exceeds planet radius {R_p}")

        lay = SimpleNamespace(name=L["name"], r_outer=r_outer)
        for key in ["rho","k","cp","alpha","K","eta","Tm"]:
            if key in L: setattr(lay, key, build_function(L[key], fns))
        for c in ["L","dT_mush"]:
            if c in L: setattr(lay, c, _to_number(L[c]))
        layers.append(lay)
        prev_outer = r_outer

    if not layers: raise ValueError("No layers defined for planet configuration")
    if not math.isclose(layers[-1].r_outer, R_p, rel_tol=0.0, abs_tol=1e-3 * R_p):
        raise ValueError(f"Outermost layer radius {layers[-1].r_outer} does not match planet radius {R_p}")

    rad = cfg.get("radiogenic",{})
    if isinstance(rad,(int,float,str)):
        val = float(_to_number(rad))
        H = constant_H(val)
    elif isinstance(rad, dict):
        rtype = str(rad.get("type", "constant")).lower()
        if rtype == "constant":
            val = float(_to_number(rad.get("H0", rad.get("value", 0.0))))
            H = constant_H(val)
        elif rtype == "multi_isotope":
            isotopes = [
                {**iso,
                 "H0": float(_to_number(iso["H0"])),
                 "half_life": float(_to_number(iso["half_life"]))}
                for iso in rad.get("isotopes", [])
            ]
            if not isotopes: raise ValueError("Radiogenic configuration 'multi_isotope' requires a non-empty 'isotopes' list")
            H = lambda t: multi_isotope(t, isotopes)
        else: raise ValueError(f"Unsupported radiogenic type '{rad.get('type')}'")
    else: raise ValueError("Invalid radiogenic configuration: must be scalar or dict")

    sim = _coerce_numbers_in(cfg.get("simulation",{}))
    integ = sim.get("integrator",{"type":"BE"})
    integrator = SimpleNamespace(
        type=integ.get("type","BE"),
        description=integ.get("description","Backward Euler")
    )

    output = cfg.get("output",{})
    save_path = output.get("save_path","results/output.npy")
    overwrite = bool(output.get("overwrite",True))
    save_dir = os.path.dirname(save_path)
    if save_dir: os.makedirs(save_dir,exist_ok=True)

    return SimpleNamespace(
        R_p=R_p, M_p=float(_to_number(planet_cfg["M_p"])), N=int(_to_number(planet_cfg.get("N",400))),
        layers=layers, radiogenic=H,
        boundary=_coerce_numbers_in(cfg.get("boundary",{})),
        initial=_coerce_numbers_in(cfg.get("initial",{})),
        convection=_coerce_numbers_in(cfg.get("convection",{})),
        constants=_coerce_numbers_in(cfg.get("constants",{})),
        sim=sim, integrator=integrator,
        output=SimpleNamespace(save_path=save_path, overwrite=overwrite)
    )

#function to ensure the directory exists and is empty
def ensure_dir(path: str): 
    if not os.path.exists(path):
        os.makedirs(path)
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

#function to run the simulation
def run(yaml_file, progress_tracker=None):
    #runs the simulation with the given YAML configuration file
    planet = load_planet_config(yaml_file) 
    results = run_simulation(planet, progress_tracker)

    #saves the results to a file
    ensure_dir(os.path.dirname(planet.output.save_path))
    results_path = planet.output.save_path
    if not results_path.endswith(".npz"):
        results_path += ".npz"
    np.savez_compressed(results_path, **results)

    #creates and returns the directory for the graphs
    graphs_dir = "graphs"
    ensure_dir(graphs_dir)
    return graphs_dir, results_path