# 🌎 PlanetaryHeatSim  
*A full-featured 1D planetary thermal evolution simulator with a modern GUI, YAML-driven physics engine, implicit/explicit ODE solvers, multiprocessing visualization, and customizable planetary models.*

---

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" />
  <img src="https://img.shields.io/badge/license-MIT-green.svg" />
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Mac%20%7C%20Linux-purple.svg" />
  <img src="https://img.shields.io/badge/gui-CustomTkinter-pink.svg" />
  <img src="https://img.shields.io/badge/numerics-ODE%20Solvers%20%7C%20Heat%20Diffusion-orange.svg" />
</p>

---

## 📌 Overview

**PlanetaryHeatSim** is a research-grade 1D spherical heat diffusion model designed to simulate the thermal evolution of terrestrial bodies (core → mantle → surface). It includes:

- A full **graphical user interface** (GUI) for editing YAML configurations  
- A modular backend with **layered material properties**, radiogenic decay models, convective scaling laws, and rich boundary condition support  
- A flexible suite of **explicit and implicit time-stepping solvers**  
- Automatic **plot generation** (core T, heatmap, flux, cooling rate, material profiles)  
- A parallelized **temperature profile viewer** with slider  
- Export features (ZIPed graphs + NPZ results)  
- Presets for Earth, Mars, Moon, and Io  

This is essentially a small research codebase disguised as a clean, user-friendly desktop app.

---

## 🎨 GUI Features (CustomTkinter)

The GUI provides:

### **✔ YAML Editor Pane**
- Full-width syntax-colored editor  
- Auto-loads `planet.yaml`  
- Buttons with documentation popups for every section:
  - Planet  
  - Layers  
  - Radiogenic  
  - Boundary conditions  
  - Initial conditions  
  - Convection  
  - Constants  
  - Simulation  
  - Output  

### **✔ Presets**
Load predefined planets:
- Earth  
- Mars  
- Moon  
- Io  

### **✔ Controls**
- Save YAML  
- Reset to file contents  
- Toggle dark/light theme  
- Start simulation  
- Export graphs  
- Export NPZ data  
- Restart app  

### **✔ Progress Tracking**
- Animated progress bar tied to backend  
- Real-time updates via polling  
- Automatic visual transition from simulation → plotting → results  

### **✔ Visualization Suite**
Six major plots generated automatically:

1. Core vs Surface Temperature  
2. Temperature Heatmap (radius × time)  
3. Material Property Profiles  
4. Surface Heat Flux  
5. Mean Cooling Rate  
6. Temperature-at-time profiles (100 time points, via multiprocessing)

### **✔ Interactive Temperature Viewer**
- Slider with 100 time steps  
- Live, auto-loading PNG images  
- High-resolution CTk rendering  

---

## 🧪 Physics & Numerical Model

The model solves the **spherical heat diffusion equation**:

```math
\frac{\partial T}{\partial t}
=
\frac{1}{\rho c_p r^2}
\frac{\partial}{\partial r}\left(
r^2 k_{\mathrm{eff}} \frac{\partial T}{\partial r}
\right)
+
\frac{H(t)}{\rho c_p}
```

with:

- Radial grid: uniform, center → surface  
- Density from:
  - constant  
  - linear EOS  
- Conductivity from:
  - constant  
  - power-law in T  
- Viscosity:
  - constant  
  - Arrhenius rheology  
- Melting temperature:
  - constant  
  - quadratic in P  

### 🔥 Radiogenic Heating
- Constant  
- Multi-isotope exponential decay  

### 🌋 Convection (Nusselt parameterization)

```math
Nu = 1 + A \left( \frac{Ra}{Ra_c} \right)^n
```

applied only to the mantle region.

---

## 🧩 YAML Configuration Schema

### **planet**
```yaml
planet:
  R_p: 6.371e6
  M_p: 5.972e24
  N: 400
  layers:
    - name: core
      r_outer: 3.48e6
      rho: 12000
      k: 40
      cp: 800
```

### **layers**
Supports rich physical models:

```yaml
rho:
  type: linear_eos
  rho0: 12000
  alpha: 3e-5
  K: 1e11
  T0: 3000

k:
  type: power_law_T
  k0: 5000
  T0: 3000
  exp: 1.0

eta:
  type: arrhenius
  eta0: 1e21
  E_ast: 300000
  V_ast: 1e-6

Tm:
  type: quadratic_Tm
  Tm0: 2000
  a: 1e-8
  b: 1e-18
```

### **radiogenic**
```yaml
radiogenic:
  type: multi_isotope
  isotopes:
    - H0: 1e-6
      half_life: 4.5e9
    - H0: 5e-7
      half_life: 7e8
```

### **boundary**
```yaml
boundary:
  surface_BC:
    type: dirichlet
    value: 250
```

Or flux-based:
```yaml
type: neumann
mode: radiative
emissivity: 0.9
T_space: 3
```

### **initial**
```yaml
initial:
  T0:
    type: adiabatic
    Tc: 3000
    cp: 1200
```

### **convection**
```yaml
convection:
  Ra_c: 1e99
  A: 0.0
  n: 1.0
```

### **simulation**
```yaml
simulation:
  t_max: 1.5e7
  dt: 1e5
  integrator:
    type: BDF4
```

### **output**
```yaml
output:
  save_path: "results/run_fastcool"
  overwrite: true
```

---

## ⚙️ Numerical Solvers (solvers/ folder)

All solvers follow the unified interface:

```python
t, Y_hist = solver(f, (t0, t1), y0, dt)
```

### **Explicit RK Solvers**
From `rk.py`:

- RK1 (Euler)  
- RK2 (Heun)  
- RK3  
- RK4 (classic)  
- RK5  
- RK6  

### **Adams–Bashforth (AB) — Explicit**
From `ab.py`:

- AB2  
- AB3  
- AB4  
- AB5  

### **Adams–Moulton (AM) — Implicit Picard**
From `am.py`:

- AM2  
- AM3  
- AM4  
- AM5  

### **Backward Differentiation Formulas (BDF) — Implicit**
From `bdf.py`:

- BDF1 (Backward Euler)  
- BDF2  
- BDF3  
- BDF4  
- BDF5  
- BDF6  

### **IRK Families (Implicit Runge–Kutta)**  
All using your quadrature-generated Butcher tables.

#### Gauss–Legendre (`gauss_legendre.py`)
- GL1  
- GL2  
- GL3  
- GL4  
- GL5  

#### Radau IIA (`radau.py`)
- R1 / BE  
- R2  
- R3  
- R4  
- R5  

#### Lobatto IIIC (`lobatto.py`)
- L1 (CN)  
- L2  
- L3  
- L4  
- L5  

### **SDIRK (diagonally implicit)**
From `sdirk.py`:

- SDIRK2  
- SDIRK3  
- SDIRK4  

All implicit methods use your **Picard–Gauss–Seidel** nonlinear relaxation method — perfectly suited for PDE-style stiff diffusion.

---

## 🧵 Multiprocessing Implementation

Temperature profiles (100 time slices) are plotted in parallel:

```python
with multiprocessing.Pool(n_cores) as pool:
    pool.map(_plot_profile_at_time_worker, args_list)
```

This dramatically improves speed and is fully safe because Matplotlib is used in non-interactive (`Agg`) mode.

---

## 📦 Folder Structure

```
PlanetaryHeatSim/
├── planetary_frontend.py
├── planetary_backend.py
├── planetary_main.py
├── planetary_visuals.py
├── solvers/
│   ├── __init__.py
│   ├── rk.py
│   ├── ab.py
│   ├── am.py
│   ├── bdf.py
│   ├── irk.py
│   ├── gauss_legendre.py
│   ├── radau.py
│   ├── lobatto.py
│   ├── sdirk.py
├── presets/
│   ├── earth.yaml
│   ├── mars.yaml
│   ├── moon.yaml
│   ├── io.yaml
├── results/
├── graphs/
└── README.md
```

---

## 🚀 Running the Program

### Install dependencies:

```bash
pip install numpy scipy pyyaml matplotlib customtkinter pillow mpmath
```

### Launch:

```bash
python planetary_frontend.py
```

---

## 🛠 Export Features

After simulation:

- **Export All Graphs** → ZIP with all PNGs  
- **Export Data** → Saves full `.npz` with arrays:
  - `t`, `r`, `T`, `rho`, `k`, `cp`, `eta`

Great for research, plotting in Jupyter, or importing into other scripts.

---

## 🧑‍🎓 Author

**Sreeram Shankar**  
High school computational engineering + numerical analysis researcher

---

## 📄 License (MIT)

```
MIT License
```

---
