import os
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

#defines the global visual settings for light mode
light_settings = {
    "font.family": "Times New Roman",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "lines.linewidth": 2.2,
    "lines.markersize": 5,
    "legend.fontsize": 10,
    "figure.figsize": (6, 4),
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "figure.facecolor": "#ffe6f2",
    "axes.facecolor": "#fff5f9",
    "axes.edgecolor": "#5e173a",
    "axes.labelcolor": "#5e173a",
    "text.color": "#5e173a",
    "xtick.color": "#5e173a",
    "ytick.color": "#5e173a",
    "grid.color": "#e0aac4",
    "figure.edgecolor": "#ffe6f2",
    "legend.facecolor": "#fff5f9",
    "legend.edgecolor": "#5e173a",
    "legend.labelcolor": "#5e173a",
}

#defines the global visual settings for dark mode
dark_settings = {
    "font.family": "Times New Roman",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "lines.linewidth": 2.2,
    "lines.markersize": 5,
    "legend.fontsize": 10,
    "figure.figsize": (6, 4),
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "figure.facecolor": "#2e0b0b",
    "axes.facecolor": "#1a0505",
    "axes.edgecolor": "#ffd8ec",
    "axes.labelcolor": "#ffd8ec",
    "text.color": "#ffd8ec",
    "xtick.color": "#ffd8ec",
    "ytick.color": "#ffd8ec",
    "grid.color": "#994c6f",
    "figure.edgecolor": "#2e0b0b",
    "legend.facecolor": "#1a0505",
    "legend.edgecolor": "#ffd8ec",
    "legend.labelcolor": "#ffd8ec",
}

#function to apply theme settings
def apply_theme(theme="light"):
    if theme.lower() == "dark":
        plt.rcParams.update(dark_settings)
    else:
        plt.rcParams.update(light_settings)

#function to ensure the directory exists
def ensure_dir(path): return os.makedirs(path, exist_ok=True)

#function to find the nearest value in an array
def find_nearest(array, value): return (np.abs(array - value)).argmin()

#plots the core and surface temperature evolution
def plot_core_surface_temperature(results, output_dir, theme="light"):
    apply_theme(theme)
    t = results["t"]
    T_central = results["T"][:, 0]
    T_surface = results["T"][:, -1]

    plt.figure()
    plt.plot(t, T_central, label="Core", color="#1b9e77")
    plt.plot(t, T_surface, label="Surface", color="#d95f02")
    plt.xlabel("Time (yr)")
    plt.ylabel("Temperature (K)")
    plt.title("Core and surface temperature evolution")
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.3e}"))
    plt.legend()
    plt.tight_layout()
    filepath = os.path.join(output_dir, "T_core_surface_vs_time.png")
    plt.savefig(filepath)
    plt.close()
    return filepath


#plots the temperature heatmap (r vs t)
def plot_temperature_heatmap(results, output_dir, theme="light"):
    apply_theme(theme)
    r = results["r"] / 1e6
    t = results["t"]
    T = results["T"]

    plt.figure(figsize=(7, 4))
    im = plt.imshow(
        T,
        extent=[r[0], r[-1], t[-1], t[0]],
        aspect="auto",
        cmap="inferno",
        origin="upper"
    )
    plt.colorbar(im, label="Temperature (K)")
    plt.xlabel("Radius (×10⁶ m)")
    plt.ylabel("Time (yr)")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.3e}"))
    plt.title("Thermal evolution (T[r,t])")
    plt.tight_layout()
    filepath = os.path.join(output_dir, "T_heatmap_r_t.png")
    plt.savefig(filepath)
    plt.close()
    return filepath


#plots the material property profiles
def plot_material_profiles(results, output_dir, theme="light"):
    apply_theme(theme)
    r = results["r"] / 1e6
    rho, k, cp = results["rho"], results["k"], results["cp"]

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].plot(r, rho, color="#1f77b4")
    ax[1].plot(r, k, color="#ff7f0e")
    ax[2].plot(r, cp, color="#2ca02c")
    labels = ["Density (kg/m³)", "Conductivity (W/m·K)", "Specific heat (J/kg·K)"]
    for i, a in enumerate(ax):
        a.set_xlabel("Radius (×10⁶ m)")
        a.set_ylabel(labels[i])
        a.grid(True, ls="--", lw=0.5)
    fig.suptitle("Material property profiles", fontsize=13)
    plt.tight_layout()
    filepath = os.path.join(output_dir, "material_profiles.png")
    plt.savefig(filepath)
    plt.close(fig)
    return filepath


#plots the surface heat flux vs time
def plot_surface_heat_flux(results, output_dir, theme="light"):
    apply_theme(theme)
    r = results["r"]
    t = results["t"]
    dr = np.diff(r)
    q_surf = -results["k"][-1] * (results["T"][:, -1] - results["T"][:, -2]) / dr[-1]

    plt.figure()
    plt.plot(t, q_surf, color="#7570b3")
    plt.xlabel("Time (yr)")
    plt.ylabel("Surface heat flux (W/m²)")
    plt.title("Surface heat flux over time")
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.3e}"))
    plt.tight_layout()
    filepath = os.path.join(output_dir, "surface_heat_flux_vs_time.png")
    plt.savefig(filepath)
    plt.close()
    return filepath


#plots the mean cooling rate vs time
def plot_mean_cooling_rate(results, output_dir, theme="light"):
    apply_theme(theme)
    t = results["t"]
    T = results["T"]
    T_mean = np.mean(T, axis=1)
    dTdt = np.gradient(T_mean, t)

    plt.figure()
    plt.plot(t, -dTdt, color="#e7298a")
    plt.xlabel("Time (yr)")
    plt.ylabel("Mean cooling rate (K/s)")
    plt.title("Global cooling rate")
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.3e}"))
    plt.tight_layout()
    filepath = os.path.join(output_dir, "mean_cooling_rate_vs_time.png")
    plt.savefig(filepath)
    plt.close()
    return filepath


#worker function for multiprocessing that plots a single temperature profile at a given time index
def _plot_profile_at_time_worker(args):
    idx, r, T, t, output_dir, theme = args
    
    #sets non-interactive backend for multiprocessing compatibility
    import matplotlib
    matplotlib.use('Agg')

    #reimports after setting backend
    import matplotlib.pyplot as plt
    
    #applies theme settings manually 
    if theme.lower() == "dark":
        settings = {
            "font.family": "Times New Roman", "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
            "lines.linewidth": 2.2, "lines.markersize": 5, "legend.fontsize": 10, "figure.figsize": (6, 4),
            "savefig.dpi": 300, "axes.grid": True, "grid.linestyle": "--", "grid.linewidth": 0.5,
            "figure.facecolor": "#2e0b0b", "axes.facecolor": "#1a0505", "axes.edgecolor": "#ffd8ec",
            "axes.labelcolor": "#ffd8ec", "text.color": "#ffd8ec", "xtick.color": "#ffd8ec", "ytick.color": "#ffd8ec",
            "grid.color": "#994c6f", "figure.edgecolor": "#2e0b0b", "legend.facecolor": "#1a0505",
            "legend.edgecolor": "#ffd8ec", "legend.labelcolor": "#ffd8ec",
        }
    else:
        settings = {
            "font.family": "Times New Roman", "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
            "lines.linewidth": 2.2, "lines.markersize": 5, "legend.fontsize": 10, "figure.figsize": (6, 4),
            "savefig.dpi": 300, "axes.grid": True, "grid.linestyle": "--", "grid.linewidth": 0.5,
            "figure.facecolor": "#ffe6f2", "axes.facecolor": "#fff5f9", "axes.edgecolor": "#5e173a",
            "axes.labelcolor": "#5e173a", "text.color": "#5e173a", "xtick.color": "#5e173a", "ytick.color": "#5e173a",
            "grid.color": "#e0aac4", "figure.edgecolor": "#ffe6f2", "legend.facecolor": "#fff5f9",
            "legend.edgecolor": "#5e173a", "legend.labelcolor": "#5e173a",
        }
    plt.rcParams.update(settings)
    
    ensure_dir(output_dir)
    actual_time = t[idx]

    fig, ax = plt.subplots()
    ax.plot(r, T[idx], color="crimson")
    ax.set_xlabel("Radius (×10⁶ m)")
    ax.set_ylabel("Temperature (K)")
    #uses scientific notation with 3 decimal places for time
    ax.set_title(f"Temperature profile at t = {actual_time:.3e} yr")
    plt.tight_layout()

    #uses index in filename to ensure uniqueness and scientific notation for time
    time_str = f"{actual_time:.3e}".replace('e+', 'ep').replace('e-', 'em')
    fname = f"T_profile_at_{idx:03d}_{time_str}yr.png"
    filepath = os.path.join(output_dir, fname)
    fig.savefig(filepath)
    plt.close(fig)

#plots the temperature profile at a given time
def plot_profile_at_time(results, time_value, output_dir, theme="light"):
    apply_theme(theme)
    ensure_dir(output_dir)
    t = results["t"]
    r = results["r"] / 1e6
    T = results["T"]

    idx = find_nearest(t, time_value)
    actual_time = t[idx]

    plt.figure()
    plt.plot(r, T[idx], color="crimson")
    plt.xlabel("Radius (×10⁶ m)")
    plt.ylabel("Temperature (K)")
    plt.title(f"Temperature profile at t = {actual_time:.3e} yr")
    plt.tight_layout()
    time_str = f"{actual_time:.3e}".replace('e+', 'ep').replace('e-', 'em')
    fname = f"T_profile_at_{time_str}yr.png"
    plt.savefig(os.path.join(output_dir, fname))
    plt.close()

#plots temperature profiles at all time points using multiprocessing
def plot_all_profiles_at_time(results, output_dir, theme="light"):
    ensure_dir(output_dir)
    t = results["t"]
    r = results["r"] / 1e6
    T = results["T"]
    
    #creates arguments list for all time indices in the subsampled data
    num_time_points = len(t)
    args_list = [(idx, r, T, t, output_dir, theme) for idx in range(num_time_points)]
    
    #defines the number of parallel operations
    n_cores = max(multiprocessing.cpu_count() - 2, 1)
    
    #uses parallel programming to plot all profiles
    with multiprocessing.Pool(n_cores) as pool: results_list = pool.map(_plot_profile_at_time_worker, args_list)

