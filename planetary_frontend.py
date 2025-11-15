import customtkinter as ctk
from tkinter import scrolledtext
import os
import mpmath as mp
import planetary_main
import threading
import numpy as np
from scipy.interpolate import interp1d
from PIL import Image
from customtkinter import CTkImage
from planetary_visuals import (
    plot_core_surface_temperature,
    plot_temperature_heatmap,
    plot_material_profiles,
    plot_surface_heat_flux,
    plot_mean_cooling_rate,
    plot_all_profiles_at_time,
)
mp.mp.dps = 200


#sets the appearance of the oveall window
ctk.set_default_color_theme("theme.json")
ctk.set_appearance_mode("system")

#class that tracks simulation progress
class ProgressTracker:
    def __init__(self):
        self.current_step = 0
        self.total_steps = 1

#class that contains the window and all widgets
class PlanetaryApp(ctk.CTk):
  #creates and configures the root
  def __init__(self):
    super().__init__()
    self.title("Planetary Heat Diffusion Simulator")
    self.geometry("1000x800")
    self.resizable(False, False)
    self.yaml_file = "planet.yaml"
    self.simulation_started = False  
    self.build_gui()

  #function that builds all the gui components of the window
  def build_gui(self):
    #configures the grid layout of the window
    self.grid_rowconfigure(0, weight=0)  
    self.grid_rowconfigure(1, weight=1)  
    self.grid_rowconfigure(2, weight=0)  
    for j in range(10): self.grid_columnconfigure(j, weight=1)
    
    #creates and places the main label
    self.main_label = ctk.CTkLabel(self, text="Planetary Heat Diffusion Simulator - YAML Configuration Editor", font=("Times New Roman", 28))
    self.main_label.grid(row=0, column=0, columnspan=10, pady=5, sticky="ew")

    #creates the text editor frame
    self.editor_frame = ctk.CTkFrame(self)
    self.editor_frame.grid(row=1, column=0, columnspan=10, padx=10, pady=5, sticky="nsew")
    self.editor_frame.grid_rowconfigure(0, weight=1)
    self.editor_frame.grid_columnconfigure(0, weight=1)

    #creates the text editor using scrolledtext wrapped in CTkFrame with theme-appropriate colors
    current_theme = ctk.get_appearance_mode().lower()
    if current_theme == "light":
      bg_color = "#fff5f9"  
      fg_color = "#5e173a"   
      insert_color = "#5e173a"
    else:
      bg_color = "#1a0505"
      fg_color = "#ffd8ec"
      insert_color = "#ffd8ec"
    
    #creates the text editor
    self.text_editor = scrolledtext.ScrolledText(self.editor_frame, wrap="none", font=("Consolas", 16), bg=bg_color, fg=fg_color, insertbackground=insert_color)
    self.text_editor.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

    #loads the yaml file into the editor
    self.load_yaml()

    #creates the button frame at the bottom
    self.button_frame = ctk.CTkFrame(self)
    self.button_frame.grid(row=2, column=0, columnspan=10, padx=10, pady=10, sticky="ew")
    self.button_frame.grid_rowconfigure(0, weight=1)
    self.button_frame.grid_rowconfigure(1, weight=1)
    for j in range(9): self.button_frame.grid_columnconfigure(j, weight=1)

    #defines the buttons for the first row
    buttons_row1 = [
      ("Info", self.show_info),
      ("Planet", self.show_planet_docs),
      ("Layers", self.show_layers_docs),
      ("Radiogenic", self.show_radiogenic_docs),
      ("Boundary", self.show_boundary_docs),
      ("Initial", self.show_initial_docs),
      ("Convection", self.show_convection_docs),
      ("Constants", self.show_constants_docs),
      ("Simulation", self.show_simulation_docs)
    ]

    #places the buttons for the first row
    for i, (text, command) in enumerate(buttons_row1):
      btn = ctk.CTkButton(self.button_frame, text=text, command=command, font=("Times New Roman", 18), width=120)
      btn.grid(row=0, column=i, padx=5, pady=5)

    #defines the buttons for the second row
    buttons_row2 = [
      ("Output", self.show_output_docs),
      ("Earth", lambda: self.load_preset("earth")),
      ("Mars", lambda: self.load_preset("mars")),
      ("Moon", lambda: self.load_preset("moon")),
      ("Io", lambda: self.load_preset("io")),
      ("Save", self.save_yaml),
      ("Reset", self.reset_yaml),
      ("Theme", self.toggle_theme),
      ("Start", self.run_simulation)
    ]

    #places the buttons for the second row
    for i, (text, command) in enumerate(buttons_row2):
      btn = ctk.CTkButton(self.button_frame, text=text, command=command, font=("Times New Roman", 18), width=120)
      btn.grid(row=1, column=i, padx=5, pady=5)

  #function that loads the yaml file into the editor
  def load_yaml(self):

    #loads the yaml file into the editor
    if os.path.exists(self.yaml_file):
      with open(self.yaml_file, "r") as f:
        content = f.read()
      self.text_editor.delete("1.0", "end")
      self.text_editor.insert("1.0", content)

    #displays an error message if the file is not found
    else:
      self.text_editor.delete("1.0", "end")
      self.text_editor.insert("1.0", "# YAML file not found. Create your configuration here.\n")

  #function that loads a preset yaml file into the editor
  def load_preset(self, preset_name):

    #loads the preset yaml file into the editor
    preset_file = os.path.join("presets", f"{preset_name}.yaml")
    if os.path.exists(preset_file):
      with open(preset_file, "r") as f:
        content = f.read()
      self.text_editor.delete("1.0", "end")
      self.text_editor.insert("1.0", content)
      self.main_label.configure(text=f"Loaded {preset_name.capitalize()} preset")
      if not self.simulation_started:
        self.after(2000, lambda: self.main_label.configure(text="Planetary Heat Diffusion Simulator - YAML Configuration Editor"))

    #displays an error message if the preset file is not found
    else:
      self.main_label.configure(text=f"Preset file not found: {preset_file}")
      if not self.simulation_started:
        self.after(2000, lambda: self.main_label.configure(text="Planetary Heat Diffusion Simulator - YAML Configuration Editor"))

  #function that saves the yaml file from the editor
  def save_yaml(self):

    #saves the yaml file
    content = self.text_editor.get("1.0", "end-1c")
    try:
      with open(self.yaml_file, "w") as f:
        f.write(content)
      self.main_label.configure(text="YAML saved successfully!")
      if not self.simulation_started:
        self.after(2000, lambda: self.main_label.configure(text="Planetary Heat Diffusion Simulator - YAML Configuration Editor"))

    #displays an error message if the file cannot be saved
    except Exception as e:
      self.main_label.configure(text=f"Error saving YAML: {str(e)}")

  #function that resets the yaml file to original
  def reset_yaml(self):
    self.load_yaml()
    self.main_label.configure(text="YAML reset to file contents")

  #function that shows planet documentation
  def show_planet_docs(self):
    docs = """PLANET PARAMETERS

Required parameters:
- R_p: Planet radius in meters (must be positive)
- M_p: Planet mass in kilograms (must be positive)
- N: Grid resolution - number of radial grid points (must be >= 2, typically 100-1000)

Example:
planet:
  R_p: 6.371e6
  M_p: 5.972e24
  N: 400

  layers:
    - name: core
      r_outer: 3.48e6
      rho: 12000
      k: 5000
      cp: 800
    - name: mantle
      r_outer: 6.371e6
      rho: 4500
      k: 5000
      cp: 1200"""
    self.show_docs_popup("Planet Parameters", docs)

  #function that shows layers documentation
  def show_layers_docs(self):
    docs = """LAYER PARAMETERS

Each layer must have:
- name: string identifier
- r_outer: outer radius in meters (must be > previous layer, <= R_p)

Optional layer properties:

DENSITY (rho):
  Option 1: Constant value
    rho: 12000  # kg/m³
  
  Option 2: Linear equation of state
    rho:
      type: linear_eos
      rho0: 12000      # Reference density (kg/m³)
      alpha: 3e-5      # Thermal expansion (1/K)
      K: 1e11          # Bulk modulus (Pa)
      T0: 3000         # Reference temperature (K)
    Formula: rho = rho0 * (1 + P/K - alpha*(T-T0))

THERMAL CONDUCTIVITY (k):
  Option 1: Constant
    k: 5000  # W/(m·K)
  
  Option 2: Power law
    k:
      type: power_law_T
      k0: 5000         # Reference conductivity
      T0: 3000         # Reference temperature (K)
      exp: 1.0         # Exponent
    Formula: k = k0 * (T0/T)^exp

SPECIFIC HEAT (cp):
  Constant only:
    cp: 800  # J/(kg·K)

VISCOSITY (eta):
  Option 1: Constant
    eta: 1e23  # Pa·s
  
  Option 2: Arrhenius law
    eta:
      type: arrhenius
      eta0: 1e21       # Pre-exponential (Pa·s)
      E_ast: 300000    # Activation energy (J/mol)
      V_ast: 1e-6      # Activation volume (m³/mol)
      R_g: 8.314       # Gas constant (optional)
    Formula: eta = eta0 * exp((E_ast + P*V_ast)/(R_g*T))

MELTING TEMPERATURE (Tm):
  Option 1: Constant
    Tm: 2000  # K
  
  Option 2: Quadratic
    Tm:
      type: quadratic_Tm
      Tm0: 2000        # Melting temp at P=0 (K)
      a: 1e-8          # Linear coefficient (K/Pa)
      b: 1e-18        # Quadratic coefficient (K/Pa²)
    Formula: Tm = Tm0 + a*P + b*P²

OPTIONAL:
  alpha: 3e-5      # Thermal expansion (1/K) - constant only
  K: 1e11          # Bulk modulus (Pa) - constant only
  L: 5e5           # Latent heat (J/kg) - constant only
  dT_mush: 100     # Mush range (K) - constant only"""
    self.show_docs_popup("Layer Parameters", docs)

  #function that shows radiogenic documentation
  def show_radiogenic_docs(self):
    docs = """RADIOGENIC HEATING

Option 1: Simple constant (scalar)
  radiogenic: 0.0  # W/m³

Option 2: Constant type (explicit)
  radiogenic:
    type: constant
    H0: 0.0  # Heating rate in W/m³
    # OR use "value" instead of "H0"

Option 3: Multi-isotope decay
  radiogenic:
    type: multi_isotope
    isotopes:
      - H0: 1e-6          # Initial heating rate (W/m³)
        half_life: 4.5e9  # Half-life in years
      - H0: 5e-7          # Can add multiple isotopes
        half_life: 7e8

Formula: H(t) = sum(H0_i * exp(-ln(2) * t / half_life_i))"""
    self.show_docs_popup("Radiogenic Heating", docs)

  #function that shows boundary documentation
  def show_boundary_docs(self):
    docs = """BOUNDARY CONDITIONS

Option 1: Dirichlet (fixed temperature)
  boundary:
    surface_BC:
      type: dirichlet
      value: 250  # Surface temperature in K

Option 2: Neumann (fixed flux)
  boundary:
    surface_BC:
      type: neumann
      mode: insulated  # No heat flux (q = 0)
      
      # OR mode: radiative
      mode: radiative
      emissivity: 0.9      # Emissivity (0-1), default 0.9
      T_space: 3.0         # Space temperature in K, default 3.0
      # Formula: q = emissivity * sigma * (T^4 - T_space^4)
      # where sigma = 5.670374419e-8 W/(m²·K⁴)
      
      # OR mode: constant
      mode: constant
      q_flux: 0.0  # Constant heat flux in W/m²"""
    self.show_docs_popup("Boundary Conditions", docs)

  #function that shows initial conditions documentation
  def show_initial_docs(self):
    docs = """INITIAL CONDITIONS

Option 1: Adiabatic profile
  initial:
    T0:
      type: adiabatic
      Tc: 3000        # Core temperature (K)
      cp: 1200        # Specific heat (J/(kg·K))
      alpha: 3e-5     # Thermal expansion (1/K)
  Formula: T(r) = Tc * exp(-alpha * g * r / cp)
  where g = G * M_p / R_p²

Option 2: Isothermal profile
  initial:
    T0:
      type: isothermal
      T: 300  # Uniform temperature in K"""
    self.show_docs_popup("Initial Conditions", docs)

  #function that shows convection documentation
  def show_convection_docs(self):
    docs = """CONVECTION PARAMETERS

Controls mantle convection enhancement of heat transport.

Formula: Nu = 1.0 + A * (Ra/Ra_c)^n
where Ra is the Rayleigh number computed from mantle properties

Parameters:
  Ra_c: Critical Rayleigh number (dimensionless)
        Set very high (e.g., 1e99) to disable convection
        Or set A=0 to disable
  
  A: Convection scaling parameter
     0.0 = no convection
     0.1 = strong convection
  
  n: Convection exponent
     Typically 0.2-1.0, often 0.333

Example:
  convection:
    Ra_c: 1e99
    A: 0.0
    n: 1.0"""
    self.show_docs_popup("Convection Parameters", docs)

  #function that shows constants documentation
  def show_constants_docs(self):
    docs = """PHYSICAL CONSTANTS

Can override default values if needed.

  constants:
    G: 6.6743e-11   # Gravitational constant (m³/(kg·s²))
    R_g: 8.314      # Gas constant (J/(mol·K))

These are optional - defaults will be used if not specified."""
    self.show_docs_popup("Physical Constants", docs)

  #function that shows simulation documentation
  def show_simulation_docs(self):
    docs = """SIMULATION PARAMETERS

  simulation:
    t_max: 1.5e7    # Maximum simulation time in years
                    # Typical: 1e6 (1 Myr) to 4.5e9 (4.5 Gyr)
    
    dt: 1e5         # Time step in years
                    # Should be << t_max, typically 1e4 to 1e6
    
    integrator:
      type: RK2      # Integrator type
      description: fast cool test  # Optional description

Available integrator types:
  Implicit methods (Gauss-Seidel Picard):
    BE, BDF2, BDF3, BDF4, BDF5, BDF6 (Backward Differentiation Formula)
    AM2, AM3, AM4, AM5 (Adams-Moulton)
    GL1, GL2, GL3, GL4, GL5 (Gauss-Legendre)
    R1, R2, R3, R4, R5 (Radau IIA)
    L1, L2, L3, L4, L5 (Lobatto IIIC)
    SDIRK2, SDIRK3, SDIRK4 (Single Diagonally Implicit Runge-Kutta)
  
  Explicit methods:
    RK1, RK2, RK3, RK4, RK5 (Runge-Kutta)
    AB2, AB3, AB4, AB5 (Adams-Bashforth)

Higher order = more accurate but may be less stable.
Implicit methods are more stable for stiff problems."""
    self.show_docs_popup("Simulation Parameters", docs)

  #function that shows output documentation
  def show_output_docs(self):
    docs = """OUTPUT PARAMETERS

  output:
    save_path: "results/run_fastcool"  # Output file path (without .npy extension)
                                       # Directory will be created if needed
    overwrite: true                    # Overwrite existing file (true/false)

The save_path should not include the .npy extension - it will be added automatically."""
    self.show_docs_popup("Output Parameters", docs)

  #function that shows program information
  def show_info(self):
    info = """PLANETARY HEAT DIFFUSION SIMULATOR
========================================

PROGRAM OVERVIEW:
This application simulates the thermal evolution of planetary bodies through 
heat diffusion processes. It solves the heat equation numerically using various 
time integration schemes, accounting for material properties, radiogenic heating, 
boundary conditions, and mantle convection.

KEY FEATURES:
- Configurable planetary structure with multiple layers
- Dynamic material properties (density, conductivity, viscosity, etc.)
- Multiple boundary condition types (Dirichlet, Neumann with various modes)
- Flexible initial temperature profiles (adiabatic, isothermal)
- Radiogenic heating with single or multi-isotope decay
- Mantle convection parameterization
- Multiple numerical integrators (implicit and explicit methods)
- YAML-based configuration for easy setup

USAGE:
1. Edit the YAML configuration file in the text editor
2. Use the documentation buttons to learn about each parameter type
3. Click "Save" to save your configuration
4. Click "Reset" to reload the original YAML file
5. Click "Start" to run the simulation
6. Use "Theme" to toggle between light and dark modes

YAML CONFIGURATION:
The configuration file (planet.yaml) contains all simulation parameters organized 
into sections:
- planet: Basic planetary properties and layer definitions
- radiogenic: Heat source configuration
- boundary: Surface boundary conditions
- initial: Initial temperature profile
- convection: Mantle convection parameters
- constants: Physical constants (optional)
- simulation: Time stepping and integrator settings
- output: Results saving options

NUMERICAL METHODS:
The simulator supports various time integration schemes:
- Implicit (Picard Gauss-Seidel relaxation): BE, BDF2-6, AM2-5 
- Explicit (unstable for stiff problems): RK1-6, AB2-5

MATERIAL PROPERTIES:
Layer properties can be constant or function-based:
- Density: constant or linear equation of state
- Conductivity: constant or power law in temperature
- Viscosity: constant or Arrhenius law
- Melting temperature: constant or quadratic in pressure

For detailed information about each parameter type, click the corresponding 
documentation button.

OUTPUT:
The simulation generates:
- Temperature evolution data (saved as .npy file)
- Visualization plots (saved in graphs/ directory)
- Material property profiles

For questions or issues, refer to the documentation buttons for each 
configuration section."""
    self.show_docs_popup("Program Information", info)

  #function that shows documentation popup
  def show_docs_popup(self, title, content):

    popup = ctk.CTkToplevel(self)
    popup.title(title)
    popup.geometry("700x600")
    popup.resizable(False, False)
    
    #creates scrollable text widget with theme-appropriate colors
    text_frame = ctk.CTkFrame(popup)
    text_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    #gets current theme to set appropriate colors
    current_theme = ctk.get_appearance_mode().lower()
    if current_theme == "light":
      bg_color = "#fff5f9"  
      fg_color = "#5e173a"  
    else:
      bg_color = "#1a0505"  
      fg_color = "#ffd8ec"   
    
    #creates the text widget
    text_widget = scrolledtext.ScrolledText(text_frame, wrap="word", font=("Consolas", 16), bg=bg_color, fg=fg_color, width=80, height=30)
    text_widget.pack(fill="both", expand=True, padx=5, pady=5)
    text_widget.insert("1.0", content)
    text_widget.configure(state="disabled")
    
    #creates close button
    close_btn = ctk.CTkButton(popup, text="Close", command=popup.destroy, font=("Times New Roman", 18))
    close_btn.pack(pady=10)

  #function that updates text editor colors based on theme
  def update_text_editor_colors(self):
    current_theme = ctk.get_appearance_mode().lower()
    if hasattr(self, 'text_editor'):
      #updates existing text editor colors
      if current_theme == "light":
        self.text_editor.configure(bg="#fff5f9", fg="#5e173a", insertbackground="#5e173a")
      else:
        self.text_editor.configure(bg="#1a0505", fg="#ffd8ec", insertbackground="#ffd8ec")

  #function that toggles the theme of the window
  def toggle_theme(self):
    current = ctk.get_appearance_mode().lower()
    ctk.set_appearance_mode("light") if current == "dark" else ctk.set_appearance_mode("dark")

    #updates text editor colors after theme change
    self.update_text_editor_colors()

  #function that begins the calculations
  def run_simulation(self):
    #marks that simulation has started (prevents default message from coming back)
    self.simulation_started = True
    
    #saves the current yaml file
    self.save_yaml()
  
    #creates a progress tracker object
    self.progress_tracker = ProgressTracker()
    
    #hides the text editor and buttons during simulation
    for widget in self.winfo_children():
      if widget != self.main_label:
        widget.grid_remove()
    
    #configures the main label for simulation
    self.main_label.configure(text="Running Simulation\nProgress: 0/0", font=("Times New Roman", 54))
    self.main_label.grid(row=0, column=0, columnspan=10, pady=10, sticky="ew")
    
    #creates a progress bar to track the backend progress 
    self.progress_bar = ctk.CTkProgressBar(self, orientation="horizontal", width=500, height=20, corner_radius=10, fg_color=("#ffe6f2", "#2e0b0b"), progress_color=("#5e173a", "#ffd8ec"))
    self.progress_bar.set(0)
    self.progress_bar.grid(row=1, column=0, columnspan=10, pady=10)
    
    #sets flag to track if simulation is running
    self.simulation_running = True
    
    #starts the progress update loop
    self.update_progress()
    
    #runs the simulation in a separate thread
    thread = threading.Thread(target=self.call_backend, daemon=True)
    thread.start()
  
  #function that calls the backend in a separate thread
  def call_backend(self):
    try:
      self.graphs_dir, self.results_path = planetary_main.run(self.yaml_file, self.progress_tracker)
      self.after(0, self.after_simulation)
      
    #displays an error message if the simulation fails
    except Exception as e: self.after(0, lambda: self.show_error(str(e)))
  
  #function that displays an error message if the simualation failed
  def show_error(self, error_msg):
    #stops the progress update loop
    self.simulation_running = False
    
    #removes progress bar if it exists
    if hasattr(self, 'progress_bar') and self.progress_bar.winfo_exists(): self.progress_bar.destroy()
    
    #configures grid for centered label and buttons
    for i in range(100): self.grid_rowconfigure(i, weight=1)
    for j in range(12): self.grid_columnconfigure(j, weight=1)
    
    #configures the main label with original font size, centered
    self.main_label.configure(text=f"Error running simulation:\n{error_msg}", font=("Times New Roman", 38))
    self.main_label.grid(row=40, rowspan=20, column=0, columnspan=12, pady=10, sticky="nsew")
    
    #adds exit and restart buttons at the bottom
    restart_btn = ctk.CTkButton(self, text="Restart", font=("Times New Roman", 26), command=self.restart_app)
    restart_btn.grid(row=99, column=0, columnspan=6, sticky="nsew", padx=10, pady=10)
    
    exit_btn = ctk.CTkButton(self, text="Exit", font=("Times New Roman", 26), command=self.exit_app)
    exit_btn.grid(row=99, column=6, columnspan=6, sticky="nsew", padx=10, pady=10)  
  
  #function that updates the progress bar
  def update_progress(self):
    try:
      current = self.progress_tracker.current_step
      total = self.progress_tracker.total_steps
      if total > 0:
        #checks if progress is complete
        if current >= total and hasattr(self, 'progress_bar') and self.progress_bar.winfo_exists():
          #removes the progress bar and updates label to show visualization phase
          self.progress_bar.destroy()
          self.main_label.configure(text="Creating visualizations", font=("Times New Roman", 54))

        elif hasattr(self, 'progress_bar') and self.progress_bar.winfo_exists():
          #updates progress bar and label
          self.progress_bar.set(current / total)
          self.main_label.configure(text=f"Running Simulation\nProgress: {current}/{total}")
    except: pass
    
    #updates the progress bar
    if self.simulation_running: self.after(200, self.update_progress)
  
  #function that processes the results after simulation
  def after_simulation(self):
    #stops the progress update loop
    self.simulation_running = False
    
    #removes progress bar if it still exists
    if hasattr(self, 'progress_bar') and self.progress_bar.winfo_exists(): self.progress_bar.destroy()
    
    #configures grid for centered label
    for i in range(10): self.grid_rowconfigure(i, weight=1)
    for j in range(10): self.grid_columnconfigure(j, weight=1)
    
    #configures the main label centered for plotting message
    self.main_label.configure(text="Simulation Completed\nPlotting Graphs", font=("Times New Roman", 67))
    self.main_label.grid(row=4, column=0, columnspan=10, pady=10, sticky="nsew")
    self.update()
    
    #loads the results from the npz file
    try:
      results = np.load(self.results_path, allow_pickle=True)
      results_dict = {key: results[key] for key in results.files}
      
      #subsamples to 100 equally spaced points if there are more than 100 points
      if len(results_dict["t"]) > 100:
        #creates indices for 100 equally spaced points
        indices = np.linspace(0, len(results_dict["t"]) - 1, 100, dtype=int)

        #subsamples time array
        t_original = results_dict["t"]
        t_subsampled = t_original[indices]
        
        #interpolates temperature array to match subsampled time points
        T_original = results_dict["T"]
        T_subsampled = np.zeros((100, T_original.shape[1]))
        for i in range(T_original.shape[1]):
          interp_func = interp1d(t_original, T_original[:, i], kind='linear', fill_value='extrapolate')
          T_subsampled[:, i] = interp_func(t_subsampled)
        
        #updates results dictionary with subsampled data
        results_dict["t"] = t_subsampled
        results_dict["T"] = T_subsampled
      
      #gets the current theme
      current_theme = ctk.get_appearance_mode().lower()
      
      #stores results for later use in viewers
      self.results_dict = results_dict
      
      #calls all visualization functions with the theme
      plot_core_surface_temperature(results_dict, self.graphs_dir, current_theme)
      plot_temperature_heatmap(results_dict, self.graphs_dir, current_theme)
      plot_material_profiles(results_dict, self.graphs_dir, current_theme)
      plot_surface_heat_flux(results_dict, self.graphs_dir, current_theme)
      plot_mean_cooling_rate(results_dict, self.graphs_dir, current_theme)
      
      #plots all temperature profiles at all time points using multiprocessing
      plot_all_profiles_at_time(results_dict, self.graphs_dir, current_theme)
      
    except Exception as e: self.show_plotting_error(str(e))

    #configures grid for top label and buttons (blank GUI) - 12 columns
    for i in range(100): 
      self.grid_rowconfigure(i, weight=1)
    for j in range(12):
      self.grid_columnconfigure(j, weight=1)
    
    #moves label to top with blank GUI - takes up all 12 columns
    self.main_label.configure(text="Simulation Results", font=("Times New Roman", 54))
    self.main_label.grid(row=0, rowspan=12, column=0, columnspan=12, pady=5, sticky="ew")
    
    #creates buttons for viewing plots (2 rows of 3 buttons)
    plot_buttons = [
      ("Core & Surface Temp", "T_core_surface_vs_time.png", "Core and surface temperature evolution"),
      ("Temperature Heatmap", "T_heatmap_r_t.png", "Thermal evolution (T[r,t])"),
      ("Material Profiles", "material_profiles.png", "Material property profiles"),
      ("Surface Heat Flux", "surface_heat_flux_vs_time.png", "Surface heat flux over time"),
      ("Mean Cooling Rate", "mean_cooling_rate_vs_time.png", "Global cooling rate"),
      ("Temperature Profiles", None, "Temperature profile viewer"), 
    ]
    
    #places the buttons to view the simulation results
    for i, (button_text, filename, title) in enumerate(plot_buttons):
      row = 40 if i < 3 else 60
      rowspan = 6
      col_start = (i % 3) * 4  
      colspan = 4
      if filename: btn = ctk.CTkButton(self, text=button_text, font=("Times New Roman", 26), command=lambda f=filename, t=title: self.open_plot_window(f, t))
      else: btn = ctk.CTkButton(self, text=button_text, font=("Times New Roman", 26), command=self.open_temperature_profile_viewer)
      btn.grid(row=row, rowspan=rowspan, column=col_start, columnspan=colspan, padx=10, pady=10, sticky="nsew")
    
    #places the buttons for export, exit, and restart
    export_graphs_btn = ctk.CTkButton(self, text="Export All Graphs", font=("Times New Roman", 26), command=self.export_all_graphs)
    export_graphs_btn.grid(row=99, column=0, columnspan=3, sticky="nsew", padx=10, pady=10)
    
    export_data_btn = ctk.CTkButton(self, text="Export All Data", font=("Times New Roman", 26), command=self.export_all_data)
    export_data_btn.grid(row=99, column=3, columnspan=3, sticky="nsew", padx=10, pady=10)
    
    restart_btn = ctk.CTkButton(self, text="Restart", font=("Times New Roman", 26), command=self.restart_app)
    restart_btn.grid(row=99, column=6, columnspan=3, sticky="nsew", padx=10, pady=10)
    
    exit_btn = ctk.CTkButton(self, text="Exit", font=("Times New Roman", 26), command=self.exit_app)
    exit_btn.grid(row=99, column=9, columnspan=3, sticky="nsew", padx=10, pady=10)
  
  #function that opens a static plot window
  def open_plot_window(self, filename, title):
    img_path = os.path.join(self.graphs_dir, filename)
    if not os.path.exists(img_path):
      return
    
    #determines window size based on plot type
    if filename == "material_profiles.png":
      win_width, win_height = 1440, 520
      img_width, img_height = 1420, 500
    elif filename == "T_heatmap_r_t.png":
      win_width, win_height = 1008, 520
      img_width, img_height = 988, 500
    else:
      win_width, win_height = 720, 520
      img_width, img_height = 700, 500
    
    #creates and configures a top level window
    win = ctk.CTkToplevel(self)
    win.title(title)
    win.geometry(f"{win_width}x{win_height}")
    win.resizable(False, False)
    
    #loads and resizes the image
    pil_img = Image.open(img_path)
    img = CTkImage(light_image=pil_img, dark_image=pil_img, size=(img_width, img_height))
    panel = ctk.CTkLabel(win, image=img, text="")
    panel.image = img
    panel.pack(pady=5)
  
  #function that opens an interactive temperature profile viewer with slider
  def open_temperature_profile_viewer(self):
    if not hasattr(self, 'results_dict'):
      return
    
    #gets the data
    t = self.results_dict["t"]
    r = self.results_dict["r"] / 1e6
    T = self.results_dict["T"]
    slider_max = len(t) - 1
    
    #creates the top level window
    viewer = ctk.CTkToplevel(self)
    viewer.title("Temperature Profile Viewer")
    viewer.geometry("900x600")
    viewer.resizable(False, False)
    
    #configures the layout
    for i in range(11):
      viewer.grid_rowconfigure(i, weight=1 if i < 10 else 0)
    for j in range(10):
      viewer.grid_columnconfigure(j, weight=1)
    
    #creates the image label
    img_label = ctk.CTkLabel(viewer, text="")
    img_label.grid(row=0, column=0, rowspan=10, columnspan=10, sticky="nsew")
    
    #function that updates the plot according to slider position
    def update_plot(t_index):
      #constructs the filename based on the index
      actual_time = t[t_index]
      time_str = f"{actual_time:.3e}".replace('e+', 'ep').replace('e-', 'em')
      filename = f"T_profile_at_{t_index:03d}_{time_str}yr.png"
      filepath = os.path.join(self.graphs_dir, filename)
      
      #loads the image if it exists
      if os.path.exists(filepath):
        pil_img = Image.open(filepath)
        img = CTkImage(light_image=pil_img, dark_image=pil_img, size=(760, 550))
        img_label.configure(image=img, text="")
        img_label.image = img
    
    #creates the slider and binds it to the updating function
    update_plot(0)
    slider = ctk.CTkSlider(viewer, from_=0, to=slider_max, number_of_steps=slider_max, orientation="horizontal", width=600)
    slider.grid(row=10, column=0, columnspan=10, sticky="swe", pady=(10, 5))
    def on_slider_change(val): update_plot(int(round(val)))
    slider.set(0)
    slider.configure(command=on_slider_change)
  
  #function that displays an error messgae if the plotting failed
  def show_plotting_error(self, error_msg):
    #configures grid for centered label and buttons
    for i in range(100): self.grid_rowconfigure(i, weight=1)
    for j in range(12): self.grid_columnconfigure(j, weight=1)
    
    #configures the main label with error message
    self.main_label.configure(text=f"Error plotting results:\n{error_msg}", font=("Times New Roman", 38))
    self.main_label.grid(row=40, rowspan=20, column=0, columnspan=12, pady=10, sticky="nsew")
    
    #adds exit and restart buttons at the bottom (
    restart_btn = ctk.CTkButton(self, text="Restart", font=("Times New Roman", 26), command=self.restart_app)
    restart_btn.grid(row=99, column=0, columnspan=6, sticky="nsew", padx=10, pady=10)
    
    exit_btn = ctk.CTkButton(self, text="Exit", font=("Times New Roman", 26), command=self.exit_app)
    exit_btn.grid(row=99, column=6, columnspan=6, sticky="nsew", padx=10, pady=10)
  
  #function that exports all graphs
  def export_all_graphs(self):
    #checks which graphs exist
    output_dir = self.graphs_dir
    if not os.path.exists(output_dir) or not os.listdir(output_dir):
      self.main_label.configure(text="No graphs to export")
      self.after(3000, lambda: self.main_label.configure(text="Simulation Results"))
      return
    
    #asks user to select the destination
    from tkinter import filedialog
    import zipfile
    zip_path = filedialog.asksaveasfilename(defaultextension=".zip", initialfile="Planetary_Graphs.zip", title="Save All Graphs", filetypes=[("ZIP Archive", "*.zip")])
    if not zip_path: return
    
    #exports all the graphs to the selected zip file
    with zipfile.ZipFile(zip_path, 'w') as z:
      for file in os.listdir(output_dir):
        if file.endswith(".png"): 
          z.write(os.path.join(output_dir, file), arcname=file)
  
  #function that exports the npz data file
  def export_all_data(self):
    #asks user to select the destination
    from tkinter import filedialog
    import shutil
    if not hasattr(self, 'results_path') or not os.path.exists(self.results_path):
      self.main_label.configure(text="No data file to export")
      self.after(3000, lambda: self.main_label.configure(text="Simulation Results"))
      return
    
    file_path = filedialog.asksaveasfilename(defaultextension=".npz", initialfile="planetary_results.npz", title="Save Data File", filetypes=[("NumPy Archive", "*.npz")])
    if not file_path: return
    
    #copies the npz file to the selected location
    shutil.copy2(self.results_path, file_path)
  
  #function that restarts the program
  def restart_app(self):
    import subprocess
    import sys
    self.destroy()
    subprocess.call([sys.executable, sys.argv[0]])
  
  #function that exits the program
  def exit_app(self):
    self.destroy()

#begins the program
if __name__ == "__main__":
    app = PlanetaryApp()
    app.mainloop()
