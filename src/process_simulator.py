"""
process_simulator.py
--------------------
Tier-1 analytical "fake factory" for the Safe-BO vs Standard-BO project.

Maps a process recipe -> doping profile -> threshold voltage (V_th)
-> {objective to maximize, constraint that must not be violated}.

Recipe (the 3 knobs the optimizer controls):
    dose  : implant dose            [cm^-2]   (log-scaled in practice)
    time  : anneal time             [s]
    temp  : anneal temperature      [deg C]

Outputs of simulate():
    V_th        : threshold voltage        [V]
    objective   : drive-current proxy (MAXIMIZE)   -- higher is better
    g           : constraint value; FEASIBLE iff g <= 0
    feasible    : bool

Physics is simplified but directionally real:
  - implant lays down a Gaussian dopant sheet
  - anneal (a) activates the dopant and (b) diffuses/broadens it (sqrt(Dt))
    with an Arrhenius diffusivity, plus dose loss at excessive thermal budget
  - resulting effective channel doping sets V_th via the textbook
    uniform-doping MOSFET threshold equation
  - objective rewards low V_th (high drive current)
  - constraint is a V_th spec WINDOW: too low -> leakage/punch-through (scrap),
    too high -> device won't switch (scrap). Outside the window = catastrophic.
"""

import numpy as np

# ----------------------------------------------------------------------
# Physical constants (CGS-ish units common in device work: cm, F/cm, C)
# ----------------------------------------------------------------------
q      = 1.602e-19        # elementary charge [C]
k_eV   = 8.617e-5         # Boltzmann constant [eV/K]
kT_q   = 0.02585          # thermal voltage at 300 K [V]
n_i    = 1.5e10           # intrinsic carrier conc. of Si [cm^-3]
eps_ox = 3.45e-13         # SiO2 permittivity [F/cm]  (3.9 * 8.854e-14)
eps_si = 1.04e-12         # Si permittivity  [F/cm]  (11.7 * 8.854e-14)

# ----------------------------------------------------------------------
# Fixed device / process parameters (not optimized)
# ----------------------------------------------------------------------
t_ox    = 2.0e-7          # gate oxide thickness [cm]  (= 2 nm)
V_FB    = -0.95           # flat-band voltage [V]  (n+ poly gate on p-Si)
dRp0    = 2.0e-6          # as-implanted straggle [cm] (= 20 nm)
N_sub   = 1.0e15          # background substrate doping [cm^-3]
V_dd    = 1.0            # supply voltage [V]

# Boron-in-Si diffusion (Arrhenius): D(T) = D0 * exp(-Ea / kT)
D0      = 0.76           # pre-exponential [cm^2/s]
Ea      = 3.46           # activation energy [eV]

C_ox    = eps_ox / t_ox  # oxide capacitance per area [F/cm^2]

# ----------------------------------------------------------------------
# Constraint spec window on V_th  (TUNE these to hit >30% infeasible)
# ----------------------------------------------------------------------
VTH_LO = 0.05            # below this -> too leaky / punch-through -> SCRAP
VTH_HI = 0.72            # above this -> won't switch well        -> SCRAP


# ----------------------------------------------------------------------
# 1. PROCESS MODEL:  recipe -> effective channel doping
# ----------------------------------------------------------------------
def effective_channel_doping(dose, time, temp_C):
    """Return effective peak channel doping N_A [cm^-3] after implant+anneal."""
    T = temp_C + 273.15                       # to Kelvin

    # --- Diffusivity (Arrhenius) and diffusion broadening ---
    D = D0 * np.exp(-Ea / (k_eV * T))         # [cm^2/s]
    dRp_eff = np.sqrt(dRp0**2 + 2.0 * D * time)   # broadened straggle [cm]

    # --- Activation: need enough thermal budget or dopant stays inactive ---
    # thermal budget proxy; f_act rises from 0 -> 1 as budget increases.
    budget = D * time                         # [cm^2], monotonic in T and t
    f_act  = 1.0 - np.exp(-budget / 3.0e-13)  # activation fraction in (0,1)

    # --- Dose loss at EXCESSIVE thermal budget (out-diffusion to surface) ---
    loss   = 1.0 - np.exp(-budget / 4.0e-12)  # grows toward 1 when over-annealed
    D_active = f_act * (1.0 - loss) * dose     # net electrically active dose

    # --- Peak concentration of the (Gaussian) active profile ---
    N_peak = D_active / (np.sqrt(2.0 * np.pi) * dRp_eff)

    return N_sub + N_peak                       # add substrate background


# ----------------------------------------------------------------------
# 2. DEVICE MODEL:  channel doping -> threshold voltage
# ----------------------------------------------------------------------
def threshold_voltage(N_A):
    """Textbook uniform-doping NMOS threshold voltage [V]."""
    phi_F = kT_q * np.log(N_A / n_i)                     # bulk potential [V]
    Q_dep = np.sqrt(2.0 * eps_si * q * N_A * (2.0 * phi_F))  # depletion charge
    return V_FB + 2.0 * phi_F + Q_dep / C_ox


# ----------------------------------------------------------------------
# 3. FULL SIMULATOR:  recipe -> {V_th, objective, constraint}
# ----------------------------------------------------------------------
def simulate(dose, time, temp_C, noise=0.0, rng=None):
    """
    Evaluate one recipe.

    noise : std-dev of Gaussian noise added to the objective (0 = noiseless).
    Returns a dict. Vectorizes over numpy-array inputs.
    """
    N_A  = effective_channel_doping(dose, time, temp_C)
    V_th = threshold_voltage(N_A)

    # Objective to MAXIMIZE: drive-current proxy  I_on ~ (V_dd - V_th)^2.
    # Lower V_th -> higher drive current -> better. (Physical value is
    # returned even for out-of-spec devices; the "catastrophe" is the
    # constraint violation, tracked by the optimizer, not hidden here.)
    objective = np.clip(V_dd - V_th, 0.0, None) ** 2

    if noise > 0.0:
        rng = np.random.default_rng() if rng is None else rng
        objective = objective + rng.normal(0.0, noise, size=np.shape(objective))

    # Constraint: FEASIBLE iff V_th in [VTH_LO, VTH_HI].
    # Continuous g (learnable by a GP): g <= 0 feasible, g > 0 violation.
    g = np.maximum(VTH_LO - V_th, V_th - VTH_HI)
    feasible = g <= 0.0

    return {"V_th": V_th, "objective": objective, "g": g, "feasible": feasible}


# ----------------------------------------------------------------------
# Design space definition (bounds the optimizer searches within)
# ----------------------------------------------------------------------
BOUNDS = {
    "log10_dose": (12.0, 14.0),   # dose 1e12 .. 1e14 cm^-2  (search in log space)
    "time":       (10.0, 100.0),  # anneal time [s]
    "temp_C":     (900.0, 1100.0) # anneal temperature [deg C]
}


# ----------------------------------------------------------------------
# Ground-truth mapping: dense grid, feasible fraction, safe optimum
# ----------------------------------------------------------------------
def ground_truth(n=60):
    """Dense grid sweep -> feasible fraction and best FEASIBLE recipe."""
    ld = np.linspace(*BOUNDS["log10_dose"], n)
    tt = np.linspace(*BOUNDS["time"], n)
    tc = np.linspace(*BOUNDS["temp_C"], n)
    LD, TT, TC = np.meshgrid(ld, tt, tc, indexing="ij")

    out  = simulate(10.0 ** LD, TT, TC, noise=0.0)
    obj  = out["objective"]
    feas = out["feasible"]

    frac_infeasible = 1.0 - feas.mean()

    obj_safe = np.where(feas, obj, -np.inf)   # mask out infeasible
    idx = np.unravel_index(np.argmax(obj_safe), obj_safe.shape)
    best = {
        "dose":   10.0 ** LD[idx],
        "time":   TT[idx],
        "temp_C": TC[idx],
        "V_th":   out["V_th"][idx],
        "objective": obj[idx],
    }
    return frac_infeasible, best, (LD, TT, TC, obj, feas, out["V_th"])


if __name__ == "__main__":
    frac, best, grid = ground_truth(n=60)
    print(f"Infeasible fraction of design space : {frac*100:5.1f}%  "
          f"(target > 30%)")
    print("Best FEASIBLE recipe (ground-truth optimum):")
    print(f"  dose   = {best['dose']:.3e} cm^-2")
    print(f"  time   = {best['time']:.1f} s")
    print(f"  temp   = {best['temp_C']:.1f} C")
    print(f"  V_th   = {best['V_th']:.3f} V   (spec window "
          f"[{VTH_LO}, {VTH_HI}] V)")
    print(f"  objective (I_on proxy) = {best['objective']:.4f}")
