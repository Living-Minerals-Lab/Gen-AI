import yaml
import numpy as np
import matplotlib.pyplot as plt


# Count the formula units (LiAlSiO4) in each calculation cell:
N_FU_ALPHA = 6       # LiAlSiO4_primitive, Li6Al6Si6O24, 42-atom cell, true ordered R3 alpha (Z = 6)
N_FU_BETA = 12      # EntryWithCollCode2929.cif, Al12Li12O48Si12, 84-atom cell (Z = 12)
N_FU_GAMMA = 4      # EntryWithCollCode66137.cif, Al4Li4O16Si4, 28-atom cell (Z = 4)
N_FU_EPSILON = 16   # ICSD_CollCode161497_anhydrous.cif, Li16Al16Si16O64, 112-atom cell (Z = 16)
N_FU_GEN1313 = 2    # gen_1313.cif (mattergen-491, cluster 4 min-e_hull rep), Al2Li2Si2O8, 14-atom cell (Z = 2)

# Static energies from MatterSim relaxation (in eV)
# Single-point MatterSim evaluation on the relaxed unit_cell stored in each
# phonopy_params.yaml (relax_and_phonon.py never saved the static energy to
# disk).
E_STATIC_ALPHA_EV = -303.527130     # LiAlSiO4_primitive (true ordered R3 alpha), 42-atom cell
E_STATIC_BETA_EV = -606.28595       # phonon_EntryWithCollCode2929_5M, 84-atom cell
E_STATIC_GAMMA_EV = -201.79683      # phonon_EntryWithCollCode66137_5M, 28-atom cell
E_STATIC_EPSILON_EV = -806.54602    # phonon_epsilon_5M, 112-atom cell
E_STATIC_GEN1313_EV = -100.652428   # phonon_gen_1313_5M, 14-atom cell (has imaginary phonon modes -- not dynamically stable)

# Paths to the Phonopy thermal properties output files
# (Generated using: phonopy load phonopy_params.yaml --mp "16 16 16" -t)
YAML_ALPHA_PATH = "/global/cfs/projectdirs/m4397/For_Atreya/Gen-AI/phonon_LiAlSiO4_primitive_5M/thermal_properties.yaml"
YAML_BETA_PATH = "/global/cfs/projectdirs/m4397/For_Atreya/Gen-AI/CS_TEST/phonon_si_example/phonon_EntryWithCollCode2929_5M/thermal_properties.yaml"
YAML_GAMMA_PATH = "/global/cfs/projectdirs/m4397/For_Atreya/Gen-AI/CS_TEST/phonon_si_example/phonon_EntryWithCollCode66137_5M/thermal_properties.yaml"
YAML_EPSILON_PATH = "/global/cfs/projectdirs/m4397/For_Atreya/Gen-AI/CS_TEST/phonon_si_example/phonon_epsilon_5M/thermal_properties.yaml"
YAML_GEN1313_PATH = "/global/cfs/projectdirs/m4397/For_Atreya/Gen-AI/CS_TEST/phonon_si_example/phonon_gen_1313_5M/thermal_properties.yaml"

# Constants
EV_TO_KJMOL = 96.4853  # Conversions from eV to kJ/mol

# =====================================================================
# 2. READ PHONOPY OUTPUTS
# =====================================================================
def parse_phonopy_thermal(file_path):
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)

    temps = []
    f_vib = []  # Vibrational Free Energy in kJ/mol (per unit cell)
    for entry in data["thermal_properties"]:
        temps.append(entry["temperature"])
        f_vib.append(entry["free_energy"])

    return np.array(temps), np.array(f_vib)

temps_alpha, f_vib_alpha = parse_phonopy_thermal(YAML_ALPHA_PATH)
temps_beta, f_vib_beta = parse_phonopy_thermal(YAML_BETA_PATH)
temps_gamma, f_vib_gamma = parse_phonopy_thermal(YAML_GAMMA_PATH)
temps_epsilon, f_vib_epsilon = parse_phonopy_thermal(YAML_EPSILON_PATH)
temps_gen1313, f_vib_gen1313 = parse_phonopy_thermal(YAML_GEN1313_PATH)

# Verify temperature alignment
if not (np.array_equal(temps_alpha, temps_beta)
        and np.array_equal(temps_alpha, temps_gamma)
        and np.array_equal(temps_alpha, temps_epsilon)
        and np.array_equal(temps_alpha, temps_gen1313)):
    raise ValueError("The temperature ranges in your Phonopy files do not match!")
temps = temps_alpha

# =====================================================================
# 3. THERMODYNAMIC NORMALIZATION (Per Formula Unit of LiAlSiO4)
# =====================================================================
# Convert MatterSim static energy from eV -> kJ/mol, then divide by N_fu
E_alpha_fu = (E_STATIC_ALPHA_EV * EV_TO_KJMOL) / N_FU_ALPHA
E_beta_fu = (E_STATIC_BETA_EV * EV_TO_KJMOL) / N_FU_BETA
E_gamma_fu = (E_STATIC_GAMMA_EV * EV_TO_KJMOL) / N_FU_GAMMA
E_epsilon_fu = (E_STATIC_EPSILON_EV * EV_TO_KJMOL) / N_FU_EPSILON
E_gen1313_fu = (E_STATIC_GEN1313_EV * EV_TO_KJMOL) / N_FU_GEN1313

# Divide Phonopy vibrational energy by N_fu
f_vib_alpha_fu = f_vib_alpha / N_FU_ALPHA
f_vib_beta_fu = f_vib_beta / N_FU_BETA
f_vib_gamma_fu = f_vib_gamma / N_FU_GAMMA
f_vib_epsilon_fu = f_vib_epsilon / N_FU_EPSILON
f_vib_gen1313_fu = f_vib_gen1313 / N_FU_GEN1313

# Compute total free energy: F(T) = E_static + F_vib(T)
F_alpha_total = E_alpha_fu + f_vib_alpha_fu
F_beta_total = E_beta_fu + f_vib_beta_fu
F_gamma_total = E_gamma_fu + f_vib_gamma_fu
F_epsilon_total = E_epsilon_fu + f_vib_epsilon_fu
F_gen1313_total = E_gen1313_fu + f_vib_gen1313_fu

# Calculate relative free energy (Beta/Gamma/Epsilon/gen_1313 relative to the Alpha baseline)
F_beta_relative = F_beta_total - F_alpha_total
F_gamma_relative = F_gamma_total - F_alpha_total
F_epsilon_relative = F_epsilon_total - F_alpha_total
F_gen1313_relative = F_gen1313_total - F_alpha_total

# =====================================================================
# 4. PLOTTING THE STABILITY CURVES
# =====================================================================
def plot_stability_curves(curves, output_path):
    """curves: list of (label, color, values) plotted relative to the Alpha baseline."""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    # Alpha baseline at y = 0
    ax.axhline(0, color="#c3c2b7", linestyle="--", linewidth=1.5, zorder=1, label="Alpha-Eucryptite (reference)")

    for label, color, values in curves:
        ax.plot(temps, values, color=color, linewidth=2.5, label=label, zorder=2)
        ax.scatter(temps[::5], values[::5], color=color, edgecolor="black", s=40, zorder=3)

    # Formatting
    ax.set_xlabel("Temperature (K)", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_ylabel("Relative Free Energy (kJ mol$^{-1}$ per f.u.)", fontsize=12, fontweight="bold", labelpad=8)

    ax.tick_params(axis="both", which="major", direction="in", length=6, width=1.2, labelsize=11)
    ax.tick_params(axis="both", which="minor", direction="in", length=3, width=1)
    ax.minorticks_on()

    ax.set_xlim(0, 1250)  # Temperature Range
    ax.legend(loc="best", fontsize=10, frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    plt.close(fig)


# Original 3-phase comparison (Beta, Gamma relative to Alpha)
plot_stability_curves(
    [
        ("Beta-Eucryptite", "#2a78d6", F_beta_relative),
        ("Gamma-Eucryptite", "#008300", F_gamma_relative),
    ],
    "free_energy_plot.png",
)

# 4-phase comparison including Epsilon (relative to Alpha)
plot_stability_curves(
    [
        ("Beta-Eucryptite", "#2a78d6", F_beta_relative),
        ("Gamma-Eucryptite", "#008300", F_gamma_relative),
        ("Epsilon-Eucryptite", "#e87ba4", F_epsilon_relative),
        ("gen_1313 (cluster 4 rep, imaginary phonon)", "#eda100", F_gen1313_relative),
    ],
    "free_energy_plot_eps.png",
)
