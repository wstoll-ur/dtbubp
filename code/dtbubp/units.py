"""Physical constants and unit conversions used everywhere in the package (one place only).

CODATA 2018 values, as in CP2K and ASE.
"""

HA_EV = 27.211386245988            # 1 Hartree in eV
BOHR_A = 0.529177210903            # 1 Bohr in Angstrom
HA_BOHR_EV_A = HA_EV / BOHR_A      # force: Hartree/Bohr -> eV/Angstrom (51.42208619...)
EV_A3_GPA = 160.21766208           # 1 eV/A^3 in GPa
BAR_PER_EV_A3 = 1.602176634e6      # 1 eV/A^3 in bar

# stress units CP2K may print -> eV/A^3
STRESS_TO_EV_A3 = {
    "bar": 1.0 / BAR_PER_EV_A3,
    "GPa": 1.0 / EV_A3_GPA,
    "MPa": 1e-3 / EV_A3_GPA,
    "Pa": 1e-9 / EV_A3_GPA,
    "atm": 1.01325 / BAR_PER_EV_A3,
}

DEBYE_EA = 1.0 / 4.80320471257     # 1 Debye in e*Angstrom (1 e*A = 4.80320471 D)

# polarizability
AU_POL_A3 = BOHR_A ** 3            # 1 atomic unit (Bohr^3) in Angstrom^3 = 0.148184711...
E2_4PIEPS0_EV_A = 14.399645478425  # e^2/(4 pi eps0) in eV*Angstrom
# MACE-MDP / SPICE-alpha store alpha in e*A^2/V. alpha[A^3] = alpha[e A^2/V] * 14.3996...
A3_TO_EA2V = 1.0 / E2_4PIEPS0_EV_A
AU_POL_EA2V = AU_POL_A3 * A3_TO_EA2V

KB_EV = 8.617333262e-5             # Boltzmann constant, eV/K
