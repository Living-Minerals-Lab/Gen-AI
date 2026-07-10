import numpy as np
import sys
from pathlib import Path
from ase.io import read
from ase.build import bulk
from ase.units import GPa
from ase.visualize import view
from mattersim.forcefield.potential import MatterSimCalculator
from mattersim.applications.phonon import PhononWorkflow
from mattersim.applications.relax import Relaxer

# initialize the structure of silicon
structure = read(sys.argv[1])
gen_id = Path(sys.argv[1]).stem

# attach the calculator to the atoms object
MODEL_LOAD_PATH = "MatterSim-v1.0.0-5M.pth"
structure.calc = MatterSimCalculator()
model_label = "5M" if "5M" in MODEL_LOAD_PATH else "1M" if "1M" in MODEL_LOAD_PATH else MODEL_LOAD_PATH


#lets relax it first
relaxer = Relaxer(
    optimizer="BFGS", # the optimization method
    filter="ExpCellFilter", # filter to apply to the cell
    constrain_symmetry=False, # whether to constrain the symmetry
)


converged, relaxed_structure = relaxer.relax(structure, steps=500, fmax=0.001)


output_base = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
output_base.mkdir(parents=True, exist_ok=True)

ph = PhononWorkflow(
    atoms=relaxed_structure,
    find_prim = False,
    work_dir = str(output_base / f"phonon_{gen_id}_{model_label}"),
    amplitude = 0.01,
    supercell_matrix = np.diag([2,2,2]),
)

has_imag, phonons = ph.run()
print(f"Has imaginary phonon: {has_imag}")
print(f"Phonon frequencies: {phonons}")