import os
import sys
import json
import argparse
import csv
import numpy as np
from typing import List, Dict, Any

# Pymatgen and ASE imports
from pymatgen.core.structure import Structure
from ase.optimize import LBFGS
from pymatgen.io.ase import AseAtomsAdaptor


# --- Environment modification for MatterSim ---
print("\n--- Updating environment for MatterSim ---")
mattersim_path = "/global/common/software/m4397/naman/code"
current_pythonpath = os.environ.get('PYTHONPATH', '')
if mattersim_path not in current_pythonpath:
    os.environ['PYTHONPATH'] = f"{mattersim_path}:{current_pythonpath}"
    print(f"✅ Added '{mattersim_path}' to PYTHONPATH for this session.")
else:
    print("✅ MatterSim path already in PYTHONPATH.")

from mattersim.forcefield.potential import MatterSimCalculator


# =================================================================
#  Utility: Custom Encoder for NumPy types (prevents JSON errors)
# =================================================================
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)

# =================================================================
#  Module 1: ML-based Geometry Optimization
# =================================================================
def relax_structure(structure: Structure, ml_calculator: MatterSimCalculator) -> Dict:
    """
    Performs a geometry optimization using MatterSim as per user settings.
    """
    formula = structure.composition.reduced_formula
    print(f"    Running ML relaxation for {formula}...")
    try:
        adaptor = AseAtomsAdaptor()
        atoms = adaptor.get_atoms(structure)
        atoms.calc = ml_calculator
        
        # User specified settings: fmax=0.001, steps=500
        optimizer = LBFGS(atoms, logfile=None)
        optimizer.run(fmax=0.001, steps=500)
        
        final_energy = atoms.get_potential_energy()
        final_pymatgen_structure = adaptor.get_structure(atoms)
        
        print(f"    ✅ Relaxation finished. Energy: {final_energy:.4f} eV")
        return {
            'final_structure': final_pymatgen_structure,
            'final_energy': final_energy
        }
    except Exception as e:
        print(f"    ❌ Error during relaxation: {e}")
        return None

# =================================================================
#  Module 2: Caching and Persistence
# =================================================================
def load_cached_results(cache_file: str) -> Dict:
    """Loads results from JSON and reconstructs Structures."""
    if os.path.exists(cache_file):
        print(f"--> Loading existing results cache from '{cache_file}'...")
        with open(cache_file, 'r') as f:
            try:
                data = json.load(f)
                for entry_id, result in data.items():
                    # Handle both 'final_structure' and 'structure' keys for robustness
                    struct_key = 'final_structure' if 'final_structure' in result else 'structure'
                    result['final_structure'] = Structure.from_dict(result[struct_key])
                    # Ensure energy key is consistent for the rest of the script
                    if 'final_energy' not in result and 'energy' in result:
                        result['final_energy'] = result['energy']
                return data
            except Exception as e:
                print(f"    Warning: Could not load cache correctly: {e}")
                return {}
    return {}

def save_results_to_cache(cache_file: str, results_dict: Dict):
    """Saves results to JSON using the user's preferred schema."""
    print(f"--> Saving updated results to '{cache_file}'...")
    data_to_save = {}
    for entry_id, result in results_dict.items():
        data_to_save[entry_id] = {
            'entry_id': entry_id,
            'final_energy': float(result['final_energy']),
            'final_structure': result['final_structure'].as_dict()
        }
    with open(cache_file, 'w') as f:
        json.dump(data_to_save, f, indent=4, cls=NumpyJSONEncoder)

# =================================================================
#  Main Execution
# =================================================================
def main():
    parser = argparse.ArgumentParser(description="Optimize unique CIFs and update Master JSON cache.")
    parser.add_argument("unique_dir", help="Directory containing unique .cif files.")
    parser.add_argument("--json", default="relaxation_results.json", help="Path to Master JSON file.")
    parser.add_argument("--model", default="MatterSim-v1.0.0-5M.pth", help="Path to MatterSim .pth model.")
    parser.add_argument("--mapping", default="id_mapping.csv", help="CSV file to save ID-Filename mapping.")
    args = parser.parse_args()

    # 1. Load existing data
    cached_results = load_cached_results(args.json)

    # 2. Determine the next available mattergen-X index
    max_idx = -1
    for eid in cached_results.keys():
        if eid.startswith("mattergen-"):
            try:
                idx = int(eid.split("-")[1])
                if idx > max_idx: max_idx = idx
            except: continue
    next_idx = max_idx + 1
    print(f"--> MatterGen IDs will start from: mattergen-{next_idx}")

    # 3. Find unique CIFs to process
    cif_files = [f for f in os.listdir(args.unique_dir) if f.endswith(".cif")]
    if not cif_files:
        print("No .cif files found in UNIQUE directory. Exiting.")
        return

    # 4. Initialize Calculator and ID mapping
    mattersim_calc = MatterSimCalculator(load_path=args.model)
    mapping_data = []
    needs_save = False

    print(f"--> Found {len(cif_files)} new unique structures to optimize.")

    # 5. Loop and Relax
    for i, filename in enumerate(cif_files):
        new_id = f"mattergen-{next_idx + i}"
        cif_path = os.path.join(args.unique_dir, filename)
        
        print(f"\n[{i+1}/{len(cif_files)}] Processing {filename} as {new_id}...")
        
        try:
            struct = Structure.from_file(cif_path)
            relaxation_result = relax_structure(struct, mattersim_calc)
            
            if relaxation_result:
                relaxation_result['entry_id'] = new_id
                cached_results[new_id] = relaxation_result
                mapping_data.append([new_id, filename])
                needs_save = True
        except Exception as e:
            print(f"    ❌ Error loading {filename}: {e}")

    # 6. Save results and Mapping CSV
    if needs_save:
        save_results_to_cache(args.json, cached_results)
        
        # Write/Append to CSV
        file_exists = os.path.isfile(args.mapping)
        with open(args.mapping, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(['entry_id', 'original_filename'])
            writer.writerows(mapping_data)
        print(f"✅ ID mapping saved to {args.mapping}")

    print("\n✅ Optimization workflow complete. You can now run your plotting script.")

if __name__ == "__main__":
    main()
