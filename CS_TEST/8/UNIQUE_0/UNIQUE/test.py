import sys
import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def get_primitive_vectors(file_path="POSCAR"):
    """
    Reads a structure file (POSCAR), prints initial lattice constants,
    converts it to its primitive representation using symmetry analysis, 
    and prints the primitive lattice vectors.
    """
    try:
        # 1. Load the structure
        # Pymatgen automatically detects the file type (VASP POSCAR)
        structure = Structure.from_file(file_path)
        print(f"--- Loaded structure from {file_path} ---")
        print(f"Input Sites: {len(structure)}")

        # --- NEW: Print Input Lattice Constants ---
        print("\n" + "-"*30)
        print("INPUT LATTICE PARAMETERS")
        print("-"*30)
        # .abc returns a tuple (a, b, c)
        print(f"Lengths (a, b, c): {structure.lattice.abc}") 
        # .angles returns a tuple (alpha, beta, gamma)
        print(f"Angles (α, β, γ):  {structure.lattice.angles}")
        print("-"*30)
        # ------------------------------------------

        # 2. Analyze Symmetry
        # We use SpacegroupAnalyzer to standardize the cell.
        # symprec=0.01 is the standard tolerance for symmetry detection.
        sga = SpacegroupAnalyzer(structure, symprec=0.001)
        
        # Get the standard primitive structure.
        # This reduces supercells or conventional cells to the smallest possible primitive cell.
        prim_structure = sga.get_primitive_standard_structure()
        print("file space group number", sga.get_space_group_symbol(), sga.get_space_group_number())
        # 3. Extract Lattice Vectors
        # The .matrix attribute returns a 3x3 numpy array where rows are a, b, and c.
        lattice_vectors = prim_structure.lattice.matrix
        
        print("\n" + "="*50)
        print("PRIMITIVE LATTICE VECTORS (Angstrom)")
        print("="*50)
        print(f"a_prim = {lattice_vectors[0]}")
        print(f"b_prim = {lattice_vectors[1]}")
        print(f"c_prim = {lattice_vectors[2]}")
        print("="*50)

        # Optional: Print angles and lengths for verification
        print(f"\nPrimitive Lattice Constants: {prim_structure.lattice.abc}")
        print(f"Primitive Lattice Angles:    {prim_structure.lattice.angles}")
        print(f"Primitive Spacegroup:        {sga.get_space_group_symbol()} ({sga.get_space_group_number()})")
        
        return lattice_vectors

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Allows usage from command line: python get_primitive_vectors.py MyPOSCAR
    filename = sys.argv[1] if len(sys.argv) > 1 else "POSCAR"
    get_primitive_vectors(filename)

