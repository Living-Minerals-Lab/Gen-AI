#read structure from file
import sys
from pymatgen.core import Structure
struct = Structure.from_file(sys.argv[1])

#get symmetrized primitive and conventional cell
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
sga = SpacegroupAnalyzer(struct,symprec=1e-5)
struct_symm_conv = sga.get_refined_structure()
struct_symm_prim = sga.find_primitive()

#write primitive and conventional cells
from pymatgen.io.vasp import Poscar
poscar = Poscar(struct_symm_conv)
poscar.write_file(filename="POSCAR-conv",significant_figures=16)
poscar = Poscar(struct_symm_prim)
poscar.write_file(filename="POSCAR-prim",significant_figures=16)