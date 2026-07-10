import pymatgen, sys
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure

structure_1 = Structure.from_file(sys.argv[1])
structure_2 = Structure.from_file(sys.argv[2])

matcher = StructureMatcher(ltol=0.5, stol=0.8, angle_tol=6, attempt_supercell=True)
print(matcher.get_rms_dist(structure_1, structure_2))
