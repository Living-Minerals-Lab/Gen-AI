"""
Diagnostic for the supercell-vs-unit-cell 4-membered-ring discrepancy in
ring_calc_eu.py. Mirrors analyze_framework_rings() but keeps the periodic
image (jimage) info that the original discards, so we can check whether
each reported 4-ring actually closes in real 3D space (net translation ==
(0,0,0)) or is a periodicity-folding graph artifact (net translation != 0).

Usage:
    python debug_rings.py
"""

import collections
import itertools
import sys

import networkx as nx
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core import Structure

STRUCTURE_FILES = [
    "/global/cfs/projectdirs/m4397/For_Atreya/Gen-AI/CS_TEST/unit_cell_euc_cifs/POSCAR_alpha.vasp",
    "/global/cfs/projectdirs/m4397/For_Atreya/Gen-AI/CS_TEST/unit_cell_euc_cifs/POSCAR_beta.vasp",
    "/global/cfs/projectdirs/m4397/For_Atreya/Gen-AI/CS_TEST/unit_cell_euc_cifs/POSCAR_gamma.vasp",
]

MAX_RING_SIZE = 6


def debug_structure(structure_file):
    print(f"\n{'='*70}\nAnalyzing {structure_file}\n{'='*70}")

    structure = Structure.from_file(structure_file)
    crystal_nn = CrystalNN()
    full_structure_graph = StructureGraph.with_local_env_strategy(structure, crystal_nn)

    framework_indices = {i for i, site in enumerate(structure) if site.species_string in {"Al", "Si"}}
    oxygen_indices = [i for i, site in enumerate(structure) if site.species_string == "O"]

    framework_graph = nx.Graph()
    framework_graph.add_nodes_from(framework_indices)

    # edge (i, j) with i < j -> list of (oxygen_idx, jimage_i, jimage_j)
    # jimage_i/jimage_j are the periodic image of atom i/j *relative to the
    # bridging oxygen's home cell*, i.e. what get_connected_sites gives us.
    edge_bridges = collections.defaultdict(list)

    for o_idx in oxygen_indices:
        neighbors = full_structure_graph.get_connected_sites(o_idx)
        framework_neighbors = [n for n in neighbors if n.index in framework_indices]

        if len(framework_neighbors) == 2:
            n1, n2 = framework_neighbors
            atom1, atom2 = n1.index, n2.index
            framework_graph.add_edge(atom1, atom2)
            key = (atom1, atom2) if atom1 < atom2 else (atom2, atom1)
            jimage1, jimage2 = (n1.jimage, n2.jimage) if atom1 < atom2 else (n2.jimage, n1.jimage)
            edge_bridges[key].append((o_idx, jimage1, jimage2))

    print(f"Framework atoms: {len(framework_indices)}, oxygens: {len(oxygen_indices)}")
    print(f"Framework graph: {framework_graph.number_of_nodes()} nodes, {framework_graph.number_of_edges()} edges")

    all_rings = list(nx.simple_cycles(framework_graph, length_bound=MAX_RING_SIZE))
    ring_counts = collections.Counter(len(r) for r in all_rings)
    print("Reproduced ring counts:")
    for size, count in sorted(ring_counts.items()):
        print(f"  - {size}-membered rings: {count}")

    four_rings = [r for r in all_rings if len(r) == 4]
    if not four_rings:
        print("No 4-membered rings to inspect.")
        return

    print(f"\nInspecting {len(four_rings)} four-membered ring(s):")
    zero_net = 0
    nonzero_net = 0
    for ring_idx, ring in enumerate(four_rings):
        species = [structure[i].species_string for i in ring]
        print(f"\n  Ring #{ring_idx}: atoms {ring} ({species})")

        cycle_edges = list(zip(ring, ring[1:] + ring[:1]))
        net_translation = [0, 0, 0]
        for a, b in cycle_edges:
            key = (a, b) if a < b else (b, a)
            bridges = edge_bridges.get(key, [])
            print(f"    edge {a}-{b}: {len(bridges)} bridging oxygen(s)")
            for o_idx, jimage1, jimage2 in bridges:
                delta = [j2 - j1 for j1, j2 in zip(jimage1, jimage2)]
                print(f"      via O{o_idx}: jimage({key[0]})={jimage1}, jimage({key[1]})={jimage2}, delta={delta}")
            # use the first bridge found for this edge to accumulate net translation
            # (direction of traversal a->b determines sign)
            if bridges:
                o_idx, jimage1, jimage2 = bridges[0]
                delta = [j2 - j1 for j1, j2 in zip(jimage1, jimage2)]
                if a != key[0]:
                    delta = [-d for d in delta]
                net_translation = [n + d for n, d in zip(net_translation, delta)]

        is_real = net_translation == [0, 0, 0]
        print(f"    net periodic translation around cycle: {net_translation} "
              f"({'REAL closed ring' if is_real else 'ARTIFACT - does not close in 3D'})")
        if is_real:
            zero_net += 1
        else:
            nonzero_net += 1

    print(f"\nSummary for {structure_file.rsplit('/', 1)[-1]}: "
          f"{zero_net} real / {nonzero_net} artifact (of {len(four_rings)} four-membered rings)")


if __name__ == "__main__":
    files = sys.argv[1:] if len(sys.argv) > 1 else STRUCTURE_FILES
    for f in files:
        debug_structure(f)
