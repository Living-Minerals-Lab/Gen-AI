import collections
import sys
import networkx as nx
from pymatgen.core import Structure
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import CrystalNN, CutOffDictNN
import itertools
import matplotlib.pyplot as plt

def visualize_graph(graph, filename="framework_graph.png"):
    """
    Uses matplotlib to draw and save a visualization of a networkx graph.

    Args:
        graph (nx.Graph): The networkx graph to visualize.
        filename (str): The path to save the output image file.
    """
    plt.figure(figsize=(12, 12))
    # Use a spring layout for a more aesthetically pleasing arrangement
    pos = nx.spring_layout(graph, iterations=100, seed=42)
    nx.draw(graph, pos, 
            with_labels=True, 
            node_color='skyblue', 
            node_size=400, 
            font_size=8,
            width=0.5)
    plt.title("Framework-Only Graph Visualization")
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Graph visualization saved to {filename}")


def analyze_framework_rings(structure_file: str, max_framework_ring_size: int = 8, visualize: bool = False):
    """
    Analyzes the aluminosilicate framework of a crystal structure to find and
    count closed polyhedral rings using an optimized graph representation.

    This function works by:
    1. Loading a crystal structure from a file (e.g., CIF).
    2. Building a bond graph of all atoms (Al, Si, O) using the chemically-
       aware CrystalNN algorithm.
    3. Creating a new, abstract 'framework-only' graph where only Al and Si
       atoms are nodes. An edge is created between two framework atoms if and
       only if they share an oxygen atom in the full structure.
    4. (Optional) Visualizing this framework-only graph.
    5. Finding all simple cycles in this highly efficient framework-only graph.
       The length of a cycle found here directly corresponds to the size of the
       polyhedral ring.
    6. Counting the rings by their size.

    Args:
        structure_file (str): Path to the crystal structure file (e.g., 'beta_spodumene.cif').
        max_framework_ring_size (int): The maximum size of Al/Si rings to search for.
        visualize (bool): If True, save a PNG image of the framework graph.

    Returns:
        collections.Counter: A dictionary-like object with ring sizes as keys
                             and their frequencies as values.
    """
    print(f"--- Analyzing {structure_file} ---")
    
    # 1. Load the structure from the provided file path
    try:
        structure = Structure.from_file(structure_file)
    except Exception as e:
        print(f"Error loading structure file: {e}")
        return collections.Counter()

    # 2. Create the full bond graph to understand connectivity.
    # We are now using the chemically-aware CrystalNN, which is more robust.
    crystal_nn = CrystalNN()

    # --- Alternative for later: CutoffDictNN ---
    # If you wanted to use explicit bond distances, you could use CutoffDictNN
    # like this. It requires knowing reasonable bond lengths beforehand.
    # cutoffs = {('Si', 'O'): 2.0, ('Al', 'O'): 2.0}
    # cutoff_nn = CutOffDictNN(cutoffs)
    # full_structure_graph = StructureGraph.with_local_env_strategy(structure, cutoff_nn)
    
    full_structure_graph = StructureGraph.with_local_env_strategy(structure, crystal_nn)

    # 3. Build the optimized, framework-only graph.
    framework_graph = nx.Graph()

    # Identify indices of framework (Al, Si) and bridging (O) atoms.
    framework_indices = {i for i, site in enumerate(structure) if site.species_string in {'Al', 'Si'}}
    oxygen_indices = [i for i, site in enumerate(structure) if site.species_string == 'O']

    # Add only framework atoms as nodes to the new graph.
    framework_graph.add_nodes_from(framework_indices)

    # Per undirected edge (atom1, atom2) with atom1 < atom2, keep every bridging
    # oxygen's periodic-image delta (jimage2 - jimage1). A bond that crosses a
    # periodic cell boundary lands on the *same atom index* as an ordinary
    # same-cell bond (every periodic copy of an atom shares one index), so a
    # cycle can look closed in this graph while actually spiraling off through
    # periodic images in real 3D space. Recording these deltas lets us verify,
    # after cycle-finding, which candidate cycles actually close (deltas sum to
    # (0, 0, 0)) versus which are periodicity artifacts.
    edge_deltas = collections.defaultdict(list)

    # For each oxygen, find its framework neighbors. Add edges between these
    # neighbors in our new framework_graph.
    for o_idx in oxygen_indices:
        neighbors = full_structure_graph.get_connected_sites(o_idx)

        # Filter neighbors to keep only Al/Si atoms
        framework_neighbors = [n for n in neighbors if n.index in framework_indices]

        # Since CrystalNN is chemically intelligent, it should correctly identify
        # bridging oxygens as having only 2 framework neighbors. We can still
        # enforce this for maximum robustness.
        if len(framework_neighbors) == 2:
            n1, n2 = framework_neighbors
            atom1, atom2 = n1.index, n2.index

            # An oxygen bridging two periodic copies of the *same* framework
            # atom (atom1 == atom2) can't be part of any real 4- or
            # 6-membered ring (a ring never revisits an atom); skip it so it
            # never becomes a self-loop edge in framework_graph.
            if atom1 == atom2:
                continue

            framework_graph.add_edge(atom1, atom2)

            key = (atom1, atom2) if atom1 < atom2 else (atom2, atom1)
            jimage1, jimage2 = (n1.jimage, n2.jimage) if atom1 < atom2 else (n2.jimage, n1.jimage)
            delta = tuple(j2 - j1 for j1, j2 in zip(jimage1, jimage2))
            edge_deltas[key].append(delta)

    # 4. (Optional) Visualize the graph before finding cycles.
    if visualize:
        output_filename = f"{structure_file.rsplit('.', 1)[0]}_graph.png"
        visualize_graph(framework_graph, filename=output_filename)

    # 5. Find all simple cycles in the much smaller, optimized graph.
    # The length_bound now corresponds directly to the framework ring size.
    all_rings = list(nx.simple_cycles(framework_graph, length_bound=max_framework_ring_size))

    # 6. Count only the rings that actually close in 3D. For each candidate
    # cycle, walk its edges and check whether any combination of bridging
    # oxygens (an edge can have more than one) sums to a net translation of
    # (0, 0, 0). If none does, the cycle never returns to its real starting
    # atom and is a periodicity artifact, not a genuine ring.
    #
    # A cycle that passes the index-space check is then independently
    # confirmed in real space: its atoms are reconstructed to actual
    # Cartesian coordinates using the winning image combination, and every
    # consecutive framework-atom pair must sit at a physically plausible
    # T...T distance. This uses real lattice geometry rather than integer
    # image bookkeeping, so it would catch any residual reference-frame bug
    # (or CrystalNN mis-bonding) that the index-space check alone could miss.
    MAX_TT_DISTANCE = 4.0  # Angstrom; corner-sharing AlO4/SiO4 T...T is ~3.0-3.3 A

    def _distance(p1, p2):
        return sum((c1 - c2) ** 2 for c1, c2 in zip(p1, p2)) ** 0.5

    framework_ring_counts = collections.Counter()
    artifact_count = 0
    geometry_artifact_count = 0
    for ring_path in all_rings:
        cycle_edges = list(zip(ring_path, ring_path[1:] + ring_path[:1]))

        per_edge_options = []
        for a, b in cycle_edges:
            key = (a, b) if a < b else (b, a)
            sign = 1 if a == key[0] else -1
            deltas = edge_deltas.get(key, [(0, 0, 0)])
            per_edge_options.append([tuple(sign * d for d in delta) for delta in deltas])

        winning_combo = next(
            (combo for combo in itertools.product(*per_edge_options)
             if all(sum(axis) == 0 for axis in zip(*combo))),
            None,
        )

        if winning_combo is None:
            artifact_count += 1
            continue

        cumulative_image = (0, 0, 0)
        cart_positions = [structure.lattice.get_cartesian_coords(structure[ring_path[0]].frac_coords)]
        for (a, b), delta in zip(cycle_edges, winning_combo):
            cumulative_image = tuple(c + d for c, d in zip(cumulative_image, delta))
            b_frac = structure[b].frac_coords + cumulative_image
            cart_positions.append(structure.lattice.get_cartesian_coords(b_frac))

        max_tt_dist = max(
            _distance(cart_positions[i], cart_positions[i + 1])
            for i in range(len(cart_positions) - 1)
        )

        if max_tt_dist > MAX_TT_DISTANCE:
            geometry_artifact_count += 1
            continue

        # The length of the path in this graph IS the ring size.
        framework_ring_counts[len(ring_path)] += 1

    if artifact_count:
        print(f"Filtered out {artifact_count} periodic-boundary artifact cycle(s) "
              f"(closed in index-space but not in real 3D space).")
    if geometry_artifact_count:
        print(f"Filtered out {geometry_artifact_count} additional cycle(s) that closed "
              f"in index-space but reconstructed to an unphysical T...T distance "
              f"(> {MAX_TT_DISTANCE} A) in real 3D space.")

    if not framework_ring_counts:
        print("No valid framework rings found.")
    else:
        print("Found framework ring counts:")
        for size, count in sorted(framework_ring_counts.items()):
            print(f"  - {size}-membered rings: {count}")

    return framework_ring_counts


if __name__ == '__main__':
    # --- Main execution block ---
    
    # Check for the --visualize flag
    visualize_output = "--visualize" in sys.argv
    
    # Get the list of cif files, excluding the flag
    cif_files = [f for f in sys.argv[1:] if f != "--visualize"]
    
    if not cif_files:
        print("Usage: python ring_analyzer.py [--visualize] <file1.cif> <file2.cif> ...")
        sys.exit(1)
        
    # Loop through all files provided as command line arguments
    for i, cif_file in enumerate(cif_files):
        # We can increase the max ring size now that the algorithm is faster
        analyze_framework_rings(cif_file, max_framework_ring_size=6, visualize=visualize_output)
        if i < len(cif_files) - 1:
             print("\n" + "="*40 + "\n")