import argparse
import pyscal3.core as pc
import sys
from ase.io import read
import matplotlib.pyplot as plt
import os

QVALS = [4, 6, 8, 10, 12]

def analyze_poscar_for_elements(file_path, elements=['Li', 'Al', 'Si'], qvals=QVALS):
    """
    Analyzes a single POSCAR file, calculates each Steinhardt q in `qvals` for
    the specified elements, and returns a nested dict {q: {element: [values]}}
    plus the file path.
    """
    system = pc.System()
    try:
        system.read.file(file_path, format="poscar")
        atoms_ase = read(file_path) # Read with ASE for chemical symbols
        print(f"Loaded {file_path} with {len(atoms_ase)} atoms.")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None # Return None if there's an error

    print(f"Finding neighbors for {file_path} with cutoff=3.0...")
    system.find.neighbors(method='cutoff', cutoff=3.0)

    print(f"Calculating q values for {file_path}...")
    system.calculate.steinhardt_parameter(qvals)

    symbols = atoms_ase.get_chemical_symbols()
    results_by_q = {}
    for q in qvals:
        q_values = system.atoms[f"q{q}"]
        results_by_element = {element: [] for element in elements}
        # Iterate through all atoms and assign this q to the correct element's list
        for i, symbol in enumerate(symbols):
            if symbol in elements:
                results_by_element[symbol].append(q_values[i])
        for element in elements:
            if not results_by_element[element]:
                print(f"No {element} atoms found in {file_path}.")
        results_by_q[q] = results_by_element

    # Return the nested q -> element -> values dict and the full file_path for labeling
    return results_by_q, file_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and plot Steinhardt q parameters for POSCAR files.")
    parser.add_argument("files", nargs="+", help="POSCAR file(s) to analyze")
    parser.add_argument("--qvals", type=str, default="4,6,8,10,12",
                         help="Comma-separated Steinhardt q values to compute/plot (default: 4,6,8,10,12)")
    parser.add_argument("--output-prefix", type=str, default="order_parameter",
                         help="Prefix for output PNG filenames, i.e. <prefix>_q<N>.png (default: order_parameter)")
    args = parser.parse_args()

    input_files = args.files
    qvals = [int(x) for x in args.qvals.split(",")]
    output_prefix = args.output_prefix
    all_structures_results = [] # Stores results for all files

    # Friendly labels for the known eucryptite reference structures, used in
    # place of the raw file path in the plot legend.
    FILE_LABELS = {
        "POSCAR_30982.vasp": "alpha",
        "POSCAR_2929.vasp": "beta",
        "POSCAR_66137.vasp": "gamma",
        "POSCAR_6302.vasp": "gen_6302",
        "POSCAR_1442.vasp": "gen_1442",
        "POSCAR_2140.vasp": "gen_2140",
        "POSCAR_2348.vasp": "gen_2348",
        "POSCAR_3490.vasp": "gen_3490",
        "POSCAR_435.vasp": "gen_435",
        "POSCAR_5507.vasp": "gen_5507",
        "POSCAR_643.vasp": "gen_643",
        "POSCAR_7022.vasp": "gen_7022",
        "POSCAR_8567.vasp": "gen_8567",
    }

    # Process each input file
    for file_path in input_files:
        q_data_by_q, label_arg = analyze_poscar_for_elements(file_path, qvals=qvals)
        if q_data_by_q is not None: # Only add if file was processed successfully
            label_arg = FILE_LABELS.get(os.path.basename(file_path), label_arg)
            all_structures_results.append((q_data_by_q, label_arg))

    if not all_structures_results:
        print("No valid POSCAR files processed. Exiting.")
        sys.exit(0)

    # Define the elements to plot
    elements_to_plot = ['Li', 'Al', 'Si']
    element_names = {'Li': 'Lithium', 'Al': 'Aluminum', 'Si': 'Silicon'}

    # Produce one 3-panel (Li/Al/Si) figure per Steinhardt q value.
    for q in qvals:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True) # sharey ensures y-axis scale is same
        axes = axes.flatten() # Ensures axes is a flat array even if there's only 1 subplot

        # Loop through each element and create its subplot
        for i, element in enumerate(elements_to_plot):
            ax = axes[i] # Get the current subplot axis

            for q_data_by_q, label_text in all_structures_results:
                # Get this q's values for the current element from the current file's data
                q_values_for_this_element = q_data_by_q[q].get(element, [])

                if q_values_for_this_element:
                    ax.plot(range(len(q_values_for_this_element)),
                            q_values_for_this_element,
                            linestyle='None',
                            marker="o",
                            label=label_text,
                            alpha=0.7)
                # else: no atoms of this type in this file, so don't plot for it

            ax.set_title(f"{element_names[element]} Atoms ($q_{{{q}}}$)")
            ax.set_xlabel(f"{element} Atom Index (sequential for each file's {element} atoms)")
            ax.set_ylabel(f"$q_{{{q}}}$ Steinhardt Parameter")
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(title="Structures") # Add legend to each subplot

        plt.tight_layout() # Adjust layout to prevent labels/titles from overlapping
        plt.savefig(f"{output_prefix}_q{q}.png", dpi=150)
        plt.close(fig)

    print(f"\nScript finished. Saved plots for q = {qvals} in the current directory.")
