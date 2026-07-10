import pyscal.core as pc
import sys
from ase.io import read, write
import matplotlib.pyplot as plt
import os

def analyze_poscar_for_elements(file_path, elements=['Li', 'Al', 'Si']):
    """
    Analyzes a single POSCAR file, calculates q4 for specified elements,
    and returns a dictionary of q4 values for each element and the file path.
    """
    system = pc.System()
    try:
        system.read_inputfile(file_path, format="poscar")
        atoms_ase = read(file_path) # Read with ASE for chemical symbols
        print(f"Loaded {file_path} with {len(atoms_ase)} atoms.")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None # Return None if there's an error

    print(f"Finding neighbors for {file_path} with cutoff=3.0...")
    system.find_neighbors(method='cutoff', cutoff=3.0)

    print(f"Calculating q values for {file_path}...")
    system.calculate_q([4,6,8,10,12])

    all_q4_values = system.get_qvals(8)
    results_q4_by_element = {element: [] for element in elements}

    # Iterate through all atoms and assign q4 to the correct element's list
    for i, symbol in enumerate(atoms_ase.get_chemical_symbols()):
        if symbol in elements:
            results_q4_by_element[symbol].append(all_q4_values[i])

    for element in elements:
        if not results_q4_by_element[element]:
            print(f"No {element} atoms found in {file_path}.")

    # Return the dictionary of q4 values by element and the full file_path for labeling
    return results_q4_by_element, file_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_multi_poscar_multi_element.py <POSCAR_file1> [POSCAR_file2] [POSCAR_file3] ...")
        sys.exit(1)

    input_files = sys.argv[1:]
    all_structures_results = [] # Stores results for all files

    # Process each input file
    for file_path in input_files:
        temp_atoms = read(file_path)
        write("POSCAR", temp_atoms, format="vasp")
        file_path = "POSCAR"
        q4_data_for_elements, label_arg = analyze_poscar_for_elements(file_path)
        if q4_data_for_elements is not None: # Only add if file was processed successfully
            all_structures_results.append((q4_data_for_elements, label_arg))

    if not all_structures_results:
        print("No valid POSCAR files processed. Exiting.")
        sys.exit(0)

    # Define the elements to plot and their subplot titles
    elements_to_plot = ['Li', 'Al', 'Si']
    subplot_titles = {
        'Li': 'Lithium Atoms ($q_{12}$)',
        'Al': 'Aluminum Atoms ($q_{12}$)',
        'Si': 'Silicon Atoms ($q_{12}$)'
    }

    # Create a figure with three subplots (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True) # sharey ensures y-axis scale is same
    axes = axes.flatten() # Ensures axes is a flat array even if there's only 1 subplot

    # Loop through each element and create its subplot
    for i, element in enumerate(elements_to_plot):
        ax = axes[i] # Get the current subplot axis

        for element_data_from_file, label_text in all_structures_results:
            # Get the q4 values for the current element from the current file's data
            q4_values_for_this_element = element_data_from_file.get(element, [])

            if q4_values_for_this_element:
                ax.plot(range(len(q4_values_for_this_element)),
                        q4_values_for_this_element,
                        linestyle='None',
                        marker="o",
                        label=label_text,
                        alpha=0.7)
            # else: no atoms of this type in this file, so don't plot for it

        ax.set_title(subplot_titles[element])
        ax.set_xlabel(f"{element} Atom Index (sequential for each file's {element} atoms)")
        ax.set_ylabel("$q_4$ Steinhardt Parameter")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(title="Structures") # Add legend to each subplot

    plt.tight_layout() # Adjust layout to prevent labels/titles from overlapping
    plt.show()

    print("\nScript finished. A plot with three subplots should be displayed.")

