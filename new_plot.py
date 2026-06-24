import json
import argparse
from typing import List

from pymatgen.core.structure import Structure
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter

def load_and_filter_entries(cache_file: str, chemical_system: str) -> List[ComputedEntry]:
    """
    Loads all results from a JSON cache file and filters them for a specific chemical system.

    Args:
        cache_file (str): Path to the JSON file containing relaxation results.
        chemical_system (str): The chemical system to filter for (e.g., 'H-O').

    Returns:
        List[ComputedEntry]: A list of pymatgen ComputedEntry objects for the specified system.
    """
    print(f"Loading results from '{cache_file}'...")
    
    # Define the set of elements we are interested in
    target_elements = set(chemical_system.split('-'))
    
    try:
        with open(cache_file, 'r') as f:
            all_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Cache file not found at '{cache_file}'")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{cache_file}'. It may be corrupted.")
        return []

    print(f"Filtering for chemical system: {chemical_system}")
    filtered_entries = []
    for entry_id, result in all_data.items():
        # --- FIX: Use the correct key 'structure' instead of 'final_structure' ---
        structure = Structure.from_dict(result['final_structure'])
        
        # Get the set of elements present in the current structure
        structure_elements = set(el.symbol for el in structure.composition.elements)
        
        # Check if the structure's elements are a subset of our target system
        if structure_elements.issubset(target_elements):
            entry = ComputedEntry(
                composition=structure.composition,
                # --- FIX: Use the correct key 'energy' instead of 'final_energy' ---
                energy=result['final_energy'],
                entry_id=entry_id
            )
            filtered_entries.append(entry)
            
    print(f"Found {len(filtered_entries)} entries matching the '{chemical_system}' system.")
    return filtered_entries

def analyze_and_plot_hull(entries: List[ComputedEntry], system_name: str):
    """
    Constructs, analyzes, and plots the convex energy hull for a given set of entries.
    """
    if not entries:
        print("No entries to analyze. Exiting.")
        return
        
    print("\nConstructing phase diagram and analyzing stability...")
    
    phase_diagram = PhaseDiagram(entries)
    
    print("\n--- Stability Analysis (Energy Above Hull) ---")
    # Sort entries by composition for clear, ordered output
    sorted_entries = sorted(entries, key=lambda e: e.composition.fractional_composition.to_pretty_string())
    for entry in sorted_entries:
        stability = phase_diagram.get_e_above_hull(entry)
        formula = entry.composition.reduced_formula
        print(f"  {entry.entry_id:<15} ({formula:<8}): {stability:.3f} eV/atom")

    plotter = PDPlotter(phase_diagram, show_unstable=True)
    plot = plotter.get_plot()
    
    print("\nDisplaying phase diagram plot. Close the plot window to exit.")
    plot.show()
    print("Analysis complete.")

def main():
    """
    Main function to parse arguments and run the analysis.
    """
    parser = argparse.ArgumentParser(
        description="Generate and plot a convex energy hull from a pre-computed JSON cache."
    )
    parser.add_argument(
        "json_file", 
        type=str, 
        help="Path to the relaxation_results.json cache file."
    )
    parser.add_argument(
        "chemical_system", 
        type=str, 
        help="Chemical system to analyze, formatted as 'El1-El2-...' (e.g., 'H-O')."
    )
    args = parser.parse_args()

    # Run the workflow
    computed_entries = load_and_filter_entries(args.json_file, args.chemical_system)
    analyze_and_plot_hull(computed_entries, args.chemical_system)

if __name__ == "__main__":
    main()
