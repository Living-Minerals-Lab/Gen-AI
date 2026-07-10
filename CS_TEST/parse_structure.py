import csv
import re
import sys
import argparse
from pathlib import Path

# Reuse make_unique_summary.py's CIF-parsing + CSV-writing logic so --output can
# produce the enriched summary (Directory, cell params, space group) directly.
sys.path.insert(0, str(Path(__file__).resolve().parent / "summaries"))
from make_unique_summary import build_rows, write_rows

def parse_and_sort(target_formula, out_filepath, mapping_filepath, output_filepath = None, cif_dir = ".", directory_label = None):
    """
    Parses the relaxation results, filters by formula, maps to gen_id, 
    and sorts by energy above hull.
    """
    # 1. Read the ID mapping into a dictionary
    # format: mattergen-X -> gen_Y.cif
    id_map = {}
    try:
        with open(mapping_filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            # Skip header if it exists
            header = next(reader, None)
            if header and "entry_id" not in header[0]:
                id_map[header[0].strip()] = header[1].strip()
                
            for row in reader:
                if len(row) >= 2:
                    id_map[row[0].strip()] = row[1].strip()
    except FileNotFoundError:
        print(f"Error: Could not find mapping file '{mapping_filepath}'")
        return

    # 2. Parse the 'out' file
    results = []
    
    # Regex updated to handle nested parentheses inside the chemical formula
    # Example match: "mattergen-10  (LiAl(SiO3)2): 0.206 eV/atom"
    pattern = re.compile(r"([a-zA-Z0-9\-]+)\s*\(\s*(.+?)\s*\)\s*:\s*([0-9.]+)\s*eV/atom")
    
    try:
        with open(out_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    structure_id = match.group(1).strip()
                    formula = match.group(2).strip()
                    energy = float(match.group(3).strip())
                    
                    if formula == target_formula:
                        # Lookup gen_id (default to 'N/A' if it's an mp- ID or missing)
                        gen_id = id_map.get(structure_id, "N/A")
                        results.append({
                            'mattergen_id': structure_id,
                            'gen_id': gen_id,
                            'energy': energy
                        })
    except FileNotFoundError:
        print(f"Error: Could not find out file '{out_filepath}'")
        return

    # 3. Sort the results by energy (increasing order)
    results.sort(key=lambda x: x['energy'])

    # 4. Output the results
    if not results:
        print(f"No structures found for formula: {target_formula}")
        return
        
    print(f"\nResults for {target_formula} (Sorted by Energy Above Hull):")
    print(f"{'-'*60}")
    print(f"{'Mattergen ID':<20} | {'Gen ID':<15} | {'Energy (eV/atom)'}")
    print(f"{'-'*60}")
    
    for res in results:
        print(f"{res['mattergen_id']:<20} | {res['gen_id']:<15} | {res['energy']:.3f}")	
    if output_filepath:
        if directory_label:
            selected_rows = [
                {"mattergen_id": res["mattergen_id"], "gen_id": res["gen_id"], "energy_above_hull": res["energy"]}
                for res in results
            ]
            rows = build_rows(selected_rows, cif_dir, directory_label)
            write_rows(rows, output_filepath)
        else:
            with open(output_filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Mattergen_ID', 'Gen_ID', 'Energy_Above_Hull'])
                for res in results:
                    writer.writerow([res['mattergen_id'], res['gen_id'], res['energy']])
            print(f"\nResults written to '{output_filepath}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract, map, and sort structure energies.")
    parser.add_argument("--formula", default="LiAl(SiO3)2", help="The chemical formula to search for (e.g., 'LiAl(SiO3)2')")
    parser.add_argument("--out", default="out.txt", help="Path to the 'out' file")
    parser.add_argument("--map", default="id_mapping.csv", help="Path to 'id_mapping.csv'")
    parser.add_argument("--output", default = None, help = "Path to output CSV file")
    parser.add_argument("--cif-dir", default = ".", help = "Folder containing CIF files (used when --directory is given)")
    parser.add_argument("--directory", default = None, help = "Directory label, e.g. 1-0 -- when given, --output is written as the enriched make_unique_summary.py-style CSV (Directory, cell params, space group) instead of the plain 3-column CSV")
    args = parser.parse_args()

    parse_and_sort(args.formula, args.out, args.map, args.output, args.cif_dir, args.directory)
