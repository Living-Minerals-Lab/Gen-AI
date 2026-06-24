# Phase Diagram and Structural Family Workflow

This directory contains four Python scripts intended to be run as a small workflow:

```text
new_plot.py -> parse_structure.py -> analyze_2.py -> cluster_reference_agent.py
```

The scripts assume that generated structure CIF files, reference structure files, a MatterGen relaxation-results JSON file, and an ID mapping CSV are available in the paths described below.

## Script Summary

### `new_plot.py`

`new_plot.py` loads a MatterGen-style JSON cache of relaxed structures, filters structures to a requested chemical system, converts them into `pymatgen` `ComputedEntry` objects, and builds a convex-hull phase diagram. It prints each matching entry's energy above hull and opens an interactive phase-diagram plot showing stable and unstable entries.

### `parse_structure.py`

`parse_structure.py` reads the text output from `new_plot.py`, filters rows for one target formula, and maps each `mattergen-*` entry ID to a generated CIF filename using `id_mapping.csv`. It prints a sorted table of `mattergen_id`, `gen_id`, and energy above hull, ordered from lowest to highest energy.

### `analyze_2.py`

`analyze_2.py` reads the sorted table produced by `parse_structure.py`, loads the matching generated CIF files and configured reference structures, and computes SOAP fingerprints for structural comparison. It uses UMAP and HDBSCAN to cluster structures, writes cluster assignment and summary CSV files, and saves UMAP plots colored by cluster and energy above hull.

### `cluster_reference_agent.py`

`cluster_reference_agent.py` reads `family_analysis_out/cluster_summary.csv` and `family_analysis_out/cluster_assignments.csv` from `analyze_2.py` and prepares a cluster-by-reference ranking table. It sends the clustering data to an OpenAI-compatible CBORG endpoint, validates the model output against the CSV-derived expected rankings, then writes both CSV and JSON versions of the normalized table.

## Expected Inputs

Before running the workflow, prepare these files and paths:

- `relaxation_results.json`: JSON file consumed by `new_plot.py`; each entry must contain `final_structure` and `final_energy`.
- `id_mapping.csv`: CSV file consumed by `parse_structure.py`; expected columns are an entry ID such as `mattergen-10` and a CIF filename such as `gen_10.cif`.
- `./../cifs`: directory consumed by `analyze_2.py`; must contain the generated CIF files listed in `id_mapping.csv`.
- `./refs`: directory consumed by `analyze_2.py`; must contain the reference files configured in the `REFERENCES` dictionary.
- `CBORG_API_KEY`: environment variable required only for `cluster_reference_agent.py`.

## Python Dependencies

The scripts use:

```text
pymatgen
ase
dscribe
numpy
pandas
matplotlib
scikit-learn
umap-learn
hdbscan
openai
```

Install these into your active Python environment before running the workflow.

## Run Order

### 1. Run `new_plot.py`

Run the phase-diagram analysis with a relaxation-results JSON file and a chemical system string:

```bash
python3 new_plot.py relaxation_results.json Li-Al-Si-O | tee out.txt
```

Replace `relaxation_results.json` with your actual JSON path and `Li-Al-Si-O` with the chemical system you want to analyze. The command writes the printed stability table to `out.txt`, which is the default input expected by `parse_structure.py`; close the plot window when you are done viewing the phase diagram.

### 2. Run `parse_structure.py`

Filter the `new_plot.py` output for the formula of interest and write the parsed table to `data.txt`:

```bash
python3 parse_structure.py --formula "LiAl(SiO3)2" --out out.txt --map id_mapping.csv | tee data.txt
```

Change `--formula` if you want a different reduced formula. `data.txt` is the default file read by `analyze_2.py`.

### 3. Run `analyze_2.py`

Check the configuration block in `analyze_2.py` before running:

```python
DATA_TXT = "data.txt"
CIF_DIR = "./../cifs"
REFERENCES = {
    "alpha": "./refs/mp-6340-standard.cif",
    "gamma": "./refs/mp-122249.cif",
    "beta_parent": "./refs/POSCAR_beta",
    "beta_partial": "./refs/beta_gen_8551.cif",
    "beta_cif": "./refs/Beta_AMS.cif",
    "Bikatite": "./refs/mp-558808.cif",
}
OUT_DIR = "./family_analysis_out"
```

Then run:

```bash
python3 analyze_2.py
```

The script writes outputs under `family_analysis_out/`, including `cluster_assignments.csv`, `cluster_summary.csv`, `umap_by_cluster.png`, `umap_by_ehull.png`, `soap_cache.pkl`, and per-reference top-20 CSV files.

### 4. Run `cluster_reference_agent.py`

Export your CBORG API key, then run the reference-table agent on the clustering outputs:

```bash
export CBORG_API_KEY="your_api_key_here"
python3 cluster_reference_agent.py \
  --summary family_analysis_out/cluster_summary.csv \
  --assignments family_analysis_out/cluster_assignments.csv \
  --out family_analysis_out/cluster_reference_table.csv \
  --json-out family_analysis_out/cluster_reference_table.json
```

Optionally set `CBORG_MODEL` or pass `--model` to choose a model other than the script default. The final outputs are `family_analysis_out/cluster_reference_table.csv` and `family_analysis_out/cluster_reference_table.json`.

## Notes

- `analyze_2.py` skips rows where `parse_structure.py` reports `gen_id` as `N/A`.
- `analyze_2.py` treats Si and Al as the same placeholder species when `MERGE_SI_AL = True`, making SOAP comparisons decoration-blind for those atoms.
- `cluster_reference_agent.py` does not blindly trust the model response; it normalizes the table back to entries that are valid according to `cluster_assignments.csv` and `cluster_summary.csv`.

## Reading Material  

```text
Mattergen: https://www.nature.com/articles/s41586-025-08628-5  
MatterSim: https://arxiv.org/abs/2405.04967
```

