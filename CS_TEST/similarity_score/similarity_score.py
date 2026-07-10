"""
Pairwise structural similarity among CIFs in cifstoexplore/ using pymatgen's
StructureMatcher.

Usage:
    python similarity_score.py
Output:
    similarity_scores.csv  (pair, rmsd, max_displacement)
"""

import itertools
import os
from pathlib import Path

import pandas as pd
from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator

CIF_DIR = "cifstoexplore"
OUT_CSV = "similarity_scores.csv"

# Only these two reference ("Entry...") structures are kept -- alpha and beta,
# per the REFERENCES dict in analyze_2.py. All other EntryWithCollCode... CIFs
# are excluded from this comparison.
KEEP_ENTRY_FILES = {"EntryWithCollCode194285.cif", "EntryWithCollCode194289.cif"}

# StructureMatcher tolerances (more permissive than pymatgen defaults
# ltol=0.2, stol=0.3, angle_tol=5) so more structurally-similar pairs match.
MATCHER_LTOL = 0.6
MATCHER_STOL = 0.9
MATCHER_ANGLE_TOL = 25

# Cluster-vs-cluster comparison config: which HDBSCAN cluster (from
# analyze_2.py, see all_unique_summary.csv "Cluster" column) to compare
# member structures against each other.
SUMMARY_CSV = "/global/cfs/projectdirs/m4397/For_Atreya/Gen-AI/CS_TEST/summaries/all_unique_summary_petalite.csv"
BASE_CIF_DIR = "."  # holds <model_weight>/UNIQUE_<unique_idx>/UNIQUE/<gen_id>.cif
CLUSTERS_TO_ANALYZE = [0, 1]
CLUSTER_OUT_CSV_TEMPLATE = "cluster_{cluster}_similarity_scores.csv"

# Reference structure paths, matching the REFERENCES dict in analyze_2.py.
REFERENCES = {
    "alpha": "/global/cfs/projectdirs/m4397/For_Atreya/Gen-AI/CS_TEST/cifs/EntryWithCollCode194285.cif",
    "beta": "/global/cfs/projectdirs/m4397/For_Atreya/Gen-AI/CS_TEST/cifs/EntryWithCollCode194289.cif",
}

# (cluster_id, reference_label) pairs to compare cluster members against a
# single reference structure.
CLUSTER_VS_REF_TO_ANALYZE = [(0, "alpha")]
CLUSTER_VS_REF_OUT_CSV_TEMPLATE = "cluster_{cluster}_vs_{ref}_similarity_scores.csv"

# Eucryptite reference structures, matching the REFERENCES dict in analyze_2.py.
EUC_REFERENCES = {
    "alpha": "/global/cfs/projectdirs/m4397/For_Atreya/Gen-AI/CS_TEST/euc_cifs/EntryWithCollCode30982.cif",
    "beta": "/global/cfs/projectdirs/m4397/For_Atreya/Gen-AI/CS_TEST/euc_cifs/EntryWithCollCode2929.cif",
    "gamma": "/global/cfs/projectdirs/m4397/For_Atreya/Gen-AI/CS_TEST/euc_cifs/EntryWithCollCode66137.cif",
}

# Generated structures (gen_id) to compare against every EUC_REFERENCES entry.
# Each one's Directory is looked up from EUC_SUMMARY_CSV to find its CIF path.
EUC_TARGET_GEN_IDS = ["gen_1442", "gen_6302", "gen_435"]
EUC_SUMMARY_CSV = "/global/cfs/projectdirs/m4397/For_Atreya/Gen-AI/CS_TEST/summaries/make_unique_summary_euc.csv"
EUC_TARGET_OUT_CSV_TEMPLATE = "{gen_id}_vs_euc_references_similarity_scores.csv"

# Pairwise comparison among these EUC_TARGET_GEN_IDS structures themselves.
EUC_PAIRWISE_GEN_IDS = ["gen_6302", "gen_435"]
EUC_PAIRWISE_OUT_CSV = "gen_6302_vs_gen_435_similarity_scores.csv"

# Pairwise comparison among these petalite-dataset (SUMMARY_CSV) gen_ids.
PETALITE_PAIRWISE_GEN_IDS = ["gen_5853", "gen_1595", "gen_1871"]
PETALITE_PAIRWISE_OUT_CSV = "gen_5853_vs_gen_1595_vs_gen_1871_similarity_scores.csv"


def is_reference(filename):
    return filename.startswith("Entry")


def make_matcher():
    return StructureMatcher(
        ltol=MATCHER_LTOL,
        stol=MATCHER_STOL,
        angle_tol=MATCHER_ANGLE_TOL,
        attempt_supercell=True,
        allow_subset=True,
        comparator=ElementComparator(),
    )


def compare_all_pairs(structures, out_csv):
    """structures: dict label -> pymatgen Structure. Runs StructureMatcher on
    every pair and writes (pair, rmsd, max_displacement) to out_csv."""
    matcher = make_matcher()
    labels = list(structures)

    rows = []
    for l1, l2 in itertools.combinations(labels, 2):
        result = matcher.get_rms_dist(structures[l1], structures[l2])
        rmsd, max_disp = (None, None) if result is None else result
        rows.append({"pair": f"{l1} vs {l2}", "rmsd": rmsd, "max_displacement": max_disp})
        print(f"  {l1} vs {l2}: rmsd={rmsd}, max_displacement={max_disp}")

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(rows)} pairs)")


def run_reference_comparison():
    cif_files = sorted(
        f for f in os.listdir(CIF_DIR)
        if f.endswith(".cif") and (not is_reference(f) or f in KEEP_ENTRY_FILES)
    )
    print(f"Found {len(cif_files)} CIF files in {CIF_DIR}/ (after filtering references)")

    structures = {f: Structure.from_file(str(Path(CIF_DIR) / f)) for f in cif_files}
    matcher = make_matcher()

    rows = []
    for f1, f2 in itertools.combinations(cif_files, 2):
        if not is_reference(f1) and not is_reference(f2):
            continue

        result = matcher.get_rms_dist(structures[f1], structures[f2])
        rmsd, max_disp = (None, None) if result is None else result

        rows.append({
            "pair": f"{f1} vs {f2}",
            "rmsd": rmsd,
            "max_displacement": max_disp,
        })
        print(f"  {f1} vs {f2}: rmsd={rmsd}, max_displacement={max_disp}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV} ({len(df)} pairs)")


def load_cluster_members(cluster_id):
    """Return {label: cif_path} for every unique structure assigned to cluster_id
    in all_unique_summary.csv's "Cluster" column (added by cross-checking
    cluster_assignments.csv)."""
    summ = pd.read_csv(SUMMARY_CSV)
    cluster_rows = summ[summ["Cluster"] == cluster_id]

    members = {}
    seen_paths = set()
    for _, row in cluster_rows.iterrows():
        model_weight, unique_idx = row["Directory"].split("-")
        path = (
            Path(BASE_CIF_DIR) / model_weight / f"UNIQUE_{unique_idx}"
            / "UNIQUE" / f"{row['Gen ID']}.cif"
        )
        if str(path) in seen_paths:
            continue  # duplicate summary row pointing at the same physical CIF
        seen_paths.add(str(path))
        members[f"{row['Gen ID']} ({row['Directory']})"] = path
    return members


def run_cluster_comparison(cluster_id):
    members = load_cluster_members(cluster_id)
    print(f"\nCluster {cluster_id}: {len(members)} unique structures")

    structures = {label: Structure.from_file(str(path)) for label, path in members.items()}
    out_csv = CLUSTER_OUT_CSV_TEMPLATE.format(cluster=cluster_id)
    compare_all_pairs(structures, out_csv)


def run_cluster_vs_reference(cluster_id, ref_label):
    members = load_cluster_members(cluster_id)
    ref_structure = Structure.from_file(REFERENCES[ref_label])
    print(f"\nCluster {cluster_id} vs {ref_label}: {len(members)} unique structures")

    matcher = make_matcher()
    rows = []
    for label, path in members.items():
        result = matcher.get_rms_dist(ref_structure, Structure.from_file(str(path)))
        rmsd, max_disp = (None, None) if result is None else result
        rows.append({"pair": f"{ref_label} vs {label}", "rmsd": rmsd, "max_displacement": max_disp})
        print(f"  {ref_label} vs {label}: rmsd={rmsd}, max_displacement={max_disp}")

    out_csv = CLUSTER_VS_REF_OUT_CSV_TEMPLATE.format(cluster=cluster_id, ref=ref_label)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(rows)} pairs)")


def resolve_gen_id_path(gen_id, summary_csv, base_cif_dir=BASE_CIF_DIR):
    """Look up gen_id's Directory in summary_csv and return its CIF path."""
    summ = pd.read_csv(summary_csv)
    rows = summ[summ["Gen ID"] == gen_id]
    if rows.empty:
        raise ValueError(f"{gen_id!r} not found in {summary_csv!r}")
    directory = rows.iloc[0]["Directory"]
    model_weight, unique_idx = directory.split("-")
    return (
        Path(base_cif_dir) / model_weight / f"UNIQUE_{unique_idx}"
        / "UNIQUE" / f"{gen_id}.cif"
    )


def run_structure_vs_references(gen_id, summary_csv, ref_dict, out_csv):
    gen_path = resolve_gen_id_path(gen_id, summary_csv)
    gen_structure = Structure.from_file(str(gen_path))
    print(f"\n{gen_id} vs {list(ref_dict)}: comparing against {gen_path}")

    matcher = make_matcher()
    rows = []
    for ref_label, ref_path in ref_dict.items():
        ref_structure = Structure.from_file(ref_path)
        result = matcher.get_rms_dist(ref_structure, gen_structure)
        rmsd, max_disp = (None, None) if result is None else result
        rows.append({"pair": f"{ref_label} vs {gen_id}", "rmsd": rmsd, "max_displacement": max_disp})
        print(f"  {ref_label} vs {gen_id}: rmsd={rmsd}, max_displacement={max_disp}")

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(rows)} pairs)")


def run_gen_ids_pairwise(gen_ids, summary_csv, out_csv):
    structures = {
        gen_id: Structure.from_file(str(resolve_gen_id_path(gen_id, summary_csv)))
        for gen_id in gen_ids
    }
    print(f"\n{gen_ids} pairwise comparison")
    compare_all_pairs(structures, out_csv)


def main():
    run_reference_comparison()
    for cluster_id in CLUSTERS_TO_ANALYZE:
        run_cluster_comparison(cluster_id)
    for cluster_id, ref_label in CLUSTER_VS_REF_TO_ANALYZE:
        run_cluster_vs_reference(cluster_id, ref_label)
    for gen_id in EUC_TARGET_GEN_IDS:
        out_csv = EUC_TARGET_OUT_CSV_TEMPLATE.format(gen_id=gen_id)
        run_structure_vs_references(gen_id, EUC_SUMMARY_CSV, EUC_REFERENCES, out_csv)
    run_gen_ids_pairwise(EUC_PAIRWISE_GEN_IDS, EUC_SUMMARY_CSV, EUC_PAIRWISE_OUT_CSV)
    run_gen_ids_pairwise(PETALITE_PAIRWISE_GEN_IDS, SUMMARY_CSV, PETALITE_PAIRWISE_OUT_CSV)


if __name__ == "__main__":
    main()
