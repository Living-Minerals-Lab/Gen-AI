"""
Filter cluster_assignments.csv (family_analysis_out_euc_total) down to stable
candidates (e_hull <= 0.03), then structurally dedup the survivors via
pairwise pymatgen StructureMatcher comparisons.

Usage:
    python dedup_filtered_euc.py
Outputs:
    family_analysis_out_euc/family_analysis_out_euc_total/filtered_euc.csv
    filtered_euc_similarity_scores.csv
    filtered_euc_deduped.csv
    filtered_euc_removed_log.csv
"""

import itertools
import time
from pathlib import Path

import pandas as pd
from pymatgen.core.structure import Structure

from similarity_score import make_matcher

E_HULL_THRESHOLD = 0.03
RMSD_DEDUP_THRESHOLD = 0.1

BASE_DIR = Path(__file__).resolve().parent.parent  # CS_TEST
ASSIGNMENTS_CSV = (
    BASE_DIR / "family_analysis_out_euc" / "family_analysis_out_euc_total" / "cluster_assignments.csv"
)
FILTERED_OUT_CSV = ASSIGNMENTS_CSV.parent / "filtered_euc.csv"

SUMMARY_CSV = "/global/cfs/projectdirs/m4397/For_Atreya/Gen-AI/CS_TEST/summaries/make_unique_summary_euc.csv"

OUT_DIR = Path(__file__).resolve().parent  # CS_TEST/similarity_score
PAIRWISE_OUT_CSV = OUT_DIR / "filtered_euc_similarity_scores.csv"
DEDUPED_OUT_CSV = OUT_DIR / "filtered_euc_deduped.csv"
REMOVED_LOG_CSV = OUT_DIR / "filtered_euc_removed_log.csv"


def build_labels(filtered):
    """One label per row of `filtered`, guaranteed unique even when two rows
    share the same (gen_id, mattergen_id) — e.g. the mattergen-112/gen_509
    coincidental collision — by suffixing repeats with an occurrence count."""
    seen = {}
    labels = []
    for _, row in filtered.iterrows():
        base = f"{row['gen_id']} ({row['mattergen_id']})"
        n = seen.get(base, 0) + 1
        seen[base] = n
        labels.append(base if n == 1 else f"{base} #{n}")
    return labels


def filter_by_e_hull():
    df = pd.read_csv(ASSIGNMENTS_CSV)
    filtered = df[df["e_hull"] <= E_HULL_THRESHOLD].reset_index(drop=True)
    filtered.to_csv(FILTERED_OUT_CSV, index=False)
    print(f"Filtered {len(df)} -> {len(filtered)} rows (e_hull <= {E_HULL_THRESHOLD})")
    print(f"Wrote {FILTERED_OUT_CSV}")
    return filtered


def resolve_cif_paths(filtered):
    """Map each filtered row to its CIF path via the (Mattergen ID, Gen ID,
    Energy Above Hull) composite key in make_unique_summary_euc.csv. That key
    is unique except for one coincidental collision (mattergen-112/gen_509),
    which is resolved positionally: the i-th occurrence of the key in
    cluster_assignments.csv maps to the i-th occurrence in the summary CSV."""
    summary = pd.read_csv(SUMMARY_CSV)
    key_cursor = {}
    paths = []
    for _, row in filtered.iterrows():
        key = (row["mattergen_id"], row["gen_id"], row["e_hull"])
        matches = summary[
            (summary["Mattergen ID"] == key[0])
            & (summary["Gen ID"] == key[1])
            & (summary["Energy Above Hull (eV/atom)"] == key[2])
        ]
        if matches.empty:
            raise ValueError(f"No summary row found for {key}")
        idx = key_cursor.get(key, 0)
        key_cursor[key] = idx + 1
        directory = matches.iloc[idx % len(matches)]["Directory"]
        model_weight, unique_idx = directory.split("-")
        path = BASE_DIR / model_weight / f"UNIQUE_{unique_idx}" / "UNIQUE" / f"{row['gen_id']}.cif"
        paths.append(path)
    return paths


def run_pairwise(filtered, labels, paths):
    structures = {lab: Structure.from_file(str(p)) for lab, p in zip(labels, paths)}
    e_hulls = dict(zip(labels, filtered["e_hull"]))

    matcher = make_matcher()
    rows = []
    pairs = list(itertools.combinations(labels, 2))
    total = len(pairs)
    print(f"Running {total} pairwise comparisons...")
    t0 = time.time()
    for i, (l1, l2) in enumerate(pairs, 1):
        result = matcher.get_rms_dist(structures[l1], structures[l2])
        rmsd, max_disp = (None, None) if result is None else result
        rows.append({
            "gen_id_1": l1, "e_hull_1": e_hulls[l1],
            "gen_id_2": l2, "e_hull_2": e_hulls[l2],
            "rmsd": rmsd, "max_displacement": max_disp,
        })
        if i % 200 == 0 or i == total:
            elapsed = time.time() - t0
            print(f"  {i}/{total} pairs done ({elapsed:.1f}s elapsed)")

    df = pd.DataFrame(rows)
    df.to_csv(PAIRWISE_OUT_CSV, index=False)
    print(f"Wrote {PAIRWISE_OUT_CSV} ({len(df)} pairs)")
    return df


def apply_dedup_rule(filtered, labels, pairwise_df):
    order = {lab: i for i, lab in enumerate(labels)}
    removed = {}  # label -> log dict

    for _, r in pairwise_df.iterrows():
        if pd.isna(r["rmsd"]) or r["rmsd"] > RMSD_DEDUP_THRESHOLD:
            continue
        l1, l2 = r["gen_id_1"], r["gen_id_2"]
        e1, e2 = r["e_hull_1"], r["e_hull_2"]
        tie_break = False
        if e1 > e2:
            loser, winner = l1, l2
        elif e2 > e1:
            loser, winner = l2, l1
        else:
            tie_break = True
            loser, winner = (l2, l1) if order[l2] > order[l1] else (l1, l2)

        if loser not in removed:
            removed[loser] = {
                "removed": loser,
                "removed_e_hull": e1 if loser == l1 else e2,
                "kept": winner,
                "rmsd": r["rmsd"],
                "tie_break_used": tie_break,
            }

    survivors = [
        row for lab, (_, row) in zip(labels, filtered.iterrows()) if lab not in removed
    ]
    survivors_df = pd.DataFrame(survivors)
    survivors_df.to_csv(DEDUPED_OUT_CSV, index=False)
    print(f"Wrote {DEDUPED_OUT_CSV} ({len(survivors_df)} survivors, {len(removed)} removed)")

    log_df = pd.DataFrame(list(removed.values()))
    log_df.to_csv(REMOVED_LOG_CSV, index=False)
    print(f"Wrote {REMOVED_LOG_CSV} ({len(log_df)} rows)")

    return survivors_df, log_df


def main():
    filtered = filter_by_e_hull()
    labels = build_labels(filtered)
    paths = resolve_cif_paths(filtered)
    pairwise_df = run_pairwise(filtered, labels, paths)
    apply_dedup_rule(filtered, labels, pairwise_df)


if __name__ == "__main__":
    main()
