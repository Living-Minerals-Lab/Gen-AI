"""
Identify structural families (e.g., β-like) among generated structures
using SOAP fingerprints + UMAP visualization + HDBSCAN clustering.

Usage:
    1. Edit CONFIG section below: paths, reference dict, SOAP params.
    2. Run: python find_families.py
    3. Outputs:
        - umap_embedding.png     (2D map colored by cluster, refs labeled)
        - umap_by_ehull.png      (2D map colored by E_hull)
        - umap_by_directory.png  (2D map colored by model weight, shaped by unique file)
        - cluster_assignments.csv  (gen_id, mattergen_id, E_hull, cluster, similarity_to_each_ref)
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

from ase.io import read
from dscribe.descriptors import SOAP
from sklearn.preprocessing import normalize
import umap
import hdbscan
import hashlib
import pickle


# =====================================================================
# CONFIG  -- edit this section
# =====================================================================

# Path to the unique-structure summary CSV (mattergen_id, gen_id, e_hull, Directory, ...)
SUMMARY_CSV = "/global/cfs/projectdirs/m4397/For_Atreya/Gen-AI/CS_TEST/summaries/make_unique_summary_euc.csv"

# Restrict analysis to a single model weight (the first number in the Directory
# column, e.g. the "1" in "1-2"). Set to None to use every model weight in the CSV.
MODEL_WEIGHT_FILTER = None

PRUNE_STALE_CACHE = True
UMAP_RANDOM_STATE = 42

# Base directory holding the per-model-weight folders, e.g. CIF_DIR/1/UNIQUE_0/UNIQUE/gen_5620.cif
# Resolved relative to this script's own location, so it works regardless of
# the shell's current working directory when the script is invoked.
CIF_DIR = str(Path(__file__).resolve().parent)

# Marker shape per "unique file" index (second number in the Directory column, e.g. "1-0")
UNIQUE_IDX_MARKERS = {0: "o", 1: "s", 2: "^", 3: "*"}
UNIQUE_IDX_LABELS = {0: "dot", 1: "square", 2: "triangle", 3: "star"}

# Reference structures: key = label (your choice), value = path to CIF/POSCAR
# These will be labeled prominently on the UMAP plot.
REFERENCES = {
    "alpha": "/global/cfs/projectdirs/m4397/For_Atreya/Gen-AI/CS_TEST/euc_cifs/EntryWithCollCode30982.cif",
    "beta": "/global/cfs/projectdirs/m4397/For_Atreya/Gen-AI/CS_TEST/euc_cifs/EntryWithCollCode2929.cif",
    "gamma": "/global/cfs/projectdirs/m4397/For_Atreya/Gen-AI/CS_TEST/euc_cifs/EntryWithCollCode66137.cif",
}

# SOAP parameters
SOAP_RCUT = 6.0      # Å, neighborhood cutoff
SOAP_NMAX = 8        # radial basis
SOAP_LMAX = 6        # angular basis
MERGE_SI_AL = True   # treat Si & Al as a single "T" species (decoration-invariant)

# UMAP parameters
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "cosine"

# HDBSCAN parameters
HDBSCAN_MIN_CLUSTER_SIZE = 10
HDBSCAN_MIN_SAMPLES = 4

# Output directory. When MODEL_WEIGHT_FILTER (or --model-weight) is set, results
# go to a per-weight subfolder inside this one, e.g. family_analysis_out_euc/family_analysis_out_euc_2.
BASE_OUT_DIR = str(Path(__file__).resolve().parent / "family_analysis_out_euc")

def parse_summary_csv(path):
    """Parse all_unique_summary.csv -> DataFrame(mattergen_id, gen_id, e_hull, model_weight, unique_idx).

    The "Directory" column holds "<model_weight>-<unique_idx>", e.g. "1-0":
    the first number is the model weight, the second is the unique-file index.
    """
    raw = pd.read_csv(path)
    directory = raw["Directory"].astype(str).str.extract(r"^(\d+)-(\d+)$")
    if directory.isna().any().any():
        bad = raw.loc[directory.isna().any(axis=1), "Directory"].unique()
        raise ValueError(f"Directory values not in '<model_weight>-<unique_idx>' form: {bad}")

    df = pd.DataFrame({
        "mattergen_id": raw["Mattergen ID"],
        "gen_id": raw["Gen ID"],
        "e_hull": raw["Energy Above Hull (eV/atom)"].astype(float),
        "model_weight": directory[0].astype(int),
        "unique_idx": directory[1].astype(int),
    })
    return df


def load_structures(df, cif_dir):
    """Returns list of (mattergen_id, gen_id, atoms, e_hull, abs_path, model_weight, unique_idx)."""
    items = []
    for _, row in df.iterrows():
        path = (
            Path(cif_dir)
            / str(row["model_weight"])
            / f"UNIQUE_{row['unique_idx']}"
            / "UNIQUE"
            / f"{row['gen_id']}.cif"
        )
        if not path.exists():
            print(f"[warn] missing CIF: {path}")
            continue
        atoms = read(str(path))
        items.append((
            row["mattergen_id"], row["gen_id"], atoms, row["e_hull"], str(path),
            int(row["model_weight"]), int(row["unique_idx"]),
        ))
    return items


def load_references(ref_dict):
    """Returns list of (label, path, atoms)."""
    items = []
    for label, path in ref_dict.items():
        if not Path(path).exists():
            print(f"[warn] missing reference: {label} -> {path}")
            continue
        atoms = read(path)
        items.append((label, path, atoms))
    return items


def maybe_merge_si_al(atoms):
    """Replace Si and Al with Ge (placeholder T-atom) so SOAP is decoration-blind."""
    a = atoms.copy()
    syms = a.get_chemical_symbols()
    new = ["Ge" if s in ("Si", "Al") else s for s in syms]
    a.set_chemical_symbols(new)
    return a


def soap_param_signature():
    return f"rcut={SOAP_RCUT}_nmax={SOAP_NMAX}_lmax={SOAP_LMAX}_merge={MERGE_SI_AL}"


def structure_cache_key(path):
    p = Path(path).resolve()
    mtime = p.stat().st_mtime
    raw = f"{p}|{mtime}|{soap_param_signature()}"
    return hashlib.md5(raw.encode()).hexdigest()


def load_cache(cache_file):
    if Path(cache_file).exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    return {}


def save_cache(cache, cache_file):
    Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(cache, f)


def compute_soap_fingerprints_cached(paths_and_atoms, soap_calc, cache, merge=True):
    """Returns (fps L2-normalized, list of cache keys used)."""
    fps = []
    used_keys = []
    n_hits, n_miss = 0, 0
    for path, atoms in paths_and_atoms:
        key = structure_cache_key(path)
        used_keys.append(key)
        if key in cache:
            fps.append(cache[key])
            n_hits += 1
        else:
            a = maybe_merge_si_al(atoms) if merge else atoms
            v = soap_calc.create(a).mean(axis=0)
            cache[key] = v
            fps.append(v)
            n_miss += 1
    print(f"      cache: {n_hits} hits, {n_miss} misses")
    fps = np.vstack(fps)
    fps = normalize(fps, norm="l2", axis=1)
    return fps, used_keys


def plot_umap_by_cluster(emb_gen, gen_clusters, ref_emb_with_labels, out_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    unique_clusters = sorted(set(gen_clusters))
    cmap = plt.get_cmap("tab20")

    for c in unique_clusters:
        mask = gen_clusters == c
        if c == -1:
            color = "lightgrey"
            label = f"noise (n={int(mask.sum())})"
        else:
            color = cmap(c % 20)
            label = f"cluster {c} (n={int(mask.sum())})"
        ax.scatter(emb_gen[mask, 0], emb_gen[mask, 1],
                   s=30, c=[color], label=label, alpha=0.8, edgecolor="none")

    for rlabel, x, y in ref_emb_with_labels:
        ax.scatter(x, y, s=260, marker="*", c="black",
                   edgecolor="yellow", linewidth=1.8, zorder=5)
        ax.annotate(rlabel, (x, y), xytext=(8, 8),
                    textcoords="offset points", fontsize=11,
                    fontweight="bold", color="black",
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor="yellow", alpha=0.75, edgecolor="black"))

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Structural families (SOAP + UMAP + HDBSCAN)")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"      wrote {out_path}")


def plot_umap_by_ehull(emb_gen, e_hulls, ref_emb_with_labels, out_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(emb_gen[:, 0], emb_gen[:, 1],
                    c=e_hulls, cmap="viridis_r",
                    s=35, alpha=0.85, edgecolor="none")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("E above hull (eV/atom)")

    for rlabel, x, y in ref_emb_with_labels:
        ax.scatter(x, y, s=260, marker="*", c="red",
                   edgecolor="black", linewidth=1.5, zorder=5)
        ax.annotate(rlabel, (x, y), xytext=(8, 8),
                    textcoords="offset points", fontsize=11,
                    fontweight="bold", color="black",
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor="white", alpha=0.85, edgecolor="black"))

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Structural families colored by E above hull")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"      wrote {out_path}")


def plot_umap_by_directory(emb_gen, model_weights, unique_idxs, ref_emb_with_labels, out_path):
    """Color points by model weight, shape points by unique-file index, legend on the right."""
    fig, ax = plt.subplots(figsize=(11, 8))

    unique_weights = sorted(set(model_weights))
    cmap = plt.get_cmap("tab10")
    weight_color = {w: cmap(i % 10) for i, w in enumerate(unique_weights)}

    for w in unique_weights:
        for idx, marker in UNIQUE_IDX_MARKERS.items():
            mask = (model_weights == w) & (unique_idxs == idx)
            if not np.any(mask):
                continue
            ax.scatter(emb_gen[mask, 0], emb_gen[mask, 1],
                       s=40, c=[weight_color[w]], marker=marker,
                       alpha=0.8, edgecolor="none")

    for rlabel, x, y in ref_emb_with_labels:
        ax.scatter(x, y, s=260, marker="*", c="black",
                   edgecolor="yellow", linewidth=1.8, zorder=5)
        ax.annotate(rlabel, (x, y), xytext=(8, 8),
                    textcoords="offset points", fontsize=11,
                    fontweight="bold", color="black",
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor="yellow", alpha=0.75, edgecolor="black"))

    color_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=8,
               markerfacecolor=weight_color[w], markeredgecolor="none",
               label=f"Model weight {w}")
        for w in unique_weights
    ]
    shape_handles = [
        Line2D([0], [0], marker=marker, linestyle="none", markersize=8,
               markerfacecolor="grey", markeredgecolor="none",
               label=f"Unique file {idx} ({UNIQUE_IDX_LABELS[idx]})")
        for idx, marker in UNIQUE_IDX_MARKERS.items()
    ]

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Structural families by model weight (color) / unique file (shape)")
    ax.legend(handles=color_handles + shape_handles, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), fontsize=8, framealpha=0.9,
              borderaxespad=0.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"      wrote {out_path}")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(model_weight_filter=MODEL_WEIGHT_FILTER):
    if model_weight_filter is not None:
        out_dir = os.path.join(BASE_OUT_DIR, f"family_analysis_out_euc_{model_weight_filter}")
    else:
        out_dir = BASE_OUT_DIR
    cache_file = os.path.join(out_dir, "soap_cache.pkl")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Parse all_unique_summary.csv
    print("[1/7] Parsing all_unique_summary.csv ...")
    df = parse_summary_csv(SUMMARY_CSV)
    print(f"      {len(df)} generated structures listed.")

    if model_weight_filter is not None:
        df = df[df["model_weight"] == model_weight_filter].reset_index(drop=True)
        print(f"      filtered to model weight {model_weight_filter}: {len(df)} structures remain.")

    # 2. Load generated structures
    print("[2/7] Loading generated CIFs ...")
    gen_items = load_structures(df, CIF_DIR)
    print(f"      {len(gen_items)} generated structures loaded.")
    if len(gen_items) == 0:
        print("[error] No generated structures loaded. Check CIF_DIR.")
        return

    # 3. Load reference structures
    print("[3/7] Loading reference structures ...")
    ref_items = load_references(REFERENCES)
    print(f"      {len(ref_items)} references loaded: {[r[0] for r in ref_items]}")
    if len(ref_items) == 0:
        print("[error] No references loaded. Check REFERENCES paths.")
        return

    # 4. Set up SOAP calculator
    all_atoms = [a for _, _, a, _, _, _, _ in gen_items] + [a for _, _, a in ref_items]
    if MERGE_SI_AL:
        all_atoms_for_species = [maybe_merge_si_al(a) for a in all_atoms]
    else:
        all_atoms_for_species = all_atoms
    species = sorted({s for a in all_atoms_for_species for s in a.get_chemical_symbols()})
    print(f"[4/7] SOAP species set: {species}")

    soap_calc = SOAP(
        species=species,
        r_cut=SOAP_RCUT,
        n_max=SOAP_NMAX,
        l_max=SOAP_LMAX,
        periodic=True,
        sparse=False,
        average="off",  # we average manually after .create()
    )

    # 5. Compute SOAP fingerprints (with cache)
    print("[5/7] Computing SOAP fingerprints (cached) ...")
    cache = load_cache(cache_file)
    print(f"      loaded cache with {len(cache)} entries")

    gen_paths_atoms = [(path, atoms) for _, _, atoms, _, path, _, _ in gen_items]
    ref_paths_atoms = [(path, atoms) for _, path, atoms in ref_items]

    gen_fps, gen_keys = compute_soap_fingerprints_cached(
        gen_paths_atoms, soap_calc, cache, merge=MERGE_SI_AL
    )
    ref_fps, ref_keys = compute_soap_fingerprints_cached(
        ref_paths_atoms, soap_calc, cache, merge=MERGE_SI_AL
    )

    save_cache(cache, cache_file)
    print(f"      saved cache with {len(cache)} entries")

    if PRUNE_STALE_CACHE:
        active = set(gen_keys) | set(ref_keys)
        stale = set(cache.keys()) - active
        if stale:
            for k in stale:
                del cache[k]
            save_cache(cache, cache_file)
            print(f"      pruned {len(stale)} stale cache entries")

    all_fps = np.vstack([gen_fps, ref_fps])
    n_gen = len(gen_items)
    n_ref = len(ref_items)
    print(f"      Feature matrix shape: {all_fps.shape}  (gen={n_gen}, ref={n_ref})")

    # 6. UMAP + HDBSCAN
    print("[6/7] UMAP + HDBSCAN ...")
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
    )
    emb = reducer.fit_transform(all_fps)
    emb_gen = emb[:n_gen]
    emb_ref = emb[n_gen:]

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
    )
    cluster_labels = clusterer.fit_predict(emb)
    gen_clusters = cluster_labels[:n_gen]
    ref_clusters = cluster_labels[n_gen:]

    # Cosine similarity of each gen to each ref (in original SOAP space; already L2-normalized)
    sim_to_refs = gen_fps @ ref_fps.T  # (n_gen, n_ref)

    # 7. Outputs
    print("[7/7] Writing outputs ...")

    # CSV
    n_ref = len(ref_items)
    out_rows = []
    for i, (mg, gid, _, eh, _, _, _) in enumerate(gen_items):
        row = {
            "mattergen_id": mg,
            "gen_id": gid,
            "e_hull": eh,
            "cluster": int(gen_clusters[i]),
            "umap_x": float(emb_gen[i, 0]),
            "umap_y": float(emb_gen[i, 1]),
        }
        # Per-reference: similarity, distance (1 - sim)
        for j, (rlabel, _, _) in enumerate(ref_items):
            sim = float(sim_to_refs[i, j])
            row[f"sim_{rlabel}"] = sim
            row[f"dist_{rlabel}"] = 1.0 - sim
        # Nearest reference
        j_best = int(np.argmax(sim_to_refs[i]))
        row["nearest_ref"] = ref_items[j_best][0]
        row["nearest_ref_sim"] = float(sim_to_refs[i, j_best])
        row["nearest_ref_dist"] = 1.0 - float(sim_to_refs[i, j_best])
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)

    # Add within-column ranks (rank 1 = most similar to that reference)
    # Lower rank number = closer to the reference.
    for j, (rlabel, _, _) in enumerate(ref_items):
        # rank by similarity descending; ties get the average rank, then cast to int
        out_df[f"rank_{rlabel}"] = (
            out_df[f"sim_{rlabel}"]
            .rank(ascending=False, method="min")
            .astype(int)
        )

    # Reorder columns for readability:
    # core info first, then per-ref blocks (sim, dist, rank), then nearest_ref summary.
    base_cols = ["mattergen_id", "gen_id", "e_hull", "cluster", "umap_x", "umap_y"]
    ref_cols = []
    for rlabel, _, _ in ref_items:
        ref_cols += [f"sim_{rlabel}", f"dist_{rlabel}", f"rank_{rlabel}"]
    nearest_cols = ["nearest_ref", "nearest_ref_sim", "nearest_ref_dist"]
    out_df = out_df[base_cols + ref_cols + nearest_cols]

    # Sort by E_hull (ascending) for the main file
    out_df = out_df.sort_values("e_hull")
    csv_path = os.path.join(out_dir, "cluster_assignments.csv")
    out_df.to_csv(csv_path, index=False)
    print(f"      wrote {csv_path}")

    # Also write per-reference "top-N most similar" files for quick inspection
    top_n = 20
    for rlabel, _, _ in ref_items:
        top_df = out_df.sort_values(f"sim_{rlabel}", ascending=False).head(top_n)
        top_path = os.path.join(out_dir, f"top{top_n}_by_{rlabel}.csv")
        top_df.to_csv(top_path, index=False)
        print(f"      wrote {top_path}")
    # Reference cluster info
    print("\n      Reference cluster assignments:")
    for j, (rlabel, _, _) in enumerate(ref_items):
        print(f"        {rlabel:20s} -> cluster {int(ref_clusters[j])}")

    # Per-cluster summary (size, mean E_hull, which refs are inside)
    print("\n      Cluster summary:")
    print(f"        {'cluster':>8s}  {'size':>5s}  {'mean Ehull':>12s}  refs_inside")
    summary_rows = []
    hull_rows = []
    for c in sorted(set(gen_clusters)):
        mask = gen_clusters == c
        size = int(mask.sum())
        cluster_items = [item for i, item in enumerate(gen_items) if mask[i]]
        e_hulls_c = [e for _, _, _, e, _, _, _ in cluster_items]
        mean_eh = float(np.mean(e_hulls_c))
        refs_in = [ref_items[j][0] for j in range(n_ref) if int(ref_clusters[j]) == c]
        tag = "noise" if c == -1 else f"{c}"
        print(f"        {tag:>8s}  {size:>5d}  {mean_eh:>12.4f}  {refs_in}")
        summary_rows.append({
            "cluster": int(c),
            "size": size,
            "mean_e_hull": mean_eh,
            "refs_inside": ",".join(refs_in),
        })

        std_eh = float(np.std(e_hulls_c))
        min_item = min(cluster_items, key=lambda item: item[3])
        min_mg, min_gid, _, min_eh, _, min_mw, min_ui = min_item
        hull_rows.append({
            "cluster": int(c),
            "size": size,
            "mean_e_hull": mean_eh,
            "std_e_hull": std_eh,
            "min_e_hull": float(min_eh),
            "min_e_hull_gen_id": min_gid,
            "min_e_hull_mattergen_id": min_mg,
            "min_e_hull_directory": f"{min_mw}-{min_ui}",
        })
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(out_dir, "cluster_summary.csv"), index=False
    )
    hull_summary_path = os.path.join(out_dir, "hull_summary.csv")
    pd.DataFrame(hull_rows).to_csv(hull_summary_path, index=False)
    print(f"      wrote {hull_summary_path}")

    # Plots
    ref_emb_with_labels = [
        (ref_items[j][0], float(emb_ref[j, 0]), float(emb_ref[j, 1]))
        for j in range(n_ref)
    ]
    e_hulls = np.array([eh for _, _, _, eh, _, _, _ in gen_items])
    model_weights = np.array([mw for _, _, _, _, _, mw, _ in gen_items])
    unique_idxs = np.array([ui for _, _, _, _, _, _, ui in gen_items])

    plot_umap_by_cluster(
        emb_gen, gen_clusters, ref_emb_with_labels,
        os.path.join(out_dir, "umap_by_cluster.png"),
    )
    plot_umap_by_ehull(
        emb_gen, e_hulls, ref_emb_with_labels,
        os.path.join(out_dir, "umap_by_ehull.png"),
    )
    plot_umap_by_directory(
        emb_gen, model_weights, unique_idxs, ref_emb_with_labels,
        os.path.join(out_dir, "umap_by_directory.png"),
    )

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-weight", type=int, default=MODEL_WEIGHT_FILTER,
        help="Restrict analysis to this model weight, e.g. 2 (the first number "
             "in the Directory column, e.g. the '2' in '2-1'). Default: use every "
             "model weight in the CSV.",
    )
    args = parser.parse_args()
    main(model_weight_filter=args.model_weight)
