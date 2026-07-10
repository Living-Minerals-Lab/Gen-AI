from pathlib import Path
import re
import math
import csv
import argparse


def get_value(text, key):
    pattern = rf"{re.escape(key)}\s+([^\s]+)"
    match = re.search(pattern, text)
    if not match:
        return None
    return match.group(1).strip("'\"")


def clean_number(x):
    if x is None:
        return None
    return float(re.sub(r"\(.*?\)", "", x))


def cell_volume(a, b, c, alpha, beta, gamma):
    alpha = math.radians(alpha)
    beta = math.radians(beta)
    gamma = math.radians(gamma)

    return a * b * c * math.sqrt(
        1
        - math.cos(alpha) ** 2
        - math.cos(beta) ** 2
        - math.cos(gamma) ** 2
        + 2 * math.cos(alpha) * math.cos(beta) * math.cos(gamma)
    )


def get_space_group(text):
    patterns = [
        r"_symmetry_space_group_name_H-M\s+['\"]?([^'\n\"]+)['\"]?",
        r"_space_group_name_H-M_alt\s+['\"]?([^'\n\"]+)['\"]?",
        r"_space_group_name_Hall\s+['\"]?([^'\n\"]+)['\"]?",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()

    num = get_value(text, "_space_group_IT_number")
    if num:
        return f"IT number {num}"

    return ""


def read_selection(selection_csv):
    selected = []

    with open(selection_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            selected.append(row)

    return selected


DEFAULT_OUT = Path(__file__).resolve().parent / "make_unique_summary_euc.csv"


def build_rows(selected_rows, cif_dir, directory_label):
    """selected_rows: iterable of dicts with mattergen_id/gen_id/energy_above_hull
    keys (in any of the casings produced by parse_structure.py or its --output CSV).
    Returns a list of enriched row dicts (cell params + space group read from each
    structure's CIF file in cif_dir)."""
    cif_dir = Path(cif_dir)
    rows = []

    for selected in selected_rows:
        mattergen_id = selected.get("mattergen_id") or selected.get("Mattergen ID") or selected.get("Mattergen_ID")
        gen_id_raw = selected.get("gen_id") or selected.get("Gen ID") or selected.get("Gen_ID")
        ehull = selected.get("energy_above_hull") or selected.get("Energy Above Hull") or selected.get("Energy_Above_Hull")

        if not gen_id_raw or gen_id_raw == "N/A":
            continue

        cif_name = gen_id_raw if gen_id_raw.endswith(".cif") else f"{gen_id_raw}.cif"
        cif_path = cif_dir / cif_name

        if not cif_path.exists():
            print(f"[warning] Missing CIF: {cif_path}")
            continue

        text = cif_path.read_text(errors="ignore")

        a = clean_number(get_value(text, "_cell_length_a"))
        b = clean_number(get_value(text, "_cell_length_b"))
        c = clean_number(get_value(text, "_cell_length_c"))

        alpha = clean_number(get_value(text, "_cell_angle_alpha"))
        beta = clean_number(get_value(text, "_cell_angle_beta"))
        gamma = clean_number(get_value(text, "_cell_angle_gamma"))

        volume = cell_volume(a, b, c, alpha, beta, gamma)
        space_group = get_space_group(text)

        rows.append({
            "Mattergen ID": mattergen_id,
            "Gen ID": Path(cif_name).stem,
            "Directory": directory_label,
            "Energy Above Hull (eV/atom)": ehull,
            "a": a,
            "b": b,
            "c": c,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "Volume": volume,
            "Space group": space_group,
        })

    return rows


def write_rows(rows, out_path):
    if not rows:
        print("[error] No rows written. Check --selection and --cif-dir.")
        return

    out_path = Path(out_path)
    file_exists = out_path.exists() and out_path.stat().st_size > 0

    with open(out_path, "a" if file_exists else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {out_path}")
    print(f"{'Appended' if file_exists else 'Wrote'} {len(rows)} rows.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cif-dir", required=True, help="Folder containing CIF files")
    parser.add_argument("--selection", required=True, help="CSV from parse_structure.py")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output CSV path (default: %(default)s)")
    parser.add_argument("--directory", required=True, help="Directory label, e.g. 1-0")

    args = parser.parse_args()

    selected_rows = read_selection(args.selection)
    rows = build_rows(selected_rows, args.cif_dir, args.directory)
    write_rows(rows, args.out)


if __name__ == "__main__":
    main()
