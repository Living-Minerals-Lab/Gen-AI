from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import openai


def _extract_json_object(text: str) -> str:
    """
    Best-effort extraction for when the model wraps JSON in prose or code fences.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response.")
    return text[start : end + 1]


def _parse_refs_inside(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_cluster_summary(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, Any]] = []
        for row in reader:
            rows.append(
                {
                    "cluster": int(row["cluster"]),
                    "size": int(row["size"]),
                    "mean_e_hull": float(row["mean_e_hull"]),
                    "refs_inside": _parse_refs_inside(row.get("refs_inside")),
                }
            )
    if not rows:
        raise ValueError(f"No rows found in cluster summary file: {path!r}")
    return rows


def _parse_cluster_assignments(path: str) -> tuple[list[str], list[dict[str, Any]]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        references = [name[len("rank_") :] for name in fieldnames if name.startswith("rank_")]
        if not references:
            raise ValueError(f"No rank_<reference> columns found in {path!r}")

        rows: list[dict[str, Any]] = []
        for row in reader:
            rows.append(
                {
                    "cluster": int(row["cluster"]),
                    "mattergen_id": str(row["mattergen_id"]).strip(),
                    "gen_id": str(row["gen_id"]).strip(),
                    "ranks": {ref: int(row[f"rank_{ref}"]) for ref in references},
                }
            )

    if not rows:
        raise ValueError(f"No rows found in cluster assignments file: {path!r}")
    return references, rows


def _group_assignments_by_cluster(
    summary_rows: Sequence[dict[str, Any]],
    assignment_rows: Sequence[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    summary_clusters = {row["cluster"] for row in summary_rows}
    grouped: dict[int, list[dict[str, Any]]] = {cluster: [] for cluster in summary_clusters}

    for row in assignment_rows:
        cluster = row["cluster"]
        if cluster not in grouped:
            raise ValueError(
                f"Cluster {cluster} exists in cluster_assignments.csv but not in cluster_summary.csv."
            )
        grouped[cluster].append(row)

    for summary in summary_rows:
        cluster = summary["cluster"]
        actual_size = len(grouped[cluster])
        if actual_size != summary["size"]:
            raise ValueError(
                f"Cluster {cluster} has size {summary['size']} in cluster_summary.csv "
                f"but {actual_size} rows in cluster_assignments.csv."
            )

    return grouped


def _top_entries(
    members: Sequence[dict[str, Any]],
    reference: str,
    limit: int,
) -> list[dict[str, Any]]:
    ranked = sorted(
        members,
        key=lambda row: (row["ranks"][reference], row["mattergen_id"], row["gen_id"]),
    )
    return [
        {
            "mattergen_id": row["mattergen_id"],
            "gen_id": row["gen_id"],
            "rank": row["ranks"][reference],
        }
        for row in ranked[:limit]
    ]


def _build_agent_input(
    summary_rows: Sequence[dict[str, Any]],
    assignment_rows: Sequence[dict[str, Any]],
    references: Sequence[str],
) -> dict[str, Any]:
    grouped = _group_assignments_by_cluster(summary_rows, assignment_rows)
    clusters: list[dict[str, Any]] = []

    for summary in summary_rows:
        cluster = summary["cluster"]
        members = grouped[cluster]
        clusters.append(
            {
                "cluster": cluster,
                "size": summary["size"],
                "mean_e_hull": summary["mean_e_hull"],
                "refs_inside": summary["refs_inside"],
                "structures": [
                    {
                        "mattergen_id": row["mattergen_id"],
                        "gen_id": row["gen_id"],
                        "ranks": {ref: row["ranks"][ref] for ref in references},
                    }
                    for row in members
                ],
            }
        )

    return {"references": list(references), "clusters": clusters}


def _build_expected_table(
    summary_rows: Sequence[dict[str, Any]],
    assignment_rows: Sequence[dict[str, Any]],
    references: Sequence[str],
) -> dict[str, Any]:
    grouped = _group_assignments_by_cluster(summary_rows, assignment_rows)
    reference_set = set(references)
    rows: list[dict[str, Any]] = []

    for summary in summary_rows:
        cluster = summary["cluster"]
        refs_inside = list(summary["refs_inside"])
        unknown_refs = [ref for ref in refs_inside if ref not in reference_set]
        if unknown_refs:
            raise ValueError(
                f"Cluster {cluster} has refs_inside entries not present in rank columns: {unknown_refs}"
            )
        members = grouped[cluster]
        columns: dict[str, list[dict[str, Any]]] = {}

        if refs_inside:
            refs_inside_set = set(refs_inside)
            for ref in references:
                columns[ref] = _top_entries(members, ref, 5) if ref in refs_inside_set else []
        else:
            for ref in references:
                columns[ref] = _top_entries(members, ref, 3)

        rows.append(
            {
                "cluster": cluster,
                "refs_inside": refs_inside,
                "columns": columns,
            }
        )

    return {"references": list(references), "rows": rows}


def _write_json(path: str, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")


def _format_cell(entries: Sequence[dict[str, Any]]) -> str:
    return "; ".join(
        f"{entry['mattergen_id']} | {entry['gen_id']} (rank {entry['rank']})"
        for entry in entries
    )


def _write_table_csv(path: str, payload: dict[str, Any]) -> None:
    references = payload["references"]
    fieldnames = ["cluster", "refs_inside"] + references
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload["rows"]:
            out_row = {
                "cluster": row["cluster"],
                "refs_inside": ",".join(row["refs_inside"]),
            }
            for ref in references:
                out_row[ref] = _format_cell(row["columns"][ref])
            writer.writerow(out_row)


def _normalize_entry_key(item: Any) -> tuple[str, str] | None:
    if not isinstance(item, dict):
        return None
    mattergen_id = str(item.get("mattergen_id", "")).strip()
    gen_id = str(item.get("gen_id", "")).strip()
    if not mattergen_id or not gen_id:
        return None
    return mattergen_id, gen_id


@dataclass(frozen=True)
class ClusterReferenceAgent:
    """
    Uses an LLM to produce a cluster-by-reference ranking table grounded in CSV data.
    """

    model: str
    base_url: str = "https://api.cborg.lbl.gov"
    api_key_env: str = "CBORG_API_KEY"

    def _client(self) -> openai.OpenAI:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing {self.api_key_env} environment variable.")
        return openai.OpenAI(api_key=api_key, base_url=self.base_url)

    def summarize(
        self,
        cluster_summary_rows: Sequence[dict[str, Any]],
        assignment_rows: Sequence[dict[str, Any]],
        references: Sequence[str],
    ) -> dict[str, Any]:
        agent_input = _build_agent_input(cluster_summary_rows, assignment_rows, references)
        expected = _build_expected_table(cluster_summary_rows, assignment_rows, references)

        prompt = (
            "You are an experienced Materials Scientist with analytical skills for materials data.\n"
            "You are analyzing clustered crystal structures.\n"
            "You are given data parsed from two CSV files:\n"
            "- cluster_summary.csv: one row per cluster with refs_inside.\n"
            "- cluster_assignments.csv: one row per generated structure with per-reference ranks.\n"
            "\n"
            "Task:\n"
            "1. If a cluster has one or more refs_inside values, populate ONLY those reference columns.\n"
            "   For each such reference, return up to the top 5 structures from that cluster with the smallest rank.\n"
            "2. If a cluster has no refs_inside values, populate EVERY reference column.\n"
            "   For each reference, return up to the top 3 structures from that cluster with the smallest rank.\n"
            "3. Every selected structure must belong to that cluster.\n"
            "4. Keep each list sorted by increasing rank.\n"
            "5. Use the reference order exactly as provided in the references array.\n"
            "\n"
            "Return ONLY valid JSON with this exact schema:\n"
            "{\n"
            '  "references": ["alpha", "gamma", "..."],\n'
            '  "rows": [\n'
            "    {\n"
            '      "cluster": 0,\n'
            '      "refs_inside": ["gamma"],\n'
            '      "columns": {\n'
            '        "alpha": [],\n'
            '        "gamma": [\n'
            '          {"mattergen_id": "mattergen-1", "gen_id": "gen_1.cif", "rank": 1}\n'
            "        ]\n"
            "      }\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "\n"
            "Input data:\n"
            + json.dumps(agent_input, indent=2, sort_keys=False)
        )

        response = self._client().chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[-1].message.content or ""
        payload = json.loads(_extract_json_object(text))
        return self._validate_and_normalize(payload, expected)

    def _validate_and_normalize(
        self,
        payload: dict[str, Any],
        expected: dict[str, Any],
    ) -> dict[str, Any]:
        references = expected["references"]
        raw_rows = payload.get("rows", []) if isinstance(payload, dict) else []
        row_lookup: dict[int, dict[str, Any]] = {}

        if isinstance(raw_rows, list):
            for row in raw_rows:
                if not isinstance(row, dict):
                    continue
                try:
                    cluster = int(row.get("cluster"))
                except (TypeError, ValueError):
                    continue
                row_lookup[cluster] = row

        normalized_rows: list[dict[str, Any]] = []
        for expected_row in expected["rows"]:
            cluster = expected_row["cluster"]
            raw_row = row_lookup.get(cluster, {})
            raw_columns = raw_row.get("columns", {}) if isinstance(raw_row, dict) else {}

            normalized_columns: dict[str, list[dict[str, Any]]] = {}
            for ref in references:
                authoritative = expected_row["columns"][ref]
                allowed_keys = {
                    (entry["mattergen_id"], entry["gen_id"]): entry for entry in authoritative
                }

                selected_keys: set[tuple[str, str]] = set()
                selected_entries: list[dict[str, Any]] = []
                if isinstance(raw_columns, dict):
                    raw_entries = raw_columns.get(ref, [])
                    if isinstance(raw_entries, list):
                        for item in raw_entries:
                            key = _normalize_entry_key(item)
                            if key in allowed_keys and key not in selected_keys:
                                selected_keys.add(key)
                                selected_entries.append(allowed_keys[key])

                for entry in authoritative:
                    key = (entry["mattergen_id"], entry["gen_id"])
                    if key not in selected_keys:
                        selected_entries.append(entry)
                        selected_keys.add(key)

                normalized_columns[ref] = sorted(
                    selected_entries,
                    key=lambda entry: (entry["rank"], entry["mattergen_id"], entry["gen_id"]),
                )

            normalized_rows.append(
                {
                    "cluster": cluster,
                    "refs_inside": expected_row["refs_inside"],
                    "columns": normalized_columns,
                }
            )

        return {"references": list(references), "rows": normalized_rows}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="LLM-based cluster/reference ranking agent for family-analysis CSV outputs."
    )
    parser.add_argument(
        "--summary",
        dest="summary_path",
        default="family_analysis_out/cluster_summary.csv",
        help="Path to cluster_summary.csv (default: family_analysis_out/cluster_summary.csv).",
    )
    parser.add_argument(
        "--assignments",
        dest="assignments_path",
        default="family_analysis_out/cluster_assignments.csv",
        help="Path to cluster_assignments.csv (default: family_analysis_out/cluster_assignments.csv).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("CBORG_MODEL", "claude-opus-4-7"),
        help="Model name (default: $CBORG_MODEL or claude-opus-4-7).",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        default="family_analysis_out/cluster_reference_table.csv",
        help="Write the table CSV to this path (default: family_analysis_out/cluster_reference_table.csv).",
    )
    parser.add_argument(
        "--json-out",
        dest="json_out_path",
        default="family_analysis_out/cluster_reference_table.json",
        help="Write the normalized JSON payload to this path (default: family_analysis_out/cluster_reference_table.json).",
    )
    args = parser.parse_args(argv)

    cluster_summary_rows = _parse_cluster_summary(args.summary_path)
    references, assignment_rows = _parse_cluster_assignments(args.assignments_path)

    agent = ClusterReferenceAgent(model=args.model)
    result = agent.summarize(cluster_summary_rows, assignment_rows, references)

    _write_table_csv(args.out_path, result)
    _write_json(args.json_out_path, result)

    print(json.dumps(result, indent=2, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
