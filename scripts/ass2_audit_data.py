"""Audit the frozen Assessment 2 snapshot and prepared history offline.

Run from the repository root:
    python scripts/ass2_audit_data.py
No downloading, model fitting or source-data rewriting is performed.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime
import csv
import gzip
import hashlib
import json
from pathlib import Path

import prepare_data as prep

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def license_status(value):
    text = prep.tidy(value)
    if not text:
        return "blank"
    if prep.identifier(text):
        return "valid"
    if text.removesuffix(".0").isdigit() and int(text.removesuffix(".0")) == 0:
        return "zero"
    return "other_invalid"


def audit(root=ROOT):
    """Verify the saved source against its pin, then inspect every retained row."""
    config = read_json(root / "config/data.json")
    pin = read_json(root / "config/assessment2_data.json")
    metadata = read_json(root / config["download_metadata"])
    raw_path = root / config["raw_csv"]
    digest = prep.sha256_file(raw_path)
    if digest != pin["raw_sha256"] or digest != metadata["csv_sha256"]:
        raise ValueError("Raw snapshot SHA-256 does not match the frozen version.")
    if metadata["dataset_id"] != config["dataset_id"] or metadata["dataset_id"] != pin["dataset_id"]:
        raise ValueError("Dataset identity mismatch.")
    if metadata["downloaded_at_utc"] != pin["downloaded_at_utc"]:
        raise ValueError("Download timestamp differs from the frozen version.")
    downloaded = datetime.fromisoformat(metadata["downloaded_at_utc"])
    if downloaded.tzinfo is None or downloaded.date().isoformat() > config["history_cutoff"]:
        raise ValueError("Invalid or post-cutoff download timestamp.")
    window = metadata["requested_window"]
    if not config["history_start"] <= window["start_date"] <= window["end_date"] <= min(
        config["history_cutoff"], downloaded.date().isoformat()
    ):
        raise ValueError("Invalid saved request window.")

    # The original checker verifies all prepared inputs/outputs and label ordering.
    prep.check(root)
    missing = Counter()
    license_counts = Counter()
    key_types = Counter()
    outcomes = Counter()
    normalization = Counter()
    seen = {}
    labelled_expected = {}
    groups = Counter()
    license_addresses = defaultdict(set)
    license_names = defaultdict(set)
    license_visits = Counter()
    fallback_keys = Counter()
    same_day = defaultdict(lambda: {"records": 0, "labels": set()})
    license_issues = []
    first = last = None
    raw_count = duplicate_count = retained = unjudged = 0

    with ExitStack() as stack:
        source = stack.enter_context(gzip.open(raw_path, "rt", encoding="utf-8-sig", newline=""))
        history_file = stack.enter_context(gzip.open(
            root / config["processed_history"], "rt", encoding="utf-8", newline=""
        ))
        reader, history_reader = csv.DictReader(source), csv.DictReader(history_file)
        if reader.fieldnames != list(prep.RAW_TO_CLEAN):
            raise ValueError("Unexpected raw schema.")
        if history_reader.fieldnames != prep.HISTORY_COLUMNS:
            raise ValueError("Unexpected prepared-history schema.")
        for raw in reader:
            raw_count += 1
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError("Malformed CSV record.")
            missing.update(field for field, value in raw.items() if not prep.tidy(value))
            inspection_id = prep.identifier(raw["Inspection ID"])
            if not inspection_id:
                raise ValueError("Missing or invalid Inspection ID.")
            signature = hashlib.sha256(json.dumps(raw, sort_keys=True).encode("utf-8")).digest()
            if inspection_id in seen:
                if seen[inspection_id] != signature:
                    raise ValueError(f"Conflicting duplicate Inspection ID: {inspection_id}")
                duplicate_count += 1
                continue
            seen[inspection_id] = signature
            row = prep.normalize_row(raw, normalization)
            expected = {field: str(row[field]) for field in prep.HISTORY_COLUMNS}
            actual = next(history_reader, None)
            if actual != expected:
                raise ValueError(f"Prepared history differs from raw record {inspection_id}.")
            date = row["inspection_date"]
            if not window["start_date"] <= date <= window["end_date"]:
                raise ValueError("Inspection lies outside the saved request window.")
            first = date if first is None else min(first, date)
            last = date if last is None else max(last, date)
            retained += 1
            outcomes[row["results"]] += 1
            if row["label"] == "":
                unjudged += 1
            else:
                labelled_expected[inspection_id] = (date, str(row["label"]))
            key = row["establishment_key"]
            groups[key] += 1
            key_types[key.split(":", 1)[0]] += 1
            status = license_status(raw["License #"])
            license_counts[status] += 1
            if status != "valid":
                license_issues.append({
                    "inspection_id": inspection_id, "inspection_date": date,
                    "raw_license": raw["License #"], "reason": status,
                    "dba_name": row["dba_name"], "address": row["address"],
                    "establishment_key": key,
                })
            if row["license_number"]:
                number = row["license_number"]
                license_visits[number] += 1
                # Includes a blank address as a recorded value, matching preparation.
                license_addresses[number].add(row["address"].casefold())
                license_names[number].add(row["dba_name"].casefold())
            elif key.startswith("fallback:"):
                fallback_keys[key] += 1
            day = same_day[(key, date)]
            day["records"] += 1
            if row["label"] != "":
                day["labels"].add(row["label"])
        if next(history_reader, None) is not None:
            raise ValueError("Prepared history has extra rows.")

    if raw_count != metadata["row_count"] or raw_count != pin["row_count"]:
        raise ValueError("Raw row count does not match the frozen snapshot.")
    actual_range = {"start": first, "end": last}
    if actual_range != pin["actual_date_range"]:
        raise ValueError("Actual inspection dates differ from the frozen version.")
    if actual_range != {
        "start": metadata["actual_date_range"]["start_date"],
        "end": metadata["actual_date_range"]["end_date"],
    }:
        raise ValueError("Actual inspection dates disagree with download metadata.")

    with gzip.open(root / config["labelled_data"], "rt", encoding="utf-8", newline="") as stream:
        labelled_reader = csv.DictReader(stream)
        for row in labelled_reader:
            expected = labelled_expected.pop(row["inspection_id"], None)
            if expected != (row["inspection_date"], row["label"]):
                raise ValueError("Labelled table contains an extra, repeated or incorrect label.")
    if labelled_expected:
        raise ValueError("Labelled table is missing valid judgements.")

    multi_address = [
        {"license_number": number, "records": license_visits[number],
         "address_count": len(addresses), "name_count": len(license_names[number]),
         "addresses_json": json.dumps(sorted(addresses), ensure_ascii=False),
         "names_json": json.dumps(sorted(license_names[number]), ensure_ascii=False)}
        for number, addresses in sorted(license_addresses.items()) if len(addresses) > 1
    ]
    same_day_review = [
        {"establishment_key": key, "inspection_date": date, "records": values["records"],
         "conflicting_binary_labels": int(len(values["labels"]) > 1)}
        for (key, date), values in sorted(same_day.items()) if values["records"] > 1
    ]
    summary = {
        "snapshot": {
            "dataset_id": metadata["dataset_id"], "raw_file": config["raw_csv"],
            "downloaded_at_utc": metadata["downloaded_at_utc"],
            "requested_window": window, "actual_date_range": actual_range,
            "course_history_cutoff": config["history_cutoff"], "raw_sha256": digest,
        },
        "counts": {
            "raw_rows": raw_count, "retained_history_rows": retained,
            "labelled_rows": retained - unjudged, "history_only_rows": unjudged,
            "exact_duplicate_rows_removed": duplicate_count,
            "conflicting_duplicate_ids": 0,
        },
        "raw_missing": {field: {"count": missing[field], "percent": 100 * missing[field] / raw_count}
                        for field in prep.RAW_TO_CLEAN},
        "license_status_retained_rows": {s: license_counts[s]
                                         for s in ["valid", "blank", "zero", "other_invalid"]},
        "matching": {
            "establishment_groups": len(groups),
            "repeated_groups": sum(n > 1 for n in groups.values()),
            "rows_by_key_type": {s: key_types[s] for s in ["license", "fallback", "inspection"]},
            "fallback_groups": len(fallback_keys),
            "licenses_with_multiple_recorded_addresses": len(multi_address),
            "same_day_multiple_visit_groups": len(same_day_review),
            "same_day_conflicting_label_groups": sum(
                row["conflicting_binary_labels"] for row in same_day_review
            ),
        },
        "retention": {
            "history_rows_matched_raw_normalization": retained,
            "history_only_rows_verified_retained": unjudged,
            "all_labelled_ids_dates_and_labels_matched": True,
        },
        "outcome_counts_for_retention_audit": dict(sorted(outcomes.items())),
        "prepared_files": {
            name: {"path": config[name], "sha256": prep.sha256_file(root / config[name])}
            for name in ["processed_history", "labelled_data"]
        },
        "limitations": [
            "Full-snapshot administrative checks only; no time-specific failure rates or model scores.",
            "Valid licence means a syntactically positive integer, not verification against a licence registry.",
            "A shared licence or fallback key is provisional identity; multiple addresses need review.",
            "Same-day visits are retained; their ordering is unknown.",
        ],
    }
    summary["review_items"] = {
        "invalid_licenses": license_issues,
        "licenses_multiple_addresses": multi_address,
        "same_day_visits": same_day_review,
    }
    print(f"Snapshot: {first} .. {last}; SHA-256 matches frozen version.")
    print(f"Exact duplicate rows removed: {duplicate_count}; conflicting IDs: 0.")
    print("Raw missing fields:")
    for field, stats in summary["raw_missing"].items():
        if stats["count"]:
            print(f"  {field}: {stats['count']:,} ({stats['percent']:.4f}%)")
    print(f"A1 audit passed: {retained:,} history rows; {unjudged:,} history-only visits retained.")
    print(f"Licence issues: {len(license_issues):,}; licences across addresses: {len(multi_address):,}.")
    print(f"Establishment groups: {len(groups):,}; fallback groups: {len(fallback_keys):,}.")
    return summary


if __name__ == "__main__":
    audit()
