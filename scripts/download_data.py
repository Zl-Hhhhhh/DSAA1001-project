#!/usr/bin/env python3
"""Download a dated, auditable snapshot of permitted Chicago inspection history.

Run from the repository root:
    python scripts/download_data.py
    python scripts/download_data.py --end-date 2026-10-30 --overwrite

Preparation and notebooks do not call this network-dependent script.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config/data.json").read_text(encoding="utf-8"))
DATASET_ID = CONFIG["dataset_id"]
BASE = "https://data.cityofchicago.org"
RESOURCE_URL = f"{BASE}/resource/{DATASET_ID}.json"
METADATA_URL = f"{BASE}/api/views/{DATASET_ID}.json"
PORTAL_PAGE = f"{BASE}/Health-Human-Services/Food-Inspections/{DATASET_ID}"
COLUMNS = [
    ("inspection_id", "Inspection ID"),
    ("dba_name", "DBA Name"),
    ("aka_name", "AKA Name"),
    ("license_", "License #"),
    ("facility_type", "Facility Type"),
    ("risk", "Risk"),
    ("address", "Address"),
    ("city", "City"),
    ("state", "State"),
    ("zip", "Zip"),
    ("inspection_date", "Inspection Date"),
    ("inspection_type", "Inspection Type"),
    ("results", "Results"),
    ("violations", "Violations"),
    ("latitude", "Latitude"),
    ("longitude", "Longitude"),
    ("location", "Location"),
]


def get_json(url: str, params: dict | None = None):
    if params:
        url += "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": "DSAA1001-data-preparation/1.0"}
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt == 4:
                raise
            delay = min(30, 2 ** (attempt + 1))
            print(f"HTTP {error.code}; retrying in {delay}s", file=sys.stderr, flush=True)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 4:
                raise
            delay = min(30, 2 ** (attempt + 1))
            print(f"Connection interrupted; retrying in {delay}s", file=sys.stderr, flush=True)
        time.sleep(delay)
    raise RuntimeError("Download retries exhausted")


def validate_window(start: str, end: str, downloaded_on: date) -> tuple[date, date]:
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    minimum = date.fromisoformat(CONFIG["history_start"])
    cutoff = date.fromisoformat(CONFIG["history_cutoff"])
    if downloaded_on > cutoff:
        raise ValueError("Publication cutoff has passed. Use the saved pre-cutoff snapshot offline.")
    if not minimum <= first <= last <= min(cutoff, downloaded_on):
        raise ValueError(
            f"Dates must satisfy {minimum} <= start <= end <= {min(cutoff, downloaded_on)}. "
            "Hidden-test dates and future dates cannot be downloaded."
        )
    return first, last


def to_portal_row(record: dict) -> dict:
    output = {}
    for api_name, display_name in COLUMNS:
        value = record.get(api_name, "")
        if api_name == "inspection_date" and value:
            value = datetime.strptime(value[:10], "%Y-%m-%d").strftime("%m/%d/%Y")
        elif api_name == "location" and isinstance(value, dict):
            latitude, longitude = value.get("latitude"), value.get("longitude")
            value = f"({latitude}, {longitude})" if latitude is not None and longitude is not None else ""
        output[display_name] = "" if value is None else value
    return output


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    today = datetime.now(timezone.utc).date()
    default_end = min(today, date.fromisoformat(CONFIG["history_cutoff"])).isoformat()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default=CONFIG["history_start"])
    parser.add_argument("--end-date", default=default_end)
    parser.add_argument("--page-size", type=int, default=10000)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace the saved snapshot")
    args = parser.parse_args()
    first, last = validate_window(args.start_date, args.end_date, today)
    if not 1 <= args.page_size <= 50000:
        parser.error("--page-size must be between 1 and 50000")

    csv_path = ROOT / CONFIG["raw_csv"]
    metadata_path = ROOT / CONFIG["download_metadata"]
    source_path = ROOT / CONFIG["source_metadata"]
    if any(path.exists() for path in (csv_path, metadata_path, source_path)) and not args.overwrite:
        parser.error("Snapshot already exists. Use --overwrite only for an intentional pre-cutoff refresh.")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = csv_path.with_name(csv_path.name + ".partial")
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"Reading source metadata; permitted window {first} .. {last}", flush=True)
    source_metadata = get_json(METADATA_URL)
    next_day = last + timedelta(days=1)
    where = (
        f"inspection_date >= '{first}T00:00:00' "
        f"AND inspection_date < '{next_day}T00:00:00'"
    )
    aggregate = get_json(
        RESOURCE_URL,
        {"$select": "count(*) as n,max(inspection_id) as max_id", "$where": where},
    )[0]
    expected_count = int(aggregate["n"])
    if expected_count == 0:
        raise ValueError("The requested window contains no inspections.")
    maximum_id = int(aggregate["max_id"])
    count, previous_id = 0, -1
    minimum_date, maximum_date = None, None

    try:
        # A fixed gzip header makes identical input produce identical compressed bytes.
        with temporary_csv.open("wb") as binary:
            with gzip.GzipFile(filename="", fileobj=binary, mode="wb", mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                    writer = csv.DictWriter(text, fieldnames=[name for _, name in COLUMNS])
                    writer.writeheader()
                    while True:
                        page = get_json(
                            RESOURCE_URL,
                            {
                                "$select": ",".join(name for name, _ in COLUMNS),
                                "$where": f"{where} AND inspection_id > {previous_id} AND inspection_id <= {maximum_id}",
                                "$order": "inspection_id ASC",
                                "$limit": str(args.page_size),
                            },
                        )
                        if not page:
                            break
                        for record in page:
                            inspection_id = int(record["inspection_id"])
                            inspection_date = date.fromisoformat(record["inspection_date"][:10])
                            if inspection_id <= previous_id or not first <= inspection_date <= last:
                                raise ValueError("API page has an invalid order, duplicate ID or out-of-window row.")
                            previous_id = inspection_id
                            minimum_date = inspection_date if minimum_date is None else min(minimum_date, inspection_date)
                            maximum_date = inspection_date if maximum_date is None else max(maximum_date, inspection_date)
                            writer.writerow(to_portal_row(record))
                            count += 1
                        print(f"Fetched {count:,} / {expected_count:,} rows", flush=True)
                        if len(page) < args.page_size:
                            break
        final_source = get_json(METADATA_URL)
        if count != expected_count:
            raise ValueError(f"Incomplete snapshot: expected {expected_count}, downloaded {count}.")
        if source_metadata.get("rowsUpdatedAt") != final_source.get("rowsUpdatedAt"):
            raise ValueError("The source changed during download; retry to obtain a consistent snapshot.")
        completed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if datetime.now(timezone.utc).date() > date.fromisoformat(CONFIG["history_cutoff"]):
            raise ValueError("Download completed after the publication cutoff; snapshot rejected.")
        metadata = {
            "dataset_name": source_metadata.get("name", "Food Inspections"),
            "dataset_id": DATASET_ID,
            "portal_page": PORTAL_PAGE,
            "api_endpoint": RESOURCE_URL,
            "download_started_at_utc": started_at,
            "downloaded_at_utc": completed_at,
            "portal_rows_updated_at_utc": datetime.fromtimestamp(
                source_metadata["rowsUpdatedAt"], tz=timezone.utc
            ).isoformat(),
            "requested_window": {"start_date": first.isoformat(), "end_date": last.isoformat()},
            "project_history_cutoff": CONFIG["history_cutoff"],
            "actual_date_range": {"start_date": minimum_date.isoformat(), "end_date": maximum_date.isoformat()},
            "row_count": count,
            "api_expected_row_count": expected_count,
            "maximum_inspection_id": maximum_id,
            "columns": [name for _, name in COLUMNS],
            "csv_file": csv_path.name,
            "csv_sha256": sha256_file(temporary_csv),
            "csv_size_bytes": temporary_csv.stat().st_size,
            "query_filter": where,
            "pagination": "inspection_id keyset, ascending, bounded by initial maximum ID",
            "compression": "gzip; portal CSV field names and MM/DD/YYYY inspection dates",
            "snapshot_note": "Observed public data at download time, not proof of complete reporting through the requested end date.",
        }
        source_temporary = source_path.with_suffix(".json.partial")
        metadata_temporary = metadata_path.with_suffix(".json.partial")
        source_temporary.write_text(json.dumps(source_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        metadata_temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary_csv.replace(csv_path)
        source_temporary.replace(source_path)
        metadata_temporary.replace(metadata_path)
        print(f"Saved {count:,} rows to {csv_path.relative_to(ROOT)} ({csv_path.stat().st_size / 1024**2:.1f} MiB)", flush=True)
        print(f"Actual inspection dates: {minimum_date} .. {maximum_date}", flush=True)
    finally:
        temporary_csv.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
