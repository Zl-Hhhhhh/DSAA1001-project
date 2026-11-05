"""A1 audit checks with small, offline raw/prepared fixtures."""
import csv
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ass2_audit_data as audit
import prepare_data as prep


class DataAuditTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / "tmp"
        parent.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="a1-audit-", dir=parent)
        self.root = Path(self.temp.name).resolve()
        self.root.relative_to(parent.resolve())
        (self.root / "config").mkdir()
        (self.root / "data/raw").mkdir(parents=True)
        (self.root / "config/data.json").write_text(json.dumps(prep.CONFIG), encoding="utf-8")
        self.raw_path = self.root / prep.CONFIG["raw_csv"]
        specs = [
            ("1", "Fail", "123", "A", "1 MAIN ST", "01/04/2016"),
            ("2", "Pass", "123", "A", "2 MAIN ST", "01/04/2016"),
            ("3", "No Entry", "0", "B", "3 MAIN ST", "01/05/2016"),
            ("4", "Out of Business", "", "C", "4 MAIN ST", "01/06/2016"),
            ("5", "Pass w/ Conditions", "broken", "D", "5 MAIN ST", "01/07/2016"),
        ]
        rows = []
        for inspection_id, result, licence, name, address, date in specs:
            row = dict.fromkeys(prep.RAW_TO_CLEAN, "")
            row.update({"Inspection ID": inspection_id, "Results": result, "License #": licence,
                        "DBA Name": name, "Address": address, "Inspection Date": date,
                        "City": "CHICAGO", "State": "IL", "Zip": "01234"})
            rows.append(row)
        self.write_gzip(self.raw_path, list(prep.RAW_TO_CLEAN), rows)
        self.metadata = {
            "dataset_id": "4ijn-s7e5", "downloaded_at_utc": "2026-10-12T07:59:09+00:00",
            "portal_page": "https://data.cityofchicago.org",
            "requested_window": {"start_date": "2016-01-01", "end_date": "2026-10-12"},
            "actual_date_range": {"start_date": "2016-01-04", "end_date": "2016-01-07"},
            "row_count": len(rows), "csv_sha256": prep.sha256_file(self.raw_path),
        }
        metadata_path = self.root / prep.CONFIG["download_metadata"]
        metadata_path.write_text(json.dumps(self.metadata), encoding="utf-8")
        self.pin = {
            "dataset_id": "4ijn-s7e5", "raw_sha256": self.metadata["csv_sha256"],
            "downloaded_at_utc": self.metadata["downloaded_at_utc"], "row_count": len(rows),
            "actual_date_range": {"start": "2016-01-04", "end": "2016-01-07"},
        }
        self.save_pin()
        prep.prepare(self.raw_path, metadata_path, self.root)

    def tearDown(self):
        self.temp.cleanup()

    def save_pin(self):
        (self.root / "config/assessment2_data.json").write_text(
            json.dumps(self.pin), encoding="utf-8"
        )

    @staticmethod
    def write_gzip(path, fields, rows):
        with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_history_retention_and_identity_evidence(self):
        files_before = {p.relative_to(self.root) for p in self.root.rglob("*") if p.is_file()}
        result = audit.audit(self.root)
        self.assertEqual(files_before, {p.relative_to(self.root) for p in self.root.rglob("*") if p.is_file()})
        self.assertEqual(result["counts"]["retained_history_rows"], 5)
        self.assertEqual(result["counts"]["history_only_rows"], 2)
        self.assertEqual(result["license_status_retained_rows"],
                         {"valid": 2, "blank": 1, "zero": 1, "other_invalid": 1})
        self.assertEqual(result["matching"]["establishment_groups"], 4)
        self.assertEqual(result["matching"]["fallback_groups"], 3)
        self.assertEqual(result["matching"]["licenses_with_multiple_recorded_addresses"], 1)
        self.assertEqual(result["matching"]["same_day_conflicting_label_groups"], 1)
        self.assertEqual(result["raw_missing"]["License #"]["count"], 1)
        issues = result["review_items"]["invalid_licenses"]
        self.assertEqual(len(issues), 3)
        self.assertEqual(len({row["establishment_key"] for row in issues}), 3)

    def test_frozen_source_hash_cannot_be_silently_changed(self):
        self.pin["raw_sha256"] = "0" * 64
        self.save_pin()
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            audit.audit(self.root)

    def test_frozen_actual_date_range_is_enforced(self):
        self.pin["actual_date_range"]["end"] = "2026-10-30"
        self.save_pin()
        with self.assertRaisesRegex(ValueError, "Actual inspection dates"):
            audit.audit(self.root)

    def test_relabelled_unjudged_visit_fails_even_with_updated_manifest_hash(self):
        path = self.root / prep.CONFIG["processed_history"]
        with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows[2]["label"] = "0"  # No Entry must remain unlabelled.
        self.write_gzip(path, prep.HISTORY_COLUMNS, rows)
        manifest_path = self.root / "data/processed/preparation_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["outputs"][prep.CONFIG["processed_history"]] = prep.sha256_file(path)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Prepared history differs"):
            audit.audit(self.root)


if __name__ == "__main__":
    unittest.main()
