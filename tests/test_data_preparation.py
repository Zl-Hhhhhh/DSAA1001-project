"""Check course-critical preparation rules using small, offline fixtures."""
import csv
from datetime import date
import gzip
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import download_data as download
import prepare_data as preparation


class PreparationTests(unittest.TestCase):
    def setUp(self):
        temporary_parent = ROOT / "tmp"
        temporary_parent.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="data-test-", dir=temporary_parent)
        self.root = Path(self.temporary.name).resolve()
        self.root.relative_to(temporary_parent.resolve())  # Verify cleanup stays inside project/tmp.
        (self.root / "data/raw").mkdir(parents=True)
        self.raw = self.root / preparation.CONFIG["raw_csv"]
        self.metadata = self.root / preparation.CONFIG["download_metadata"]

    def tearDown(self):
        self.temporary.cleanup()

    def row(self, inspection_id="1", result="Fail", inspected_on="01/04/2016", **changes):
        row = {column: "" for column in preparation.RAW_TO_CLEAN}
        row.update({
            "Inspection ID": inspection_id, "Inspection Date": inspected_on,
            "Results": result, "DBA Name": "EXAMPLE SHOP", "License #": "123",
            "Address": "1 MAIN ST", "City": "CHICAGO", "State": "IL",
            "Zip": "01234", "Facility Type": " Restaurant ", "Risk": "Risk 1 (High)",
            "Inspection Type": "Canvass", "Violations": "CURRENT OUTCOME - MUST NOT BE A FEATURE",
        })
        row.update(changes)
        return row

    def save(self, rows):
        with gzip.open(self.raw, "wt", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(preparation.RAW_TO_CLEAN))
            writer.writeheader()
            writer.writerows(rows)
        metadata = {
            "dataset_id": "4ijn-s7e5", "downloaded_at_utc": "2026-10-12T08:00:00+00:00",
            "portal_page": download.PORTAL_PAGE,
            "requested_window": {"start_date": "2016-01-01", "end_date": "2026-10-12"},
            "row_count": len(rows), "csv_sha256": preparation.sha256_file(self.raw),
        }
        self.metadata.write_text(json.dumps(metadata), encoding="utf-8")

    def read(self, name):
        with gzip.open(self.root / preparation.CONFIG[name], "rt", encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    def test_labels_history_and_current_outcome_exclusion(self):
        results = ["Fail", "Pass", "Pass w/ Conditions", "Out of Business",
                   "No Entry", "Not Ready", "Business Not Located", ""]
        self.save([self.row(str(8-i), value, f"01/{i+4:02}/2016") for i, value in enumerate(results)])
        report = preparation.prepare(self.raw, self.metadata, self.root)
        labelled = self.read("labelled_data")
        history = self.read("processed_history")
        self.assertEqual(report["history_rows"], 8)
        self.assertEqual(report["labelled_rows"], 3)
        self.assertEqual([row["label"] for row in labelled], ["1", "0", "0"])
        self.assertEqual([row["inspection_date"] for row in labelled],
                         ["2016-01-04", "2016-01-05", "2016-01-06"])
        self.assertTrue(all(row["zip"] == "01234" for row in labelled))
        self.assertTrue(all("results" not in row and "violations" not in row for row in labelled))
        self.assertEqual(history[-1]["label"], "")
        self.assertEqual(history[-2]["results"], "Business Not Located")
        preparation.check(self.root)

    def test_only_exact_duplicate_inspections_are_removed(self):
        first = self.row()
        self.save([first, first, self.row("2", "Pass", "01/05/2016")])
        report = preparation.prepare(self.raw, self.metadata, self.root)
        self.assertEqual(report["history_rows"], 2)
        self.assertEqual(report["exact_duplicate_rows_removed"], 1)
        self.assertEqual(report["repeated_establishment_groups"], 1)

    def test_conflicting_duplicate_id_is_rejected(self):
        self.save([self.row(), self.row(result="Pass")])
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            preparation.prepare(self.raw, self.metadata, self.root)
        self.assertFalse((self.root / preparation.CONFIG["processed_history"]).exists())

    def test_hidden_test_record_is_rejected(self):
        self.save([self.row(inspected_on="10/31/2026")])
        with self.assertRaisesRegex(ValueError, "permitted"):
            preparation.prepare(self.raw, self.metadata, self.root)

    def test_unknown_result_is_not_silently_labelled(self):
        self.save([self.row(result="Unexpected outcome")])
        with self.assertRaisesRegex(ValueError, "Unrecognised"):
            preparation.prepare(self.raw, self.metadata, self.root)

    def test_missing_licenses_do_not_form_one_shared_establishment(self):
        first = self.row(**{"License #": "", "DBA Name": "", "Address": ""})
        second = self.row("2", **{"License #": "0", "DBA Name": "", "Address": ""})
        self.save([first, second])
        report = preparation.prepare(self.raw, self.metadata, self.root)
        rows = self.read("labelled_data")
        self.assertNotEqual(rows[0]["establishment_key"], rows[1]["establishment_key"])
        self.assertEqual(report["establishment_groups"], 2)
        self.assertTrue(all(row["license_number"] == "" for row in rows))

    def test_raw_corruption_is_rejected(self):
        self.save([self.row()])
        with self.raw.open("ab") as stream:
            stream.write(b"unexpected change")
        with self.assertRaisesRegex(ValueError, "checksum"):
            preparation.prepare(self.raw, self.metadata, self.root)

    def test_repeated_offline_preparation_is_deterministic(self):
        self.save([self.row("2", inspected_on="01/05/2016"), self.row("1")])
        first = preparation.prepare(self.raw, self.metadata, self.root)
        first_hashes = [preparation.sha256_file(self.root / preparation.CONFIG[name])
                        for name in ["processed_history", "labelled_data"]]
        second = preparation.prepare(self.raw, self.metadata, self.root)
        second_hashes = [preparation.sha256_file(self.root / preparation.CONFIG[name])
                         for name in ["processed_history", "labelled_data"]]
        self.assertEqual(first, second)
        self.assertEqual(first_hashes, second_hashes)

    def test_downloader_rejects_future_hidden_and_post_cutoff_downloads(self):
        for start, end, today in [
            ("2015-12-31", "2016-01-04", date(2026, 10, 12)),
            ("2016-01-01", "2026-10-13", date(2026, 10, 12)),
            ("2016-01-01", "2026-10-31", date(2026, 10, 31)),
            ("2016-01-01", "2026-10-30", date(2026, 12, 4)),
        ]:
            with self.subTest(start=start, end=end, today=today):
                with self.assertRaises(ValueError):
                    download.validate_window(start, end, today)


if __name__ == "__main__":
    unittest.main()
