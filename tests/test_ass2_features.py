"""Behavioural tests for static features, history boundaries and identity choices."""
from collections import Counter
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ass2_features import (
    build_features, normalize_inspections, establishment_keys,
    BASE_CATEGORICAL_FEATURES, BASE_NUMERIC_FEATURES,
)
from prepare_data import normalize_row, RAW_TO_CLEAN


class FeatureTests(unittest.TestCase):
    def frame(self):
        specs = [
            ("1", "10", "SHOP A", "1 MAIN ST", "2020-01-01", 1),
            ("2", "10", "SHOP A", "1 MAIN ST", "2020-01-02", 0),
            ("3", "10", "SHOP A", "1 MAIN ST", "2020-01-02", 1),
            ("4", "10", "SHOP A", "1 MAIN ST", "2020-01-03", None),
            ("5", "10", "SHOP A", "2 MAIN ST", "2020-01-04", 0),
            ("6", "20", "SHOP B", "6 MAIN ST", "2020-01-02", 1),
            ("7", "0", "SHOP C", "7 MAIN ST", "2020-01-01", None),
            ("8", "", "SHOP C", "7 MAIN ST", "2020-01-02", 0),
            ("9", "", "", "", "2020-01-02", None),
        ]
        frame = pd.DataFrame(specs, columns=[
            "inspection_id", "license_number", "dba_name", "address", "inspection_date", "label",
        ])
        frame["label"] = frame.label.astype("Int8")
        for key, value in {
            "facility_type": " Restaurant ", "risk": "Risk 3 (Low)",
            "inspection_type": "Canvass", "zip": "01234", "city": "CHICAGO", "state": "IL",
            "latitude": "41.8", "longitude": "-87.6",
        }.items():
            frame[key] = value
        return frame

    def by_id(self, result):
        return result.X.set_axis(result.meta.inspection_id).sort_index()

    def test_same_day_isolation_mixed_day_and_unjudged_retention(self):
        result = build_features(self.frame())
        X = self.by_id(result)
        for key in ["2", "3"]:
            self.assertEqual(X.loc[key, "prior_visits"], 1)
            self.assertEqual(X.loc[key, "prior_fail_rate"], 1)
            self.assertEqual(X.loc[key, "previous_outcome"], "Previous Fail")
        self.assertEqual(X.loc["5", "prior_visits"], 4)
        self.assertEqual(X.loc["5", "prior_judgements"], 3)
        self.assertEqual(X.loc["5", "prior_failures"], 2)
        self.assertAlmostEqual(X.loc["5", "prior_fail_rate"], 2/3)
        self.assertEqual(X.loc["5", "previous_outcome"], "Mixed previous date")
        self.assertEqual(len(result.y), 9)
        self.assertEqual(result.y.isna().sum(), 3)

    def test_unjudged_only_history_is_known_but_has_no_failure_rate(self):
        result = build_features(self.frame())
        X = self.by_id(result)
        self.assertEqual(X.loc["8", "prior_visits"], 1)
        self.assertEqual(X.loc["8", "prior_judgements"], 0)
        self.assertTrue(np.isnan(X.loc["8", "prior_fail_rate"]))
        self.assertEqual(X.loc["8", "previous_outcome"], "No earlier judgement")
        self.assertTrue(result.meta.set_index("inspection_id").loc["8", "known_establishment"])
        self.assertEqual(X.loc["9", "prior_visits"], 0)

    def test_batch_history_excludes_start_day_and_all_later_labels_and_visits(self):
        frame = self.frame()
        targets = frame.iloc[[1, 2, 3, 4]]
        expected = build_features(targets, history=frame, batch_start="2020-01-02")
        self.assertTrue(expected.X.prior_visits.eq(1).all())
        self.assertTrue(expected.X.prior_fail_rate.eq(1).all())
        changed = frame.copy()
        changed.loc[changed.inspection_date.ge("2020-01-02"), "label"] = 1
        actual = build_features(changed.iloc[[1, 2, 3, 4]], history=changed, batch_start="2020-01-02")
        pd.testing.assert_frame_equal(expected.X, actual.X)
        pd.testing.assert_frame_equal(expected.meta, actual.meta)

    def test_future_history_and_input_order_do_not_change_earlier_features(self):
        frame = self.frame()
        targets = frame.iloc[[4, 0, 2, 1]].copy()
        expected = build_features(targets, history=frame)
        future = frame.iloc[[0]].assign(inspection_id="99", inspection_date="2020-01-05", label=0)
        actual = build_features(
            targets, history=pd.concat([frame, future], ignore_index=True).sample(frac=1, random_state=3)
        )
        pd.testing.assert_frame_equal(expected.X, actual.X)
        self.assertEqual(actual.meta.inspection_id.tolist(), ["5", "1", "3", "2"])
        shuffled = build_features(targets.iloc[::-1], history=frame)
        pd.testing.assert_frame_equal(self.by_id(expected), self.by_id(shuffled))

    def test_license_address_separates_changed_premises_without_changing_basic_inputs(self):
        primary = build_features(self.frame(), identity="license")
        alternate = build_features(self.frame(), identity="license_address")
        self.assertEqual(self.by_id(primary).loc["5", "prior_visits"], 4)
        self.assertEqual(self.by_id(alternate).loc["5", "prior_visits"], 0)
        static = BASE_CATEGORICAL_FEATURES + BASE_NUMERIC_FEATURES
        pd.testing.assert_frame_equal(primary.X[static], alternate.X[static])
        missing_address = self.frame().iloc[[0, 1]].assign(address="")
        keys = establishment_keys(normalize_inspections(missing_address), "license_address")
        self.assertEqual(keys.nunique(), 2)

    def test_static_categories_calendar_missing_numbers_and_outcome_exclusion(self):
        frame = self.frame()
        frame["Violations"] = "unavailable current text"
        frame.loc[0, "latitude"] = "999"
        frame.loc[1, "longitude"] = "not a number"
        result = build_features(frame, include_history=False)
        self.assertEqual(set(result.X), set(BASE_CATEGORICAL_FEATURES + BASE_NUMERIC_FEATURES))
        self.assertEqual(result.X.loc[0, "zip"], "01234")
        self.assertEqual(result.X.loc[0, "risk"], "RISK 3 (LOW)")
        self.assertEqual(result.X.loc[0, "inspection_month"], "1")
        self.assertEqual(result.X.loc[0, "inspection_weekday"], "2")
        self.assertTrue(np.isnan(result.X.loc[0, "latitude"]))
        self.assertTrue(np.isnan(result.X.loc[1, "longitude"]))
        changed = frame.copy()
        changed["label"] = 1
        changed["Violations"] = "changed current outcome text"
        pd.testing.assert_frame_equal(result.X, build_features(changed, include_history=False).X)

    def test_raw_unlabelled_hidden_input_and_empty_history(self):
        target = self.frame().iloc[[0]].drop(columns="label")
        target["inspection_id"] = "100"
        target["inspection_date"] = "10/31/2026"
        target = target.rename(columns={clean: raw for raw, clean in RAW_TO_CLEAN.items()})
        result = build_features(target, history=self.frame(), batch_start="2026-10-31")
        self.assertTrue(result.y.isna().all())
        self.assertEqual(result.meta.inspection_id.tolist(), ["100"])
        self.assertEqual(result.X.prior_visits.tolist(), [5])
        empty = self.frame().iloc[:0]
        unseen = build_features(target, history=empty, batch_start="2026-10-31")
        self.assertEqual(unseen.X.prior_visits.tolist(), [0])
        self.assertTrue(unseen.X.prior_fail_rate.isna().all())

    def test_fallback_keys_match_preparation_and_zeros_are_not_shared(self):
        data = normalize_inspections(self.frame())
        self.assertEqual(establishment_keys(data).iloc[6], establishment_keys(data).iloc[7])
        raw = dict.fromkeys(RAW_TO_CLEAN, "")
        raw.update({"Inspection ID": "7", "Inspection Date": "01/01/2020", "License #": "0",
                    "DBA Name": "SHOP C", "Address": "7 MAIN ST", "City": "CHICAGO",
                    "State": "IL", "Zip": "01234"})
        expected = normalize_row(raw, Counter())["establishment_key"]
        self.assertEqual(establishment_keys(data).iloc[6], expected)
        data.loc[7, "address"] = "OTHER ADDRESS"
        self.assertNotEqual(establishment_keys(data).iloc[6], establishment_keys(data).iloc[7])

    def test_invalid_ids_labels_and_batch_contracts_are_rejected(self):
        frame = self.frame()
        with self.assertRaisesRegex(ValueError, "unique"):
            build_features(pd.concat([frame, frame.iloc[[0]]]))
        bad = frame.copy()
        bad.loc[0, "label"] = 2
        with self.assertRaisesRegex(ValueError, "Labels"):
            build_features(bad)
        with self.assertRaisesRegex(ValueError, "explicit history"):
            build_features(frame, batch_start="2020-01-01")
        with self.assertRaisesRegex(ValueError, "precedes"):
            build_features(frame, history=frame, batch_start="2020-01-02")
        with self.assertRaisesRegex(ValueError, "calendar date"):
            build_features(frame, history=frame, batch_start="2020-01-01 12:00:00")
        bad = frame.iloc[[0]].assign(inspection_date="2019-12-31")
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            build_features(frame, history=bad)

    def test_history_matches_independent_row_by_row_reference(self):
        rng = np.random.default_rng(1001)
        rows = self.frame().iloc[[0] * 120].copy().reset_index(drop=True)
        rows["inspection_id"] = [str(i+1000) for i in range(len(rows))]
        rows["license_number"] = rng.choice(["10", "20", "30"], len(rows))
        rows["address"] = rng.choice(["1 MAIN ST", "2 MAIN ST"], len(rows))
        rows["inspection_date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 12, len(rows)), unit="D")
        rows["label"] = pd.Series(rng.choice([0, 1, None], len(rows)), dtype="Int8")
        for identity in ["license", "license_address"]:
            for batch_start in [None, "2020-01-07"]:
                targets = rows if batch_start is None else rows.loc[rows.inspection_date.ge(batch_start)]
                targets = targets.sample(frac=1, random_state=2)
                result = build_features(targets, history=rows, batch_start=batch_start, identity=identity)
                normalized = normalize_inspections(rows)
                keys = establishment_keys(normalized, identity)
                for i, target in result.meta.iterrows():
                    before = normalized.loc[keys.eq(target.establishment_key) & normalized.inspection_date.lt(target.history_before)]
                    judged = before.loc[before.label.notna()]
                    self.assertEqual(result.X.loc[i, "prior_visits"], len(before))
                    self.assertEqual(result.X.loc[i, "prior_judgements"], len(judged))
                    self.assertEqual(result.X.loc[i, "prior_failures"], int(judged.label.sum()))
                    if judged.empty:
                        expected = "No earlier judgement"
                        self.assertTrue(np.isnan(result.X.loc[i, "prior_fail_rate"]))
                    else:
                        last = set(judged.loc[judged.inspection_date.eq(judged.inspection_date.max()), "label"])
                        expected = "Mixed previous date" if len(last) > 1 else "Previous Fail" if 1 in last else "Previous non-Fail"
                        self.assertAlmostEqual(result.X.loc[i, "prior_fail_rate"], float(judged.label.mean()))
                    self.assertEqual(result.X.loc[i, "previous_outcome"], expected)

    def test_empty_targets_and_same_day_datetime_are_handled(self):
        frame = self.frame()
        result = build_features(frame.iloc[:0], history=frame)
        self.assertEqual(len(result.X), 0)
        self.assertEqual(len(result.meta), 0)
        frame.loc[1, "inspection_date"] = "2020-01-01T15:00:00"
        X = self.by_id(build_features(frame))
        self.assertEqual(X.loc["2", "prior_visits"], 0)


if __name__ == "__main__":
    unittest.main()
