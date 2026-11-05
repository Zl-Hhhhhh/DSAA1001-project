"""Regression checks for time-respecting inspection history."""
from pathlib import Path
import sys
import unittest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ass1_utils import strict_past_features


class StrictHistoryTests(unittest.TestCase):
    def frame(self):
        rows = [
            ("1", "A", "2020-01-01", 1),
            ("2", "A", "2020-01-02", 0),
            ("3", "A", "2020-01-02", 1),
            ("4", "A", "2020-01-03", None),
            ("5", "A", "2020-01-04", 0),
            ("6", "B", "2020-01-02", 1),
        ]
        frame = pd.DataFrame(rows, columns=["inspection_id", "establishment_key", "inspection_date", "label"])
        frame["inspection_date"] = pd.to_datetime(frame["inspection_date"])
        frame["label"] = frame["label"].astype("Int8")
        return frame

    def features(self, frame):
        return strict_past_features(frame).set_index("inspection_id").sort_index()

    def test_same_day_outcomes_are_not_visible_to_each_other(self):
        features = self.features(self.frame())
        for inspection_id in ["2", "3"]:
            self.assertEqual(features.loc[inspection_id, "prior_visits"], 1)
            self.assertEqual(features.loc[inspection_id, "prior_failures"], 1)
            self.assertEqual(features.loc[inspection_id, "previous_outcome"], "Previous Fail")
        changed = self.frame()
        changed.loc[changed.inspection_id.eq("2"), "label"] = 1
        pd.testing.assert_frame_equal(features.loc[["2", "3"]], self.features(changed).loc[["2", "3"]])

    def test_unjudged_visits_count_without_overwriting_last_judgement(self):
        features = self.features(self.frame())
        self.assertEqual(features.loc["5", "prior_visits"], 4)
        self.assertEqual(features.loc["5", "prior_judgements"], 3)
        self.assertEqual(features.loc["5", "prior_failures"], 2)
        self.assertEqual(features.loc["5", "previous_outcome"], "Mixed previous date")
        self.assertEqual(features.loc["6", "prior_visits"], 0)
        self.assertEqual(features.loc["6", "previous_outcome"], "No earlier judgement")

    def test_shuffling_input_does_not_change_any_history(self):
        frame = self.frame()
        pd.testing.assert_frame_equal(self.features(frame), self.features(frame.sample(frac=1, random_state=1001)))

    def test_later_outcomes_do_not_change_earlier_history(self):
        frame = self.frame()
        expected = self.features(frame)
        future = pd.DataFrame([{
            "inspection_id": "7", "establishment_key": "A",
            "inspection_date": pd.Timestamp("2020-01-05"), "label": 1,
        }])
        extended = pd.concat([frame, future], ignore_index=True)
        extended["label"] = extended["label"].astype("Int8")
        pd.testing.assert_frame_equal(expected, self.features(extended).loc[expected.index])


if __name__ == "__main__":
    unittest.main()
