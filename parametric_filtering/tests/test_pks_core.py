"""Offline tests: no GPU, API calls, network, or model weights required."""
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pks_core as c


def load_script(rel):
    spec = importlib.util.spec_from_file_location("test_script_" + rel.replace("/", "_"), ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.stats = {
            "common": {"mean": 8.0, "std": 1.0},
            "plausible": {"mean": 0.0, "std": 1.0},
            "nonsense": {"mean": -8.0, "std": 1.0},
        }

    def test_pmi_sum_formula(self):
        self.assertEqual(c.classic_pmi_from_nll(10, 30, 35), 5)

    def test_prompt_template(self):
        self.assertEqual(c.TEMPLATE_NO_CONTEXT.format(q="Q", a="A"),
                         "Instruction: You must respond based *only* on the information that you are trained with and the knowledge that you know. Answer the following question in just One to Three words.\n\nQuestion: Q\n\nAnswer: A")

    def test_nll_scoring_calls_three_prompts(self):
        strings = []
        def fake_nll(text):
            strings.append(text)
            return {0: 10, 1: 20, 2: 25}[len(strings)-1]
        self.assertEqual(c.score_qa(fake_nll, "Q", "A"), 5)
        self.assertTrue(strings[0].endswith("Answer: "))
        self.assertTrue(strings[1].endswith("Answer: A"))
        self.assertTrue(strings[2].endswith("Answer: A"))

    def test_population_standard_deviation(self):
        stat = c.calibration_stats([0, 2, 4])
        self.assertAlmostEqual(stat["mean"], 2)
        self.assertAlmostEqual(stat["std"], math.sqrt(8/3))
        self.assertEqual(stat["n"], 3)

    def test_zero_variance_rejected(self):
        with self.assertRaises(ValueError):
            c.calibration_stats([2, 2, 2])

    def test_classification(self):
        self.assertEqual(c.classify_pks(7.9, self.stats)[0], "common")
        self.assertEqual(c.classify_pks(0.1, self.stats)[0], "plausible")
        self.assertEqual(c.classify_pks(-8.1, self.stats)[0], "nonsense")
        # Ties are resolved in original class insertion order.
        self.assertEqual(c.classify_pks(4.0, self.stats)[0], "common")

    def test_load_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            for cat, name in c.DEFAULT_CALIBRATION_FILENAMES.items():
                c.write_json(Path(tmp) / name, {"pmi_classic": [0, 2, 4]})
            stats = c.load_calibration(tmp)
            self.assertEqual(stats["nonsense"]["n"], 3)

    def test_missing_or_conflicting_question(self):
        with self.assertRaises(ValueError):
            c.normalized_questions([{"original_question": "", "answer": "A"}])
        with self.assertRaises(ValueError):
            c.normalized_questions([{"original_question": "Q", "answer": "A"},
                                    {"original_question": "Q", "answer": "B"}])

    def test_filter_unique_question_and_rows(self):
        f = load_script("pks_filtering/00_para14.py")
        rows = [{"original_question": "Q1", "answer": "A1", "hop": 1},
                {"original_question": "Q2", "answer": "A2", "hop": 1},
                {"original_question": "Q1", "answer": "A1", "hop": 2}]
        calls = []
        def score(q, a):
            calls.append(q)
            return 8 if q == "Q1" else 0
        kept, removed, details = f.filter_items(rows, self.stats, score)
        self.assertEqual(calls, ["Q1", "Q2"])
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(removed), 2)
        self.assertEqual(details["Q1"]["action"], "REMOVE")

    def test_fail_safe_policy(self):
        f = load_script("pks_filtering/00_para14.py")
        rows = [{"original_question": "Q", "answer": "A"}]
        def bad(*args):
            raise RuntimeError("mock inference error")
        kept, _, details = f.filter_items(rows, self.stats, bad, on_error="keep")
        self.assertEqual(len(kept), 1)
        self.assertEqual(details["Q"]["category"], "nonsense")
        with self.assertRaises(RuntimeError):
            f.filter_items(rows, self.stats, bad, on_error="error")

    def test_independent_verification_and_categories(self):
        f = load_script("validation/para19.py")
        rows = [{"original_question": "Q1", "answer": "A1"},
                {"original_question": "Q2", "answer": "A2"}]
        kept, removed, scores = f.independently_filter(rows, self.stats, lambda q, _: 8 if q == "Q1" else 0)
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(removed), 1)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "categories.csv"
            p.write_text("question,category\nQ1,common\nQ2,plausible\n", encoding="utf-8")
            f.verify_categories(scores, p)
            p.write_text("question,category\nQ1,nonsense\nQ2,plausible\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                f.verify_categories(scores, p)

    def test_baseline_json_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            c.write_json(Path(tmp) / "calibration.json", {"pmi_classic": [1.0, 3.0]})
            self.assertEqual(c.read_json(Path(tmp) / "calibration.json")["pmi_classic"], [1.0, 3.0])


if __name__ == "__main__":
    unittest.main()
