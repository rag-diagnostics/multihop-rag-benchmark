"""Tests on synthetic examples only; never contacts a hosted model."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, file):
    spec=importlib.util.spec_from_file_location(name, file)
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


select = load("random_selector", ROOT / "distractor_selection" / "select_random_distractors.py")
audit = load("context_audit", ROOT / "multihop_construction" / "audit_contexts.py")
filter_code = load("filter_hops", ROOT / "multihop_construction" / "filter_multihop.py")


class TestRandomSelection(unittest.TestCase):
    def test_gold_answer_span_exclusion(self):
        self.assertTrue(select.contains_gold_answer("Venice was mentioned.", "Venice"))
        self.assertTrue(select.contains_gold_answer("Paris is mentioned.", "Rome, Paris"))
        self.assertFalse(select.contains_gold_answer("Parison was mentioned.", "Paris"))

    def test_random_is_repeatable_and_unique(self):
        docs = ["The gold answer is Venice."] + [f"Independent report {i}" for i in range(20)]
        record = {"question":"Where is the answer?", "answer":"Venice", "Document":docs[0]}
        _, first, _ = select.select_for_row(record, docs, 10, 42, "span-only", None, "test")
        _, second, _ = select.select_for_row(record, docs, 10, 42, "span-only", None, "test")
        self.assertEqual(first["distractors"], second["distractors"])
        self.assertEqual(len(set(first["distractors"])), 10)
        self.assertNotIn(docs[0], first["distractors"])

    def test_rejects_non_unknown_and_api_errors(self):
        with patch.object(select,"call_gemini",return_value="some answer"):
            self.assertEqual(select.check_candidate("q", "doc", "llm", "placeholder", "test")[0],False)
        with patch.object(select,"call_gemini",side_effect=RuntimeError("failed")):
            self.assertEqual(select.check_candidate("q", "doc", "llm", "placeholder", "test")[0],False)
        with patch.object(select,"call_gemini",return_value="UNKNOWN"):
            self.assertEqual(select.check_candidate("q", "doc", "llm", "placeholder", "test")[0],True)


class TestMultihopAudit(unittest.TestCase):
    def test_context_integrity(self):
        gold = ["gold"]
        distractors = [f"other{i}" for i in range(10)]
        entry = {"hops":{"n1":{"golden_docs":gold,"context_distractors":distractors}}}
        self.assertEqual(audit.audit_one(entry),[])
        entry["hops"]["n1"]["context_distractors"] = distractors[:-1]
        self.assertTrue(audit.audit_one(entry))

    def test_filter_failed_hops(self):
        from tests.fixtures import chain
        source=[chain(family="random"),
                {"original_question":"bad", "hops":{"n2":{"error":"failed"}}}]
        with tempfile.TemporaryDirectory() as d:
            inp = Path(d) / "in.json"
            out = Path(d) / "out.json"
            inp.write_text(json.dumps(source),encoding="utf-8")
            filter_code.filter_json_remove_entries_with_hops_error(str(inp),str(out))
            self.assertEqual(len(json.loads(out.read_text(encoding="utf-8"))),1)


if __name__ == "__main__":
    unittest.main()
