"""Offline tests for Section 3.7, no API calls or real data needed."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import json

PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PARENT))
from construct_contexts import build_one, build_all, stable_seed


def example_chain():
    gold = [f'gold-{i}' for i in range(1, 5)]
    distracting = [f'distractor-{i}' for i in range(1, 11)]
    hops = {'n1': {
        'question': 'first question',
        'golden_docs': gold[:1],
        'context_distractors': distracting[:],
    }}
    for depth in (2, 3, 4):
        hops[f'n{depth}'] = {
            'question': f'question n{depth}',
            'all_golden_docs': gold[:depth],
            'context_distractors_remaining': distracting[depth - 1:],
        }
    return {'original_question': 'first question',
            'answer': 'gold answer', 'decomposition': [], 'hops': hops}


class TestContextFormation(unittest.TestCase):
    def setUp(self):
        self.records = build_one(example_chain(), 0, seed=42)
        self.by_key = {(r['hop'], r['configuration']): r for r in self.records}

    def test_twelve_conditions(self):
        self.assertEqual(len(self.records), 12)
        self.assertEqual(set(self.by_key),
            {('n1', a) for a in ('A1', 'A2', 'A3')} |
            {(f'n{n}', b) for n in (2, 3, 4) for b in ('B1', 'B2', 'B3')})

    def test_singlehop_placement(self):
        self.assertEqual(self.by_key['n1','A1']['gold_positions_1_based'], [1])
        self.assertEqual(self.by_key['n1','A2']['gold_positions_1_based'], [6])
        self.assertEqual(self.by_key['n1','A3']['gold_positions_1_based'], [11])

    def test_multihop_order(self):
        for depth in (2, 3, 4):
            canon = self.by_key[f'n{depth}', 'B1']['documents']
            rev = self.by_key[f'n{depth}', 'B2']['documents']
            g = [f'gold-{i}' for i in range(1, depth+1)]
            positions = self.by_key[f'n{depth}', 'B1']['gold_positions_1_based']
            self.assertEqual(positions, list(range(positions[0], positions[0] + depth)))
            self.assertEqual([canon[p-1] for p in positions], g)
            self.assertEqual([rev[p-1] for p in positions], g[::-1])

    def test_all_contexts_have_eleven_unique_documents(self):
        for r in self.records:
            self.assertEqual(len(r['documents']), 11)
            self.assertEqual(len(set(r['documents'])), 11)
            self.assertEqual(len(r['gold_positions_1_based']), int(r['hop'][1:]))

    def test_shuffle_reproducible_with_seed(self):
        other = build_one(example_chain(), 0, seed=42)
        self.assertEqual(self.records, other)
        self.assertNotEqual(stable_seed(42, 0, 4, 'B3'),
                            stable_seed(43, 0, 4, 'B3'))
        self.assertNotEqual(stable_seed(42, 0, 4, 'B3'),
                            stable_seed(42, 1, 4, 'B3'))

    def test_bad_inputs_fail(self):
        corrupted = example_chain()
        corrupted['hops']['n2']['context_distractors_remaining'] = ['x']
        with self.assertRaisesRegex(ValueError, 'expected 9 distractors'):
            build_one(corrupted, 0)
        corrupted = example_chain()
        corrupted['hops']['n1']['context_distractors'][0] = 'gold-1'
        with self.assertRaisesRegex(ValueError, 'duplicate text'):
            build_one(corrupted, 0)
        corrupted = example_chain()
        corrupted['hops']['n4'] = {'error':'failed'}
        with self.assertRaisesRegex(ValueError, 'Missing or failed'):
            build_all([corrupted], seed=42)

    def test_json_serializable(self):
        self.assertEqual(len(json.loads(json.dumps(self.records))), 12)


if __name__ == '__main__':
    unittest.main()
