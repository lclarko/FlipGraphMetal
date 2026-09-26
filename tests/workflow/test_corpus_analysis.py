"""Native bounded corpus summaries preserve domain and cost distinctions."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import identity_oracle as oracle

ROOT = Path(__file__).resolve().parents[2]
BINARY = Path(os.environ.get('FGM_SCHEME_TOOL', ROOT / 'build/metal/scheme_tool'))


class CorpusAnalysisTests(unittest.TestCase):
    def execute(self, records, options=(), command='analyze', summary=True, expected=0):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source, output = directory / 'records.jsonl', directory / 'summary.json'
            source.write_text(''.join(json.dumps(record) + '\n' for record in records))
            original = source.read_bytes()
            args = [str(BINARY), command, '--input', str(source), '--format', 'jsonl',
                    '--output', str(output), *options]
            if summary:
                args.append('--summary')
            result = subprocess.run(args, text=True, capture_output=True, timeout=20)
            self.assertEqual(result.returncode, expected, result.stderr)
            self.assertEqual(source.read_bytes(), original)
            if expected:
                self.assertFalse(output.exists(), result.stderr)
                return result
            return [json.loads(line) for line in output.read_text().splitlines()]

    def test_presentation_counts_domains_and_existing_descriptors(self):
        signed = oracle.schoolbook((1, 1, 1))
        signed['metadata'] = {'bound': -500, 'label': 'supplied only'}
        records = [signed, signed, oracle.schoolbook((1, 1, 1), 'F2'),
                   oracle.schoolbook((2, 1, 1), 'F2')]
        summary, = self.execute(records)
        self.assertEqual(summary['schema'], 'fgm-corpus-summary-v1')
        self.assertEqual(summary['record_count'], 4)
        self.assertEqual(len(summary['groups']), 3)
        groups = {(value['domain'], tuple(value['dimensions']), value['rank']): value
                  for value in summary['groups'].values()}
        group = groups['ZT', (1, 1, 1), 1]
        self.assertEqual(group['record_count'], 2)
        self.assertEqual(group['records_with_supplied_metadata'], 2)
        self.assertEqual(group['naive_additions'], {'count': 2, 'sum': 0, 'min': 0, 'max': 0})
        self.assertNotIn('verified_circuit_additions', group)
        self.assertEqual(group['verified_circuit_records'], 0)
        self.assertEqual(group['search_eligible'], 2)
        self.assertEqual(groups['F2', (1, 1, 1), 1]['signed_reducer_eligible'], 0)
        original_reports = self.execute(records, summary=False)
        self.assertEqual(len(original_reports), 4)
        self.assertEqual(original_reports[0]['metadata']['bound'], -500)
        self.assertIn('eligibility', original_reports[0])

    def test_verified_circuit_cost_separate_from_naive(self):
        circuit = dict(n=[1, 1, 1], m=1, z2=False, complexity={'naive': 0, 'reduced': 0})
        for factor in 'uvw':
            circuit[factor] = [[{'index': 0, 'value': 1}]]
            circuit[factor + '_fresh'] = []
        summary, = self.execute([circuit, oracle.schoolbook((1, 1, 1))])
        group, = summary['groups'].values()
        self.assertEqual(group['naive_additions']['count'], 2)
        self.assertEqual(group['verified_circuit_additions'], {'count': 1, 'sum': 0, 'min': 0, 'max': 0})
        self.assertEqual(group['verified_circuit_records'], 1)

    def test_invalid_late_record_never_publishes_partial_summary(self):
        valid = oracle.schoolbook((1, 1, 1))
        invalid = oracle.schoolbook((1, 1, 1))
        invalid['w'] = [[0]]
        self.execute([valid, invalid], expected=1)

    def test_summary_limits_and_command_scope(self):
        records = [oracle.schoolbook((1, 1, 1))]
        self.execute(records, options=('--selection-memory', '1024'), expected=2)
        self.execute(records, options=('--verification-work', '1'), expected=2)
        self.execute(records, command='verify', expected=1)
        self.execute(records, options=('--summary',), expected=1)
