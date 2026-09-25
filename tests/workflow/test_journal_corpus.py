"""Native journal corpus reads and resume admission regression checks; no GPU."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import identity_oracle as oracle
from test_run_config import settings

ROOT = Path(__file__).resolve().parents[2]
JOURNAL = Path(os.environ.get('FGM_JOURNAL_DRIVER', ROOT / 'build/workflow/test_journal'))
TOOL = Path(os.environ.get('FGM_SCHEME_TOOL', ROOT / 'build/metal/scheme_tool'))
EXECUTION = Path(os.environ.get('FGM_EXECUTION_DRIVER', ROOT / 'build/workflow/test_execution'))


def member(scheme):
    return dict(scheme_id=oracle.identity(scheme), scheme=scheme, weight=0)


def admission(scheme, origin='import'):
    return dict(scheme_id=oracle.identity(scheme), scheme=scheme, origin=origin,
                rank=scheme['rank'], domain=scheme['domain'])


def transaction(scheme):
    rank = dict(rank=1, members=[member(scheme)])
    return dict(schema='fgm-search-transaction-v1', kind='run_start', stage=1,
                workflow=dict(mode='alternatives', domain='ZT', dimensions=[1, 1, 1], collection_rank=1),
                admissions=[admission(scheme)],
                pools=dict(schema='active-rank-pool-v1', active=[rank], reserves=[copy.deepcopy(rank)]))


class JournalCorpusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.history = self.root / 'history'
        self.scheme = oracle.schoolbook((1, 1, 1))
        self.serial = 0

    def fixture(self, transactions):
        result = subprocess.run([str(JOURNAL), 'fixture', str(self.history)],
                                input=json.dumps(transactions), text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def inventory(self):
        return {str(path.relative_to(self.history)): path.read_bytes()
                for path in self.history.iterdir() if path.is_file()}

    def corpus(self, *options, expected=0, command='verify', output_format=None):
        self.serial += 1
        output = self.root / ('corpus' + str(self.serial) + '.jsonl')
        args = [str(TOOL), command, '--input', str(self.history), '--format', 'journal',
                '--output', str(output), *options]
        if output_format:
            args += ['--output-format', output_format]
        before = self.inventory()
        result = subprocess.run(args, capture_output=True, text=True, timeout=20, env={**os.environ, 'PATH': ''})
        self.assertEqual(result.returncode, expected, result.stderr)
        self.assertEqual(self.inventory(), before, 'corpus read mutated journal')
        if expected:
            self.assertFalse(output.exists())
            return None
        return [json.loads(line) for line in output.read_text().splitlines()]

    def preflight(self, expected=0):
        self.serial += 1
        output = self.root / ('receipt' + str(self.serial) + '.json')
        config = dict(schema='fgm-run-v1', operation='search',
                      policy=dict(settings(), dimensions=[1, 1, 1], collection_rank=1, target_rank=1),
                      input=dict(kind='resume', journal=str(self.history)),
                      execution=dict(workers=2, batch_steps=1, block_size=32, backend='general', memory_bytes=256*1024*1024),
                      pool=dict(capacity_per_rank=4, reserve_per_rank=4, memory_bytes=1024*1024, stage_threshold=1, selector='uniform'),
                      history=dict(path=str(self.history), storage_bytes=4*1024*1024, transaction_bytes=32768, index_memory_bytes=4096),
                      output=str(output))
        path = self.root / 'run.json'
        path.write_text(json.dumps(config))
        before = self.inventory()
        result = subprocess.run([str(EXECUTION), '--run-config', str(path), '--validate-only'],
                                text=True, capture_output=True, timeout=20, env={**os.environ, 'PATH': ''})
        self.assertEqual(result.returncode, expected, result.stderr)
        self.assertEqual(self.inventory(), before, 'resume preflight mutated journal')
        if expected:
            self.assertFalse(output.exists())
            return None
        return json.loads(output.read_text())

    def test_unique_admissions_export_and_summary_without_cache(self):
        first = transaction(self.scheme)
        duplicate = copy.deepcopy(first)
        duplicate['admissions'][0]['origin'] = 'discovery'
        second = copy.deepcopy(self.scheme)
        second['rank'] = 2
        second['u'] = [[1], [0]]
        second['v'] = second['w'] = [[1], [1]]
        last = copy.deepcopy(first)
        last['admissions'] = [admission(second, 'discovery')]
        self.fixture([first, duplicate, last])
        (self.history / 'index.ids').unlink()
        (self.history / 'index.json').unlink()
        records = self.corpus()
        self.assertEqual([record['scheme_id'] for record in records], [oracle.identity(self.scheme), oracle.identity(second)])
        self.assertEqual([record['source_bindings'][-1]['sequence'] for record in records], [1, 3])
        self.assertEqual(records[0]['source_bindings'][-1]['origin'], 'import')
        exported = self.corpus(command='export', output_format='jsonl')
        self.assertEqual([record['factors_id'] for record in exported], [record['factors_id'] for record in records])
        summary, = self.corpus('--summary', command='analyze')
        self.assertEqual(summary['record_count'], 2)
        self.assertEqual(len(summary['groups']), 2)
        self.corpus('--domain', 'F2', expected=1)
        self.corpus('--scan-bytes', str((self.history / 'journal.bin').stat().st_size), expected=2)
        self.corpus('--record-bytes', '128', expected=2)

    def test_reserve_only_resume_preflight_refills_without_writes(self):
        tx = transaction(self.scheme)
        tx['pools']['active'] = []
        self.fixture([tx])
        receipt = self.preflight()
        self.assertEqual(receipt['admitted_inputs'], 1)
        self.assertFalse(receipt['execution_started'])
        self.assertEqual(receipt['presentations'][0]['scheme_id'], oracle.identity(self.scheme))

    def test_empty_resume_preflight_admits_no_parents(self):
        tx = transaction(self.scheme)
        tx['pools']['active'] = tx['pools']['reserves'] = []
        self.fixture([tx])
        receipt = self.preflight()
        self.assertEqual(receipt['admitted_inputs'], 0)
        self.assertNotIn('workers', receipt)

    def test_invalid_evicted_history_rejected_by_resume_and_corpus(self):
        tx = transaction(self.scheme)
        invalid = copy.deepcopy(self.scheme)
        invalid['w'] = [[0]]
        bad = copy.deepcopy(tx)
        bad['admissions'] = [admission(invalid, 'discovery')]
        self.fixture([tx, bad])
        self.preflight(expected=1)
        self.corpus(expected=1)

    def test_offstage_reserve_domain_rejected(self):
        tx = transaction(self.scheme)
        wrong = oracle.schoolbook((1, 1, 1), 'F2')
        tx['pools']['reserves'][0]['members'] = [member(wrong)]
        self.fixture([tx])
        self.preflight(expected=1)

    def test_retained_parent_requires_admission_history(self):
        tx = transaction(self.scheme)
        tx['admissions'] = []
        self.fixture([tx])
        self.preflight(expected=1)

    def test_historical_capture_factor_binding_is_verified(self):
        tx = transaction(self.scheme)
        tx['observations'] = [dict(scheme=self.scheme, scheme_id=oracle.identity(self.scheme),
                                   factors_id='fgm-factors-v1:' + '0'*64)]
        self.fixture([tx])
        self.preflight(expected=1)
