"""Selected-trace and report consistency, without GPU or closure-search claims."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'benchmarks/workflow'))
import baseline
from test_two_aux_search import fixture_result


class TwoAuxReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = Path(os.environ.get('FGM_REDUCTION_RESULT_DRIVER',
                                        ROOT/'build/workflow/test_reduction_result'))
        cls.source = baseline.normalized_input(json.loads(
            (ROOT/'benchmarks/workflow/fixtures/fgm1/factors/sun.json').read_text()))
        source = dict(dimensions=[3,3,3], rank=23, domain='ZT', orientation='cyclic-w',
                      **{k: cls.source[k] for k in 'uvw'})
        factors_id = baseline.host_oracle().identity(source, False)
        request = dict(two_aux=True, inputs=9, targets=cls.source['v'], raw_slots=[32877])
        response = cls.invoke(request)
        if response.returncode:
            raise RuntimeError(response.stderr)
        selected = json.loads(response.stdout)['results'][0]
        census = cls.invoke(dict(two_aux=True, inputs=9, targets=cls.source['v'],
                                max_pair_slots=32878, prepare_only=True, summary_only=True))
        if census.returncode:
            raise RuntimeError(census.stderr)
        stats = json.loads(census.stdout)['stats']
        value = fixture_result(cls.source, factors_id)
        value['settings'] = dict(constructor=dict(family='signed-two-aux-distinct-v1', max_pair_slots=65536))
        value['baseline_additions_by_stage'] = dict(u=13, v=37, w=30)
        value['baseline_additions'] = 80
        value['pre_two_aux_additions_by_stage'] = dict(u=13, v=37, w=30)
        value['additions_by_stage'] = dict(u=13, v=13, w=30)
        for stage, out, cost, status in [('u','u',13,'one-auxiliary'), ('wt','w',30,'target-only')]:
            s = value['two_auxiliary']['stages'][stage]
            s.update(max_pair_slots=65536, prefix_limit=min(s['raw_family_size'],65536),
                     baseline_cost=cost, incumbent_cost=cost, final_cost=cost,
                     stop_reason='smaller-family-succeeded', coverage='not-searched',
                     raw_scanned=0, second_invalid=0, logical_prefix=0,
                     unvisited_raw_slots=s['raw_family_size'])
            value['construction'][stage]['status'] = status
        s = value['two_auxiliary']['stages']['v']
        s.update(stats)
        prepared = stats['prepared']
        # The route census and selected trace are checked. Other work counters
        # are synthetic protocol fields, not a claimed run of every candidate.
        s.update(max_pair_slots=65536, prefix_limit=65536, baseline_cost=37,
                 incumbent_cost=37, final_cost=13, stop_reason='bound-attained',
                 used_auxiliaries=2, logical_prefix=32878, unvisited_raw_slots=s['raw_family_size']-32878,
                 dispatched=prepared, completed=prepared, validated=prepared,
                 gpu_sweeps=prepared, gpu_rule_checks=2*prepared,
                 negative_validation_rule_checks=2*(prepared-1))
        s['selected_witness'] = {k: selected[k] for k in
                                ('raw_slot','helpers','creations','gates','count','available')}
        s['selected_witness'].update(first_route=32877//420, second_route=32877%420)
        value['stage_sources']['v'] = 'two-auxiliary'
        cls.valid = value

    @classmethod
    def invoke(cls, request):
        return subprocess.run([str(cls.driver)], input=json.dumps(request),
                              capture_output=True, text=True, timeout=15)

    def check(self, value, success):
        result = self.invoke(dict(effective=self.source, two_aux_report=value))
        self.assertEqual(result.returncode, 0 if success else 1, result.stderr)

    def test_selected_sun_trace_is_valid(self):
        self.check(self.valid, True)

    def test_selected_route_trace_and_counter_corruption(self):
        mutations = [
            lambda s: s['selected_witness'].update(raw_slot=32876),
            lambda s: s['selected_witness'].update(first_route=0),
            lambda s: s['selected_witness'].update(second_route=0),
            lambda s: s['selected_witness'].update(available=0),
            lambda s: s['selected_witness'].update(count=26),
            lambda s: s['selected_witness']['helpers'][0].__setitem__(0,99),
            lambda s: s['selected_witness']['creations'][0].__setitem__(3,0),
            lambda s: s['selected_witness']['gates'][0].__setitem__(1,33),
            lambda s: s['selected_witness']['gates'][0].__setitem__(4,-s['selected_witness']['gates'][0][4]),
            lambda s: s.update(batch_tail_candidates=s['prepared']),
            lambda s: s.update(batch_tail_candidates=1),  # No raw slots after the selected lane.
            lambda s: s.update(used_auxiliaries=1),
        ]
        for i, mutate in enumerate(mutations):
            with self.subTest(case=i):
                value = deepcopy(self.valid)
                mutate(value['two_auxiliary']['stages']['v'])
                self.check(value, False)

    def test_false_stage_source_and_negative_validation_counts(self):
        value = deepcopy(self.valid)
        value['baseline_additions_by_stage']['u'] += 1
        value['baseline_additions'] += 1
        value['two_auxiliary']['stages']['u']['baseline_cost'] += 1
        self.check(value, False)  # A baseline source cannot claim a reduced cost.
        for source in ('baseline', 'cancellation', 'transpose-two-auxiliary'):
            with self.subTest(source=source):
                value = deepcopy(self.valid)
                value['stage_sources']['v'] = source
                self.check(value, False)
        value = deepcopy(self.valid)
        s = value['two_auxiliary']['stages']['v']
        s.update(stop_reason='budget-exhausted', max_pair_slots=32878, prefix_limit=32878,
                 final_cost=37, used_auxiliaries=0, selected_witness=None,
                 negative_validation_rule_checks=0)
        value['settings']['constructor']['max_pair_slots'] = 32878
        for other in ('u','wt'):
            value['two_auxiliary']['stages'][other].update(max_pair_slots=32878, prefix_limit=32878)
        value['additions_by_stage']['v'] = 37
        value['stage_sources']['v'] = 'baseline'
        self.check(value, False)
        s['negative_validation_rule_checks'] = 2*s['prepared']
        self.check(value, True)
        value['baseline_additions_by_stage']['v'] += 1
        value['baseline_additions'] += 1
        s['baseline_cost'] += 1
        value['stage_sources']['v'] = 'cancellation'
        self.check(value, False)  # Exhausted smaller construction cannot be the source.


if __name__ == '__main__':
    unittest.main()
