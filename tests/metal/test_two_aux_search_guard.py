"""Host-only admission checks for native two-auxiliary qualification."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import two_aux_search as qualification


class QualificationAdmissionTests(unittest.TestCase):
    def test_outer_reserve_refuses_new_work_before_headroom(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(qualification.time, 'monotonic', return_value=805), \
             patch.object(qualification, 'wait_for_headroom') as wait, \
             patch.object(qualification, 'run') as child:
            with self.assertRaisesRegex(ValueError, 'no child allowance'):
                qualification.run_child('late', ['unused'], Path(directory), {}, 900)
            wait.assert_not_called()
            child.assert_not_called()

    def test_headroom_crossing_outer_admission_boundary_refuses_child(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(qualification.time, 'monotonic', side_effect=[804.8, 805.1]), \
             patch.object(qualification, 'wait_for_headroom',
                          return_value=dict(admitted=True, seconds=.3, samples=[])) as wait, \
             patch.object(qualification, 'run') as child:
            output = Path(directory)
            with self.assertRaisesRegex(ValueError, 'consumed child allowance'):
                qualification.run_child('late', ['unused'], output, {}, 900)
            self.assertAlmostEqual(wait.call_args.args[0], .2)
            child.assert_not_called()
            self.assertEqual(json.loads((output/'late-headroom.json').read_text())['seconds'], .3)

    def test_on_time_child_uses_prospective_guard(self):
        complete = dict(complete=True, forced_termination=False, cleanup_failure=False)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(qualification.time, 'monotonic', side_effect=[10, 11]), \
             patch.object(qualification, 'wait_for_headroom',
                          return_value=dict(admitted=True, seconds=1, samples=[])) as wait, \
             patch.object(qualification, 'run', return_value=complete) as child:
            output = Path(directory)
            self.assertEqual(qualification.run_child('ok', ['unused'], output, {}, 900), complete)
            wait.assert_called_once_with(45.)
            child.assert_called_once_with(['/usr/bin/env', 'PATH=', 'unused'], output/'ok-guard',
                                          wired_limit_bytes=4294967296)

    def test_expected_rejection_still_requires_clean_termination(self):
        cases = [('forced_termination', True), ('cleanup_failure', True),
                 ('error', 'time limit at completion'), ('wall_seconds', 45)]
        for field, value in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory, \
                 patch.object(qualification.time, 'monotonic', return_value=10), \
                 patch.object(qualification, 'wait_for_headroom',
                              return_value=dict(admitted=True, seconds=0, samples=[])):
                output = Path(directory)
                (output/'reject-guard').mkdir()
                (output/'reject-guard/run.log').write_text(
                    'additive resume policy or evaluation settings changed\n')
                result = dict(complete=False, exit_code=1,
                              forced_termination=False, cleanup_failure=False,
                              error='exit 1', wall_seconds=1)
                result[field] = value
                with patch.object(qualification, 'run', return_value=result):
                    with self.assertRaisesRegex(ValueError, 'rejection did not happen'):
                        qualification.run_child('reject', ['unused'], output, {}, 900,
                                                expected_rejection=True)


if __name__ == '__main__':
    unittest.main()
