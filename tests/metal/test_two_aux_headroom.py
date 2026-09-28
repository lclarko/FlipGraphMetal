"""Host-only deadline and launch-admission checks for two-aux qualification."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import two_aux


AVAILABLE = 0
UNAVAILABLE = two_aux.LAUNCH_WIRED_CEILING_BYTES + 1


class Clock:
    def __init__(self):
        self.now = 0.
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class HeadroomDeadlineTests(unittest.TestCase):
    def test_sample_finishing_after_deadline_is_rejected(self):
        clock = Mock(side_effect=[0., 44.8, 45.1])
        sample = Mock(return_value=AVAILABLE)
        sleep = Mock()
        result = two_aux.wait_for_headroom(45, monotonic=clock, sample=sample, sleep=sleep)
        self.assertFalse(result['admitted'])
        self.assertEqual(result['samples'], [dict(seconds=45.1, wired_bytes=AVAILABLE)])
        self.assertAlmostEqual(sample.call_args.kwargs['timeout'], .2)
        sleep.assert_not_called()

    def test_sample_finishing_at_deadline_is_rejected(self):
        sample = Mock(return_value=AVAILABLE)
        result = two_aux.wait_for_headroom(
            45, monotonic=Mock(side_effect=[0., 44.8, 45.]),
            sample=sample, sleep=Mock())
        self.assertFalse(result['admitted'])
        self.assertEqual(result['seconds'], 45.)
        sample.assert_called_once()

    def test_expiry_before_sampling(self):
        sample, sleep = Mock(), Mock()
        result = two_aux.wait_for_headroom(
            45, monotonic=Mock(side_effect=[0., 45.]), sample=sample, sleep=sleep)
        self.assertEqual(result, dict(seconds=45., samples=[], admitted=False))
        sample.assert_not_called()
        sleep.assert_not_called()

    def test_remaining_time_caps_sample_timeout(self):
        sample = Mock(return_value=AVAILABLE)
        result = two_aux.wait_for_headroom(
            45, monotonic=Mock(side_effect=[0., 44.8, 44.9]),
            sample=sample, sleep=Mock())
        self.assertTrue(result['admitted'])
        self.assertAlmostEqual(sample.call_args.kwargs['timeout'], .2)

    def test_expiry_between_sample_check_and_sleep(self):
        sleep = Mock()
        result = two_aux.wait_for_headroom(
            45, monotonic=Mock(side_effect=[0., 44.8, 44.9, 45.1]),
            sample=Mock(return_value=UNAVAILABLE), sleep=sleep)
        self.assertFalse(result['admitted'])
        self.assertEqual(result['seconds'], 45.1)
        self.assertEqual(len(result['samples']), 1)
        sleep.assert_not_called()

    def test_sleep_is_capped_by_recomputed_remaining_time(self):
        sleep = Mock()
        result = two_aux.wait_for_headroom(
            45, monotonic=Mock(side_effect=[0., 44., 44.1, 44.8, 45.]),
            sample=Mock(return_value=UNAVAILABLE), sleep=sleep)
        self.assertFalse(result['admitted'])
        self.assertAlmostEqual(sleep.call_args.args[0], .2)
        self.assertGreater(sleep.call_args.args[0], 0)

    def test_unavailable_retry_then_success(self):
        clock = Clock()
        values = iter((UNAVAILABLE, AVAILABLE))
        timeouts = []

        def sample(*, timeout):
            timeouts.append(timeout)
            clock.now += .1
            return next(values)

        result = two_aux.wait_for_headroom(
            45, monotonic=clock.monotonic, sample=sample, sleep=clock.sleep)
        self.assertTrue(result['admitted'])
        self.assertEqual([entry['wired_bytes'] for entry in result['samples']],
                         [UNAVAILABLE, AVAILABLE])
        self.assertEqual(timeouts, [2, 2])
        self.assertEqual(clock.sleeps, [.5])
        self.assertAlmostEqual(result['seconds'], .7)

    def test_acceptable_sample_finishing_before_deadline(self):
        clock = Clock()
        sample = Mock(return_value=AVAILABLE)
        result = two_aux.wait_for_headroom(
            45, monotonic=clock.monotonic, sample=sample, sleep=clock.sleep)
        self.assertTrue(result['admitted'])
        self.assertEqual(result['samples'], [dict(seconds=0., wired_bytes=AVAILABLE)])
        sample.assert_called_once_with(timeout=2)
        self.assertEqual(clock.sleeps, [])

    def test_zero_second_wait_expires_without_sampling_or_sleep(self):
        sample, sleep = Mock(), Mock()
        result = two_aux.wait_for_headroom(
            0, monotonic=Mock(side_effect=[0., 0.]), sample=sample, sleep=sleep)
        self.assertEqual(result, dict(seconds=0., samples=[], admitted=False))
        sample.assert_not_called()
        sleep.assert_not_called()

    def test_run_group_retains_rejected_headroom_and_refuses_child(self):
        requests = [dict(inputs=2, targets=[[1, 1]])]
        payload = (json.dumps(requests, separators=(',', ':'), sort_keys=True) + '\n').encode()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / 'case-input.json').write_bytes(payload)
            for seconds, evidence in (
                (45, dict(seconds=45.1, samples=[dict(seconds=45.1,
                                                       wired_bytes=AVAILABLE)], admitted=False)),
                (0, dict(seconds=0., samples=[], admitted=False)),
            ):
                with self.subTest(wait_seconds=seconds):
                    with patch.object(two_aux, 'wait_for_headroom', return_value=evidence) as wait, \
                         patch.object(two_aux, 'run') as child:
                        with self.assertRaisesRegex(AssertionError, 'prelaunch wired-memory ceiling'):
                            two_aux.run_group('case', requests, output, Path('unused'), seconds)
                    wait.assert_called_once_with(seconds)
                    child.assert_not_called()
                    self.assertEqual(json.loads((output / 'case-headroom.json').read_text()), evidence)


if __name__ == '__main__':
    unittest.main()
