from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks/metal'))
import capture
from counters import check_dispatches, select_samples


class CaptureEvidence(unittest.TestCase):
    def test_cleanup_reaps_exited_leader(self):
        process = subprocess.Popen(['/usr/bin/true'], start_new_session=True)
        try:
            time.sleep(.1)
            capture.stop([process])
            self.assertEqual(process.returncode, 0)
        finally:
            process.wait(timeout=2)

    def test_cleanup_bounds_wait_and_attempts_all_groups(self):
        first, second = Mock(pid=101), Mock(pid=102)
        first.wait.side_effect = subprocess.TimeoutExpired('first', 2)
        with patch('capture.os.killpg', side_effect=ProcessLookupError):
            with self.assertRaises(RuntimeError):
                capture.stop([first, second])
        first.wait.assert_called_once_with(timeout=2)
        second.wait.assert_called_once_with(timeout=2)

    def test_cleanup_preserves_permission_error(self):
        process = Mock(pid=101)
        process.poll.return_value = None
        with patch('capture.os.killpg', side_effect=PermissionError('denied')):
            with self.assertRaisesRegex(RuntimeError, 'denied'):
                capture.stop([process])

    def test_segmented_dispatches(self):
        groups = {(1, 2, 3): [(10, 20), (30, 40)], (4, 5, 6): [(50, 60)], (7, 8, 9): [(70, 80)]}
        check_dispatches(groups, True)
        del groups[(7, 8, 9)]
        with self.assertRaisesRegex(ValueError, 'three distinct'):
            check_dispatches(groups, True)

    def test_each_segment_requires_counter_samples(self):
        intervals = [(10, 20), (30, 40), (50, 60)]
        selected, coverage = select_samples(intervals, [(10, 1), (20, 9), (31, 2), (51, 3)])
        self.assertEqual(selected, [1, 2, 3])
        self.assertEqual(coverage, [1, 1, 1])
        with self.assertRaisesRegex(ValueError, 'every target compute segment'):
            select_samples(intervals, [(11, 1), (51, 3)])


if __name__ == '__main__':
    unittest.main()
