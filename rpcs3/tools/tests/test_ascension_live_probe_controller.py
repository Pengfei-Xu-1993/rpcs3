#!/usr/bin/env python3
"""Bounded-controller tests that never launch RPCS3 or open a real capture."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock


TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import ascension_live_probe as probe  # noqa: E402


def _stats(**values: object) -> dict[str, object]:
    return {
        "ok": True,
        "authorized": True,
        "armed": False,
        "capturing": True,
        "max_events": 10,
        "frame": 1,
        "observed": 0,
        "accepted": 0,
        "filtered": 0,
        "written": 0,
        "snapshots": 0,
        "dropped": 0,
        **values,
    }


class FakeProcess:
    pid = 4242

    def __init__(self) -> None:
        self.alive = True
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return None if self.alive else -15

    def terminate(self) -> None:
        self.terminated = True
        self.alive = False

    def kill(self) -> None:
        self.killed = True
        self.alive = False

    def wait(self, timeout: float | None = None) -> int:
        if self.alive:
            raise probe.subprocess.TimeoutExpired("fake-rpcs3", timeout)
        return -15


class FakePipe:
    def __init__(self, _timeout: float) -> None:
        self.commands: list[str] = []
        self.max_events = probe.DEFAULT_MAX_EVENTS
        self.armed = False
        self.capturing = False

    def __enter__(self) -> "FakePipe":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def command(self, text: str) -> dict[str, object]:
        self.commands.append(text)
        verb = text.split(maxsplit=1)[0]
        if verb == "START_CAPTURE":
            path = pathlib.Path(text.split('path="', 1)[1].rsplit('"', 1)[0])
            path.write_bytes(bytes(probe.FILE_HEADER_SIZE))
            self.capturing = True
        elif verb == "SET_MAX_EVENTS":
            self.max_events = int(text.split()[1])
        elif verb == "ARM":
            self.armed = True
        elif verb == "DISARM":
            self.armed = False
        elif verb == "STOP_CAPTURE":
            self.armed = False
            self.capturing = False
        return _stats(
            armed=self.armed,
            capturing=self.capturing,
            max_events=self.max_events,
        )


class ControllerHelperTests(unittest.TestCase):
    def test_command_limit_replacement_and_guard_detection(self) -> None:
        commands = probe._commands_for_max_events(50_000)
        self.assertEqual(commands.count("SET_MAX_EVENTS 50000"), 1)
        self.assertFalse(any(command == "SET_MAX_EVENTS 250000" for command in commands))
        self.assertTrue(probe._changes_capture_guard("SET_MAX_EVENTS 1"))
        self.assertTrue(probe._changes_capture_guard("SET_FILTER FIRST_HITS=1"))
        self.assertTrue(probe._changes_capture_guard("STOP_CAPTURE"))
        self.assertFalse(probe._changes_capture_guard("GET_STATS"))

    def test_positive_argument_contract(self) -> None:
        self.assertEqual(probe._positive_int("50000"), 50_000)
        self.assertEqual(probe._positive_finite_float("45"), 45.0)
        for value in ("0", "-1"):
            with self.assertRaises(argparse.ArgumentTypeError):
                probe._positive_int(value)
        for value in ("0", "-1", "nan", "inf"):
            with self.assertRaises(argparse.ArgumentTypeError):
                probe._positive_finite_float(value)

    def test_guard_accepts_only_closed_converged_bounded_output(self) -> None:
        final_probe = _stats(armed=False, capturing=False, accepted=10, written=10)
        aggregate = {
            "counts": {
                "ppu_event_count": 0,
                "spu_task_count": 4,
                "rsx_draw_count": 6,
                "snapshot_count": 0,
                "dropped_count": 0,
                "truncated": False,
            }
        }
        self.assertEqual(
            probe._capture_guard_violations(
                max_events=10,
                arm_seconds=45.0,
                arm_duration_seconds=44.0,
                stop_reason="max_events",
                final_probe=final_probe,
                capture_exists=True,
                analysis_exit_code=0,
                summary_lines=20,
                aggregate=aggregate,
            ),
            [],
        )
        final_probe["accepted"] = 11
        self.assertIn(
            "event_limit_exceeded_or_missing",
            probe._capture_guard_violations(
                max_events=10,
                arm_seconds=45.0,
                arm_duration_seconds=46.0,
                stop_reason="max_events",
                final_probe=final_probe,
                capture_exists=True,
                analysis_exit_code=0,
                summary_lines=20,
                aggregate=aggregate,
            ),
        )

    def test_guard_rejects_missing_analyzer_and_decoded_count_mismatch(self) -> None:
        final_probe = _stats(armed=False, capturing=False, accepted=10, written=10)
        aggregate = {
            "counts": {
                "ppu_event_count": 0,
                "spu_task_count": 4,
                "rsx_draw_count": 5,
                "snapshot_count": 0,
                "dropped_count": 0,
                "truncated": False,
            }
        }
        violations = probe._capture_guard_violations(
            max_events=10,
            arm_seconds=45.0,
            arm_duration_seconds=44.0,
            stop_reason="arm_deadline",
            final_probe=final_probe,
            capture_exists=True,
            analysis_exit_code=None,
            summary_lines=None,
            aggregate=aggregate,
        )
        self.assertIn("analyzer_failed_or_missing", violations)
        self.assertIn("bounded_summary_missing_or_oversized", violations)
        self.assertIn("decoded_event_count_mismatch", violations)


class BoundedControllerTests(unittest.TestCase):
    def test_deadline_disarms_flushes_closes_and_records_guard(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ascension-controller-test-") as directory:
            root = pathlib.Path(directory)
            executable = root / "rpcs3.exe"
            executable.write_bytes(b"fake")
            session_root = root / "sessions"
            fake_process = FakeProcess()
            fake_pipe = FakePipe(1.0)

            def fake_analyzer(argv: list[str], check: bool) -> types.SimpleNamespace:
                self.assertFalse(check)
                output = pathlib.Path(argv[argv.index("--output-dir") + 1])
                output.mkdir(parents=True)
                (output / "model-summary.txt").write_text("bounded aggregate\n", encoding="utf-8")
                (output / "top-candidates.json").write_text(
                    json.dumps(
                        {
                            "counts": {
                                "ppu_event_count": 0,
                                "spu_task_count": 0,
                                "rsx_draw_count": 0,
                                "snapshot_count": 0,
                                "dropped_count": 0,
                                "truncated": False,
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                return types.SimpleNamespace(returncode=0)

            args = argparse.Namespace(
                rpcs3=str(executable),
                session_root=str(session_root),
                game=None,
                pipe_timeout=1.0,
                max_events=10,
                arm_seconds=0.10,
                authorization_timeout=1.0,
            )
            with (
                mock.patch.object(probe.subprocess, "Popen", return_value=fake_process),
                mock.patch.object(probe, "ProbePipe", return_value=fake_pipe),
                mock.patch.object(probe.subprocess, "run", side_effect=fake_analyzer),
            ):
                self.assertEqual(probe.run_controller(args), 0)

            session = pathlib.Path((session_root / "LATEST.txt").read_text(encoding="utf-8"))
            status = json.loads((session / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["phase"], "complete")
            self.assertEqual(status["stop_reason"], "arm_deadline")
            self.assertLessEqual(status["arm_duration_seconds"], 0.10)
            self.assertTrue(status["capture_guard_valid"])
            self.assertEqual(status["capture_guard_violations"], [])
            self.assertTrue(fake_process.terminated)
            self.assertFalse(fake_process.killed)
            start_index = next(
                index
                for index, command in enumerate(fake_pipe.commands)
                if command.startswith("START_CAPTURE ")
            )
            self.assertLess(fake_pipe.commands.index("SET_MAX_EVENTS 10"), start_index)
            self.assertLess(start_index, fake_pipe.commands.index("ARM"))
            self.assertEqual(fake_pipe.commands.count("DISARM"), 1)
            self.assertGreaterEqual(fake_pipe.commands.count("FLUSH"), 2)
            self.assertEqual(fake_pipe.commands.count("STOP_CAPTURE"), 1)


if __name__ == "__main__":
    unittest.main()
