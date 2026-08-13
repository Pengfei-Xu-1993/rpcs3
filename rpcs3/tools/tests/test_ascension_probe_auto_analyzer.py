#!/usr/bin/env python3
"""Synthetic contract tests for the compact Ascension probe analyzer.

These tests deliberately construct dataclasses in memory.  They must never open a
real capture: that keeps raw probe data out of test output and model context.
"""

from __future__ import annotations

import collections
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import ascension_live_probe as probe  # noqa: E402
import ascension_probe_auto_analyzer as compact  # noqa: E402


SECRET_SAMPLE = b"DO-NOT-LEAK-RAW!"
SECRET_TOKEN_BASE = 0x7E5700000000


def _pointer(token: int | None, rule_id: int = 1) -> tuple[probe.PointerResult, ...]:
    if token is None:
        return ()
    return (
        probe.PointerResult(
            rule_id=rule_id,
            flags=1,
            source=0x100000,
            address=token,
            content_hash=token ^ 0xABCDEF,
            requested_size=len(SECRET_SAMPLE),
            captured_size=len(SECRET_SAMPLE),
            sample=SECRET_SAMPLE,
        ),
    )


def _spu(
    *,
    sequence: int,
    frame: int,
    instance: int,
    token: int | None,
    arena_base: int,
) -> probe.SPUEvent:
    return probe.SPUEvent(
        sequence=sequence,
        time_us=1_000_000 + frame * 10_000 + instance * 100,
        frame=frame,
        thread=instance,
        pc=0x928C,
        target=0xC490,
        registers={},
        task_header_lsa=0x10650,
        task_context_lsa=0x10700,
        dma_descriptor_lsa=0x10800,
        task_sequence=frame * 10 + instance,
        task_format=0x871C0C00,
        packed_count=256,
        task_header_word_04=0x300000 + frame * 0x100,
        auxiliary=0,
        # Deliberately identical: it is a mesh family, not an instance token.
        source0=0x100000,
        source1=0,
        output_start=arena_base + instance * 0x4000,
        output_end=arena_base + instance * 0x4000 + 0x2000,
        pointers=_pointer(token),
    )


def _rsx(event: probe.SPUEvent, sequence: int) -> probe.RSXEvent:
    return probe.RSXEvent(
        sequence=sequence,
        time_us=event.time_us + 50,
        frame=event.frame,
        producer_sequence=event.sequence,
        draw_sequence=sequence,
        vp=11,
        fp=22,
        vertex_count=512,
        stream_vertex_count=512,
        first_vertex=0,
        stream_address=event.output_start,
        stream_size=0x2000,
        index_address=0x900000,
        index_count=768,
        attribute_mask=0xC3B5,
        stride=32,
        primitive=5,
        command=0,
        index_type=0,
        index_hash=0x1111,
        layout_hash=0x2222,
        task_sequence=event.task_sequence,
        task_format=event.task_format,
        task_source0=event.source0,
        output_relative=0,
        overlap=0x2000,
    )


def make_capture(
    *,
    frames: int = 6,
    missing_after: int | None = None,
    swap_tokens: bool = False,
) -> probe.Capture:
    spu: list[probe.SPUEvent] = []
    rsx: list[probe.RSXEvent] = []
    sequence = 1
    for frame in range(1, frames + 1):
        # Rotate the output arena every frame.  A useful identity token must
        # survive this, while an absolute output address must not.
        arena_base = 0x800000 + frame * 0x20000
        for instance in range(2):
            token_instance = (1 - instance) if swap_tokens and frame % 2 == 0 else instance
            token = SECRET_TOKEN_BASE + token_instance * 0x1000
            if missing_after is not None and frame > missing_after:
                token = None
            event = _spu(
                sequence=sequence,
                frame=frame,
                instance=instance,
                token=token,
                arena_base=arena_base,
            )
            spu.append(event)
            sequence += 1
            rsx.append(_rsx(event, sequence))
            sequence += 1
    capture = probe.Capture(
        header={
            "title_id": "SYNTHETIC",
            "app_version": "00.00",
            "written_events": len(spu) + len(rsx),
            "dropped_events": 0,
        },
        ppu=[],
        spu=spu,
        rsx=rsx,
        snapshots=[],
    )
    probe._decorate_events(capture)
    return capture


def _candidate(analysis: dict[str, object], name: str) -> dict[str, object]:
    return next(item for item in analysis["candidates"] if item["candidate"] == name)


def _walk_keys(value: object):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


class MetricTests(unittest.TestCase):
    def test_stable_instance_token_has_full_pairing_without_collision(self) -> None:
        analysis = probe.analyze_capture(make_capture())
        candidate = _candidate(analysis, "pointer[1].address")
        self.assertEqual(candidate["cross_frame_stability"], 1.0)
        self.assertEqual(candidate["same_mesh_collision"], 0.0)
        self.assertEqual(candidate["pair_rate"], 1.0)
        self.assertEqual(candidate["buffer_rotation_stability"], 1.0)
        self.assertEqual(candidate["rsx_mapping_coverage"], 1.0)

    def test_constant_mesh_token_is_collision_prone_and_ambiguous(self) -> None:
        analysis = probe.analyze_capture(make_capture())
        candidate = _candidate(analysis, "source0.absolute")
        self.assertEqual(candidate["cross_frame_stability"], 1.0)
        self.assertEqual(candidate["same_mesh_collision"], 0.5)
        self.assertEqual(candidate["pair_rate"], 0.0)
        self.assertGreater(candidate["ambiguous"], 0)

    def test_frame_relative_normalization_survives_arena_rotation(self) -> None:
        analysis = probe.analyze_capture(make_capture())
        absolute = _candidate(analysis, "output.absolute")
        normalized = _candidate(analysis, "output.frame_relative")
        self.assertEqual(absolute["cross_frame_stability"], 0.0)
        self.assertEqual(normalized["cross_frame_stability"], 1.0)
        self.assertEqual(normalized["same_mesh_collision"], 0.0)
        self.assertEqual(normalized["pair_rate"], 1.0)

    def test_missing_rate_is_computed_from_all_task_samples(self) -> None:
        analysis = probe.analyze_capture(make_capture(frames=6, missing_after=3))
        candidate = _candidate(analysis, "pointer[1].address")
        self.assertEqual(candidate["missing_rate"], 0.5)
        self.assertEqual(candidate["samples"], 6)


class CompactOutputContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.capture = make_capture(frames=8)

    def test_top_n_and_required_aggregate_fields(self) -> None:
        result = compact.analyze_capture_compact(self.capture, top_n=3, anomaly_limit=2)
        self.assertEqual(result["schema_version"], 1)
        self.assertLessEqual(len(result["top_candidates"]), 3)
        self.assertIn("generated", result["filters"])
        self.assertIn("accepted", result["filters"])
        self.assertIn("rejected", result["filters"])
        required = {
            "candidate",
            "category",
            "score",
            "decision",
            "reject_reasons",
            "samples",
            "availability",
            "cross_frame_stability",
            "same_mesh_collision",
            "pair_rate",
            "output_mapping",
            "arena_rotation_stability",
            "arena_token_survival",
            "unique_values",
            "eligible_adjacent_pairs",
            "eligible_collision_groups",
        }
        for candidate in result["top_candidates"]:
            self.assertTrue(required.issubset(candidate), candidate)

    def test_compact_analysis_is_a_pure_in_memory_operation(self) -> None:
        with mock.patch("builtins.open", side_effect=AssertionError("compact analysis attempted file I/O")):
            result = compact.analyze_capture_compact(self.capture, top_n=3, anomaly_limit=1)
        self.assertEqual(result["schema_version"], 1)

    def test_hard_rejects_constant_and_low_support_tokens(self) -> None:
        result = compact.analyze_capture_compact(self.capture, top_n=1000, anomaly_limit=2)
        by_name = {item["candidate"]: item for item in result["top_candidates"]}
        stable = by_name["pointer[1].address"]
        constant = by_name["source0.absolute"]
        self.assertEqual(stable["decision"], "accept")
        self.assertEqual(constant["decision"], "reject")
        self.assertTrue(constant["reject_reasons"])

        low_support = compact.analyze_capture_compact(
            make_capture(frames=2), top_n=1000, anomaly_limit=2
        )
        low_by_name = {item["candidate"]: item for item in low_support["top_candidates"]}
        self.assertEqual(low_by_name["pointer[1].address"]["decision"], "reject")
        self.assertTrue(low_by_name["pointer[1].address"]["reject_reasons"])

    def test_rotation_metric_distinguishes_identity_from_absolute_buffer(self) -> None:
        result = compact.analyze_capture_compact(self.capture, top_n=1000, anomaly_limit=2)
        by_name = {item["candidate"]: item for item in result["top_candidates"]}
        identity = by_name["pointer[1].address"]
        absolute_buffer = by_name["output.absolute"]
        self.assertIsNone(identity["arena_rotation_stability"])
        self.assertIsNone(absolute_buffer["arena_rotation_stability"])
        self.assertEqual(identity["arena_token_survival"], 1.0)
        self.assertEqual(absolute_buffer["arena_token_survival"], 0.0)
        self.assertEqual(absolute_buffer["decision"], "reject")

    def test_output_mapping_is_not_evaluated_without_rsx_draws(self) -> None:
        capture = make_capture(frames=8)
        capture.rsx.clear()
        result = compact.analyze_capture_compact(capture, top_n=1000, anomaly_limit=2)
        by_name = {item["candidate"]: item for item in result["top_candidates"]}
        self.assertIsNone(by_name["pointer[1].address"]["output_mapping"])

    def test_token_survival_does_not_claim_object_ground_truth(self) -> None:
        result = compact.analyze_capture_compact(
            make_capture(frames=8, swap_tokens=True), top_n=1000, anomaly_limit=2
        )
        by_name = {item["candidate"]: item for item in result["top_candidates"]}
        swapped = by_name["pointer[1].address"]
        # The token multiset is perfectly stable, but it alternates between the
        # two output-relative slots.  Without an independent object anchor the
        # analyzer may report survival, but must not fabricate object stability.
        self.assertEqual(swapped["cross_frame_stability"], 1.0)
        self.assertEqual(swapped["pair_rate"], 1.0)
        self.assertEqual(swapped["arena_token_survival"], 1.0)
        self.assertIsNone(swapped["arena_rotation_stability"])

    def test_compact_result_never_contains_raw_events_or_memory_samples(self) -> None:
        result = compact.analyze_capture_compact(self.capture, top_n=10, anomaly_limit=2)
        forbidden_keys = {
            "event",
            "events",
            "ppu",
            "spu",
            "rsx",
            "registers",
            "pointers",
            "stack",
            "sample",
            "raw",
            "memory",
        }
        self.assertFalse(forbidden_keys.intersection(_walk_keys(result)))
        encoded = json.dumps(result, ensure_ascii=False, sort_keys=True)
        self.assertNotIn(SECRET_SAMPLE.hex(), encoded)
        self.assertNotIn(SECRET_SAMPLE.decode("ascii"), encoded)
        self.assertNotIn(str(SECRET_TOKEN_BASE), encoded)
        self.assertNotIn(f"0x{SECRET_TOKEN_BASE:x}", encoded.lower())

    def test_withdrawn_header_word_is_not_an_identity_candidate(self) -> None:
        result = compact.analyze_capture_compact(
            self.capture, top_n=1000, anomaly_limit=2
        )
        names = {item["candidate"] for item in result["top_candidates"]}
        self.assertFalse(
            any(
                "descriptor" in name or "task_header_word_04" in name
                for name in names
            ),
            names,
        )

    def test_anomalies_are_capped_per_kind(self) -> None:
        result = compact.analyze_capture_compact(self.capture, top_n=10, anomaly_limit=1)
        anomalies = result["anomalies"]
        self.assertTrue(anomalies, "synthetic collision/rotation cases should emit aggregate anomalies")
        if isinstance(anomalies, dict):
            counts = {kind: len(items) for kind, items in anomalies.items()}
        else:
            counts = collections.Counter(item["kind"] for item in anomalies)
        self.assertTrue(all(count <= 1 for count in counts.values()), counts)

    def test_model_summary_respects_line_budget_and_privacy_boundary(self) -> None:
        result = compact.analyze_capture_compact(self.capture, top_n=20, anomaly_limit=3)
        summary = compact.format_model_summary(result, max_lines=24)
        self.assertLessEqual(len(summary.splitlines()), 24)
        self.assertNotIn(SECRET_SAMPLE.hex(), summary)
        self.assertNotIn(SECRET_SAMPLE.decode("ascii"), summary)
        self.assertNotIn(str(SECRET_TOKEN_BASE), summary)
        self.assertNotIn(f"0x{SECRET_TOKEN_BASE:x}", summary.lower())
        self.assertNotIn("SPUEvent", summary)
        self.assertNotIn("PointerResult", summary)

    def test_output_writer_emits_only_bounded_aggregate_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = pathlib.Path(directory) / "analysis"
            compact.write_compact_outputs(
                self.capture,
                output_dir,
                top_n=3,
                anomaly_limit=1,
                max_summary_lines=12,
            )
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {
                    "model-summary.txt",
                    "top-candidates.json",
                    "anomalies.json",
                    "full-ranking.local-only.json",
                    "next-probe-plan.json",
                },
            )
            summary = (output_dir / "model-summary.txt").read_text(encoding="utf-8")
            model_facing = "\n".join(
                (output_dir / name).read_text(encoding="utf-8")
                for name in (
                    "model-summary.txt",
                    "top-candidates.json",
                    "anomalies.json",
                    "next-probe-plan.json",
                )
            )
            local_ranking = json.loads(
                (output_dir / "full-ranking.local-only.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertLessEqual(len(summary.splitlines()), 12)
        self.assertNotIn(SECRET_SAMPLE.hex(), model_facing)
        self.assertNotIn(SECRET_SAMPLE.decode("ascii"), model_facing)
        self.assertNotIn(str(SECRET_TOKEN_BASE), model_facing)
        self.assertEqual(
            local_ranking["warning"],
            "LOCAL_ONLY_AGGREGATES_DO_NOT_PASTE_WHOLE_FILE_INTO_MODEL",
        )


class LocalProbePlanTests(unittest.TestCase):
    def _result(self, candidate: str) -> dict[str, object]:
        return {
            "top_candidates": [
                {
                    "candidate": candidate,
                    "decision": "accept",
                }
            ]
        }

    def test_recorded_rule_generates_exact_one_level_deeper_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = pathlib.Path(directory) / "probe-config.json"
            config.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "effective_commands": [
                            "ADD_POINTER_FOLLOW ID=1 SOURCE=SOURCE0 SPACE=GUEST DEPTH=0 OFFSET0=0 SIZE=64"
                        ],
                    }
                ),
                encoding="utf-8",
            )
            plan = compact.build_next_probe_plan(
                self._result("pointer[1].be32+0x4"), config
            )
        self.assertEqual(
            plan["generated_commands"],
            [
                "ADD_POINTER_FOLLOW ID=2 SOURCE=SOURCE0 SPACE=GUEST DEPTH=1 "
                "OFFSET0=0x4 OFFSET1=0x0 SIZE=64"
            ],
        )
        self.assertFalse(plan["pointer_follow"][0]["requires_recorded_rule_definition"])

    def test_controller_config_round_trips_effective_rule_history(self) -> None:
        commands = [
            "ADD_POINTER_FOLLOW ID=1 SOURCE=SOURCE0 SPACE=GUEST DEPTH=0 OFFSET0=0 SIZE=64",
            "ADD_POINTER_FOLLOW ID=2 SOURCE=PPU_R7 SPACE=GUEST DEPTH=0 OFFSET0=0 SIZE=64",
            "REMOVE_POINTER_FOLLOW 1",
        ]
        with tempfile.TemporaryDirectory() as directory:
            config = pathlib.Path(directory) / "probe-config.json"
            probe._write_probe_config(config, commands)
            rules = compact._parse_pointer_rules(config)
            self.assertFalse(config.with_suffix(".tmp").exists())
        self.assertEqual(set(rules), {2})
        self.assertEqual(rules[2]["SOURCE"], "PPU_R7")

    def test_removed_rule_is_not_replayed_as_active(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = pathlib.Path(directory) / "probe-config.json"
            config.write_text(
                json.dumps(
                    {
                        "effective_commands": [
                            "ADD_POINTER_FOLLOW ID=1 SOURCE=SOURCE0 SPACE=GUEST DEPTH=0 OFFSET0=0 SIZE=64",
                            "REMOVE_POINTER_FOLLOW 1",
                        ]
                    }
                ),
                encoding="utf-8",
            )
            plan = compact.build_next_probe_plan(
                self._result("pointer[1].be32+0x4"), config
            )
        self.assertEqual(plan["generated_commands"], [])
        self.assertTrue(plan["pointer_follow"][0]["requires_recorded_rule_definition"])

    def test_full_rule_table_and_ls_rule_never_generate_a_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = pathlib.Path(directory) / "probe-config.json"
            commands = [
                f"ADD_POINTER_FOLLOW ID={rule_id} SOURCE=SOURCE0 SPACE=GUEST DEPTH=0 OFFSET0=0 SIZE=64"
                for rule_id in range(1, 9)
            ]
            config.write_text(json.dumps({"effective_commands": commands}), encoding="utf-8")
            full = compact.build_next_probe_plan(
                self._result("pointer[1].be32+0x4"), config
            )
            self.assertEqual(full["generated_commands"], [])
            self.assertTrue(full["pointer_follow"][0]["requires_free_rule_slot"])

            config.write_text(
                json.dumps(
                    {
                        "effective_commands": [
                            "ADD_POINTER_FOLLOW ID=1 SOURCE=TASK_CONTEXT SPACE=LS DEPTH=0 OFFSET0=0 SIZE=64"
                        ]
                    }
                ),
                encoding="utf-8",
            )
            local_store = compact.build_next_probe_plan(
                self._result("pointer[1].be32+0x4"), config
            )
        self.assertEqual(local_store["generated_commands"], [])
        self.assertTrue(
            local_store["pointer_follow"][0]["requires_address_space_confirmation"]
        )

    def test_unrecorded_or_non_be32_field_is_never_guessed(self) -> None:
        missing = compact.build_next_probe_plan(
            self._result("pointer[7].be32+0x0"), None
        )
        self.assertEqual(missing["generated_commands"], [])
        self.assertTrue(missing["pointer_follow"][0]["requires_recorded_rule_definition"])

        with tempfile.TemporaryDirectory() as directory:
            config = pathlib.Path(directory) / "probe-config.json"
            config.write_text(
                json.dumps(
                    {
                        "effective_commands": [
                            "ADD_POINTER_FOLLOW ID=1 SOURCE=SOURCE0 SPACE=GUEST DEPTH=0 OFFSET0=0 SIZE=64"
                        ]
                    }
                ),
                encoding="utf-8",
            )
            little_endian = compact.build_next_probe_plan(
                self._result("pointer[1].le32+0x4"), config
            )
        self.assertEqual(little_endian["generated_commands"], [])

    def test_invalid_recorded_source_is_not_propagated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = pathlib.Path(directory) / "probe-config.json"
            config.write_text(
                json.dumps(
                    {
                        "effective_commands": [
                            "ADD_POINTER_FOLLOW ID=1 SOURCE=NOT_A_SOURCE SPACE=GUEST DEPTH=0 OFFSET0=0 SIZE=64"
                        ]
                    }
                ),
                encoding="utf-8",
            )
            plan = compact.build_next_probe_plan(
                self._result("pointer[1].be32+0x4"), config
            )
        self.assertEqual(plan["generated_commands"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
