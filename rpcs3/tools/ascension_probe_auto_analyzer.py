#!/usr/bin/env python3
"""Local, compact candidate search for Ascension Live Probe captures.

The analyzer intentionally keeps raw events, registers, pointer samples, and
memory bytes on the local machine.  Its model-facing outputs contain aggregate
metrics only: a bounded summary, Top-N candidates, and capped anomaly classes.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import dataclasses
import json
import math
import pathlib
import re
import shutil
import tempfile
from typing import Callable, Hashable

import ascension_live_probe as probe


MIN_SAMPLES = 8
MIN_ADJACENT_PAIRS = 3
MIN_AVAILABILITY = 0.90
MAX_COLLISION = 0.05
MIN_STABILITY = 0.95
MIN_PAIR_RATE = 0.90
MIN_ARENA_SURVIVAL = 0.90
MAX_PPU_CALLSITE_AGGREGATES = 8
PPU_JOIN_WINDOW_US = 50_000
PPU_POINTER_CANDIDATE = re.compile(r"^nearest_ppu\.pointer\[(\d+)]\.")
PPU_REGISTER_CANDIDATE = re.compile(r"^nearest_ppu\.r(\d+)(?:$|\.)")
MFC_PROVENANCE_FIELDS = (
    ("task_header", 1, "task_header_guest_ea", "task_header_mfc_age"),
    ("task_context", 2, "task_context_guest_ea", "task_context_mfc_age"),
    ("dma_descriptor", 4, "dma_descriptor_guest_ea", "dma_descriptor_mfc_age"),
)


@dataclasses.dataclass(frozen=True, slots=True)
class CandidateSpec:
    name: str
    category: str
    extractor: Callable[[probe.SPUEvent], Hashable | None]
    complexity_penalty: float = 1.0


def _pointer(
    event: probe.SPUEvent | probe.PPUEvent, rule_id: int
) -> probe.PointerResult | None:
    return next(
        (item for item in event.pointers if item.rule_id == rule_id and item.flags),
        None,
    )


def _nearest_ppu_pointer(
    event: probe.SPUEvent, rule_id: int
) -> probe.PointerResult | None:
    return _pointer(event.nearest_ppu, rule_id) if event.nearest_ppu else None


def _nonzero(value: int) -> int | None:
    return value or None


def _candidate_specs(capture: probe.Capture) -> list[CandidateSpec]:
    """Generate candidates from every field already present in the capture.

    This function does not dereference memory.  It evaluates all captured
    registers, normalized addresses, relative addresses, pointer target hashes,
    and every aligned integer in the pointer rule's bounded 16-byte preview.
    """

    specs: dict[str, CandidateSpec] = {}

    def add(
        name: str,
        category: str,
        extractor: Callable[[probe.SPUEvent], Hashable | None],
        penalty: float = 1.0,
    ) -> None:
        specs.setdefault(name, CandidateSpec(name, category, extractor, penalty))

    def add_address(
        prefix: str,
        getter: Callable[[probe.SPUEvent], int],
        category: str = "address",
    ) -> None:
        add(f"{prefix}.absolute", category, lambda e, get=getter: _nonzero(get(e)))
        add(f"{prefix}.aligned16", category, lambda e, get=getter: (get(e) & ~0xF) if get(e) else None, 0.99)
        add(f"{prefix}.page", category, lambda e, get=getter: (get(e) >> 12) if get(e) else None, 0.98)
        add(f"{prefix}.page_offset", category, lambda e, get=getter: (get(e) & 0xFFF) if get(e) else None, 0.98)

    add_address("source0", lambda e: e.source0)
    add_address("source1", lambda e: e.source1)
    add_address("auxiliary", lambda e: e.auxiliary)
    add_address("output", lambda e: e.output_start, "output_address")
    add_address(
        "mfc.task_header",
        lambda e: e.task_header_guest_ea if e.mfc_provenance_mask & 1 else 0,
        "mfc_provenance",
    )
    add_address(
        "mfc.task_context",
        lambda e: e.task_context_guest_ea if e.mfc_provenance_mask & 2 else 0,
        "mfc_provenance",
    )
    add_address(
        "mfc.dma_descriptor",
        lambda e: e.dma_descriptor_guest_ea if e.mfc_provenance_mask & 4 else 0,
        "mfc_provenance",
    )
    add("output.frame_relative", "output_derived", lambda e: e.output_frame_relative if e.output_start else None)
    add("output.size", "pipeline_metadata", lambda e: e.output_end - e.output_start if e.output_end > e.output_start else None)
    add("task.sequence", "sequence_metadata", lambda e: e.task_sequence)
    add("task.local_occurrence", "order_heuristic", lambda e: e.occurrence, 0.85)
    add("task.format", "pipeline_metadata", lambda e: e.task_format)
    add("task.packed_count", "pipeline_metadata", lambda e: e.packed_count)

    relative_fields = {
        "source1-source0": lambda e: (e.source1 - e.source0) & 0xFFFFFFFF if e.source1 and e.source0 else None,
        "auxiliary-source0": lambda e: (e.auxiliary - e.source0) & 0xFFFFFFFF if e.auxiliary and e.source0 else None,
        "output-source0": lambda e: (e.output_start - e.source0) & 0xFFFFFFFF if e.output_start and e.source0 else None,
    }
    for name, extractor in relative_fields.items():
        add(name, "relative_address", extractor, 0.97)

    spu_regs = sorted({reg for event in capture.spu for reg in event.registers})
    for reg in spu_regs:
        add(f"spu.r{reg}", "spu_register", lambda e, reg=reg: e.registers.get(reg))
        add(f"spu.r{reg}.low12", "spu_register_normalized", lambda e, reg=reg: (value & 0xFFF) if (value := e.registers.get(reg)) is not None else None, 0.98)
        add(f"spu.r{reg}.page", "spu_register_normalized", lambda e, reg=reg: (value >> 12) if (value := e.registers.get(reg)) else None, 0.98)

    ppu_regs = sorted({reg for event in capture.ppu for reg in event.registers})
    for reg in ppu_regs:
        add(
            f"nearest_ppu.r{reg}",
            "ppu_register",
            lambda e, reg=reg: e.nearest_ppu.registers.get(reg) if e.nearest_ppu else None,
            0.94,
        )
        add(
            f"nearest_ppu.r{reg}.low12",
            "ppu_register_normalized",
            lambda e, reg=reg: (value & 0xFFF) if e.nearest_ppu and (value := e.nearest_ppu.registers.get(reg)) is not None else None,
            0.92,
        )
        add(
            f"nearest_ppu.r{reg}.page",
            "ppu_register_normalized",
            lambda e, reg=reg: (value >> 12) if e.nearest_ppu and (value := e.nearest_ppu.registers.get(reg)) else None,
            0.92,
        )

    ppu_rule_ids = sorted(
        {
            item.rule_id
            for event in capture.ppu
            for item in event.pointers
            if item.flags
        }
    )
    for rule_id in ppu_rule_ids:
        add(
            f"nearest_ppu.pointer[{rule_id}].source",
            "ppu_pointer_source",
            lambda e, rule_id=rule_id: (
                item.source
                if (item := _nearest_ppu_pointer(e, rule_id))
                else None
            ),
            0.91,
        )
        add(
            f"nearest_ppu.pointer[{rule_id}].address",
            "ppu_pointer_address",
            lambda e, rule_id=rule_id: (
                item.address
                if (item := _nearest_ppu_pointer(e, rule_id))
                else None
            ),
            0.92,
        )
        add(
            f"nearest_ppu.pointer[{rule_id}].address.page",
            "ppu_pointer_address",
            lambda e, rule_id=rule_id: (
                item.address >> 12
                if (item := _nearest_ppu_pointer(e, rule_id)) and item.address
                else None
            ),
            0.90,
        )
        add(
            f"nearest_ppu.pointer[{rule_id}].address.low12",
            "ppu_pointer_address",
            lambda e, rule_id=rule_id: (
                item.address & 0xFFF
                if (item := _nearest_ppu_pointer(e, rule_id)) and item.address
                else None
            ),
            0.90,
        )
        add(
            f"nearest_ppu.pointer[{rule_id}].address-source",
            "ppu_pointer_address",
            lambda e, rule_id=rule_id: (
                (item.address - item.source) & 0xFFFFFFFF
                if (item := _nearest_ppu_pointer(e, rule_id))
                and item.address
                and item.source
                else None
            ),
            0.89,
        )
        add(
            f"nearest_ppu.pointer[{rule_id}].content_hash",
            "ppu_pointer_target_hash",
            lambda e, rule_id=rule_id: (
                item.content_hash
                if (item := _nearest_ppu_pointer(e, rule_id))
                else None
            ),
            0.88,
        )
        for offset in range(0, 16, 4):
            add(
                f"nearest_ppu.pointer[{rule_id}].be32+0x{offset:x}",
                "ppu_pointer_field",
                lambda e, rule_id=rule_id, offset=offset: (
                    int.from_bytes(item.sample[offset:offset + 4], "big")
                    if (item := _nearest_ppu_pointer(e, rule_id))
                    and item.captured_size >= offset + 4
                    else None
                ),
                0.87,
            )
            add(
                f"nearest_ppu.pointer[{rule_id}].le32+0x{offset:x}",
                "ppu_pointer_field",
                lambda e, rule_id=rule_id, offset=offset: (
                    int.from_bytes(item.sample[offset:offset + 4], "little")
                    if (item := _nearest_ppu_pointer(e, rule_id))
                    and item.captured_size >= offset + 4
                    else None
                ),
                0.85,
            )
        for offset in (0, 8):
            add(
                f"nearest_ppu.pointer[{rule_id}].be64+0x{offset:x}",
                "ppu_pointer_field",
                lambda e, rule_id=rule_id, offset=offset: (
                    int.from_bytes(item.sample[offset:offset + 8], "big")
                    if (item := _nearest_ppu_pointer(e, rule_id))
                    and item.captured_size >= offset + 8
                    else None
                ),
                0.84,
            )
            add(
                f"nearest_ppu.pointer[{rule_id}].le64+0x{offset:x}",
                "ppu_pointer_field",
                lambda e, rule_id=rule_id, offset=offset: (
                    int.from_bytes(item.sample[offset:offset + 8], "little")
                    if (item := _nearest_ppu_pointer(e, rule_id))
                    and item.captured_size >= offset + 8
                    else None
                ),
                0.83,
            )

    rule_ids = sorted(
        {
            item.rule_id
            for event in capture.spu
            for item in event.pointers
            if item.flags
        }
    )
    for rule_id in rule_ids:
        add(
            f"pointer[{rule_id}].source",
            "pointer_source",
            lambda e, rule_id=rule_id: (item.source if (item := _pointer(e, rule_id)) else None),
            0.97,
        )
        add(
            f"pointer[{rule_id}].address",
            "pointer_address",
            lambda e, rule_id=rule_id: (item.address if (item := _pointer(e, rule_id)) else None),
            0.98,
        )
        add(
            f"pointer[{rule_id}].address.page",
            "pointer_address",
            lambda e, rule_id=rule_id: (item.address >> 12 if (item := _pointer(e, rule_id)) and item.address else None),
            0.96,
        )
        add(
            f"pointer[{rule_id}].address.low12",
            "pointer_address",
            lambda e, rule_id=rule_id: (item.address & 0xFFF if (item := _pointer(e, rule_id)) and item.address else None),
            0.96,
        )
        add(
            f"pointer[{rule_id}].address-source",
            "relative_address",
            lambda e, rule_id=rule_id: ((item.address - item.source) & 0xFFFFFFFF if (item := _pointer(e, rule_id)) and item.address and item.source else None),
            0.95,
        )
        add(
            f"pointer[{rule_id}].content_hash",
            "pointer_target_hash",
            lambda e, rule_id=rule_id: (item.content_hash if (item := _pointer(e, rule_id)) else None),
            0.94,
        )
        for offset in range(0, 16, 4):
            add(
                f"pointer[{rule_id}].be32+0x{offset:x}",
                "pointer_field",
                lambda e, rule_id=rule_id, offset=offset: (
                    int.from_bytes(item.sample[offset:offset + 4], "big")
                    if (item := _pointer(e, rule_id)) and item.captured_size >= offset + 4 else None
                ),
                0.91,
            )
            add(
                f"pointer[{rule_id}].le32+0x{offset:x}",
                "pointer_field",
                lambda e, rule_id=rule_id, offset=offset: (
                    int.from_bytes(item.sample[offset:offset + 4], "little")
                    if (item := _pointer(e, rule_id)) and item.captured_size >= offset + 4 else None
                ),
                0.89,
            )
        for offset in (0, 8):
            add(
                f"pointer[{rule_id}].be64+0x{offset:x}",
                "pointer_field",
                lambda e, rule_id=rule_id, offset=offset: (
                    int.from_bytes(item.sample[offset:offset + 8], "big")
                    if (item := _pointer(e, rule_id)) and item.captured_size >= offset + 8 else None
                ),
                0.88,
            )
            add(
                f"pointer[{rule_id}].le64+0x{offset:x}",
                "pointer_field",
                lambda e, rule_id=rule_id, offset=offset: (
                    int.from_bytes(item.sample[offset:offset + 8], "little")
                    if (item := _pointer(e, rule_id)) and item.captured_size >= offset + 8 else None
                ),
                0.87,
            )
    return list(specs.values())


def _draw_signature(event: probe.RSXEvent) -> tuple[int, ...]:
    return (
        event.vp,
        event.fp,
        event.layout_hash,
        event.attribute_mask,
        event.stride,
        event.primitive,
        event.index_hash,
        event.vertex_count,
    )


def _families(capture: probe.Capture) -> tuple[dict[int, Hashable], set[int]]:
    by_producer: dict[int, list[probe.RSXEvent]] = collections.defaultdict(list)
    for draw in capture.rsx:
        if draw.producer_sequence:
            by_producer[draw.producer_sequence].append(draw)
    result: dict[int, Hashable] = {}
    for event in capture.spu:
        draws = by_producer.get(event.sequence)
        if draws:
            result[event.sequence] = ("rsx", tuple(sorted({_draw_signature(draw) for draw in draws})))
        else:
            output_size = max(0, event.output_end - event.output_start)
            result[event.sequence] = ("spu", event.task_format, event.packed_count, output_size)
    return result, set(by_producer)


def _frame_arenas(capture: probe.Capture) -> dict[int, int]:
    arenas: dict[int, int] = {}
    for event in capture.spu:
        if event.output_start:
            arenas[event.frame] = min(arenas.get(event.frame, event.output_start), event.output_start)
    return arenas


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def _score_spec(
    capture: probe.Capture,
    spec: CandidateSpec,
    family_by_sequence: dict[int, Hashable],
    mapped_sequences: set[int],
    frame_arenas: dict[int, int],
) -> dict[str, object]:
    grouped: dict[Hashable, dict[int, list[tuple[Hashable, probe.SPUEvent]]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    values: set[Hashable] = set()
    missing = 0
    mapped = 0
    for event in capture.spu:
        token = spec.extractor(event)
        if token is None:
            missing += 1
            continue
        values.add(token)
        grouped[family_by_sequence[event.sequence]][event.frame].append((token, event))
        mapped += event.sequence in mapped_sequences

    collision_duplicates = 0
    collision_population = 0
    eligible_collision_groups = 0
    eligible_collision_tasks = 0
    adjacent_matches = 0
    adjacent_population = 0
    eligible_adjacent_pairs = 0
    paired = 0
    ambiguous = 0
    unmatched = 0
    arena_matches = 0
    arena_population = 0
    eligible_arena_pairs = 0
    max_concurrency = 0

    for frames in grouped.values():
        ordered = sorted(frames)
        for frame in ordered:
            tokens = [token for token, _ in frames[frame]]
            max_concurrency = max(max_concurrency, len(tokens))
            if len(tokens) >= 2:
                eligible_collision_groups += 1
                eligible_collision_tasks += len(tokens)
                collision_duplicates += len(tokens) - len(set(tokens))
                collision_population += len(tokens)
        for left_frame, right_frame in zip(ordered, ordered[1:]):
            if right_frame - left_frame > 2:
                continue
            left = frames[left_frame]
            right = frames[right_frame]
            left_counts = collections.Counter(token for token, _ in left)
            right_counts = collections.Counter(token for token, _ in right)
            population = max(len(left), len(right))
            intersection = sum((left_counts & right_counts).values())
            adjacent_matches += intersection
            adjacent_population += population
            eligible_adjacent_pairs += 1

            for token in left_counts.keys() | right_counts.keys():
                left_count = left_counts[token]
                right_count = right_counts[token]
                if left_count == 1 and right_count == 1:
                    paired += 1
                elif left_count and right_count:
                    ambiguous += max(left_count, right_count)
                else:
                    unmatched += max(left_count, right_count)

            left_arena = frame_arenas.get(left_frame)
            right_arena = frame_arenas.get(right_frame)
            if left_arena is not None and right_arena is not None and left_arena != right_arena:
                arena_matches += intersection
                arena_population += population
                eligible_arena_pairs += 1

    total = len(capture.spu)
    samples = total - missing
    availability = samples / total if total else 0.0
    stability = _ratio(adjacent_matches, adjacent_population)
    collision = _ratio(collision_duplicates, collision_population)
    pair_rate = _ratio(paired, paired + ambiguous + unmatched)
    output_mapping = _ratio(mapped, samples) if capture.rsx else None
    arena_survival = _ratio(arena_matches, arena_population)

    reject_reasons: list[str] = []
    if samples < MIN_SAMPLES:
        reject_reasons.append("insufficient_samples")
    if eligible_adjacent_pairs < MIN_ADJACENT_PAIRS:
        reject_reasons.append("insufficient_adjacent_frames")
    if availability < MIN_AVAILABILITY:
        reject_reasons.append("low_availability")
    if len(values) <= 1:
        reject_reasons.append("constant_token")
    if collision is not None and eligible_collision_groups and collision > MAX_COLLISION:
        reject_reasons.append("simultaneous_collision")
    if stability is not None and stability < MIN_STABILITY:
        reject_reasons.append("low_cross_frame_stability")
    if pair_rate is not None and pair_rate < MIN_PAIR_RATE:
        reject_reasons.append("ambiguous_pairing")
    if arena_survival is not None and arena_survival < MIN_ARENA_SURVIVAL:
        reject_reasons.append("arena_dependent")
    if spec.category == "order_heuristic":
        reject_reasons.append("submission_order_only")
    if spec.category == "output_derived":
        reject_reasons.append("output_slot_only")
    if spec.category == "output_address":
        reject_reasons.append("output_buffer_address")
    if spec.category == "pipeline_metadata":
        reject_reasons.append("pipeline_metadata_only")
    if spec.category == "sequence_metadata":
        reject_reasons.append("sequence_counter_only")
    if spec.category.startswith("ppu_"):
        reject_reasons.append("weak_temporal_join")

    identity_factors = [
        max(stability or 0.0, 1e-6),
        max(1.0 - (collision or 0.0), 1e-6),
        max(pair_rate or 0.0, 1e-6),
    ]
    identity_score = math.prod(identity_factors) ** (1.0 / len(identity_factors))
    evidence_factors = [max(availability, 1e-6)]
    if output_mapping is not None:
        evidence_factors.append(max(output_mapping, 1e-6))
    support_factor = min(1.0, samples / 32.0) * min(1.0, eligible_adjacent_pairs / 7.0)
    if max_concurrency >= 2:
        support_factor *= 0.5 + 0.5 * min(1.0, eligible_collision_groups / 8.0)
    evidence_factors.append(max(support_factor, 1e-6))
    evidence_quality = math.prod(evidence_factors) ** (1.0 / len(evidence_factors))
    score = identity_score * evidence_quality * spec.complexity_penalty
    if reject_reasons:
        score *= 0.20

    confidence = "high" if samples >= 64 and eligible_adjacent_pairs >= 16 else "medium" if samples >= 16 and eligible_adjacent_pairs >= 7 else "low"
    return {
        "candidate": spec.name,
        "category": spec.category,
        "score": round(score, 6),
        "identity_score": round(identity_score, 6),
        "evidence_quality": round(evidence_quality, 6),
        "decision": "reject" if reject_reasons else "accept",
        "reject_reasons": reject_reasons,
        "confidence": confidence,
        "samples": samples,
        "availability": _round(availability),
        "cross_frame_stability": _round(stability),
        "same_mesh_collision": _round(collision),
        "pair_rate": _round(pair_rate),
        "output_mapping": _round(output_mapping),
        # No independent game-object anchor exists in the capture.  Reporting a
        # number here would fabricate ground truth from the candidate itself.
        "arena_rotation_stability": None,
        "arena_token_survival": _round(arena_survival),
        "unique_values": len(values),
        "eligible_adjacent_pairs": eligible_adjacent_pairs,
        "eligible_collision_groups": eligible_collision_groups,
        "eligible_collision_tasks": eligible_collision_tasks,
        "eligible_arena_pairs": eligible_arena_pairs,
        "paired": paired,
        "ambiguous": ambiguous,
        "unmatched": unmatched,
        "max_concurrency": max_concurrency,
        "support_factor": round(support_factor, 6),
    }


def _rank_capture(capture: probe.Capture) -> list[dict[str, object]]:
    family_by_sequence, mapped_sequences = _families(capture)
    arenas = _frame_arenas(capture)
    ranking = [
        _score_spec(capture, spec, family_by_sequence, mapped_sequences, arenas)
        for spec in _candidate_specs(capture)
    ]
    semantic_reasons = {
        "submission_order_only",
        "output_slot_only",
        "output_buffer_address",
        "pipeline_metadata_only",
        "sequence_counter_only",
        "weak_temporal_join",
    }
    ranking.sort(
        key=lambda item: (
            item["decision"] == "accept",
            not bool(semantic_reasons.intersection(item["reject_reasons"])),
            -len(item["reject_reasons"]),
            item["score"],
            item["pair_rate"] if item["pair_rate"] is not None else -1.0,
            -(item["same_mesh_collision"] if item["same_mesh_collision"] is not None else 1.0),
        ),
        reverse=True,
    )
    return ranking


def _aggregate_anomalies(
    ranking: list[dict[str, object]], anomaly_limit: int
) -> list[dict[str, object]]:
    by_kind: dict[str, list[dict[str, object]]] = collections.defaultdict(list)

    def add(kind: str, item: dict[str, object], summary: str) -> None:
        if len(by_kind[kind]) >= anomaly_limit:
            return
        by_kind[kind].append(
            {
                "kind": kind,
                "candidate": item["candidate"],
                "count": int(item.get("samples", 0)),
                "summary": summary,
            }
        )

    for item in ranking:
        reasons = set(item["reject_reasons"])
        if "simultaneous_collision" in reasons:
            add("collision", item, "Duplicate tokens occur among simultaneous tasks in the same draw family.")
        if "insufficient_samples" in reasons or "insufficient_adjacent_frames" in reasons:
            add("low_support", item, "There is not enough adjacent-frame evidence for a reliable identity claim.")
        if "arena_dependent" in reasons:
            add("arena_dependency", item, "The token does not survive output-arena address rotation.")
        if "low_availability" in reasons:
            add("missing_data", item, "The candidate is absent from too much of the capture.")
        if "ambiguous_pairing" in reasons:
            add("ambiguous_pairing", item, "Token reuse prevents unambiguous adjacent-frame pairing.")
        if "constant_token" in reasons:
            add("constant_token", item, "A constant value cannot distinguish simultaneous instances.")
    return [entry for kind in sorted(by_kind) for entry in by_kind[kind]]


def _nearest_rank(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def _ppu_join_diagnostics(
    capture: probe.Capture, ranking: list[dict[str, object]]
) -> dict[str, object]:
    """Summarize temporal-join quality without exposing event payloads."""

    ppu_times = sorted(event.time_us for event in capture.ppu)
    deltas: list[int] = []
    window_counts: list[int] = []
    nearest_sequences: list[int] = []
    for event in capture.spu:
        right = bisect.bisect_right(ppu_times, event.time_us)
        left = bisect.bisect_left(
            ppu_times, event.time_us - PPU_JOIN_WINDOW_US, 0, right
        )
        window_counts.append(right - left)
        if event.nearest_ppu is not None:
            deltas.append(event.time_us - event.nearest_ppu.time_us)
            nearest_sequences.append(event.nearest_ppu.sequence)

    reuse = collections.Counter(nearest_sequences)
    best_candidate = next(
        (
            item
            for item in ranking
            if str(item["category"]).startswith("ppu_")
        ),
        None,
    )
    best_pointer_by_rule: dict[int, dict[str, object]] = {}
    for item in ranking:
        match = PPU_POINTER_CANDIDATE.match(str(item["candidate"]))
        if match:
            best_pointer_by_rule.setdefault(int(match.group(1)), item)
    best_pointer_candidates = [
        best_pointer_by_rule[rule_id] for rule_id in sorted(best_pointer_by_rule)
    ]
    best_argument_by_register: dict[int, dict[str, object]] = {}
    for item in ranking:
        match = PPU_REGISTER_CANDIDATE.match(str(item["candidate"]))
        if match:
            register = int(match.group(1))
            if 3 <= register <= 10:
                best_argument_by_register.setdefault(register, item)
    best_argument_candidates = [
        best_argument_by_register[register]
        for register in sorted(best_argument_by_register)
    ]
    return {
        "window_us": PPU_JOIN_WINDOW_US,
        "spu_with_nearest_ppu": len(deltas),
        "coverage": (len(deltas) / len(capture.spu)) if capture.spu else None,
        "delta_us_p50": _nearest_rank(deltas, 0.50),
        "delta_us_p95": _nearest_rank(deltas, 0.95),
        "delta_us_max": max(deltas) if deltas else None,
        "preceding_ppu_count_p50": _nearest_rank(window_counts, 0.50),
        "preceding_ppu_count_p95": _nearest_rank(window_counts, 0.95),
        "preceding_ppu_count_max": max(window_counts) if window_counts else None,
        "distinct_nearest_ppu": len(reuse),
        "maximum_spu_reuse": max(reuse.values()) if reuse else 0,
        "best_candidate": best_candidate,
        "best_argument_candidates": best_argument_candidates,
        "best_pointer_candidates": best_pointer_candidates,
    }


def _mfc_provenance_diagnostics(
    capture: probe.Capture, ranking: list[dict[str, object]]
) -> dict[str, object]:
    """Summarize same-SPU DMA source coverage without exposing guest EAs."""

    total = len(capture.spu)
    fields: list[dict[str, object]] = []
    for name, bit, ea_attribute, age_attribute in MFC_PROVENANCE_FIELDS:
        mapped = [event for event in capture.spu if event.mfc_provenance_mask & bit]
        eas = [int(getattr(event, ea_attribute)) for event in mapped]
        ages = [int(getattr(event, age_attribute)) for event in mapped]
        prefix = f"mfc.{name}."
        best_candidate = next(
            (
                item
                for item in ranking
                if str(item["candidate"]).startswith(prefix)
            ),
            None,
        )
        fields.append(
            {
                "field": name,
                "mapped": len(mapped),
                "coverage": (len(mapped) / total) if total else None,
                "distinct_eas": len(set(eas)),
                "distinct_pages": len({ea >> 12 for ea in eas}),
                "age_p50": _nearest_rank(ages, 0.50),
                "age_p95": _nearest_rank(ages, 0.95),
                "age_max": max(ages) if ages else None,
                "best_candidate": best_candidate,
            }
        )
    any_mapped = sum(bool(event.mfc_provenance_mask & 7) for event in capture.spu)
    return {
        "spu_with_any_mapping": any_mapped,
        "coverage": (any_mapped / total) if total else None,
        "fields": fields,
    }


def _compact_from_ranking(
    capture: probe.Capture,
    ranking: list[dict[str, object]],
    top_n: int,
    anomaly_limit: int,
) -> dict[str, object]:
    top_n = max(0, int(top_n))
    anomaly_limit = max(0, int(anomaly_limit))
    accepted = sum(item["decision"] == "accept" for item in ranking)
    target = str(capture.header.get("title_id", ""))[:32]
    version = str(capture.header.get("app_version", ""))[:16]
    ppu_callsites = collections.Counter(event.pc for event in capture.ppu)
    return {
        "schema_version": 1,
        "target": {"title_id": target, "app_version": version},
        "counts": {
            "ppu_event_count": len(capture.ppu),
            "spu_task_count": len(capture.spu),
            "rsx_draw_count": len(capture.rsx),
            "snapshot_count": len(capture.snapshots),
            "dropped_count": int(capture.header.get("dropped_events", 0) or 0),
            "truncated": bool(capture.truncated),
        },
        "ppu_callsites": [
            {"pc": f"0x{pc:08x}", "count": count}
            for pc, count in sorted(
                ppu_callsites.items(), key=lambda item: (-item[1], item[0])
            )[:MAX_PPU_CALLSITE_AGGREGATES]
        ],
        "ppu_join": _ppu_join_diagnostics(capture, ranking),
        "mfc_provenance": _mfc_provenance_diagnostics(capture, ranking),
        "filters": {
            "generated": len(ranking),
            "accepted": accepted,
            "rejected": len(ranking) - accepted,
            "thresholds": {
                "minimum_samples": MIN_SAMPLES,
                "minimum_adjacent_pairs": MIN_ADJACENT_PAIRS,
                "minimum_availability": MIN_AVAILABILITY,
                "maximum_collision": MAX_COLLISION,
                "minimum_stability": MIN_STABILITY,
                "minimum_pair_rate": MIN_PAIR_RATE,
                "minimum_arena_survival": MIN_ARENA_SURVIVAL,
            },
        },
        "top_candidates": ranking[:top_n],
        "anomalies": _aggregate_anomalies(ranking, anomaly_limit),
        "notes": [
            "All raw probe records and memory previews stayed local; this result contains aggregate metrics only.",
            "Draw-family grouping is independent of source0 and of the candidate being scored.",
            "arena_token_survival is only an address-rotation proxy, not proof of persistent game-object identity.",
            "arena_rotation_stability remains null until an independent object anchor is captured.",
            "MFC provenance is a same-SPU DMA source mapping, not proof of a PPU owner or game object.",
            "Accepted candidates are evidence for validation, not automatic approval for DLSS motion vectors.",
        ],
    }


def analyze_capture_compact(
    capture: probe.Capture, top_n: int = 10, anomaly_limit: int = 5
) -> dict[str, object]:
    """Return aggregate-only data safe to place in a model context."""

    ranking = _rank_capture(capture) if capture.spu else []
    return _compact_from_ranking(capture, ranking, top_n, anomaly_limit)


def format_model_summary(result: dict[str, object], max_lines: int = 120) -> str:
    """Format a strictly bounded, aggregate-only model handoff."""

    max_lines = max(4, int(max_lines))
    counts = result["counts"]
    filters = result["filters"]
    ppu_join = result["ppu_join"]
    mfc_provenance = result["mfc_provenance"]
    target = result["target"]
    fmt_percent = lambda value: "n/e" if value is None else f"{value:.1%}"
    fmt_value = lambda value: "n/e" if value is None else str(value)
    lines = [
        "Ascension local analyzer - model summary",
        f"target: {target['title_id']} v{target['app_version']}",
        (
            "local counts: "
            f"PPU={counts['ppu_event_count']} SPU={counts['spu_task_count']} "
            f"RSX={counts['rsx_draw_count']} snapshots={counts['snapshot_count']}"
        ),
        "PPU callsites (aggregate): "
        + (
            ", ".join(
                f"{item['pc']}={item['count']}" for item in result["ppu_callsites"]
            )
            or "none"
        ),
        (
            "PPU->SPU temporal join: "
            f"coverage={fmt_percent(ppu_join['coverage'])} "
            f"delta-us p50/p95/max={fmt_value(ppu_join['delta_us_p50'])}/"
            f"{fmt_value(ppu_join['delta_us_p95'])}/{fmt_value(ppu_join['delta_us_max'])} "
            f"preceding-{ppu_join['window_us']}us p50/p95/max="
            f"{fmt_value(ppu_join['preceding_ppu_count_p50'])}/"
            f"{fmt_value(ppu_join['preceding_ppu_count_p95'])}/"
            f"{fmt_value(ppu_join['preceding_ppu_count_max'])}"
        ),
        f"MFC GET provenance: any-coverage={fmt_percent(mfc_provenance['coverage'])}",
        f"quality: dropped={counts['dropped_count']} truncated={counts['truncated']}",
        f"candidates: generated={filters['generated']} accepted={filters['accepted']} rejected={filters['rejected']}",
        "Top candidates (aggregate metrics only):",
    ]
    for field in mfc_provenance["fields"]:
        best = field["best_candidate"]
        best_text = "none"
        if best is not None:
            best_text = (
                f"{best['candidate']} [{best['decision']}] "
                f"reasons={','.join(best['reject_reasons']) or 'none'}"
            )
        lines.insert(
            -3,
            f"MFC {field['field']}: coverage={fmt_percent(field['coverage'])} "
            f"distinct-ea/page={field['distinct_eas']}/{field['distinct_pages']} "
            f"age p50/p95/max={fmt_value(field['age_p50'])}/"
            f"{fmt_value(field['age_p95'])}/{fmt_value(field['age_max'])}; best={best_text}",
        )
    best_ppu = ppu_join["best_candidate"]
    if best_ppu is not None and len(lines) < max_lines:
        lines.insert(
            -1,
            f"Best PPU-derived candidate: {best_ppu['candidate']} [{best_ppu['decision']}] "
            f"score={best_ppu['score']:.4f} support={best_ppu['samples']}; "
            f"reasons={','.join(best_ppu['reject_reasons']) or 'none'}",
        )
    for best_ppu_argument in ppu_join["best_argument_candidates"]:
        if len(lines) >= max_lines:
            break
        lines.insert(
            -1,
            f"Best PPU argument candidate: {best_ppu_argument['candidate']} "
            f"[{best_ppu_argument['decision']}] score={best_ppu_argument['score']:.4f} "
            f"support={best_ppu_argument['samples']}; "
            f"reasons={','.join(best_ppu_argument['reject_reasons']) or 'none'}",
        )
    for best_ppu_pointer in ppu_join["best_pointer_candidates"]:
        if len(lines) >= max_lines:
            break
        lines.insert(
            -1,
            f"Best PPU pointer candidate: {best_ppu_pointer['candidate']} "
            f"[{best_ppu_pointer['decision']}] score={best_ppu_pointer['score']:.4f} "
            f"support={best_ppu_pointer['samples']}; "
            f"reasons={','.join(best_ppu_pointer['reject_reasons']) or 'none'}",
        )
    for index, item in enumerate(result["top_candidates"], 1):
        if len(lines) + 2 > max_lines:
            break
        stability = item["cross_frame_stability"]
        collision = item["same_mesh_collision"]
        pair_rate = item["pair_rate"]
        survival = item["arena_token_survival"]
        fmt = lambda value: "n/e" if value is None else f"{value:.1%}"
        lines.append(
            f"{index:02d}. {item['candidate']} [{item['decision']}] score={item['score']:.4f} confidence={item['confidence']}"
        )
        lines.append(
            f"    stability={fmt(stability)} collision={fmt(collision)} pair={fmt(pair_rate)} "
            f"arena-survival={fmt(survival)} support={item['samples']}; reasons={','.join(item['reject_reasons']) or 'none'}"
        )
    anomaly_counts = collections.Counter(item["kind"] for item in result["anomalies"])
    if len(lines) < max_lines:
        lines.append(
            "Anomaly classes: "
            + (", ".join(f"{kind}={count}" for kind, count in sorted(anomaly_counts.items())) or "none")
        )
    if len(lines) < max_lines:
        lines.append("Boundary: true object stability is not evaluated without an independent anchor.")
    return "\n".join(lines[:max_lines]) + "\n"


_POINTER_FIELD = re.compile(r"^pointer\[(\d+)]\.(be|le)(32|64)\+0x([0-9a-f]+)$")


def _parse_pointer_rules(config_path: pathlib.Path | None) -> dict[int, dict[str, str]]:
    if config_path is None or not config_path.is_file():
        return {}
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(document, dict):
        return {}
    command_history = document.get("effective_commands", [])
    if not isinstance(command_history, list):
        return {}
    rules: dict[int, dict[str, str]] = {}
    for command in command_history:
        if not isinstance(command, str):
            continue
        tokens = command.strip().split()
        if not tokens:
            continue
        operation = tokens[0].upper()
        values = {
            key.upper(): value
            for token in tokens[1:]
            if "=" in token
            for key, value in [token.split("=", 1)]
        }
        rule_id_text = values.get("ID")
        # REMOVE_POINTER_FOLLOW uses a positional ID in the live-probe
        # protocol (for example, ``REMOVE_POINTER_FOLLOW 2``).  Replaying only
        # ID= forms leaves removed rules falsely active.
        if operation == "REMOVE_POINTER_FOLLOW" and rule_id_text is None and len(tokens) == 2:
            rule_id_text = tokens[1]
        try:
            rule_id = int(rule_id_text or "0", 0)
        except ValueError:
            continue
        if operation == "REMOVE_POINTER_FOLLOW":
            rules.pop(rule_id, None)
        elif operation == "ADD_POINTER_FOLLOW" and 1 <= rule_id <= 8:
            rules[rule_id] = values
    return rules


def _format_offset(value: int) -> str:
    return f"-0x{-value:x}" if value < 0 else f"0x{value:x}"


def _valid_pointer_source(source: str) -> bool:
    fixed = {
        "TASK_HEADER",
        "TASK_CONTEXT",
        "DMA_DESCRIPTOR",
        "SOURCE0",
        "SOURCE1",
        "AUXILIARY",
        "OUTPUT",
    }
    source = source.upper()
    if source in fixed:
        return True
    match = re.fullmatch(r"(PPU|SPU)_R(\d+)", source)
    if not match:
        return False
    register = int(match.group(2))
    return register < (32 if match.group(1) == "PPU" else 128)


def _deeper_follow_command(
    output_rule_id: int, source_rule: dict[str, str], field_offset: int
) -> str | None:
    try:
        depth = min(4, max(0, int(source_rule.get("DEPTH", "0"), 0)))
        if depth >= 4:
            return None
        offsets = [int(source_rule.get(f"OFFSET{index}", "0"), 0) for index in range(depth + 1)]
    except ValueError:
        return None
    source = source_rule.get("SOURCE", "").upper()
    space = source_rule.get("SPACE", "GUEST").upper()
    if not _valid_pointer_source(source) or space not in {"GUEST", "LS"}:
        return None
    new_offsets = offsets[:depth] + [offsets[depth] + field_offset, 0]
    parts = [
        "ADD_POINTER_FOLLOW",
        f"ID={output_rule_id}",
        f"SOURCE={source}",
        f"SPACE={space}",
        f"DEPTH={depth + 1}",
    ]
    parts.extend(f"OFFSET{index}={_format_offset(value)}" for index, value in enumerate(new_offsets))
    parts.append("SIZE=64")
    return " ".join(parts)


def build_next_probe_plan(
    result: dict[str, object], config_path: pathlib.Path | None = None
) -> dict[str, object]:
    """Build a bounded local follow-up plan without inventing memory semantics."""

    validation: list[dict[str, object]] = []
    pointer_follow: list[dict[str, object]] = []
    generated_commands: list[str] = []
    rules = _parse_pointer_rules(config_path)
    free_rule_ids = [rule_id for rule_id in range(1, 9) if rule_id not in rules]
    for item in result["top_candidates"]:
        if item["decision"] == "accept" and len(validation) < 5:
            validation.append(
                {
                    "candidate": item["candidate"],
                    "action": "validate_in_independent_short_capture",
                    "minimum_adjacent_pairs": 7,
                }
            )
        match = _POINTER_FIELD.match(str(item["candidate"]))
        if match and len(pointer_follow) < 8:
            source_rule_id = int(match.group(1))
            field_offset = int(match.group(4), 16)
            # PS3 pointers are big-endian 32-bit values.  LE/64-bit fields stay
            # ranked as tokens but are never guessed to be pointers.
            command = None
            source_rule = rules.get(source_rule_id)
            # A deeper rule consumes a new live-probe slot.  Reusing the
            # source ID creates duplicate IDs in the C++ rule table, while an
            # LS rule cannot safely assume that a captured BE32 field is
            # another LS address.  Generate only the unambiguous guest-space
            # case and leave all other cases as explicit review items.
            if (
                match.group(2) == "be"
                and match.group(3) == "32"
                and source_rule is not None
                and source_rule.get("SPACE", "GUEST").upper() == "GUEST"
            ):
                if free_rule_ids:
                    command = _deeper_follow_command(
                        free_rule_ids[0], source_rule, field_offset
                    )
                    # Do not consume a rule slot when the source definition is
                    # malformed or already at the maximum follow depth.
                    if command:
                        free_rule_ids.pop(0)
            pointer_follow.append(
                {
                    "source_rule_id": source_rule_id,
                    "field_offset": int(match.group(4), 16),
                    "endianness": "big" if match.group(2) == "be" else "little",
                    "width_bits": int(match.group(3)),
                    "action": "try_one_deeper_pointer_follow_locally",
                    "requires_recorded_rule_definition": source_rule_id not in rules,
                    "requires_address_space_confirmation": bool(
                        source_rule
                        and source_rule.get("SPACE", "GUEST").upper() != "GUEST"
                    ),
                    "requires_free_rule_slot": bool(
                        source_rule
                        and source_rule.get("SPACE", "GUEST").upper() == "GUEST"
                        and match.group(2) == "be"
                        and match.group(3) == "32"
                        and command is None
                        and not free_rule_ids
                    ),
                    "generated_command": command,
                }
            )
            if command:
                generated_commands.append(command)
    return {
        "schema_version": 1,
        "validation": validation,
        "pointer_follow": pointer_follow,
        "generated_commands": generated_commands,
        "limits": {
            "short_capture_only": True,
            "raw_records_to_model": False,
            "automatic_apply": False,
        },
        "note": (
            "Commands are generated only from a recorded probe-config.json and big-endian 32-bit fields. "
            "Missing definitions, LE fields, and 64-bit fields are never guessed."
        ),
    }


def _write_json(path: pathlib.Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_compact_outputs(
    capture: probe.Capture,
    output_dir: pathlib.Path,
    top_n: int = 10,
    anomaly_limit: int = 5,
    max_summary_lines: int = 120,
    config_path: pathlib.Path | None = None,
) -> dict[str, object]:
    """Run once locally and write only aggregate reports."""

    output_dir.mkdir(parents=True, exist_ok=True)
    ranking = _rank_capture(capture) if capture.spu else []
    result = _compact_from_ranking(capture, ranking, top_n, anomaly_limit)
    summary = format_model_summary(result, max_lines=max_summary_lines)
    (output_dir / "model-summary.txt").write_text(summary, encoding="utf-8")
    _write_json(
        output_dir / "top-candidates.json",
        {
            "schema_version": result["schema_version"],
            "target": result["target"],
            "counts": result["counts"],
            "ppu_callsites": result["ppu_callsites"],
            "ppu_join": result["ppu_join"],
            "mfc_provenance": result["mfc_provenance"],
            "filters": result["filters"],
            "top_candidates": result["top_candidates"],
        },
    )
    _write_json(output_dir / "anomalies.json", result["anomalies"])
    _write_json(
        output_dir / "full-ranking.local-only.json",
        {
            "schema_version": 1,
            "warning": "LOCAL_ONLY_AGGREGATES_DO_NOT_PASTE_WHOLE_FILE_INTO_MODEL",
            "ranking": ranking,
        },
    )
    plan_input = dict(result)
    plan_input["top_candidates"] = ranking[:100]
    _write_json(output_dir / "next-probe-plan.json", build_next_probe_plan(plan_input, config_path))
    return result


def _self_test() -> int:
    directory = pathlib.Path(tempfile.mkdtemp(prefix="ascension-auto-analyzer-"))
    try:
        capture_path = directory / "synthetic.bin"
        probe.write_synthetic_capture(capture_path)
        capture = probe.read_capture(capture_path)
        result = analyze_capture_compact(capture, top_n=100, anomaly_limit=2)
        by_name = {item["candidate"]: item for item in result["top_candidates"]}
        assert by_name["pointer[1].address"]["decision"] == "accept"
        assert len(format_model_summary(result, 24).splitlines()) <= 24
        print(format_model_summary(result, 12), end="")
        print("self-test passed; no real capture was opened")
        return 0
    finally:
        shutil.rmtree(directory)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", nargs="?", help="local ALPROBE1 capture; never emitted in model-facing output")
    parser.add_argument("--output-dir")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--anomaly-limit", type=int, default=5)
    parser.add_argument("--max-summary-lines", type=int, default=120)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return _self_test()
    if not args.capture:
        parser.error("capture is required unless --self-test is used")
    capture_path = pathlib.Path(args.capture).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve() if args.output_dir else capture_path.with_suffix("").with_name(capture_path.stem + "-auto-analysis")
    result = write_compact_outputs(
        probe.read_capture(capture_path),
        output_dir,
        top_n=args.top,
        anomaly_limit=args.anomaly_limit,
        max_summary_lines=args.max_summary_lines,
        config_path=capture_path.parent / "probe-config.json",
    )
    print(format_model_summary(result, max_lines=args.max_summary_lines), end="")
    print(f"aggregate reports: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
