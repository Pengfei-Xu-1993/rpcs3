#!/usr/bin/env python3
"""Controller, ABI decoder, and identity analyzer for Ascension Live Probe v1."""

from __future__ import annotations

import argparse
import bisect
import collections
import ctypes
import dataclasses
import datetime as dt
import json
import math
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
from typing import BinaryIO, Callable, Iterable


PIPE_NAME = r"\\.\pipe\RPCS3AscensionLiveProbe"
FILE_HEADER = struct.Struct("<8sIIIIQQQQ16s16s64s64s296s")
RECORD_HEADER = struct.Struct("<IHHIIQQQQIIII")
VALUES = struct.Struct("<32Q")
WORDS = struct.Struct("<64I")
POINTER = struct.Struct("<IIQQQII16s")
EVENT_SIZE = 1024
FILE_HEADER_SIZE = 512
EVENT_MAGIC = 0x45504C41
EVENT_PPU = 1
EVENT_SPU = 2
EVENT_RSX = 3
EVENT_FLAG_AUTHORIZED = 1 << 0
EVENT_FLAG_MFC_PROVENANCE = 1 << 5
SNAPSHOT_SPU_LS = 0x100
SNAPSHOT_PPU_STACK = 0x101
DEFAULT_MAX_EVENTS = 250_000
GRACEFUL_STOP_LEAD_SECONDS = 1.0
WATCHDOG_LEAD_SECONDS = 0.25
POST_DISARM_QUIESCE_SECONDS = 0.25
FLUSH_CONVERGENCE_SECONDS = 2.0


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _positive_finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive finite number")
    return value


def _cstring(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("utf-8", "replace")


@dataclasses.dataclass(slots=True)
class PointerResult:
    rule_id: int
    flags: int
    source: int
    address: int
    content_hash: int
    requested_size: int
    captured_size: int
    sample: bytes


@dataclasses.dataclass(slots=True)
class PPUEvent:
    sequence: int
    time_us: int
    frame: int
    thread: int
    pc: int
    caller: int
    target: int
    registers: dict[int, int]
    stack_address: int
    stack: bytes
    pointers: tuple[PointerResult, ...]


@dataclasses.dataclass(slots=True)
class SPUEvent:
    sequence: int
    time_us: int
    frame: int
    thread: int
    pc: int
    target: int
    registers: dict[int, int]
    task_header_lsa: int
    task_context_lsa: int
    dma_descriptor_lsa: int
    task_sequence: int
    task_format: int
    packed_count: int
    # Raw word at task_header_lsa + 0x04. It is preserved for ABI
    # compatibility but is not a verified guest pointer or identity field.
    task_header_word_04: int
    auxiliary: int
    source0: int
    source1: int
    output_start: int
    output_end: int
    pointers: tuple[PointerResult, ...]
    mfc_provenance_mask: int = 0
    task_header_guest_ea: int = 0
    task_context_guest_ea: int = 0
    dma_descriptor_guest_ea: int = 0
    task_header_mfc_age: int = 0
    task_context_mfc_age: int = 0
    dma_descriptor_mfc_age: int = 0
    occurrence: int = 0
    output_frame_relative: int = 0
    nearest_ppu: PPUEvent | None = None


@dataclasses.dataclass(slots=True)
class RSXEvent:
    sequence: int
    time_us: int
    frame: int
    producer_sequence: int
    draw_sequence: int
    vp: int
    fp: int
    vertex_count: int
    stream_vertex_count: int
    first_vertex: int
    stream_address: int
    stream_size: int
    index_address: int
    index_count: int
    attribute_mask: int
    stride: int
    primitive: int
    command: int
    index_type: int
    index_hash: int
    layout_hash: int
    task_sequence: int
    task_format: int
    task_source0: int
    output_relative: int
    overlap: int


@dataclasses.dataclass(slots=True)
class Capture:
    header: dict[str, object]
    ppu: list[PPUEvent]
    spu: list[SPUEvent]
    rsx: list[RSXEvent]
    snapshots: list[dict[str, int]]
    truncated: bool = False


def _decode_pointers(blob: bytes, count: int) -> tuple[PointerResult, ...]:
    results: list[PointerResult] = []
    offset = 64 + VALUES.size + WORDS.size
    for index in range(min(count, 8)):
        fields = POINTER.unpack_from(blob, offset + index * POINTER.size)
        if fields[0]:
            results.append(PointerResult(*fields))
    return tuple(results)


def read_capture(path: os.PathLike[str] | str) -> Capture:
    ppu_events: list[PPUEvent] = []
    spu_events: list[SPUEvent] = []
    rsx_events: list[RSXEvent] = []
    snapshots: list[dict[str, int]] = []
    truncated = False
    with open(path, "rb") as stream:
        raw = stream.read(FILE_HEADER_SIZE)
        if len(raw) != FILE_HEADER_SIZE:
            raise ValueError("capture is shorter than its 512-byte header")
        fields = FILE_HEADER.unpack(raw)
        if fields[0] != b"ALPROBE1" or fields[1] != 1 or fields[2] != FILE_HEADER_SIZE:
            raise ValueError(f"unsupported capture header: magic={fields[0]!r} version={fields[1]}")
        header: dict[str, object] = {
            "version": fields[1],
            "event_size": fields[3],
            "record_header_size": fields[4],
            "start_time_us": fields[5],
            "written_events": fields[6],
            "dropped_events": fields[7],
            "written_snapshots": fields[8],
            "title_id": _cstring(fields[9]),
            "app_version": _cstring(fields[10]),
            "executable_hash": _cstring(fields[11]),
            "build_label": _cstring(fields[12]),
        }
        while True:
            common = stream.read(RECORD_HEADER.size)
            if not common:
                break
            if len(common) != RECORD_HEADER.size:
                truncated = True
                break
            h = RECORD_HEADER.unpack(common)
            magic, version, event_type, total_size, flags = h[:5]
            if magic != EVENT_MAGIC or version != 1 or total_size < RECORD_HEADER.size:
                raise ValueError(f"bad record at file offset {stream.tell() - RECORD_HEADER.size:#x}")
            payload = stream.read(total_size - RECORD_HEADER.size)
            if len(payload) != total_size - RECORD_HEADER.size:
                truncated = True
                break
            if event_type >= 0x100:
                snapshots.append({
                    "type": event_type,
                    "sequence": h[5],
                    "time_us": h[6],
                    "frame": h[7],
                    "thread": h[9],
                    "pc": h[10],
                    "payload_size": len(payload),
                })
                continue
            if total_size != EVENT_SIZE:
                raise ValueError(f"lightweight event size is {total_size}, expected {EVENT_SIZE}")
            blob = common + payload
            values = VALUES.unpack_from(blob, RECORD_HEADER.size)
            words = WORDS.unpack_from(blob, RECORD_HEADER.size + VALUES.size)
            sequence, time_us, frame, producer = h[5:9]
            thread, pc, caller, target = h[9:13]
            if event_type == EVENT_PPU:
                mask = words[0] | (words[1] << 32)
                registers = {reg: values[reg] for reg in range(32) if mask & (1 << reg)}
                stack_size = min(words[4], 128)
                stack = blob[RECORD_HEADER.size + VALUES.size + 32 * 4:
                             RECORD_HEADER.size + VALUES.size + 32 * 4 + stack_size]
                ppu_events.append(PPUEvent(
                    sequence, time_us, frame, thread, pc, caller, target,
                    registers, words[2], stack, _decode_pointers(blob, words[5])))
            elif event_type == EVENT_SPU:
                count = min(words[32], 32)
                registers = {words[index]: values[index] for index in range(count)}
                spu_events.append(SPUEvent(
                    sequence, time_us, frame, thread, pc, target, registers,
                    words[33], words[34], words[35], words[36], words[37], words[38],
                    words[39], words[40], words[41], words[42], words[43], words[44],
                    _decode_pointers(blob, words[45]), words[46], words[47], words[48],
                    words[49], words[50], words[51], words[52]))
            elif event_type == EVENT_RSX:
                rsx_events.append(RSXEvent(
                    sequence, time_us, frame, producer, words[0], words[1], words[2],
                    words[3], words[4], words[5], words[6], words[7], words[8], words[9],
                    words[10], words[11], words[12], words[13], words[14], values[0],
                    values[1], words[18], words[19], words[22], words[27], words[28]))
    capture = Capture(header, ppu_events, spu_events, rsx_events, snapshots, truncated)
    _decorate_events(capture)
    return capture


def _decorate_events(capture: Capture) -> None:
    occurrence: collections.Counter[tuple[int, tuple[int, int, int]]] = collections.Counter()
    frame_min_output: dict[int, int] = {}
    for event in capture.spu:
        if event.output_start:
            frame_min_output[event.frame] = min(frame_min_output.get(event.frame, event.output_start), event.output_start)
    for event in capture.spu:
        mesh = (event.source0, event.task_format, event.packed_count)
        key = (event.frame, mesh)
        event.occurrence = occurrence[key]
        occurrence[key] += 1
        if event.output_start:
            event.output_frame_relative = event.output_start - frame_min_output.get(event.frame, event.output_start)
    if capture.ppu:
        capture.ppu.sort(key=lambda event: event.time_us)
        times = [event.time_us for event in capture.ppu]
        for event in capture.spu:
            index = bisect.bisect_right(times, event.time_us) - 1
            if index >= 0 and event.time_us - times[index] <= 50_000:
                event.nearest_ppu = capture.ppu[index]


def _pointer(event: SPUEvent, rule_id: int) -> PointerResult | None:
    return next((item for item in event.pointers if item.rule_id == rule_id and item.flags), None)


def candidate_extractors(capture: Capture) -> dict[str, Callable[[SPUEvent], object | None]]:
    candidates: dict[str, Callable[[SPUEvent], object | None]] = {
        "source0.absolute": lambda e: e.source0 or None,
        "source1.absolute": lambda e: e.source1 or None,
        "auxiliary.absolute": lambda e: e.auxiliary or None,
        "output.absolute": lambda e: e.output_start or None,
        "output.page": lambda e: (e.output_start >> 12) if e.output_start else None,
        "output.page_offset": lambda e: (e.output_start & 0xFFF) if e.output_start else None,
        "output.frame_relative": lambda e: e.output_frame_relative if e.output_start else None,
        "task.sequence": lambda e: e.task_sequence,
        "task.local_occurrence": lambda e: e.occurrence,
    }
    rule_ids = sorted({pointer.rule_id for event in capture.spu for pointer in event.pointers})
    for rule_id in rule_ids:
        candidates[f"pointer[{rule_id}].source"] = lambda e, rule_id=rule_id: (p.source if (p := _pointer(e, rule_id)) else None)
        candidates[f"pointer[{rule_id}].address"] = lambda e, rule_id=rule_id: (p.address if (p := _pointer(e, rule_id)) else None)
        candidates[f"pointer[{rule_id}].content_hash"] = lambda e, rule_id=rule_id: (p.content_hash if (p := _pointer(e, rule_id)) else None)
        for offset in range(0, 16, 4):
            candidates[f"pointer[{rule_id}].be32+0x{offset:x}"] = (
                lambda e, rule_id=rule_id, offset=offset:
                    int.from_bytes(p.sample[offset:offset + 4], "big")
                    if (p := _pointer(e, rule_id)) and p.captured_size >= offset + 4 else None)
    ppu_regs = sorted({reg for event in capture.ppu for reg in event.registers if 3 <= reg <= 10})
    for reg in ppu_regs:
        candidates[f"nearest_ppu.r{reg}"] = (
            lambda e, reg=reg: e.nearest_ppu.registers.get(reg) if e.nearest_ppu else None)
        candidates[f"nearest_ppu.r{reg}+mesh"] = (
            lambda e, reg=reg: (e.nearest_ppu.registers.get(reg), e.source0)
            if e.nearest_ppu and e.nearest_ppu.registers.get(reg) else None)
    return candidates


def _score_candidate(capture: Capture, name: str, extractor: Callable[[SPUEvent], object | None]) -> dict[str, object]:
    groups: dict[tuple[int, int, int], dict[int, list[tuple[object, SPUEvent]]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    missing = 0
    values: set[object] = set()
    for event in capture.spu:
        token = extractor(event)
        if token is None:
            missing += 1
            continue
        mesh = (event.source0, event.task_format, event.packed_count)
        groups[mesh][event.frame].append((token, event))
        values.add(token)

    collision_numerator = 0
    collision_denominator = 0
    adjacent_matched = 0
    adjacent_opportunities = 0
    paired = ambiguous = unmatched = reassociated = rotation_pairs = 0
    for frames in groups.values():
        ordered = sorted(frames)
        for frame in ordered:
            tokens = [token for token, _ in frames[frame]]
            if len(tokens) > 1:
                collision_numerator += len(tokens) - len(set(tokens))
                collision_denominator += len(tokens)
        for left_frame, right_frame in zip(ordered, ordered[1:]):
            if right_frame - left_frame > 2:
                continue
            left = frames[left_frame]
            right = frames[right_frame]
            left_by_token: dict[object, list[SPUEvent]] = collections.defaultdict(list)
            right_by_token: dict[object, list[SPUEvent]] = collections.defaultdict(list)
            for token, event in left: left_by_token[token].append(event)
            for token, event in right: right_by_token[token].append(event)
            all_tokens = set(left_by_token) | set(right_by_token)
            adjacent_opportunities += max(len(left), len(right))
            adjacent_matched += sum(min(len(left_by_token[token]), len(right_by_token[token])) for token in all_tokens)
            for token in all_tokens:
                a = left_by_token[token]
                b = right_by_token[token]
                if len(a) == 1 and len(b) == 1:
                    paired += 1
                    if a[0].output_start != b[0].output_start:
                        rotation_pairs += 1
                        reassociated += 1
                elif a and b:
                    ambiguous += max(len(a), len(b))
                else:
                    unmatched += max(len(a), len(b))

    total = len(capture.spu)
    missing_rate = missing / total if total else 1.0
    collision = collision_numerator / collision_denominator if collision_denominator else 0.0
    stability = adjacent_matched / adjacent_opportunities if adjacent_opportunities else 0.0
    pair_total = paired + ambiguous + unmatched
    pair_rate = paired / pair_total if pair_total else 0.0
    mapped_sequences = {event.producer_sequence for event in capture.rsx if event.producer_sequence}
    mapped_tasks = sum(event.sequence in mapped_sequences for event in capture.spu)
    rsx_mapping = mapped_tasks / total if total else 0.0
    rotation_stability = reassociated / rotation_pairs if rotation_pairs else None
    score = stability * (1.0 - collision) * (1.0 - missing_rate) * (0.5 + 0.5 * rsx_mapping)
    return {
        "candidate": name,
        "score": score,
        "cross_frame_stability": stability,
        "same_mesh_collision": collision,
        "missing_rate": missing_rate,
        "unique_values": len(values),
        "buffer_rotation_stability": rotation_stability,
        "rsx_mapping_coverage": rsx_mapping,
        "pair_rate": pair_rate,
        "paired": paired,
        "ambiguous": ambiguous,
        "unmatched": unmatched,
        "reassociated": reassociated,
        "samples": total - missing,
    }


def analyze_capture(capture: Capture) -> dict[str, object]:
    candidates = ([_score_candidate(capture, name, extractor)
                   for name, extractor in candidate_extractors(capture).items()]
                  if capture.spu else [])
    candidates.sort(key=lambda item: (item["score"], item["pair_rate"], -item["same_mesh_collision"]), reverse=True)
    mapped_draws = sum(bool(event.producer_sequence) for event in capture.rsx)
    return {
        "capture": capture.header,
        "truncated": capture.truncated,
        "counts": {
            "ppu_events": len(capture.ppu),
            "spu_events": len(capture.spu),
            "rsx_events": len(capture.rsx),
            "snapshots": len(capture.snapshots),
            "mapped_rsx_draws": mapped_draws,
            "rsx_draw_mapping_rate": mapped_draws / len(capture.rsx) if capture.rsx else 0.0,
        },
        "candidates": candidates,
        "notes": [
            *(["No SPU events were captured; identity candidates were not scored."] if not capture.spu else []),
            "Scores are comparative evidence, not proof of game-level instance semantics.",
            "Cross-frame stability uses same-mesh multisets in adjacent observed frames.",
            "Collision measures duplicate tokens among simultaneous tasks sharing source0/format/count.",
            "A high task.local_occurrence score can still fail when submission order changes.",
        ],
    }


def format_report(analysis: dict[str, object], limit: int = 20) -> str:
    counts = analysis["counts"]
    capture = analysis["capture"]
    lines = [
        "Ascension Live Probe identity report",
        "====================================",
        f"target: {capture.get('title_id')} v{capture.get('app_version')}",
        f"PPU/SPU/RSX events: {counts['ppu_events']} / {counts['spu_events']} / {counts['rsx_events']}",
        f"RSX output mapping: {counts['mapped_rsx_draws']}/{counts['rsx_events']} ({counts['rsx_draw_mapping_rate']:.2%})",
        f"snapshots: {counts['snapshots']}  dropped: {capture.get('dropped_events', 0)}  truncated: {analysis['truncated']}",
        "",
        "Ranked candidate tokens",
        "-----------------------",
    ]
    if not analysis["candidates"]:
        lines.append("No candidate ranking is available because the capture contains no SPU task events.")
    for index, candidate in enumerate(analysis["candidates"][:limit], 1):
        rotation = candidate["buffer_rotation_stability"]
        lines.extend([
            f"{index:2}. {candidate['candidate']}  score={candidate['score']:.4f}",
            f"    stability={candidate['cross_frame_stability']:.2%}  collision={candidate['same_mesh_collision']:.2%}  missing={candidate['missing_rate']:.2%}",
            f"    pair_rate={candidate['pair_rate']:.2%}  paired/ambiguous/unmatched={candidate['paired']}/{candidate['ambiguous']}/{candidate['unmatched']}",
            f"    unique={candidate['unique_values']}  RSX-mapped={candidate['rsx_mapping_coverage']:.2%}  rotation={'n/a' if rotation is None else f'{rotation:.2%}'}",
        ])
    lines.extend(["", "Interpretation boundary", "-----------------------"])
    lines.extend(f"- {note}" for note in analysis["notes"])
    return "\n".join(lines) + "\n"


class ProbePipe:
    def __init__(self, timeout: float = 90.0):
        if os.name != "nt":
            raise OSError("the RPCS3 Live Probe named pipe is Windows-only")
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32 = kernel32
        kernel32.CreateFileW.restype = ctypes.c_void_p
        kernel32.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                                         ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
        kernel32.SetNamedPipeHandleState.restype = ctypes.c_int
        kernel32.SetNamedPipeHandleState.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32),
                                                      ctypes.c_void_p, ctypes.c_void_p]
        kernel32.WriteFile.restype = ctypes.c_int
        kernel32.WriteFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
                                       ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
        kernel32.ReadFile.restype = ctypes.c_int
        kernel32.ReadFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
                                      ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        deadline = time.monotonic() + timeout
        handle = None
        while time.monotonic() < deadline:
            handle = kernel32.CreateFileW(PIPE_NAME, 0xC0000000, 0, None, 3, 0, None)
            if handle != ctypes.c_void_p(-1).value:
                break
            time.sleep(0.1)
        if handle == ctypes.c_void_p(-1).value or handle is None:
            raise TimeoutError(f"timed out waiting for {PIPE_NAME}")
        self.handle = handle
        mode = ctypes.c_uint32(2)  # PIPE_READMODE_MESSAGE
        if not kernel32.SetNamedPipeHandleState(ctypes.c_void_p(handle), ctypes.byref(mode), None, None):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())

    def command(self, text: str) -> dict[str, object]:
        payload = text.encode("utf-8")
        written = ctypes.c_uint32()
        if not self._kernel32.WriteFile(ctypes.c_void_p(self.handle), payload, len(payload), ctypes.byref(written), None):
            raise ctypes.WinError(ctypes.get_last_error())
        buffer = ctypes.create_string_buffer(8192)
        received = ctypes.c_uint32()
        if not self._kernel32.ReadFile(ctypes.c_void_p(self.handle), buffer, len(buffer), ctypes.byref(received), None):
            raise ctypes.WinError(ctypes.get_last_error())
        response = buffer.raw[:received.value].decode("utf-8", "replace")
        return json.loads(response)

    def close(self) -> None:
        if getattr(self, "handle", None) not in (None, ctypes.c_void_p(-1).value):
            self._kernel32.CloseHandle(ctypes.c_void_p(self.handle))
            self.handle = None

    def __enter__(self) -> "ProbePipe":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


DEFAULT_COMMANDS = [
    "SET_FILTER PPU_PC=0x073c80c PPU_CALLER=0 PPU_THREAD=0",
    "SET_FILTER SPU_PC=0x0928c SPU_TARGET=0x0c490 SPU_FORMAT=0x871c0c00 SPU_THREAD=0 SPU_SEQUENCE=0 SPU_SOURCE=0 SPU_OUTPUT=0",
    "SET_FILTER FRAME_START=0 FRAME_END=0xffffffffffffffff RSX_MASK=0xc3b5 RSX_VP=0 RSX_FP=0 RSX_MIN_VERTICES=128 RSX_MAX_VERTICES=8192 RSX_ADDRESS=0 RSX_PRIMITIVE=0xffffffff",
    "SET_SAMPLE_RATE 1",
    f"SET_MAX_EVENTS {DEFAULT_MAX_EVENTS}",
    "SET_REGISTER_SET PPU 0,1,3,4,5,6,7,8,9,10",
    "SET_REGISTER_SET SPU 0,1,3,4,5,6,7,8,12,20,21,23,32,70,72,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95",
    "SET_STACK_WINDOW 128",
    "ADD_POINTER_FOLLOW ID=1 SOURCE=SOURCE0 SPACE=GUEST DEPTH=0 OFFSET0=0 SIZE=64",
    "ADD_POINTER_FOLLOW ID=2 SOURCE=SOURCE1 SPACE=GUEST DEPTH=0 OFFSET0=0 SIZE=64",
    "ADD_POINTER_FOLLOW ID=3 SOURCE=AUXILIARY SPACE=GUEST DEPTH=0 OFFSET0=0 SIZE=64",
    "ADD_POINTER_FOLLOW ID=4 SOURCE=DMA_DESCRIPTOR SPACE=LS DEPTH=0 OFFSET0=0 SIZE=64",
    "ADD_POINTER_FOLLOW ID=5 SOURCE=PPU_R3 SPACE=GUEST DEPTH=0 OFFSET0=0 SIZE=64",
    "ADD_POINTER_FOLLOW ID=6 SOURCE=PPU_R4 SPACE=GUEST DEPTH=0 OFFSET0=0 SIZE=64",
    "ADD_POINTER_FOLLOW ID=7 SOURCE=PPU_R7 SPACE=GUEST DEPTH=0 OFFSET0=0 SIZE=64",
    "ADD_POINTER_FOLLOW ID=8 SOURCE=PPU_R8 SPACE=GUEST DEPTH=0 OFFSET0=0 SIZE=64",
]


def _commands_for_max_events(max_events: int) -> list[str]:
    replacement = f"SET_MAX_EVENTS {max_events}"
    commands = [
        replacement if command.startswith("SET_MAX_EVENTS ") else command
        for command in DEFAULT_COMMANDS
    ]
    if commands.count(replacement) != 1:
        raise RuntimeError("default command list must contain exactly one event limit")
    return commands


def _require_ok(command: str, response: dict[str, object]) -> dict[str, object]:
    if not response.get("ok"):
        raise RuntimeError(f"command failed: {command}: {response}")
    return response


def _changes_capture_guard(command: str) -> bool:
    tokens = command.strip().upper().split()
    if not tokens:
        return False
    if tokens[0] in {"ARM", "DISARM", "START_CAPTURE", "STOP_CAPTURE", "SET_MAX_EVENTS"}:
        return True
    return tokens[0] == "SET_FILTER" and any(
        token.split("=", 1)[0] == "FIRST_HITS" for token in tokens[1:]
    )


def _capture_guard_violations(
    *,
    max_events: int,
    arm_seconds: float | None,
    arm_duration_seconds: float | None,
    stop_reason: str | None,
    final_probe: dict[str, object] | None,
    capture_exists: bool,
    analysis_exit_code: int | None,
    summary_lines: int | None,
    aggregate: dict[str, object] | None,
) -> list[str]:
    violations: list[str] = []
    if arm_seconds is not None and (
        arm_duration_seconds is None or arm_duration_seconds > arm_seconds
    ):
        violations.append("arm_duration_exceeded_or_missing")
    if not final_probe:
        violations.append("missing_final_probe")
    else:
        accepted = int(final_probe.get("accepted", -1))
        written = int(final_probe.get("written", -1))
        dropped = int(final_probe.get("dropped", -1))
        if int(final_probe.get("max_events", -1)) != max_events:
            violations.append("event_limit_configuration_mismatch")
        if accepted < 0 or accepted > max_events:
            violations.append("event_limit_exceeded_or_missing")
        if stop_reason == "max_events" and accepted != max_events:
            violations.append("event_limit_stop_not_exact")
        if final_probe.get("armed") or final_probe.get("capturing"):
            violations.append("probe_not_closed")
        if final_probe.get("authorized") is not True:
            violations.append("target_not_authorized")
        if dropped != 0:
            violations.append("probe_reported_drops")
        if accepted != written + dropped:
            violations.append("probe_counts_not_converged")
    if not capture_exists:
        violations.append("capture_missing")
    if analysis_exit_code != 0:
        violations.append("analyzer_failed_or_missing")
    if summary_lines is None or summary_lines > 120:
        violations.append("bounded_summary_missing_or_oversized")
    if not aggregate:
        violations.append("aggregate_report_missing")
    else:
        counts = aggregate.get("counts", {})
        if not isinstance(counts, dict):
            violations.append("aggregate_counts_missing")
        else:
            if int(counts.get("dropped_count", -1)) != 0:
                violations.append("capture_header_reported_drops")
            if counts.get("truncated") is not False:
                violations.append("capture_truncated_or_unknown")
            required_count_keys = {
                "ppu_event_count",
                "spu_task_count",
                "rsx_draw_count",
                "snapshot_count",
            }
            if not required_count_keys.issubset(counts):
                violations.append("aggregate_event_counts_missing")
            if final_probe and required_count_keys.issubset(counts):
                decoded_events = sum(
                    int(counts.get(key, 0))
                    for key in ("ppu_event_count", "spu_task_count", "rsx_draw_count")
                )
                if decoded_events != int(final_probe.get("written", -1)):
                    violations.append("decoded_event_count_mismatch")
                if int(counts["snapshot_count"]) != int(final_probe.get("snapshots", -1)):
                    violations.append("decoded_snapshot_count_mismatch")
    if stop_reason not in {"arm_deadline", "max_events"}:
        violations.append("unexpected_stop_reason")
    return violations


def _append_jsonl(path: pathlib.Path, item: object) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def _write_controller_status(path: pathlib.Path, phase: str, **values: object) -> None:
    status = {
        "phase": phase,
        "updated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        **values,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _write_probe_config(path: pathlib.Path, commands: list[str]) -> None:
    """Persist the exact rule history so local analyzers never guess it."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"schema_version": 1, "effective_commands": commands}, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_controller(args: argparse.Namespace) -> int:
    executable = pathlib.Path(args.rpcs3).resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)
    requested_max_events = getattr(args, "max_events", None)
    max_events = requested_max_events or DEFAULT_MAX_EVENTS
    arm_seconds = getattr(args, "arm_seconds", None)
    authorization_timeout = getattr(args, "authorization_timeout", None)
    bounded = requested_max_events is not None or arm_seconds is not None
    graceful_lead = (
        min(GRACEFUL_STOP_LEAD_SECONDS, arm_seconds / 2.0)
        if arm_seconds is not None
        else None
    )
    watchdog_lead = (
        min(
            WATCHDOG_LEAD_SECONDS,
            max(0.0, arm_seconds - (graceful_lead or 0.0)) / 2.0,
        )
        if arm_seconds is not None
        else None
    )
    limits = {
        "max_events": max_events,
        "max_events_enforced": requested_max_events is not None,
        "arm_seconds": arm_seconds,
        "authorization_timeout_seconds": authorization_timeout,
        "graceful_stop_lead_seconds": graceful_lead,
        "watchdog_lead_seconds": watchdog_lead,
    }
    root = pathlib.Path(args.session_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    session = root / f"session-{stamp}"
    session.mkdir()
    capture_path = session / "capture.bin"
    control_path = session / "control.in"
    response_path = session / "control-responses.jsonl"
    status_path = session / "status.json"
    config_path = session / "probe-config.json"
    control_path.write_text("", encoding="utf-8")
    effective_commands: list[str] = []
    _write_probe_config(config_path, effective_commands)
    (root / "LATEST.txt").write_text(str(session), encoding="utf-8")

    environment = os.environ.copy()
    environment.update({
        "RPCS3_ASCENSION_LIVE_PROBE": "1",
        "RPCS3_ASCENSION_LIVE_PROBE_OUTPUT": str(capture_path),
        "RPCS3_ASCENSION_SPU_TASK_PROBE": "0",
        "RPCS3_ASCENSION_MFC_TASK_IDENTITY": "0",
        "RPCS3_CHARACTER_VERTEX_PROBE": "0",
        "RPCS3_TEMPORAL_CAPTURE": "1",
        "RPCS3_TEMPORAL_CAPTURE_VERBOSE": "0",
    })
    command = [str(executable)]
    if args.game:
        command.append(args.game)
    process = subprocess.Popen(command, cwd=executable.parent, env=environment)
    print(f"[live-probe] RPCS3 PID {process.pid}")
    print(f"[live-probe] session: {session}")
    _write_controller_status(status_path, "waiting_for_pipe", pid=process.pid, limits=limits)
    controller_error: BaseException | None = None
    stop_reason: str | None = None
    armed_at: str | None = None
    arm_acknowledged_at: str | None = None
    disarmed_at: str | None = None
    stopped_at: str | None = None
    arm_started_monotonic: float | None = None
    arm_duration_seconds: float | None = None
    final_probe: dict[str, object] | None = None
    planned_shutdown = False
    forced_shutdown = False
    guard_rejections = 0
    watchdog_cancel = threading.Event()
    watchdog_state: dict[str, object] = {"fired": False}
    watchdog_thread: threading.Thread | None = None

    def record_response(command_text: str, response: dict[str, object]) -> None:
        _append_jsonl(response_path, {"command": command_text, "response": response})

    try:
        with ProbePipe(args.pipe_timeout) as pipe:
            _write_controller_status(status_path, "configuring", pid=process.pid, limits=limits)
            for command_text in _commands_for_max_events(max_events):
                response = _require_ok(command_text, pipe.command(command_text))
                record_response(command_text, response)
                effective_commands.append(command_text)
                _write_probe_config(config_path, effective_commands)

            configured_probe = _require_ok("GET_STATS", pipe.command("GET_STATS"))
            if int(configured_probe.get("max_events", -1)) != max_events:
                raise RuntimeError(f"probe did not retain max_events={max_events}: {configured_probe}")
            if bounded or authorization_timeout is not None:
                authorization_deadline = time.monotonic() + (authorization_timeout or args.pipe_timeout)
                while not configured_probe.get("authorized"):
                    if process.poll() is not None:
                        raise RuntimeError("RPCS3 exited before the target executable was authorized")
                    if time.monotonic() >= authorization_deadline:
                        raise TimeoutError("timed out waiting for the authorized target executable")
                    _write_controller_status(
                        status_path,
                        "waiting_for_authorization",
                        pid=process.pid,
                        limits=limits,
                        probe=configured_probe,
                    )
                    time.sleep(0.25)
                    configured_probe = _require_ok("GET_STATS", pipe.command("GET_STATS"))

            start_command = f'START_CAPTURE path="{capture_path}"'
            start_response = _require_ok(start_command, pipe.command(start_command))
            record_response(start_command, start_response)
            if int(start_response.get("max_events", -1)) != max_events or not start_response.get("capturing"):
                raise RuntimeError(f"probe did not open with max_events={max_events}: {start_response}")

            armed_at = dt.datetime.now().astimezone().isoformat(timespec="milliseconds")
            arm_started_monotonic = time.monotonic()
            if arm_seconds is not None:
                watchdog_delay = max(0.01, arm_seconds - (watchdog_lead or 0.0))

                def enforce_arm_deadline() -> None:
                    if not watchdog_cancel.wait(watchdog_delay):
                        watchdog_state["fired"] = True
                        watchdog_state["fired_at"] = dt.datetime.now().astimezone().isoformat(timespec="milliseconds")
                        watchdog_state["fired_after_seconds"] = time.monotonic() - (arm_started_monotonic or 0.0)
                        if process.poll() is None:
                            nonlocal forced_shutdown
                            forced_shutdown = True
                            process.terminate()

                watchdog_thread = threading.Thread(
                    target=enforce_arm_deadline,
                    name="ascension-probe-arm-watchdog",
                    daemon=True,
                )
                watchdog_thread.start()

            arm_response = _require_ok("ARM", pipe.command("ARM"))
            record_response("ARM", arm_response)
            if not arm_response.get("armed"):
                raise RuntimeError(f"probe did not enter armed state: {arm_response}")
            arm_acknowledged_at = dt.datetime.now().astimezone().isoformat(timespec="milliseconds")
            print("[live-probe] capture is armed; bounded controller is monitoring it.")
            _write_controller_status(
                status_path,
                "armed",
                pid=process.pid,
                limits=limits,
                armed_at=armed_at,
                arm_acknowledged_at=arm_acknowledged_at,
                probe=arm_response,
            )

            control_offset = 0
            next_stats = 0.0
            next_status = 0.0
            while process.poll() is None:
                with control_path.open("r", encoding="utf-8") as stream:
                    stream.seek(control_offset)
                    for line in stream:
                        command_text = line.strip()
                        if not command_text or command_text.startswith("#"):
                            continue
                        if bounded and _changes_capture_guard(command_text):
                            guard_rejections += 1
                            record_response(
                                command_text,
                                {"ok": False, "error": "bounded capture guard is immutable while armed"},
                            )
                            continue
                        response = pipe.command(command_text)
                        if response.get("ok"):
                            effective_commands.append(command_text)
                            _write_probe_config(config_path, effective_commands)
                        record_response(command_text, response)
                    control_offset = stream.tell()
                now = time.monotonic()
                if watchdog_state.get("fired"):
                    stop_reason = "arm_watchdog"
                    break
                if arm_seconds is not None and arm_started_monotonic is not None:
                    graceful_deadline = arm_started_monotonic + arm_seconds - (graceful_lead or 0.0)
                    if now >= graceful_deadline:
                        stop_reason = "arm_deadline"
                        break
                if now >= next_stats:
                    status = _require_ok("GET_STATS", pipe.command("GET_STATS"))
                    final_probe = status
                    if requested_max_events is not None and int(status.get("accepted", 0)) >= max_events:
                        stop_reason = "max_events"
                        break
                    next_stats = time.monotonic() + 0.25
                    if now >= next_status:
                        _write_controller_status(
                            status_path,
                            "armed",
                            pid=process.pid,
                            limits=limits,
                            armed_at=armed_at,
                            arm_acknowledged_at=arm_acknowledged_at,
                            probe=status,
                        )
                        print(f"[live-probe] frame={status.get('frame')} accepted={status.get('accepted')} written={status.get('written')} dropped={status.get('dropped')}")
                        next_status = now + 5.0
                sleep_for = 0.05
                if arm_seconds is not None and arm_started_monotonic is not None:
                    sleep_for = min(sleep_for, max(0.0, graceful_deadline - time.monotonic()))
                time.sleep(sleep_for)

            if process.poll() is not None and stop_reason is None:
                if watchdog_state.get("fired"):
                    stop_reason = "arm_watchdog"
                    arm_duration_seconds = float(watchdog_state.get("fired_after_seconds", 0.0))
                else:
                    stop_reason = "rpcs3_exit"

            if stop_reason in {"arm_deadline", "max_events"} and process.poll() is None:
                disarm_response = _require_ok("DISARM", pipe.command("DISARM"))
                record_response("DISARM", disarm_response)
                disarmed_at = dt.datetime.now().astimezone().isoformat(timespec="milliseconds")
                arm_duration_seconds = time.monotonic() - (arm_started_monotonic or time.monotonic())
                watchdog_cancel.set()
                if watchdog_thread is not None:
                    watchdog_thread.join(timeout=1.0)
                time.sleep(POST_DISARM_QUIESCE_SECONDS)

                convergence_deadline = time.monotonic() + FLUSH_CONVERGENCE_SECONDS
                previous_counts: tuple[int, int, int] | None = None
                stable_count = 0
                while True:
                    flush_response = _require_ok("FLUSH", pipe.command("FLUSH"))
                    record_response("FLUSH", flush_response)
                    counts = (
                        int(flush_response.get("accepted", -1)),
                        int(flush_response.get("written", -1)),
                        int(flush_response.get("dropped", -1)),
                    )
                    stable_count = stable_count + 1 if counts == previous_counts else 1
                    if counts[0] == counts[1] + counts[2] and stable_count >= 2:
                        break
                    if time.monotonic() >= convergence_deadline:
                        raise RuntimeError(f"capture counters did not converge after DISARM: {flush_response}")
                    previous_counts = counts
                    time.sleep(0.05)

                final_probe = _require_ok("STOP_CAPTURE", pipe.command("STOP_CAPTURE"))
                record_response("STOP_CAPTURE", final_probe)
                stopped_at = dt.datetime.now().astimezone().isoformat(timespec="milliseconds")
                planned_shutdown = True
            elif arm_started_monotonic is not None:
                watchdog_cancel.set()
                arm_duration_seconds = time.monotonic() - arm_started_monotonic
    except (KeyboardInterrupt, Exception) as error:
        controller_error = error
        print(f"[live-probe] control channel ended: {error}", file=sys.stderr)
        if stop_reason is None:
            stop_reason = "controller_error"
        if watchdog_state.get("fired"):
            stop_reason = "arm_watchdog"
            arm_duration_seconds = float(watchdog_state.get("fired_after_seconds", 0.0))
        _write_controller_status(
            status_path,
            "controller_error",
            pid=process.pid,
            limits=limits,
            armed_at=armed_at,
            arm_acknowledged_at=arm_acknowledged_at,
            arm_duration_seconds=arm_duration_seconds,
            stop_reason=stop_reason,
            watchdog=watchdog_state,
            error=str(error),
        )
    finally:
        watchdog_cancel.set()
        if watchdog_thread is not None:
            watchdog_thread.join(timeout=1.0)
        if (controller_error is not None or planned_shutdown) and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
        exit_code = process.wait()

    analysis_exit_code: int | None = None
    aggregate: dict[str, object] | None = None
    summary_lines: int | None = None
    postprocess_error: str | None = None
    try:
        log_path = executable.parent / "log" / "RPCS3.log"
        if log_path.is_file():
            shutil.copy2(log_path, session / "RPCS3.log")
        if capture_path.is_file() and capture_path.stat().st_size >= FILE_HEADER_SIZE:
            analyzer = pathlib.Path(__file__).with_name("ascension_probe_auto_analyzer.py")
            if analyzer.is_file():
                analysis_process = subprocess.run(
                    [sys.executable, str(analyzer), str(capture_path), "--output-dir", str(session / "auto-analysis"),
                     "--top", "10", "--anomaly-limit", "5", "--max-summary-lines", "120"],
                    check=False,
                )
                analysis_exit_code = analysis_process.returncode
                summary_path = session / "auto-analysis" / "model-summary.txt"
                aggregate_path = session / "auto-analysis" / "top-candidates.json"
                if summary_path.is_file():
                    summary_lines = len(summary_path.read_text(encoding="utf-8").splitlines())
                if aggregate_path.is_file():
                    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
            else:
                print(f"[live-probe] compact analyzer is missing: {analyzer}", file=sys.stderr)
        else:
            print("[live-probe] no valid capture was produced", file=sys.stderr)
    except Exception as error:
        postprocess_error = str(error)
        print(f"[live-probe] post-processing failed: {error}", file=sys.stderr)

    capture_exists = capture_path.is_file()
    guard_violations = _capture_guard_violations(
        max_events=max_events,
        arm_seconds=arm_seconds,
        arm_duration_seconds=arm_duration_seconds,
        stop_reason=stop_reason,
        final_probe=final_probe,
        capture_exists=capture_exists,
        analysis_exit_code=analysis_exit_code,
        summary_lines=summary_lines,
        aggregate=aggregate,
    ) if bounded else []
    operational_failure = (
        controller_error is not None
        or postprocess_error is not None
        or (bounded and not planned_shutdown)
        or (capture_exists and analysis_exit_code != 0)
    )
    guard_failure = bounded and bool(guard_violations)
    _write_controller_status(
        status_path,
        "failed" if operational_failure or guard_failure else "complete",
        pid=process.pid,
        limits=limits,
        armed_at=armed_at,
        arm_acknowledged_at=arm_acknowledged_at,
        disarmed_at=disarmed_at,
        stopped_at=stopped_at,
        arm_duration_seconds=arm_duration_seconds,
        stop_reason=stop_reason,
        watchdog=watchdog_state,
        guard_rejections=guard_rejections,
        final_probe=final_probe,
        rpcs3_exit_code=exit_code,
        rpcs3_exit_was_controller_requested=planned_shutdown or forced_shutdown or bool(watchdog_state.get("fired")),
        controller_error=str(controller_error) if controller_error is not None else None,
        postprocess_error=postprocess_error,
        capture_exists=capture_exists,
        analysis_exit_code=analysis_exit_code,
        aggregate_summary_lines=summary_lines,
        capture_quality=aggregate.get("counts") if aggregate else None,
        capture_guard_valid=bounded and not guard_failure and not operational_failure,
        capture_guard_violations=guard_violations,
    )
    print(f"[live-probe] RPCS3 exit code: {exit_code}; all outputs: {session}")
    if operational_failure:
        return 1
    if guard_failure:
        return 2
    if planned_shutdown:
        return 0
    return exit_code


def append_command(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.session_root).resolve()
    if args.session:
        session = pathlib.Path(args.session).resolve()
    else:
        session = pathlib.Path((root / "LATEST.txt").read_text(encoding="utf-8").strip())
    with (session / "control.in").open("a", encoding="utf-8") as stream:
        stream.write(args.text.strip() + "\n")
    print(session / "control.in")
    return 0


def decode_to_jsonl(capture: Capture, output: pathlib.Path) -> None:
    def serializable(event: object) -> dict[str, object]:
        result = dataclasses.asdict(event)
        for pointer in result.get("pointers", []):
            pointer["sample"] = pointer["sample"].hex()
        if "stack" in result:
            result["stack"] = result["stack"].hex()
        if result.get("nearest_ppu"):
            result["nearest_ppu"] = result["nearest_ppu"]["sequence"]
        return result
    with output.open("w", encoding="utf-8") as stream:
        for kind, events in (("ppu", capture.ppu), ("spu", capture.spu), ("rsx", capture.rsx)):
            for event in events:
                stream.write(json.dumps({"type": kind, **serializable(event)}, ensure_ascii=False) + "\n")


def _pack_pointer(rule_id: int = 0, source: int = 0, address: int = 0, content_hash: int = 0) -> bytes:
    sample = struct.pack(">4I", address & 0xFFFFFFFF, source & 0xFFFFFFFF, rule_id, 0)
    return POINTER.pack(rule_id, 1 if rule_id else 0, source, address, content_hash, 16, 16, sample)


def _pack_event(event_type: int, sequence: int, time_us: int, frame: int,
                producer: int = 0, values: Iterable[int] = (), words: Iterable[int] = (),
                pointers: Iterable[bytes] = (), flags: int = EVENT_FLAG_AUTHORIZED) -> bytes:
    values_list = list(values)[:32] + [0] * 32
    words_list = list(words)[:64] + [0] * 64
    pointer_list = list(pointers)[:8] + [_pack_pointer()] * 8
    common = RECORD_HEADER.pack(EVENT_MAGIC, 1, event_type, EVENT_SIZE, flags, sequence,
                                time_us, frame, producer, 0x02000000, 0x928C, 0, 0xC490)
    return (common + VALUES.pack(*values_list[:32]) + WORDS.pack(*words_list[:64]) +
            b"".join(pointer_list[:8]))


def write_synthetic_capture(path: pathlib.Path) -> None:
    events: list[bytes] = []
    sequence = 1
    for frame in range(1, 5):
        for instance in range(2):
            time_us = 1_000_000 + frame * 10_000 + instance * 100
            ppu_values = [0] * 32
            ppu_values[7] = 0x500000 + instance * 0x1000
            ppu_words = [0] * 64
            ppu_words[0] = (1 << 7)
            events.append(_pack_event(EVENT_PPU, sequence, time_us - 20, frame,
                                      values=ppu_values, words=ppu_words))
            sequence += 1
            producer_sequence = sequence
            values = [0] * 32
            words = [0] * 64
            words[32] = 0
            words[36:45] = [frame * 10 + instance, 0x871C0C00, 256, 0x300000 + frame * 0x100,
                            0, 0x100000, 0, 0x800000 + frame * 0x20000 + instance * 0x4000,
                            0x800000 + frame * 0x20000 + instance * 0x4000 + 0x2000]
            words[45] = 1
            token = 0x500000 + instance * 0x1000
            words[46:53] = [7, token, token + 0x100, token + 0x200, 2, 1, 0]
            events.append(_pack_event(EVENT_SPU, sequence, time_us, frame,
                                      values=values, words=words,
                                      pointers=[_pack_pointer(1, 0x100000, token, token ^ 0xABCDEF)],
                                      flags=EVENT_FLAG_AUTHORIZED | EVENT_FLAG_MFC_PROVENANCE))
            sequence += 1
            rsx_words = [0] * 64
            rsx_words[1:4] = [11, 22, 512]
            rsx_words[6:10] = [words[43], 0x2000, 0x900000, 768]
            rsx_words[10:13] = [0xC3B5, 32, 5]
            rsx_words[18:29] = [words[36], words[37], words[38], words[39], words[40], words[41],
                                 words[42], words[43], words[44], 0, 0x2000]
            events.append(_pack_event(EVENT_RSX, sequence, time_us + 50, frame,
                                      producer=producer_sequence, words=rsx_words))
            sequence += 1
    title = b"BCAS25016\0".ljust(16, b"\0")
    app = b"01.12\0".ljust(16, b"\0")
    executable_hash = b"PPU-3a0b43e4a5f4bfea64f53612ee7c5d990f88129c\0".ljust(64, b"\0")
    label = b"synthetic self-test\0".ljust(64, b"\0")
    header = FILE_HEADER.pack(b"ALPROBE1", 1, 512, 1024, 64, 1_000_000,
                              len(events), 0, 0, title, app, executable_hash, label, bytes(296))
    with path.open("wb") as stream:
        stream.write(header)
        stream.writelines(events)


def self_test(args: argparse.Namespace) -> int:
    if args.output:
        path = pathlib.Path(args.output).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        directory = pathlib.Path(tempfile.mkdtemp(prefix="ascension-live-probe-"))
        path = directory / "synthetic.bin"
        cleanup = True
    write_synthetic_capture(path)
    capture = read_capture(path)
    analysis = analyze_capture(capture)
    assert len(capture.ppu) == 8 and len(capture.spu) == 8 and len(capture.rsx) == 8
    assert analysis["counts"]["rsx_draw_mapping_rate"] == 1.0
    stable = next(item for item in analysis["candidates"] if item["candidate"] == "pointer[1].address")
    assert stable["same_mesh_collision"] == 0.0 and stable["cross_frame_stability"] == 1.0
    print(format_report(analysis, 8))
    print(f"self-test passed: {path}")
    if cleanup:
        shutil.rmtree(path.parent)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    run = sub.add_parser("run", help="launch RPCS3 and manage one capture session")
    run.add_argument("--rpcs3", required=True)
    run.add_argument("--session-root", required=True)
    run.add_argument("--game", help="optional boot path; omit to select the game in the RPCS3 GUI")
    run.add_argument("--pipe-timeout", type=float, default=90.0)
    run.add_argument(
        "--max-events",
        type=_positive_int,
        help=f"strict accepted-event limit; probe default is {DEFAULT_MAX_EVENTS}",
    )
    run.add_argument(
        "--arm-seconds",
        type=_positive_finite_float,
        help="strict ARM-to-DISARM bound; cleanup and analyzer run after producers stop",
    )
    run.add_argument(
        "--authorization-timeout",
        type=_positive_finite_float,
        help="wait this long for the exact authorized title before ARM (boot time is excluded)",
    )
    run.set_defaults(func=run_controller)
    command = sub.add_parser("command", help="append a runtime command for the active controller")
    command.add_argument("--session-root", required=True)
    command.add_argument("--session")
    command.add_argument("text")
    command.set_defaults(func=append_command)
    decode = sub.add_parser("decode", help="decode a binary capture to JSONL")
    decode.add_argument("capture")
    decode.add_argument("--output", required=True)
    decode.set_defaults(func=lambda args: (decode_to_jsonl(read_capture(args.capture), pathlib.Path(args.output)), 0)[1])
    analyze = sub.add_parser("analyze", help="write JSON and concise text identity reports")
    analyze.add_argument("capture")
    analyze.add_argument("--json")
    analyze.add_argument("--text")
    def analyze_command(args: argparse.Namespace) -> int:
        result = analyze_capture(read_capture(args.capture))
        report = format_report(result)
        if args.json: pathlib.Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.text: pathlib.Path(args.text).write_text(report, encoding="utf-8")
        print(report)
        return 0
    analyze.set_defaults(func=analyze_command)
    test = sub.add_parser("self-test", help="exercise the ABI decoder and analyzer with synthetic events")
    test.add_argument("--output")
    test.set_defaults(func=self_test)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
