#!/usr/bin/env python3
"""Build a mounted RPCS3 texture-pack index from a Resource Studio bundle.

The index contains content keys and absolute paths to the edited DDS files. It
does not copy or modify the source project, and the output is replaced
atomically only after the complete bundle has passed validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from typing import NamedTuple


INDEX_HEADER = "RPCS3_TEXTURE_PACK_V1"
HASH_DOMAIN = b"RPCS3_TEXTURE_PACK_BC_V1"
MAX_DIMENSION = 16384
MAX_MIPS = 16
MAX_SCALE = 8
MAX_COMPRESSED_PIXELS = 128 * 1024 * 1024
FORMATS = {
    b"DXT1": (0x86, 8),
    b"DXT3": (0x87, 16),
    b"DXT5": (0x88, 16),
}


class DdsInfo(NamedTuple):
    width: int
    height: int
    mipmaps: int
    fourcc: str
    gcm_format: int
    levels: tuple[tuple[int, int, memoryview], ...]
    file_sha256: str


class IndexEntry(NamedTuple):
    semantic: int
    edit_sha256: str
    edit_path: Path


def _read_u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def parse_dds(path: Path) -> DdsInfo:
    data = path.read_bytes()
    if len(data) < 128 or data[:4] != b"DDS " or _read_u32(data, 4) != 124:
        raise ValueError(f"not a supported DDS file: {path}")
    if _read_u32(data, 76) != 32 or not (_read_u32(data, 80) & 0x4):
        raise ValueError(f"DDS does not use a FourCC format: {path}")
    if _read_u32(data, 112) != 0:
        raise ValueError(f"cubemap/volume DDS is not supported: {path}")

    width = _read_u32(data, 16)
    height = _read_u32(data, 12)
    mipmaps = max(1, _read_u32(data, 28))
    fourcc_bytes = data[84:88]
    if fourcc_bytes not in FORMATS:
        raise ValueError(f"unsupported DDS FourCC {fourcc_bytes!r}: {path}")
    if (
        width <= 0
        or height <= 0
        or width > MAX_DIMENSION
        or height > MAX_DIMENSION
        or width * height > MAX_COMPRESSED_PIXELS
        or mipmaps > MAX_MIPS
    ):
        raise ValueError(f"DDS dimensions or mip count exceed safety limits: {path}")

    gcm_format, block_size = FORMATS[fourcc_bytes]
    levels: list[tuple[int, int, memoryview]] = []
    view = memoryview(data)
    offset = 128
    for level in range(mipmaps):
        level_width = max(1, width >> level)
        level_height = max(1, height >> level)
        blocks_x = max(1, (level_width + 3) // 4)
        blocks_y = max(1, (level_height + 3) // 4)
        level_size = blocks_x * blocks_y * block_size
        if offset + level_size > len(data):
            raise ValueError(f"truncated DDS mip chain: {path}")
        levels.append((level_width, level_height, view[offset : offset + level_size]))
        offset += level_size
    if offset != len(data):
        raise ValueError(f"DDS has unexpected trailing data: {path}")

    return DdsInfo(
        width=width,
        height=height,
        mipmaps=mipmaps,
        fourcc=fourcc_bytes.decode("ascii"),
        gcm_format=gcm_format,
        levels=tuple(levels),
        file_sha256=hashlib.sha256(data).hexdigest().upper(),
    )


def content_key(dds: DdsInfo) -> str:
    digest = hashlib.sha1()
    digest.update(HASH_DOMAIN)
    digest.update(struct.pack("<IIII", dds.gcm_format, dds.width, dds.height, dds.mipmaps))
    for width, height, payload in dds.levels:
        digest.update(struct.pack("<III", width, height, len(payload)))
        digest.update(payload)
    return digest.hexdigest()


def _resolve_relative(root: Path, value: str) -> Path:
    return root.joinpath(*value.replace("\\", "/").split("/"))


def build_index(bundle_path: Path) -> tuple[dict[str, IndexEntry], dict[str, int]]:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle.get("schema") != "gowmod.ascension-bundle-texture-project.v2":
        raise ValueError(f"unsupported Resource Studio bundle schema: {bundle.get('schema')!r}")

    entries: dict[str, IndexEntry] = {}
    stats = {
        "groups": 0,
        "occurrences": 0,
        "duplicates": 0,
        "color_occurrences": 0,
        "normal_occurrences": 0,
    }
    groups = bundle.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("bundle contains no resource groups")

    for group_number, group in enumerate(groups, 1):
        relative_project = group.get("pairProjectRelativePath")
        if not isinstance(relative_project, str):
            raise ValueError("bundle group is missing pairProjectRelativePath")
        pair_directory = _resolve_relative(bundle_path.parent, relative_project)
        pair_manifest_path = pair_directory / "ascension_texture_project.json"
        pair = json.loads(pair_manifest_path.read_text(encoding="utf-8"))
        if pair.get("Schema") != "gowmod.ascension-texture-project.v2":
            raise ValueError(f"unsupported pair schema: {pair_manifest_path}")
        textures = pair.get("Textures")
        if not isinstance(textures, list):
            raise ValueError(f"pair has no texture list: {pair_manifest_path}")

        stats["groups"] += 1
        print(
            f"[{group_number:02d}/{len(groups):02d}] {pair_directory.name}: "
            f"validating {len(textures)} textures",
            flush=True,
        )
        for texture in textures:
            semantic = int(texture.get("Semantic", -1))
            if semantic not in (0, 1):
                raise ValueError(f"unsupported texture semantic in {pair_manifest_path}")
            source_path = _resolve_relative(pair_directory, texture["SourceRelativePath"])
            edit_path = _resolve_relative(pair_directory, texture["EditRelativePath"])
            if "\t" in str(edit_path) or "\n" in str(edit_path) or "\r" in str(edit_path):
                raise ValueError(f"DDS path cannot be represented in the TSV index: {edit_path}")

            source = parse_dds(source_path)
            edit = parse_dds(edit_path)
            manifest_format = str(texture.get("Format", ""))
            if (
                source.width != int(texture.get("Width", -1))
                or source.height != int(texture.get("Height", -1))
                or source.mipmaps != int(texture.get("MipCount", -1))
                or source.fourcc != manifest_format
                or source.file_sha256 != str(texture.get("SourceDdsSha256", "")).upper()
            ):
                raise ValueError(f"source DDS does not match its manifest: {source_path}")
            if edit.gcm_format != source.gcm_format or edit.mipmaps != source.mipmaps:
                raise ValueError(f"edited DDS changed format or mip count: {edit_path}")
            if (
                edit.width < source.width
                or edit.height < source.height
                or edit.width % source.width
                or edit.height % source.height
            ):
                raise ValueError(f"edited DDS is not an integer upscale: {edit_path}")
            scale_x = edit.width // source.width
            scale_y = edit.height // source.height
            if scale_x != scale_y or not 1 <= scale_x <= MAX_SCALE:
                raise ValueError(f"edited DDS scale is outside 1x-{MAX_SCALE}x: {edit_path}")

            key = content_key(source)
            candidate = IndexEntry(semantic, edit.file_sha256, edit_path.resolve())
            previous = entries.get(key)
            if previous:
                if previous.edit_sha256 != candidate.edit_sha256:
                    raise ValueError(f"one source texture maps to multiple edited DDS payloads: {source_path}")
                entries[key] = IndexEntry(
                    max(previous.semantic, candidate.semantic),
                    previous.edit_sha256,
                    previous.edit_path,
                )
                stats["duplicates"] += 1
            else:
                entries[key] = candidate

            stats["occurrences"] += 1
            stats["normal_occurrences" if semantic == 1 else "color_occurrences"] += 1

    return entries, stats


def write_index(path: Path, entries: dict[str, IndexEntry], stats: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        INDEX_HEADER,
        f"# groups={stats['groups']}",
        f"# occurrences={stats['occurrences']}",
        f"# unique_entries={len(entries)}",
        f"# duplicate_occurrences={stats['duplicates']}",
        f"# color_occurrences={stats['color_occurrences']}",
        f"# normal_occurrences={stats['normal_occurrences']}",
    ]
    for key in sorted(entries):
        item = entries[key]
        lines.append(f"{key}\t{item.semantic}\t{item.edit_path}")
    contents = "\n".join(lines) + "\n"

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=path.name + ".",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary.write(contents)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="ascension_bundle_project.json")
    parser.add_argument("--output", type=Path, help="destination .tsv pack index")
    parser.add_argument("--dry-run", action="store_true", help="validate and report without writing an index")
    args = parser.parse_args()
    if not args.dry_run and args.output is None:
        parser.error("--output is required unless --dry-run is used")

    try:
        entries, stats = build_index(args.bundle.resolve())
        if not args.dry_run:
            write_index(args.output.resolve(), entries, stats)
        print(
            "Validated "
            f"{stats['occurrences']} occurrences in {stats['groups']} groups; "
            f"{len(entries)} unique content keys, {stats['duplicates']} duplicates, "
            f"{stats['normal_occurrences']} normal candidates."
        )
        if args.dry_run:
            print("Dry run complete; no files were written.")
        else:
            print(f"Wrote mounted texture-pack index: {args.output.resolve()}")
        return 0
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
