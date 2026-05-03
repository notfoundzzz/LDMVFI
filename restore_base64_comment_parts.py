#!/usr/bin/env python3
"""Restore a binary file from base64 text parts copied through comments.

This is the local-side fallback for cases where only `.part*.b64.txt` content
can be copied back. It does not require the manifest produced by
split_base64_for_gist.py; it simply sorts part files by filename, strips
whitespace, concatenates base64, decodes, and writes the output.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import re
import sys
from pathlib import Path


BASE64_RE = re.compile(r"^[A-Za-z0-9+/]*={0,2}$")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_part(path: Path, allow_fenced: bool) -> str:
    text = path.read_text(encoding="ascii")
    lines = text.splitlines()

    if allow_fenced:
        kept: list[str] = []
        in_fence = False
        saw_fence = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                saw_fence = True
                in_fence = not in_fence
                continue
            if in_fence:
                kept.append(stripped)
        if saw_fence:
            lines = kept

    compact = re.sub(r"\s+", "", "\n".join(lines))
    if not compact:
        raise ValueError(f"{path} is empty")
    if not BASE64_RE.match(compact):
        raise ValueError(
            f"{path} contains non-base64 characters. Save only the part body, "
            "or wrap the part in a markdown code fence and pass --allow-fenced."
        )
    return compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore an image/binary file from base64 txt chunks copied through comments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--parts-dir", default="comment_base64_parts", help="Directory containing copied txt parts.")
    parser.add_argument("--pattern", default="*.txt", help="Glob pattern for part files.")
    parser.add_argument("--output", required=True, help="Restored output path, e.g. restored_demo.png.")
    parser.add_argument("--expected-sha256", default="", help="Optional expected final SHA256.")
    parser.add_argument("--allow-fenced", action="store_true", help="Read only content inside markdown ``` fences.")
    parser.add_argument("--force", action="store_true", help="Overwrite output if it exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parts_dir = Path(args.parts_dir).expanduser().resolve()
    if not parts_dir.is_dir():
        raise FileNotFoundError(f"parts dir not found: {parts_dir}")

    part_paths = sorted(path for path in parts_dir.glob(args.pattern) if path.is_file())
    if not part_paths:
        raise FileNotFoundError(f"no files match {args.pattern!r} under {parts_dir}")

    chunks = [read_part(path, args.allow_fenced) for path in part_paths]
    encoded = "".join(chunks)
    data = base64.b64decode(encoded.encode("ascii"), validate=True)

    actual_sha = sha256_bytes(data)
    if args.expected_sha256 and actual_sha.lower() != args.expected_sha256.lower():
        raise RuntimeError(
            f"SHA256 mismatch: expected={args.expected_sha256.lower()} actual={actual_sha}"
        )

    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"{output} exists; pass --force to overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)

    print(f"parts={len(part_paths)}")
    print(f"base64_chars={len(encoded)}")
    print(f"wrote={output}")
    print(f"bytes={len(data)}")
    print(f"sha256={actual_sha}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
