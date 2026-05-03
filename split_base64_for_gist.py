#!/usr/bin/env python3
"""Split binary files into base64 text chunks suitable for GitHub Gist.

GitHub/Gist limits can vary by client and are not a good place to rely on one
huge text file. This script uses a conservative per-part size and writes:

  - <name>.manifest.json
  - <name>.part001ofNNN.b64.txt
  - <name>.restore.py

Upload the manifest, all part files, and the restore script to a Gist. On the
other side, run the restore script from the directory containing the parts.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_MAX_PART_MIB = 20.0
DEFAULT_LINE_WIDTH = 76


RESTORE_SCRIPT = r'''#!/usr/bin/env python3
"""Restore a binary file from split_base64_for_gist.py output."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="{manifest_name}",
        help="Manifest JSON path. Defaults to the manifest generated with this script.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path. Defaults to manifest original_filename.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output if it already exists.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent

    chunks = []
    for part in manifest["parts"]:
        part_path = root / part["filename"]
        text = part_path.read_text(encoding="ascii")
        compact = re.sub(r"\s+", "", text)
        actual_sha = sha256_bytes(compact.encode("ascii"))
        if actual_sha != part["base64_sha256"]:
            raise RuntimeError(
                f"Part checksum mismatch: {part_path} "
                f"expected={part['base64_sha256']} actual={actual_sha}"
            )
        chunks.append(compact)

    data = base64.b64decode("".join(chunks).encode("ascii"), validate=True)
    actual_file_sha = sha256_bytes(data)
    if actual_file_sha != manifest["sha256"]:
        raise RuntimeError(
            f"File checksum mismatch: expected={manifest['sha256']} actual={actual_file_sha}"
        )
    if len(data) != manifest["size_bytes"]:
        raise RuntimeError(
            f"File size mismatch: expected={manifest['size_bytes']} actual={len(data)}"
        )

    output = Path(args.output) if args.output else root / manifest["original_filename"]
    if output.exists() and not args.force:
        raise FileExistsError(f"{output} already exists; pass --force to overwrite")
    output.write_bytes(data)
    print(f"restored {output} ({len(data)} bytes, sha256={actual_file_sha})")


if __name__ == "__main__":
    main()
'''


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_name(path: Path) -> str:
    name = path.name.strip() or path.stem
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    name = name.strip("._")
    return name or "file"


def wrap_base64(text: str, line_width: int) -> str:
    if line_width <= 0:
        return text
    return "\n".join(text[i : i + line_width] for i in range(0, len(text), line_width)) + "\n"


def wrapped_size(num_chars: int, line_width: int) -> int:
    if line_width <= 0 or num_chars == 0:
        return num_chars
    return num_chars + math.ceil(num_chars / line_width)


def max_base64_chars_for_part(max_part_bytes: int, line_width: int) -> int:
    """Return the largest base64-char count whose wrapped text fits in bytes."""
    lo, hi = 0, max_part_bytes
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if wrapped_size(mid, line_width) <= max_part_bytes:
            lo = mid
        else:
            hi = mid - 1

    # Splitting at multiples of four keeps each part readable and padding-safe.
    lo -= lo % 4
    if lo <= 0:
        raise ValueError("max part size is too small for base64 splitting")
    return lo


def mib_to_bytes(value: float) -> int:
    if value <= 0:
        raise argparse.ArgumentTypeError("MiB value must be positive")
    return int(value * 1024 * 1024)


def write_index(out_dir: Path, manifests: list[dict]) -> None:
    upload_files = ["README_GIST_TRANSFER.md"]
    for manifest in manifests:
        upload_files.append(manifest["manifest_filename"])
        upload_files.append(manifest["restore_script"])
        upload_files.extend(part["filename"] for part in manifest["parts"])
    upload_cmd = "gh gist create --public " + " ".join(upload_files)

    lines = [
        "# Gist Base64 Transfer Index",
        "",
        "This directory is intended for cloud-to-local transfer via GitHub Gist.",
        "",
        "## Cloud upload",
        "",
        "Run this from the generated output directory on the cloud machine:",
        "",
        "```bash",
        upload_cmd,
        "```",
        "",
        "If the Gist should be private, remove `--public`.",
        "If `gh` is unavailable, manually upload every file in this directory to one Gist.",
        "",
        "## Local restore",
        "",
        "Clone or download the Gist locally, then run the matching restore command below.",
        "Base64 expands binary data by about 4/3, so large images may produce multiple parts.",
        "",
    ]
    for manifest in manifests:
        lines.extend(
            [
                f"## {manifest['original_filename']}",
                "",
                f"- Original size: {manifest['size_bytes']} bytes",
                f"- SHA256: `{manifest['sha256']}`",
                f"- Parts: {manifest['num_parts']}",
                f"- Manifest: `{manifest['manifest_filename']}`",
                f"- Restore: `{manifest['restore_script']}`",
                "",
                "Restore command:",
                "",
                "```bash",
                f"python {manifest['restore_script']} --manifest {manifest['manifest_filename']}",
                "```",
                "",
            ]
        )
    (out_dir / "README_GIST_TRANSFER.md").write_text("\n".join(lines), encoding="utf-8")


def split_file(path: Path, out_dir: Path, max_part_bytes: int, line_width: int, stem: str) -> dict:
    data = path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    max_chars = max_base64_chars_for_part(max_part_bytes, line_width)
    chunks = [encoded[i : i + max_chars] for i in range(0, len(encoded), max_chars)] or [""]

    original_filename = path.name
    num_parts = len(chunks)
    manifest_name = f"{stem}.manifest.json"
    restore_name = f"{stem}.restore.py"
    parts = []

    for idx, chunk in enumerate(chunks, start=1):
        part_name = f"{stem}.part{idx:03d}of{num_parts:03d}.b64.txt"
        part_text = wrap_base64(chunk, line_width)
        part_bytes = part_text.encode("ascii")
        if len(part_bytes) > max_part_bytes:
            raise RuntimeError(f"internal error: {part_name} exceeds max part size")
        (out_dir / part_name).write_bytes(part_bytes)
        compact = re.sub(r"\s+", "", part_text).encode("ascii")
        parts.append(
            {
                "index": idx,
                "filename": part_name,
                "base64_chars": len(chunk),
                "text_bytes": len(part_bytes),
                "base64_sha256": sha256_bytes(compact),
            }
        )

    manifest = {
        "format": "split_base64_for_gist.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "original_path": str(path),
        "original_filename": original_filename,
        "size_bytes": len(data),
        "sha256": sha256_bytes(data),
        "base64_chars": len(encoded),
        "max_part_bytes": max_part_bytes,
        "max_part_mib": round(max_part_bytes / 1024 / 1024, 4),
        "line_width": line_width,
        "num_parts": num_parts,
        "manifest_filename": manifest_name,
        "restore_script": restore_name,
        "parts": parts,
    }
    (out_dir / manifest_name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / restore_name).write_text(
        RESTORE_SCRIPT.replace("{manifest_name}", manifest_name),
        encoding="utf-8",
    )
    try:
        os.chmod(out_dir / restore_name, 0o755)
    except OSError:
        pass
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split image/binary files into base64 chunks for GitHub Gist transfer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", help="Image or binary files to encode.")
    parser.add_argument("--out-dir", default="gist_base64_parts", help="Directory for generated parts.")
    parser.add_argument(
        "--max-part-mib",
        type=float,
        default=DEFAULT_MAX_PART_MIB,
        help="Maximum size of each generated text part in MiB. Use a value safely below your Gist/client limit.",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=DEFAULT_LINE_WIDTH,
        help="Wrap base64 lines at this width. Use 0 for no wrapping.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Optional filename prefix. With multiple inputs, the input stem is appended if needed.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite the output directory if it exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [Path(item).expanduser().resolve() for item in args.inputs]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Input file not found: " + ", ".join(missing))
    if args.line_width < 0:
        raise ValueError("--line-width must be >= 0")

    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists():
        if not args.force and any(out_dir.iterdir()):
            raise FileExistsError(f"{out_dir} is not empty; pass --force or choose another --out-dir")
        if args.force:
            shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_part_bytes = mib_to_bytes(args.max_part_mib)
    stems: list[str] = []
    used_stems: set[str] = set()
    prefix_stem = safe_name(Path(args.prefix)) if args.prefix else None
    for path in paths:
        if prefix_stem and len(paths) == 1:
            base_stem = prefix_stem
        elif prefix_stem:
            base_stem = f"{prefix_stem}_{safe_name(Path(path.stem))}"
        else:
            base_stem = safe_name(Path(path.stem))

        stem = base_stem
        suffix = 2
        while stem in used_stems:
            stem = f"{base_stem}_{suffix}"
            suffix += 1
        used_stems.add(stem)
        stems.append(stem)

    manifests = [
        split_file(path, out_dir, max_part_bytes, args.line_width, stem)
        for path, stem in zip(paths, stems)
    ]
    write_index(out_dir, manifests)

    print(f"wrote {out_dir}")
    for manifest in manifests:
        print(
            f"{manifest['original_filename']}: "
            f"{manifest['size_bytes']} bytes -> {manifest['num_parts']} parts "
            f"(max {manifest['max_part_mib']} MiB/part)"
        )
    print("upload manifest + part files + restore script to Gist")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
