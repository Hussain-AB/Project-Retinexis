#!/usr/bin/env python3
"""Apply the locked ISBI retrain decisions to the phase-2 configs.

Line-level edits, so the comments in each YAML survive. Idempotent: running it
twice changes nothing the second time. Prints a unified diff and, by default,
changes nothing -- pass --write to actually edit.

    python scripts/apply_isbi_config_edits.py            # dry run, shows the diff
    python scripts/apply_isbi_config_edits.py --write

What it sets, and why (decision numbers refer to the retrain decision record):

  output_dir: outputs_isbi        the discarded runs stay in outputs/ as the record,
                                  and the trainer's restart guard does not fire
  use_weighted_bce: false         decision 8 -- pos_weight ~7.0 on hypertension is the
  pos_weight: null                documented cause of ECE 0.16-0.22
  labels_csv: ..._dedup.csv       hypertension only -- 6,029 duplicate rows removed
  source_col: source_dataset      hypertension only -- lets the loader and the
                                  predictions CSV record which cohort each image is from
  input_size_override: 224        dinov3 only -- it otherwise reads its input size from
                                  the HF processor at runtime, so a model-card update
                                  could change resolution in the middle of the grid
"""
from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEDUP_CSV = "data/hypertension_combined/metadata_processed/combined_metadata_dedup.csv"
OUTPUT_DIR = "outputs_isbi"
DINOV3_INPUT_SIZE = 224

TARGETS = [
    "tasks/cimt/configs/phase2/*.yaml",
    "tasks/hypertension/configs/phase2/*.yaml",
]


def set_scalar(lines: list[str], key: str, value: str) -> tuple[list[str], bool]:
    """Replace `key: ...` at column 0. Returns (lines, changed)."""
    pat = re.compile(rf"^{re.escape(key)}\s*:\s*(.*?)\s*(#.*)?$")
    out, changed = [], False
    for ln in lines:
        m = pat.match(ln)
        if m and m.group(1) != value:
            comment = f"  {m.group(2)}" if m.group(2) else ""
            out.append(f"{key}: {value}{comment}")
            changed = True
        else:
            out.append(ln)
    return out, changed


def insert_after(lines: list[str], anchor_key: str, new_line: str) -> tuple[list[str], bool]:
    """Insert new_line right after the `anchor_key:` line, if its key is absent."""
    key = new_line.split(":", 1)[0]
    if any(re.match(rf"^{re.escape(key)}\s*:", ln) for ln in lines):
        return lines, False
    out, done = [], False
    for ln in lines:
        out.append(ln)
        if not done and re.match(rf"^{re.escape(anchor_key)}\s*:", ln):
            out.append(new_line)
            done = True
    return (out, True) if done else (lines, False)


def set_model_kwarg(lines: list[str], key: str, value: str) -> tuple[list[str], bool]:
    """Set `key: value` inside the top-level model_kwargs: block."""
    start = next((i for i, ln in enumerate(lines) if re.match(r"^model_kwargs\s*:", ln)), None)
    if start is None:
        return lines, False
    end = start + 1
    while end < len(lines) and (lines[end].startswith((" ", "\t")) or not lines[end].strip()):
        end += 1
    block = lines[start + 1:end]
    pat = re.compile(rf"^(\s+){re.escape(key)}\s*:\s*(.*?)\s*$")
    for i, ln in enumerate(block):
        m = pat.match(ln)
        if m:
            if m.group(2) == value:
                return lines, False
            block[i] = f"{m.group(1)}{key}: {value}"
            return lines[:start + 1] + block + lines[end:], True
    indent = next((re.match(r"^(\s+)", ln).group(1) for ln in block if ln.strip()), "  ")
    last = max((i for i, ln in enumerate(block) if ln.strip()), default=-1)
    block.insert(last + 1, f"{indent}{key}: {value}")
    return lines[:start + 1] + block + lines[end:], True


def process(path: Path) -> tuple[str, str, list[str]]:
    original = path.read_text()
    lines = original.splitlines()
    notes: list[str] = []
    is_hyp = "hypertension" in str(path)

    for fn, args, note in [
        (set_scalar, ("output_dir", OUTPUT_DIR), f"output_dir -> {OUTPUT_DIR}"),
        (set_scalar, ("use_weighted_bce", "false"), "use_weighted_bce -> false"),
        (set_scalar, ("pos_weight", "null"), "pos_weight -> null"),
    ]:
        lines, ch = fn(lines, *args)
        if ch:
            notes.append(note)

    if is_hyp:
        lines, ch = set_scalar(lines, "labels_csv", DEDUP_CSV)
        if ch:
            notes.append("labels_csv -> dedup CSV")
        lines, ch = insert_after(lines, "patient_id_col", "source_col: source_dataset")
        if ch:
            notes.append("source_col: source_dataset added")

    if re.search(r"^architecture\s*:\s*dinov3\s*$", original, re.M):
        lines, ch = set_model_kwarg(lines, "input_size_override", str(DINOV3_INPUT_SIZE))
        if ch:
            notes.append(f"model_kwargs.input_size_override -> {DINOV3_INPUT_SIZE}")

    updated = "\n".join(lines) + ("\n" if original.endswith("\n") else "")
    return original, updated, notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="apply the edits (default: dry run)")
    ap.add_argument("--configs", nargs="*", default=None)
    args = ap.parse_args()

    paths: list[Path] = []
    for pat in (args.configs or TARGETS):
        paths.extend(sorted(ROOT.glob(pat)))
    paths = sorted(set(paths))
    if not paths:
        print("no configs matched")
        return 1

    touched = 0
    for p in paths:
        original, updated, notes = process(p)
        rel = p.relative_to(ROOT)
        if original == updated:
            print(f"  ok        {rel}")
            continue
        touched += 1
        print(f"\n=== {rel}")
        for n in notes:
            print(f"    * {n}")
        print("".join(difflib.unified_diff(
            original.splitlines(keepends=True), updated.splitlines(keepends=True),
            fromfile=str(rel), tofile=str(rel), n=1)))
        if args.write:
            p.write_text(updated)

    print(f"\n{touched}/{len(paths)} configs {'updated' if args.write else 'would change'}")
    if touched and not args.write:
        print("dry run -- rerun with --write to apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
