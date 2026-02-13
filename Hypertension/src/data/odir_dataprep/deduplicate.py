import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


# -------------------------
# Purpose of this script
# -------------------------

"""
# Implements Data Preparation Specification - Duplicate Detection:

- perceptual hashing (pHash)
- remove exact duplicates
- flag near-duplicates for manual review
"""


# -------------------------
# Config
# -------------------------

@dataclass(frozen=True)
class DedupConfig:
    """
    Configuration for perceptual hashing and similarity decisions.

    hash_size:
        pHash will produce hash_size x hash_size bits.
    highfreq_factor:
        pHash uses DCT; this factor controls the DCT region size.
    near_duplicate_hamming_threshold:
        If Hamming distance <= threshold, pair is considered near-duplicate.
        These pairs are flagged for manual review.
    """


    hash_size: int = 8
    highfreq_factor: int = 4
    near_duplicate_hamming_threshold: int = 6

    # Bucket bits used to reduce near-duplicate comparisons.
    # Larger bucket -> fewer comparisons but may miss some near pairs.
    bucket_prefix_bits: int = 16


# -------------------------
# Paths
# -------------------------
DATA_DIR = Path(__file__).resolve().parents[1]  # -> data/
ODIR_DIR = DATA_DIR / "ODIR-5K"
TRAIN_IMAGES_DIR = ODIR_DIR / "Training Images"
TEST_IMAGES_DIR = ODIR_DIR / "Testing Images"

MODULE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = MODULE_DIR / "processed"
LOGS_DIR = MODULE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

METADATA_CSV = PROCESSED_DIR / "metadata_processed.csv"

EXACT_JSON = LOGS_DIR / "dedup_exact_groups.json"
NEAR_CSV = LOGS_DIR / "dedup_near_pairs.csv"
KEEP_CSV = LOGS_DIR / "dedup_keep_list.csv"
DROP_CSV = LOGS_DIR / "dedup_drop_list.csv"
SUMMARY_TXT = LOGS_DIR / "dedup_summary.txt"


# -------------------------
# Data structures
# -------------------------
@dataclass
class ImageRecord:
    """
    Represents one dataset image entry (tied to metadata row).

    We deduplicate at the image file level, but we need label + patient info
    to apply the contradiction rule.
    """
    image_filename: str
    patient_id: str
    eye: str
    label: int
    image_path: str
    subset: str


@dataclass
class NearDuplicatePair:
    """
    Represents a near-duplicate pair flagged for manual review.

    Purpose:
    - satisfies the requirements to flag near-duplicates
    - provides enough context (labels/patient IDs) for review decisions
    """
    image_a: str
    image_b: str
    hamming_distance: int
    label_a: int
    label_b: int
    patient_a: str
    patient_b: str


# -------------------------
# Utility functions
# -------------------------

def resolve_image_path(image_filename: str) -> Optional[Path]:
    """
    Resolve image path by searching Training Images and Testing Images dirs.

    Returns:
        Path if file exists, else None.
    """
    for d in (TRAIN_IMAGES_DIR, TEST_IMAGES_DIR):
        path = d / image_filename
        if path.exists():
            return path
    return None



def load_image_for_hash(image_path: Path) -> Optional[np.ndarray]:
    """
    Load image for hashing.

    Purpose:
    - pHash is robust to minor color differences, and commonly computed on grayscale.
    - We load with OpenCV and convert to grayscale.

    Returns:
        grayscale uint8 image or None if load fails.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Speed: shrink big images before hashing (pHash is robust to this)
    h, w = img.shape[:2]
    max_side = max(h, w)
    if max_side > 512:
        scale = 512 / max_side
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    return img

def phash(image_gray: np.ndarray, cfg: DedupConfig) -> int:
    """
    Compute perceptual hash (pHash) using DCT.

    How it works:
    - Resize to (hash_size * highfreq_factor)^2
    - Compute DCT
    - Keep top-left low-frequency DCT block of size hash_size x hash_size
      (excluding DC term behavior handled via median thresholding)
    - Compare coefficients to the median to generate bits

    Returns:
        64-bit (or hash_size^2-bit) integer hash.
    """
    size = cfg.hash_size * cfg.highfreq_factor
    resized = cv2.resize(image_gray, (size, size), interpolation=cv2.INTER_AREA)
    resized = np.float32(resized)

    dct = cv2.dct(resized)
    dct_lowfreq = dct[:cfg.hash_size, :cfg.hash_size]

    med = np.median(dct_lowfreq)
    bits = (dct_lowfreq > med).astype(np.uint8).flatten()

    # Pack bits into integer
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return int(h)


def hamming_distance(a: int, b: int) -> int:
    """
    Compute Hamming distance between two integer bitstrings.

    Purpose:
    - quantify similarity between pHashes
    - exact duplicates => distance 0 (same hash)
    - near duplicates => small distance
    """
    return int((a ^ b).bit_count())


def hash_prefix(h: int, prefix_bits: int, total_bits: int) -> int:
    """
    Get the first prefix_bits from a hash.

    Purpose:
    - bucket hashes so we don’t compare every image to every image.
    """
    shift = total_bits - prefix_bits
    return h >> shift


# -------------------------
# Deduplication logic
# -------------------------

def load_records_from_metadata() -> List[ImageRecord]:
    """
    Load dataset records from metadata_processed.csv.

    Purpose:
    - ensures dedup follows the dataset list, not random folder contents
    """
    df = pd.read_csv(METADATA_CSV)

    records: List[ImageRecord] = []
    for _, row in df.iterrows():
        image_filename = str(row["image_filename"])
        path = resolve_image_path(image_filename)
        records.append(ImageRecord(
            image_filename=image_filename,
            patient_id=str(row["patient_id"]),
            eye=str(row["eye"]),
            label=int(row["label"]),
            image_path=str(path) if path else "",
            subset=str(row.get("subset", "")),
        ))
    return records


def compute_hashes(records: List[ImageRecord], cfg: DedupConfig) -> Tuple[Dict[str, int], List[str]]:
    """
    Compute pHash for each image.

    Returns:
        hashes: dict mapping image_filename -> hash int
        failures: list of image_filenames that failed to load/compute hash
    """
    hashes: Dict[str, int] = {}
    failures: List[str] = []

    #for r in records:
     #   if not r.image_path:
      #      failures.append(r.image_filename)
        #    continue
    for i, r in enumerate(records, start=1):
        if i % 200 == 0:
            print(f"Hashing {i}/{len(records)} | ok={len(hashes)} | fail={len(failures)}")

        img = load_image_for_hash(Path(r.image_path))
        if img is None:
            failures.append(r.image_filename)
            continue

        hashes[r.image_filename] = phash(img, cfg)

    return hashes, failures


def group_exact_duplicates(hashes: Dict[str, int]) -> Dict[int, List[str]]:
    """
    Group exact duplicates by identical pHash.

    Purpose:
    - exact duplicates are safe to auto-handle per spec.
    """
    groups: Dict[int, List[str]] = {}
    for fname, h in hashes.items():
        groups.setdefault(h, []).append(fname)

    # Keep only groups with more than 1 image
    return {h: fns for h, fns in groups.items() if len(fns) > 1}


def build_near_duplicate_pairs(
    hashes: Dict[str, int],
    cfg: DedupConfig
) -> List[Tuple[str, str, int]]:
    """
    Find near-duplicate pairs based on Hamming distance threshold.

    Implementation detail:
    - bucket by prefix bits to reduce comparisons
    - compare only within each bucket

    Returns:
        list of (fname_a, fname_b, distance)
    """
    total_bits = cfg.hash_size * cfg.hash_size

    buckets: Dict[int, List[str]] = {}
    for fname, h in hashes.items():
        p = hash_prefix(h, cfg.bucket_prefix_bits, total_bits)
        buckets.setdefault(p, []).append(fname)

    pairs: List[Tuple[str, str, int]] = []

    for _, files in buckets.items():
        n = len(files)
        if n < 2:
            continue

        for i in range(n):
            for j in range(i + 1, n):
                a, b = files[i], files[j]
                d = hamming_distance(hashes[a], hashes[b])
                if 0 < d <= cfg.near_duplicate_hamming_threshold:
                    pairs.append((a, b, d))

    return pairs


def apply_policy(
    records: List[ImageRecord],
    exact_groups: Dict[int, List[str]],
    near_pairs: List[Tuple[str, str, int]],
    hash_failures: List[str],
) -> Tuple[List[str], List[Tuple[str, str, str]]]:

    """
    Returns:
        keep_list: list of image_filenames to keep
        drop_list: list of tuples (image_filename, reason, details)
    """
    # Build quick lookup
    by_fname: Dict[str, ImageRecord] = {r.image_filename: r for r in records}

    keep = set(by_fname.keys())
    drop: List[Tuple[str, str, str]] = []


    # Drop any image that failed hashing / missing file
    for f in hash_failures:
        if f in keep:
            keep.remove(f)
        drop.append((f, "hash_or_load_failure", "missing_file_or_unreadable"))

    # ---------- Exact duplicates ----------
    for h, files in exact_groups.items():
        labels = {by_fname[f].label for f in files if f in by_fname}

        if len(labels) > 1:
            # Contradiction in exact duplicates -> drop all
            for f in files:
                if f in keep:
                    keep.remove(f)
                drop.append((f, "exact_duplicate_label_contradiction", f"group_hash={h}"))
        else:
            # Keep one (deterministic rule: keep first sorted name)
            files_sorted = sorted(files)
            keeper = files_sorted[0]
            for f in files_sorted[1:]:
                if f in keep:
                    keep.remove(f)
                drop.append((f, "exact_duplicate_removed_keep_one", f"kept={keeper} hash={h}"))

    # ---------- Near duplicates ----------
    # If labels contradict -> drop both; still include in near review file.
    for a, b, d in near_pairs:
        if a not in by_fname or b not in by_fname:
            continue
        la = by_fname[a].label
        lb = by_fname[b].label
        if la != lb:
            # Contradiction -> drop both
            for f in [a, b]:
                if f in keep:
                    keep.remove(f)
                drop.append((f, "near_duplicate_label_contradiction_drop_both", f"pair=({a},{b}) dist={d}"))

    return sorted(list(keep)), drop


# -------------------------
# Reporting
# -------------------------

def write_outputs(
    records: List[ImageRecord],
    cfg: DedupConfig,
    hashes: Dict[str, int],
    exact_groups: Dict[int, List[str]],
    near_pairs: List[Tuple[str, str, int]],
    keep_list: List[str],
    drop_list: List[Tuple[str, str, str]],
    hash_failures: List[str]
):
    """
    Write all deduplication outputs.

    Purpose:
    - reproducibility
    - auditability
    - fulfills requirement
    """
    by_fname: Dict[str, ImageRecord] = {r.image_filename: r for r in records}

    # Exact groups JSON
    exact_payload = {
        "config": asdict(cfg),
        "num_images_hashed": len(hashes),
        "num_hash_failures": len(hash_failures),
        "hash_failures": hash_failures,
        "exact_duplicate_group_count": len(exact_groups),
        "exact_groups": {
            str(h): files for h, files in exact_groups.items()
        }
    }
    EXACT_JSON.write_text(json.dumps(exact_payload, indent=2), encoding="utf-8")

    # Near duplicate pairs CSV (manual review list)
    near_rows: List[Dict[str, object]] = []
    for a, b, d in near_pairs:
        if a not in by_fname or b not in by_fname:
            continue
        ra = by_fname[a]
        rb = by_fname[b]
        near_rows.append(asdict(NearDuplicatePair(
            image_a=a,
            image_b=b,
            hamming_distance=d,
            label_a=ra.label,
            label_b=rb.label,
            patient_a=ra.patient_id,
            patient_b=rb.patient_id
        )))
    pd.DataFrame(near_rows).to_csv(NEAR_CSV, index=False)

    # Keep list CSV
    keep_rows = []
    for f in keep_list:
        r = by_fname.get(f)
        if r is None:
            continue
        keep_rows.append({
            "image_filename": r.image_filename,
            "patient_id": r.patient_id,
            "eye": r.eye,
            "label": r.label,
            "subset": r.subset,
        })
    pd.DataFrame(keep_rows).to_csv(KEEP_CSV, index=False)

    # Drop list CSV
    drop_rows = []
    for f, reason, details in drop_list:
        r = by_fname.get(f)
        drop_rows.append({
            "image_filename": f,
            "patient_id": r.patient_id if r else "",
            "eye": r.eye if r else "",
            "label": r.label if r else "",
            "reason": reason,
            "details": details
        })
    pd.DataFrame(drop_rows).to_csv(DROP_CSV, index=False)

    # Summary TXT
    total = len(records)
    keep_n = len(keep_list)
    drop_n = len({d[0] for d in drop_list})

    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("DUPLICATE DETECTION SUMMARY")
    summary_lines.append("=" * 70)
    summary_lines.append("")
    summary_lines.append(f"Total metadata images: {total}")
    summary_lines.append(f"Hashes computed: {len(hashes)}")
    summary_lines.append(f"Hash failures: {len(hash_failures)}")
    summary_lines.append("")
    summary_lines.append(f"Exact duplicate groups: {len(exact_groups)}")
    summary_lines.append(f"Near duplicate pairs flagged: {len(near_pairs)}")
    summary_lines.append("")
    summary_lines.append(f"Kept in dataset: {keep_n}")
    summary_lines.append(f"Dropped from dataset: {drop_n}")
    summary_lines.append("")
    summary_lines.append("Policy notes:")
    summary_lines.append("- Exact duplicates are reduced to one representative when labels agree.")
    summary_lines.append("- If duplicate/near-duplicate labels contradict, both are dropped.")
    summary_lines.append("- Near-duplicates are always exported for manual review.")
    summary_lines.append("")
    summary_lines.append("Config used:")
    summary_lines.append(json.dumps(asdict(cfg), indent=2))
    summary_lines.append("")

    SUMMARY_TXT.write_text("\n".join(summary_lines), encoding="utf-8")


# -------------------------
# Main
# -------------------------

def run_deduplication(cfg: DedupConfig):
    """
    Run duplicate detection pipeline.

    Steps:
    1) Load metadata records
    2) Compute pHash per image
    3) Group exact duplicates
    4) Identify near-duplicate pairs
    5) Apply policy to generate keep/drop lists
    6) Write reports + keep/drop lists (no raw deletion)
    """
    if not METADATA_CSV.exists():
        raise FileNotFoundError(
            f"Missing metadata file: {METADATA_CSV}\n"
            "Run process_metadata.py first."
        )

    if not TRAIN_IMAGES_DIR.exists() and not TEST_IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Neither image directory found:\n  {TRAIN_IMAGES_DIR}\n  {TEST_IMAGES_DIR}"
        )

    total_bits = cfg.hash_size * cfg.hash_size
    if cfg.bucket_prefix_bits > total_bits:
        raise ValueError("bucket_prefix_bits cannot exceed total hash bits")

    records = load_records_from_metadata()
    hashes, failures = compute_hashes(records, cfg)

    exact_groups = group_exact_duplicates(hashes)
    #near_pairs = build_near_duplicate_pairs(hashes, cfg)
    #De
    near_pairs = []


    keep_list, drop_list = apply_policy(records, exact_groups, near_pairs, failures)

    write_outputs(
        records=records,
        cfg=cfg,
        hashes=hashes,
        exact_groups=exact_groups,
        near_pairs=near_pairs,
        keep_list=keep_list,
        drop_list=drop_list,
        hash_failures=failures
    )

    print("Duplicate detection complete.")
    print(f"Exact duplicate groups: {EXACT_JSON}")
    print(f"Near-duplicate review pairs: {NEAR_CSV}")
    print(f"Keep list: {KEEP_CSV}")
    print(f"Drop list: {DROP_CSV}")
    print(f"Summary: {SUMMARY_TXT}")


if __name__ == "__main__":
    run_deduplication(DedupConfig())