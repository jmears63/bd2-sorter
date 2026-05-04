"""Command-line interface for bd2_sorter."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import batdetect2.api as bd2_api

COMBINED_CSV_COLUMNS = [
    "filename",
    "time_exp",
    "duration",
    "class_name",
    "start_time",
    "end_time",
    "low_freq",
    "high_freq",
    "class",
    "class_prob",
    "det_prob",
    "individual",
    "event",
]


def parse_input_dir(raw_value: str) -> Path:
    """Parse and validate required input directory argument."""
    value = raw_value.strip()
    if not value:
        raise argparse.ArgumentTypeError("input_dir must be a non-empty directory path")
    return Path(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="bd2_sorter",
        description=(
            "Run BatDetect2 on each .wav file in a directory and "
            "write per-file JSON output into a raw subdirectory."
        ),
    )
    parser.add_argument(
        "input_dir",
        type=parse_input_dir,
        help="Path to a directory containing .wav files.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print active BatDetect2 configuration before processing files.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for downstream analysis (default: 0.5).",
    )
    parser.add_argument(
        "-j",
        "--junk",
        action="store_true",
        help="Move files with no over-threshold detections into junk directory.",
    )
    parser.add_argument(
        "-r",
        "--replace-symlinks",
        action="store_true",
        help="Replace existing symlinks when creating matched links.",
    )
    return parser.parse_args(argv)


def find_wav_files(input_dir: Path) -> list[Path]:
    """Return sorted .wav files in directory (case-insensitive extension)."""
    files = [path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".wav"]
    return sorted(files)


def write_raw_json(output_path: Path, prediction_dict: dict[str, Any]) -> None:
    """Write prediction JSON to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(prediction_dict, handle, indent=2)


def initialize_raw_combined_csv(raw_combined_csv_path: Path) -> None:
    """Initialize raw combined CSV file with a fresh header."""
    with raw_combined_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMBINED_CSV_COLUMNS)
        writer.writeheader()


def build_annotation_rows(
    prediction_dict: dict[str, Any],
    filename: str,
    threshold: float,
) -> list[dict[str, Any]]:
    """Build one CSV row per annotation with class_prob >= threshold."""
    base_fields = {
        "filename": filename,
        "time_exp": prediction_dict.get("time_exp"),
        "duration": prediction_dict.get("duration"),
        "class_name": prediction_dict.get("class_name"),
    }
    rows: list[dict[str, Any]] = []
    for annotation in prediction_dict.get("annotation", []):
        if not isinstance(annotation, dict):
            continue
        class_prob = parse_probability(annotation.get("class_prob"))
        if class_prob is None or class_prob < threshold:
            continue
        row = {
            **base_fields,
            "start_time": annotation.get("start_time"),
            "end_time": annotation.get("end_time"),
            "low_freq": annotation.get("low_freq"),
            "high_freq": annotation.get("high_freq"),
            "class": annotation.get("class"),
            "class_prob": class_prob,
            "det_prob": annotation.get("det_prob"),
            "individual": annotation.get("individual"),
            "event": annotation.get("event"),
        }
        rows.append(row)
    return rows


def append_rows_to_raw_combined_csv(raw_combined_csv_path: Path, rows: list[dict[str, Any]]) -> None:
    """Append rows to raw combined CSV."""
    if not rows:
        return
    with raw_combined_csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMBINED_CSV_COLUMNS)
        writer.writerows(rows)


def write_summary_csv(
    summary_csv_path: Path,
    per_file_class_probs: list[tuple[str, dict[str, float]]],
) -> None:
    """Write one row per input file with dynamic class probability columns."""
    detected_file_class_probs = [
        (filename, class_prob_map)
        for filename, class_prob_map in per_file_class_probs
        if class_prob_map
    ]
    class_columns = sorted(
        {
            class_key
            for _, class_prob_map in detected_file_class_probs
            for class_key in class_prob_map.keys()
        }
    )
    fieldnames = ["filename", *class_columns]
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for filename, class_prob_map in detected_file_class_probs:
            row: dict[str, Any] = {"filename": filename}
            for class_key in class_columns:
                row[class_key] = class_prob_map.get(class_key, "")
            writer.writerow(row)


def extract_file_timestamp(filename: str) -> str | None:
    """Extract YYYYMMDD_HHMMSS timestamp from filename."""
    stem = Path(filename).stem
    match = re.search(r"(\d{8}_\d{6})", stem)
    if not match:
        return None
    timestamp_str = match.group(1)
    try:
        datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
    except ValueError:
        return None
    return timestamp_str


def build_file_class_chirp_counts(
    prediction_dict: dict[str, Any],
    threshold: float,
) -> dict[str, int]:
    """Return class_key -> chirp_count for a file."""
    chirp_counts: dict[str, int] = {}

    for annotation in prediction_dict.get("annotation", []):
        if not isinstance(annotation, dict):
            continue
        class_name = annotation.get("class")
        if not isinstance(class_name, str) or not class_name.strip():
            continue
        class_prob = parse_probability(annotation.get("class_prob"))
        if class_prob is None or class_prob < threshold:
            continue

        class_key = to_short_class_key(class_name)
        if not class_key:
            continue

        chirp_counts[class_key] = chirp_counts.get(class_key, 0) + 1
    return chirp_counts


def compute_bucket_bounds(file_time: str, bucket_minutes: int = 15) -> tuple[str, str]:
    """Compute bucket start/end strings for a YYYYMMDD_HHMMSS timestamp."""
    parsed = datetime.strptime(file_time, "%Y%m%d_%H%M%S")
    minute_bucket = (parsed.minute // bucket_minutes) * bucket_minutes
    bucket_start = parsed.replace(minute=minute_bucket, second=0, microsecond=0)
    bucket_end = bucket_start + timedelta(minutes=bucket_minutes)
    return (
        bucket_start.strftime("%Y-%m-%d %H:%M:%S"),
        bucket_end.strftime("%Y-%m-%d %H:%M:%S"),
    )


def write_time_profile_csv(
    time_profile_csv_path: Path,
    per_file_time_profile: list[tuple[str, str, int]],
) -> None:
    """Write time-profile rows, one row per 15-minute time bucket."""
    class_columns = sorted({class_key for _, class_key, _ in per_file_time_profile})
    bucket_counts: dict[datetime, dict[str, int]] = {}
    for file_time, class_key, chirp_count in per_file_time_profile:
        bucket_start, bucket_end = compute_bucket_bounds(file_time)
        del bucket_end  # bucket_end is derived from bucket_start during write-out
        bucket_start_dt = datetime.strptime(bucket_start, "%Y-%m-%d %H:%M:%S")
        if bucket_start_dt not in bucket_counts:
            bucket_counts[bucket_start_dt] = {}
        bucket_counts[bucket_start_dt][class_key] = (
            bucket_counts[bucket_start_dt].get(class_key, 0) + chirp_count
        )

    class_fieldnames: list[str] = []
    for class_key in class_columns:
        class_fieldnames.append(class_key)
        class_fieldnames.append(f"{class_key}_ma_1h")
    fieldnames = ["bucket_start", "bucket_end", *class_fieldnames]

    if bucket_counts:
        min_bucket = min(bucket_counts.keys())
        max_bucket = max(bucket_counts.keys())
        ordered_buckets: list[datetime] = []
        cursor = min_bucket
        while cursor <= max_bucket:
            ordered_buckets.append(cursor)
            cursor += timedelta(minutes=15)
    else:
        ordered_buckets = []

    moving_avg_window_size = 4  # 4x15min = 1 hour trailing window
    with time_profile_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, bucket_start in enumerate(ordered_buckets):
            bucket_end = bucket_start + timedelta(minutes=15)
            class_count_map = bucket_counts.get(bucket_start, {})
            row: dict[str, Any] = {
                "bucket_start": bucket_start.strftime("%Y-%m-%d %H:%M:%S"),
                "bucket_end": bucket_end.strftime("%Y-%m-%d %H:%M:%S"),
            }
            for class_key in class_columns:
                row[class_key] = class_count_map.get(class_key, 0)
                window_start = max(0, index - moving_avg_window_size + 1)
                window = ordered_buckets[window_start : index + 1]
                window_sum = sum(
                    bucket_counts.get(bucket_dt, {}).get(class_key, 0) for bucket_dt in window
                )
                row[f"{class_key}_ma_1h"] = round(window_sum / len(window), 6)
            writer.writerow(
                row
            )


def print_class_detection_summary(per_file_class_probs: list[tuple[str, dict[str, float]]]) -> None:
    """Print class-level detection summary rows to stdout."""
    class_counts: dict[str, int] = {}
    for _, class_prob_map in per_file_class_probs:
        for class_key in class_prob_map.keys():
            class_counts[class_key] = class_counts.get(class_key, 0) + 1

    print("Class Summary")
    for class_key in sorted(class_counts.keys()):
        print(f"{class_key}: {class_counts[class_key]}")


def parse_probability(value: Any) -> float | None:
    """Parse a numeric probability value, returning None when invalid."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_short_class_key(class_name: str) -> str:
    """Convert class name to compact key (first 3 chars/word, lowercase)."""
    words = class_name.split()
    return "".join(word[:3].lower() for word in words if word)


def build_thresholded_class_prob_map(
    prediction_dict: dict[str, Any],
    threshold: float,
) -> dict[str, float]:
    """Return short_class_key -> max class_prob filtered by threshold."""
    max_prob_by_class: dict[str, float] = {}
    annotations = prediction_dict.get("annotation", [])
    if not isinstance(annotations, list):
        return {}

    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        class_name = annotation.get("class")
        if not isinstance(class_name, str) or not class_name.strip():
            continue
        short_class_key = to_short_class_key(class_name)
        if not short_class_key:
            continue
        class_prob = parse_probability(annotation.get("class_prob"))
        if class_prob is None:
            continue

        current_max = max_prob_by_class.get(short_class_key)
        if current_max is None or class_prob > current_max:
            max_prob_by_class[short_class_key] = class_prob

    return {
        class_key: class_prob
        for class_key, class_prob in max_prob_by_class.items()
        if class_prob >= threshold
    }


def ensure_matched_class_dirs(base_dir: Path, class_keys: list[str]) -> None:
    """Ensure matched, all, and per-class subdirectories exist."""
    matched_dir = base_dir / "matched"
    matched_dir.mkdir(parents=True, exist_ok=True)
    (matched_dir / "all").mkdir(parents=True, exist_ok=True)
    for class_key in class_keys:
        (matched_dir / class_key).mkdir(parents=True, exist_ok=True)


def create_symlink_in_dir(
    target_dir: Path,
    symlink_name: str,
    source_wav: Path,
    replace_symlinks: bool = False,
) -> None:
    """Create a symlink in target_dir, skipping conflicting existing paths."""
    symlink_path = target_dir / symlink_name
    relative_target = Path(os.path.relpath(source_wav, start=target_dir.resolve()))
    if symlink_path.exists() or symlink_path.is_symlink():
        if symlink_path.is_symlink() and replace_symlinks:
            symlink_path.unlink()
            symlink_path.symlink_to(relative_target)
            return
        if symlink_path.is_symlink() and symlink_path.resolve() == source_wav:
            return
        print(
            f"Skipping symlink creation, path exists: {symlink_path}",
            file=sys.stderr,
        )
        return
    symlink_path.symlink_to(relative_target)


def format_prob_score(probability: float) -> str:
    """Format probability as zero-padded score (prob * 1000)."""
    score = int(probability * 1000)
    if score < 0:
        score = 0
    return f"{score:03d}"


def create_class_symlinks(
    base_dir: Path,
    wav_file: Path,
    sorted_class_probs: list[tuple[str, float]],
    replace_symlinks: bool = False,
) -> None:
    """Create per-class and all symlinks to original wav file."""
    source_wav = wav_file.resolve()
    matched_dir = base_dir / "matched"
    top_class, _ = sorted_class_probs[0]
    all_symlink_name = f"ln_{top_class}_{wav_file.name}"
    create_symlink_in_dir(
        matched_dir / "all",
        all_symlink_name,
        source_wav,
        replace_symlinks=replace_symlinks,
    )
    for class_key, class_prob in sorted_class_probs:
        class_dir = matched_dir / class_key
        class_score = format_prob_score(class_prob)
        class_symlink_name = f"ln_{wav_file.stem}_{class_score}{wav_file.suffix}"
        create_symlink_in_dir(
            class_dir,
            class_symlink_name,
            source_wav,
            replace_symlinks=replace_symlinks,
        )


def print_processing_row(wav_file: Path, subdir_names: list[str], moved_to_junk: bool = False) -> None:
    """Print one standard processing row per wav file."""
    line = f"Processing {wav_file.name} {json.dumps(subdir_names)}"
    if moved_to_junk:
        line += " --> junk"
    print(line)


def move_wav_to_junk(input_dir: Path, wav_file: Path) -> Path:
    """Move wav file into input_dir/junk, avoiding name collisions."""
    junk_dir = input_dir / "junk"
    junk_dir.mkdir(parents=True, exist_ok=True)

    destination = junk_dir / wav_file.name
    if destination.exists() or destination.is_symlink():
        stem = wav_file.stem
        suffix = wav_file.suffix
        counter = 1
        while True:
            candidate = junk_dir / f"{stem}_{counter}{suffix}"
            if not candidate.exists() and not candidate.is_symlink():
                destination = candidate
                break
            counter += 1

    return wav_file.replace(destination)


def run(
    input_dir: Path,
    verbose: bool = False,
    threshold: float = 0.5,
    junk: bool = False,
    replace_symlinks: bool = False,
) -> int:
    """Execute bat detection pipeline for all .wav files in input_dir."""
    input_dir = input_dir.expanduser()
    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 2
    if not input_dir.is_dir():
        print(f"Input path is not a directory: {input_dir}", file=sys.stderr)
        return 2
    input_dir = input_dir.resolve()
    csv_dir = input_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    raw_combined_csv_path = csv_dir / "raw_combined.csv"
    summary_csv_path = csv_dir / "summary.csv"
    time_profile_csv_path = csv_dir / "time_profile.csv"
    initialize_raw_combined_csv(raw_combined_csv_path)
    (input_dir / "matched").mkdir(parents=True, exist_ok=True)

    wav_files = find_wav_files(input_dir)
    if not wav_files:
        print(f"No .wav files found in: {input_dir}")
        return 0

    raw_dir = input_dir / "raw"
    config = bd2_api.get_config()
    if verbose:
        print("Active BatDetect2 config:")
        print(json.dumps(config, indent=2, sort_keys=True, default=str))
        print(f"Analysis threshold: {threshold}")

    failures = 0
    per_file_class_probs: list[tuple[str, dict[str, float]]] = []
    per_file_time_profile: list[tuple[str, str, int]] = []
    for wav_file in wav_files:
        output_path = raw_dir / f"{wav_file.name}.json"
        output_path_resolved = output_path.resolve()
        if output_path_resolved.parent != raw_dir.resolve():
            failures += 1
            print(
                f"Failed {wav_file.name}: output path escaped raw directory",
                file=sys.stderr,
            )
            continue
        try:
            results = bd2_api.process_file(str(wav_file), config=config)
            prediction_dict = results["pred_dict"]
            thresholded_class_prob_map = build_thresholded_class_prob_map(
                prediction_dict,
                threshold=threshold,
            )
            per_file_class_probs.append((wav_file.name, thresholded_class_prob_map))
            file_timestamp = extract_file_timestamp(wav_file.name)
            if file_timestamp is None:
                print(
                    f"Failed {wav_file.name}: unable to extract YYYYMMDD_HHMMSS timestamp",
                    file=sys.stderr,
                )
            else:
                file_class_chirp_counts = build_file_class_chirp_counts(
                    prediction_dict,
                    threshold=threshold,
                )
                for class_key, chirp_count in file_class_chirp_counts.items():
                    per_file_time_profile.append(
                        (
                            file_timestamp,
                            class_key,
                            chirp_count,
                        )
                    )
            sorted_class_probs = sorted(
                thresholded_class_prob_map.items(),
                key=lambda item: (-item[1], item[0]),
            )
            sorted_class_keys = [class_key for class_key, _ in sorted_class_probs]
            moved_to_junk = False
            ensure_matched_class_dirs(input_dir, sorted_class_keys)
            if sorted_class_keys:
                create_class_symlinks(
                    input_dir,
                    wav_file,
                    sorted_class_probs,
                    replace_symlinks=replace_symlinks,
                )
                symlink_subdirs = ["all", *sorted_class_keys]
            elif junk:
                move_wav_to_junk(input_dir, wav_file)
                symlink_subdirs = []
                moved_to_junk = True
            else:
                symlink_subdirs = []
            if verbose:
                sorted_thresholded_class_prob_map = {
                    class_name: class_prob for class_name, class_prob in sorted_class_probs
                }
                print(json.dumps(sorted_thresholded_class_prob_map))
            print_processing_row(wav_file, symlink_subdirs, moved_to_junk=moved_to_junk)
            write_raw_json(output_path_resolved, prediction_dict)
            rows = build_annotation_rows(
                prediction_dict,
                filename=wav_file.name,
                threshold=threshold,
            )
            append_rows_to_raw_combined_csv(raw_combined_csv_path, rows)
        except Exception as err:  # noqa: BLE001
            failures += 1
            print(f"Failed {wav_file.name}: {err}", file=sys.stderr)

    if failures:
        write_summary_csv(summary_csv_path, per_file_class_probs)
        write_time_profile_csv(time_profile_csv_path, per_file_time_profile)
        print_class_detection_summary(per_file_class_probs)
        print(f"Completed with {failures} failure(s).", file=sys.stderr)
        return 1
    write_summary_csv(summary_csv_path, per_file_class_probs)
    write_time_profile_csv(time_profile_csv_path, per_file_time_profile)
    print_class_detection_summary(per_file_class_probs)
    return 0


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    args = parse_args(argv)
    raise SystemExit(
        run(
            args.input_dir,
            verbose=args.verbose,
            threshold=args.threshold,
            junk=args.junk,
            replace_symlinks=args.replace_symlinks,
        )
    )
