#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create an AudioCraft‑ready subset of the MTG‑Jamendo dataset.

Usage (example):
python make_jamendo_subset.py \
       --jamendo_audio_root  /data/jamendo/audio \
       --meta_tsv            mtg-jamendo-dataset/data/raw_30s_cleantags.tsv \
       --genres              "rock pop hiphop electronic jazz classical reggae metal folk blues" \
       --num_per_genre       50 \
       --out_root            ./dataset/jamendo_subset \
       --split_pct_valid     0.1
"""

import argparse
import csv
import gzip
import json
import random
import shutil
import subprocess
import textwrap
import wave
from collections import defaultdict
from math import ceil, isfinite
from pathlib import Path


def deduplicate_tracks(rows):
    """Keep one metadata row per track, ordered independently of genre sampling."""
    by_id = {}
    for row in rows:
        by_id.setdefault(row["track_id"], row)
    return [by_id[track_id] for track_id in sorted(by_id)]


def validate_split_fraction(value):
    """Allow a training-only dataset, but always reserve room for training."""
    if not 0 <= value < 1:
        raise ValueError("split_pct_valid must be at least 0 and less than 1")
    return value


def split_track_ids(rows, wanted_genres, valid_fraction, seed):
    """Return disjoint training/validation IDs, stratified by first matching genre.

    A multi-genre track belongs to one stratum and one split. Each nonempty
    stratum retains at least one training track; singleton strata stay in training.
    The seed and input records determine the split without changing global RNG state.
    """
    validate_split_fraction(valid_fraction)
    wanted_genres = set(wanted_genres)
    by_genre = defaultdict(list)
    for row in deduplicate_tracks(rows):
        genre = next((g for g in row["genres"] if g in wanted_genres), None)
        if genre is None:
            raise ValueError(f"Track {row['track_id']} has no requested genre")
        by_genre[genre].append(row["track_id"])

    rng = random.Random(seed)
    train_ids, valid_ids = set(), set()
    for genre in sorted(by_genre):
        track_ids = by_genre[genre]
        n_valid = min(len(track_ids) - 1, ceil(len(track_ids) * valid_fraction))
        genre_valid = set(rng.sample(track_ids, n_valid))
        valid_ids.update(genre_valid)
        train_ids.update(set(track_ids) - genre_valid)
    return train_ids, valid_ids


def locate_mp3(track_id: int, base_folder: Path, rel_path: str) -> Path:
    """
    Return an existing .mp3 file for *track_id*.

    1.  Try the exact path listed in the Jamendo TSV.
    2.  Fallback: look in the *same directory* for any file that begins
        with the numeric track‑id and ends with '.mp3'
        (e.g. 123456.low.mp3, 123456.preview.mp3, …).
    3.  As a last resort (very unlikely) search the whole tree once.
    """
    exact = base_folder / rel_path
    if exact.exists():
        return exact

    folder = exact.parent
    candidates = sorted(folder.glob(f"{track_id}*.mp3"))
    if candidates:
        return candidates[0]

    # very slow, but runs only if the dataset is organised differently
    deep = sorted(base_folder.rglob(f"{track_id}*.mp3"))
    if deep:
        return deep[0]

    raise FileNotFoundError(f"Could not find any *.mp3 for track {track_id}")


def mp3_exists(row, base_folder):
    """
    Return True if an .mp3 file for this TSV row is present on disk,
    False otherwise (no exception raised).
    """
    try:
        locate_mp3(row["track_id"], base_folder, row["path"])
        return True
    except FileNotFoundError:
        return False


# ───────────────────────────── ffmpeg helpers ──────────────────────────────
def ffmpeg(cmd):
    result = subprocess.run(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(result.stdout.decode())


def chunk_and_resample(src: Path, dst_folder: Path, chunk_sec: int):
    """Cut *src* into ≤chunk_sec clips at 32 kHz mono WAV."""
    proc = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(src)
    ],
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    if proc.returncode:
        raise RuntimeError(f"Could not inspect {src}: {proc.stderr.decode(errors='replace')}")
    duration = float(proc.stdout)
    if not isfinite(duration) or duration <= 0:
        raise ValueError(f"Invalid audio duration for {src}: {duration}")
    n_clips = ceil(duration / chunk_sec)

    clips = []
    for i in range(n_clips):
        out_wav = dst_folder / f"{src.stem}_{i:03d}.wav"
        ffmpeg([
            "ffmpeg", "-y", "-v", "error", "-ss",
            str(i * chunk_sec), "-t",
            str(chunk_sec), "-i",
            str(src), "-ar", "32000", "-ac", "1", "-c:a", "pcm_s16le",
            str(out_wav)
        ])
        # MP3 container durations can include encoder padding beyond the audio.
        if wav_duration(out_wav) > 0:
            clips.append(out_wav)
        else:
            out_wav.unlink()
    if not clips:
        raise RuntimeError(f"No audio was decoded from {src}")
    return clips


def wav_duration(path):
    """Read the actual duration of a generated PCM WAV, including short tails."""
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


def prepare_output(out_root, overwrite=False):
    """Require opt-in before replacing this exporter's known output paths."""
    outputs = [out_root / name for name in (
        "train", "valid", "train.jsonl.gz", "valid.jsonl.gz",
        "train.jsonl.gz.partial", "valid.jsonl.gz.partial",
    )]
    existing = [path for path in outputs if path.exists() or path.is_symlink()]
    if existing and not overwrite:
        raise ValueError("Export outputs already exist; use --overwrite to replace train/valid outputs")
    for path in existing:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    (out_root / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "valid").mkdir(exist_ok=True)


def write_sidecar(wav: Path, info: dict):
    with open(wav.with_suffix(".json"), "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def append_manifest(line: dict, manifest_path: Path):
    with gzip.open(manifest_path, "at") as f:
        f.write(json.dumps(line) + "\n")


# ───────────────────────────── TSV reader ──────────────────────────────────
def read_jamendo_meta(tsv_file: Path):
    """
    Minimal re‑implementation of scripts/commons.py::read_file
    that only keeps what we need for this subset script.
    """
    data = []
    with open(tsv_file, newline="") as fp:
        reader = csv.reader(fp, delimiter="\t")
        next(reader)  # skip header row
        for row in reader:
            track_id = int(row[0].split("_")[1])
            artist_id = int(row[1].split("_")[1])
            title = f"track_{track_id}"
            duration = float(row[4])
            tags = row[5:]  # the variable‑length tail
            genres = [
                t.split("---")[1] for t in tags if t.startswith("genre---")
            ]
            data.append({
                "track_id": track_id,
                "artist_id": artist_id,
                "title": title,
                "duration": duration,
                "tags": tags,
                "genres": genres,
                "path":
                    row[3]  # relative MP3 path inside the Jamendo dump
            })
    return data


# ───────────────────────────── main logic ─────────────────────────────────
def main():
    from tqdm import tqdm

    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=textwrap.dedent(__doc__))
    p.add_argument("--jamendo_audio_root", required=True)
    p.add_argument("--meta_tsv", required=True)
    p.add_argument("--genres",
                   required=True,
                   help="Space-separated genre tags to keep (or 'all')")
    p.add_argument("--num_per_genre",
                   type=int,
                   default=50,
                   help="Tracks per genre. Use -1 for all available.")
    p.add_argument("--out_root", default="./dataset/jamendo_subset")
    p.add_argument("--overwrite", action="store_true",
                   help="Replace existing train/valid directories and manifests in out_root.")
    p.add_argument("--split_pct_valid", type=float, default=0.1,
                   help="Validation fraction in [0, 1); singleton genres stay in training.")
    p.add_argument("--chunk", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    try:
        validate_split_fraction(args.split_pct_valid)
    except ValueError as error:
        p.error(str(error))
    if args.num_per_genre != -1 and args.num_per_genre <= 0:
        p.error("num_per_genre must be positive or -1 for all available tracks")
    if args.chunk <= 0:
        p.error("chunk must be positive")

    print("▶  Parsing Jamendo metadata …")
    rows = deduplicate_tracks(read_jamendo_meta(Path(args.meta_tsv)))

    if args.genres == "all":
        # We need to scan rows to find all genres
        all_genres = set()
        for r in rows:
            all_genres.update(r["genres"])
        wanted_genres = sorted(all_genres)
        print(f"▶  Found {len(wanted_genres)} unique genres.")
    else:
        wanted_genres = sorted(set(args.genres.split()))
    if not wanted_genres:
        p.error("No genres were selected")
    rng = random.Random(args.seed)

    # bucket rows by genre
    buckets = defaultdict(list)
    for r in rows:
        for g in sorted(set(r["genres"])):
            if g in wanted_genres:
                buckets[g].append(r)

    selected = []  # will hold ONLY rows with existing audio
    warnings_issued = set()  # to print each missing‑file warning once

    for g in wanted_genres:
        rng.shuffle(buckets[g])  # shuffle so replacements are random
        kept = []

        for row in buckets[g]:
            if mp3_exists(row, Path(args.jamendo_audio_root)):
                kept.append(row)
            else:
                if row["track_id"] not in warnings_issued:
                    print(
                        f"⚠️  Missing audio for track {row['track_id']} – skipping"
                    )
                    warnings_issued.add(row["track_id"])

            if args.num_per_genre > 0 and len(kept) == args.num_per_genre:
                break  # we have enough valid rows for this genre

        if args.num_per_genre > 0 and len(kept) < args.num_per_genre:
            raise ValueError(f"After skipping missing files, genre “{g}” has "
                             f"only {len(kept)} usable tracks "
                             f"(need {args.num_per_genre})")

        selected.extend(kept)

    selected = deduplicate_tracks(selected)
    if not selected:
        p.error("No usable tracks were found for the requested genres")
    train_ids, valid_ids = split_track_ids(
        selected, wanted_genres, args.split_pct_valid, args.seed)
    assert train_ids.isdisjoint(valid_ids)
    print(f"▶  Selected {len(selected)} unique tracks: "
          f"{len(train_ids)} training, {len(valid_ids)} validation.")

    out_root = Path(args.out_root).resolve()
    try:
        prepare_output(out_root, args.overwrite)
    except ValueError as error:
        p.error(str(error))

    # Incomplete runs never publish loadable manifests.
    m_train = out_root / "train.jsonl.gz.partial"
    m_valid = out_root / "valid.jsonl.gz.partial"
    for m in (m_train, m_valid):
        with gzip.open(m, "wt"):
            pass

    print("▶  Converting audio and writing manifests …")
    for row in tqdm(selected, total=len(selected)):
        # Stop if free space is less than 20 GB
        if shutil.disk_usage(out_root).free < 20 * (1024**3):
            raise RuntimeError("Export incomplete: less than 20 GB of disk space remaining")

        split = "valid" if row["track_id"] in valid_ids else "train"

        # Use the same primary genre as the split stratification.
        g = next((tag for tag in row["genres"] if tag in wanted_genres),
                 row["genres"][0])

        dest_dir = out_root / split / g
        dest_dir.mkdir(parents=True, exist_ok=True)

        src_mp3 = locate_mp3(row["track_id"], Path(args.jamendo_audio_root),
                             row["path"])
        if not src_mp3.exists():
            raise FileNotFoundError(src_mp3)

        clips = chunk_and_resample(src_mp3, dest_dir, args.chunk)

        for clip in clips:
            side_info = {
                "title": row["title"],
                "artist": f"artist_{row['artist_id']}",
                "genre": g,
                "tags": row["tags"],
                "description": f"{g} track featuring {' ,'.join(row['tags'])}"
            }
            write_sidecar(clip, side_info)
            append_manifest(
                {
                    "path": str(clip),
                    "duration": wav_duration(clip),
                    "sample_rate": 32000
                },
                m_valid if split == "valid" else m_train)

    final_manifests = [out_root / "train.jsonl.gz", out_root / "valid.jsonl.gz"]
    for temporary, final in zip((m_train, m_valid), final_manifests):
        temporary.replace(final)
    print("✅  Done!\n   • audio tree :", out_root, "\n   • manifests  :", *final_manifests)


if __name__ == "__main__":
    main()
