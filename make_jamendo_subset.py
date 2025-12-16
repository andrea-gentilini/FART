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

"""
FART
python make_jamendo_subset.py \
--jamendo_audio_root  /home/andleo/jamendo_full_508gb/jamendo \
--meta_tsv            /home/andleo/jamendo_full_508gb/raw_30s_cleantags.tsv \
--genres              "all" \
--num_per_genre       -1 \
--out_root            /home/andleo/temp \
--split_pct_valid     0.1
"""
import argparse, csv, gzip, json, os, random, subprocess, textwrap, shutil
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm

from math import ceil
from collections import defaultdict
import itertools


# ── add this helper near the top of the script ────────────────────────────
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
    candidates = list(folder.glob(f"{track_id}*.mp3"))
    if candidates:
        return candidates[0]

    # very slow, but runs only if the dataset is organised differently
    deep = next(base_folder.rglob(f"{track_id}*.mp3"), None)
    if deep:
        return deep

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
    duration = float(proc.stdout)
    n_clips = max(1, int(duration // chunk_sec))

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
        clips.append(out_wav)
    return clips


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
        header = next(reader)  # skip header row
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
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=textwrap.dedent(__doc__))
    p.add_argument("--jamendo_audio_root", required=True)
    p.add_argument("--meta_tsv", required=True)
    p.add_argument("--genres",
                   required=True,
                   help="Space‑separated list of 10 genre tags to keep (or 'all')")  # FART
    p.add_argument("--num_per_genre",
                   type=int,
                   default=50,
                   help="Tracks per genre (50 or 100). Use -1 for all available.")  # FART
    p.add_argument("--out_root", default="./dataset/jamendo_subset")
    p.add_argument("--split_pct_valid", type=float, default=0.1)
    p.add_argument("--chunk", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print("▶  Parsing Jamendo metadata …")
    rows = read_jamendo_meta(Path(args.meta_tsv))

    # FART: Handle 'all' genres
    if args.genres == "all":
        # We need to scan rows to find all genres
        all_genres = set()
        for r in rows:
            all_genres.update(r["genres"])
        wanted_genres = list(all_genres)
        print(f"▶  Found {len(wanted_genres)} unique genres.")
    else:
        wanted_genres = args.genres.split()
    random.seed(args.seed)

    # bucket rows by genre
    buckets = defaultdict(list)
    for r in rows:
        for g in r["genres"]:
            if g in wanted_genres:
                buckets[g].append(r)

    # sample N per genre
    # selected = []
    # for g in wanted_genres:
    #     if len(buckets[g]) < args.num_per_genre:
    #         raise ValueError(
    #             f"Not enough tracks for genre “{g}” "
    #             f"({len(buckets[g])} available, need {args.num_per_genre})")
    #     selected.extend(random.sample(buckets[g], args.num_per_genre))

    selected = []  # will hold ONLY rows with existing audio
    warnings_issued = set()  # to print each missing‑file warning once

    for g in wanted_genres:
        random.shuffle(buckets[g])  # shuffle so replacements are random
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

            if args.num_per_genre > 0 and len(kept) == args.num_per_genre:  # FART
                break  # we have enough valid rows for this genre

        if args.num_per_genre > 0 and len(kept) < args.num_per_genre:  # FART
            raise ValueError(f"After skipping missing files, genre “{g}” has "
                             f"only {len(kept)} usable tracks "
                             f"(need {args.num_per_genre})")

        selected.extend(kept)

    print(
        f"▶  Selected {len(selected)} tracks (target: {args.num_per_genre} per genre × {len(wanted_genres)} genres)"  # FART
    )

    # ------------------------------------------------------------------
    # 1)  build per‑genre buckets of *indexes* inside the `selected` list
    # ------------------------------------------------------------------
    genre_to_idxs = defaultdict(list)
    for idx, row in enumerate(selected):
        g = next((tag for tag in row["genres"] if tag in wanted_genres),
                 row["genres"][0])
        genre_to_idxs[g].append(idx)

    # ------------------------------------------------------------------
    # 2)  sample validation indexes *per genre* with ceiling(round)
    # ------------------------------------------------------------------
    valid_ids = set()
    for g, idxs in genre_to_idxs.items():
        n_val = max(1, ceil(len(idxs) * args.split_pct_valid))
        valid_ids.update(random.sample(idxs, n_val))

    # you may print a tiny report if you like:
    print("Validation tracks per genre:", {
        g: sum(i in valid_ids for i in idxs)
        for g, idxs in genre_to_idxs.items()
    })

    # validation split
    # n_valid = int(len(selected) * args.split_pct_valid)
    # valid_ids = set(random.sample(range(len(selected)), n_valid))

    out_root = Path(args.out_root)
    (out_root / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "valid").mkdir(exist_ok=True)

    m_train = out_root / "train.jsonl.gz"
    m_valid = out_root / "valid.jsonl.gz"
    for m in (m_train, m_valid):
        if m.exists():
            m.unlink()

    print("▶  Converting audio and writing manifests …")
    for idx, row in tqdm(list(enumerate(selected)), total=len(selected)):
        # Stop if free space is less than 20 GB
        if shutil.disk_usage(out_root).free < 20 * (1024**3):
            print("\n🛑  Stopping: Less than 20 GB of disk space remaining.")
            break

        split = "valid" if idx in valid_ids else "train"

        # g = row["genres"][0]  # first matching genre
        # choose the first genre that is actually in the user‑supplied list
        g = next((tag for tag in row["genres"] if tag in wanted_genres),
                 row["genres"][0])

        dest_dir = out_root / split / g
        dest_dir.mkdir(parents=True, exist_ok=True)

        # src_mp3 = Path(args.jamendo_audio_root) / row["path"]
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
            # append_manifest({
            #     "path": str(clip),
            #     "duration": args.chunk
            # }, m_valid if split == "valid" else m_train)
            append_manifest(
                {
                    "path": str(clip),
                    "duration": args.chunk,
                    "sample_rate": 32000
                },  # ← add this
                m_valid if split == "valid" else m_train)

    print("✅  Done!\n   • audio tree :", out_root, "\n   • manifests  :",
          m_train, m_valid)


if __name__ == "__main__":
    main()
