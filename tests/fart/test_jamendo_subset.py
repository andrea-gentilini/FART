"""Offline checks for track selection and leakage-free Jamendo manifests."""

import gzip
import io
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

import make_jamendo_subset as jamendo


def track(track_id, genres=None):
    return {
        "track_id": track_id,
        "artist_id": track_id,
        "title": f"track_{track_id}",
        "genres": genres or ["rock", "pop"],
        "tags": ["genre---rock", "genre---pop"],
        "path": f"{track_id}.mp3",
    }


class SplitTracksTest(unittest.TestCase):
    def test_multi_genre_duplicates_cannot_leak(self):
        rows = [track(i) for i in range(10)]
        train_ids, valid_ids = jamendo.split_track_ids(
            rows + rows, ["rock", "pop"], 0.2, seed=42)
        self.assertEqual(train_ids | valid_ids, set(range(10)))
        self.assertFalse(train_ids & valid_ids)
        self.assertEqual(len(valid_ids), 2)
        self.assertEqual(len(jamendo.deduplicate_tracks(rows + rows)), 10)

    def test_split_is_independent_of_row_and_requested_genre_order(self):
        rows = [track(i, ["rock"] if i < 10 else ["pop"]) for i in range(20)]
        expected = jamendo.split_track_ids(rows, ["rock", "pop"], 0.2, seed=42)
        actual = jamendo.split_track_ids(rows[::-1], ["pop", "rock"], 0.2, seed=42)
        self.assertEqual(actual, expected)

    def test_split_does_not_change_global_random_state(self):
        state = random.getstate()
        jamendo.split_track_ids([track(i) for i in range(10)], ["rock"], 0.2, 42)
        self.assertEqual(random.getstate(), state)

    def test_zero_validation_keeps_every_track_in_training(self):
        train_ids, valid_ids = jamendo.split_track_ids([track(1), track(2)], ["rock"], 0, 42)
        self.assertEqual(train_ids, {1, 2})
        self.assertFalse(valid_ids)

    def test_small_genres_keep_training_examples(self):
        rows = [track(1, ["rock"]), track(2, ["pop"]), track(3, ["pop"])]
        train_ids, valid_ids = jamendo.split_track_ids(rows, ["rock", "pop"], 0.99, 42)
        self.assertIn(1, train_ids)
        self.assertEqual(len(train_ids), 2)
        self.assertEqual(len(valid_ids), 1)

    def test_invalid_validation_fractions_are_rejected(self):
        for value in (-0.1, 1, 1.1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                jamendo.split_track_ids([track(1)], ["rock"], value, 42)

    def test_missing_requested_genre_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no requested genre"):
            jamendo.split_track_ids([track(1, ["jazz"])], ["rock"], 0.1, 42)


class ManifestIntegrationTest(unittest.TestCase):
    def make_manifests(self, out_root, fraction, free_bytes=100 * 1024**3):
        rows = [track(i) for i in range(6)]
        for row in rows:
            (out_root / row["path"]).write_bytes(b"source audio")

        def fake_conversion(src, destination, chunk_sec):
            output = destination / f"{src.stem}_000.wav"
            with wave.open(str(output), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(32000)
                audio.writeframes(b"\0\0" * 3200)
            return [output]

        args = [
            "make_jamendo_subset.py", "--jamendo_audio_root", str(out_root),
            "--meta_tsv", "unused.tsv", "--genres", "all", "--num_per_genre", "-1",
            "--out_root", str(out_root), "--split_pct_valid", str(fraction),
        ]
        progress = SimpleNamespace(tqdm=lambda items, **kwargs: items)
        with patch.object(sys, "argv", args), \
                patch.dict(sys.modules, {"tqdm": progress}), \
                patch.object(jamendo, "read_jamendo_meta", return_value=rows), \
                patch.object(jamendo, "mp3_exists", return_value=True), \
                patch.object(jamendo, "locate_mp3", side_effect=lambda tid, base, rel: base / rel), \
                patch.object(jamendo, "chunk_and_resample", side_effect=fake_conversion), \
                patch.object(jamendo.shutil, "disk_usage", return_value=SimpleNamespace(free=free_bytes)), \
                patch("builtins.print"):
            jamendo.main()

        manifests = []
        for split in ("train", "valid"):
            with gzip.open(out_root / f"{split}.jsonl.gz", "rt") as stream:
                manifests.append([json.loads(line) for line in stream])
        return manifests

    def test_main_writes_each_multi_genre_track_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            train, valid = self.make_manifests(Path(temporary), 0.3)
            train_ids = {Path(row["path"]).stem for row in train}
            valid_ids = {Path(row["path"]).stem for row in valid}
            self.assertFalse(train_ids & valid_ids)
            self.assertEqual(len(train) + len(valid), 6)
            self.assertEqual(len(train_ids | valid_ids), 6)
            self.assertEqual(len(valid), 2)
            for row in train + valid:
                self.assertEqual(row["sample_rate"], 32000)
                self.assertEqual(row["duration"], 0.1)
                self.assertTrue(Path(row["path"]).with_suffix(".json").exists())

    def test_main_creates_readable_empty_validation_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            train, valid = self.make_manifests(Path(temporary), 0)
            self.assertEqual(len(train), 6)
            self.assertEqual(valid, [])

    def test_low_disk_fails_without_publishing_partial_manifests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "Export incomplete"):
                self.make_manifests(root, 0.3, free_bytes=0)
            self.assertFalse((root / "train.jsonl.gz").exists())
            self.assertFalse((root / "valid.jsonl.gz").exists())


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg and ffprobe required")
class SyntheticAudioExportTest(unittest.TestCase):
    def test_real_conversion_and_explicit_repeated_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio_root, out_root = root / "audio", root / "export"
            audio_root.mkdir()
            for track_id in range(4):
                subprocess.run([
                    "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                    f"sine=frequency={440 + track_id * 20}:sample_rate=32000:duration=1.25",
                    str(audio_root / f"{track_id}.mp3"),
                ], check=True, capture_output=True)
            meta = root / "metadata.tsv"
            meta.write_text("TRACK_ID\tARTIST_ID\tALBUM_ID\tPATH\tDURATION\tTAGS\n" + "".join(
                f"track_{i}\tartist_{i}\talbum_{i}\t{i}.mp3\t1.25\tgenre---rock\tgenre---pop\n"
                for i in range(4)))
            args = [
                "make_jamendo_subset.py", "--jamendo_audio_root", str(audio_root),
                "--meta_tsv", str(meta), "--genres", "all", "--num_per_genre", "-1",
                "--out_root", str(out_root), "--split_pct_valid", "0.25", "--chunk", "1",
            ]

            def run(extra_args=()):
                progress = SimpleNamespace(tqdm=lambda items, **kwargs: items)
                with patch.object(sys, "argv", args + list(extra_args)), \
                        patch.dict(sys.modules, {"tqdm": progress}), \
                        patch.object(jamendo.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * 1024**3)), \
                        patch("builtins.print"), patch("sys.stderr", new_callable=io.StringIO):
                    jamendo.main()

            def read_manifests():
                result = []
                for split in ("train", "valid"):
                    with gzip.open(out_root / f"{split}.jsonl.gz", "rt") as stream:
                        result.append([json.loads(line) for line in stream])
                return result

            run()
            first = read_manifests()
            train, valid = first
            train_tracks = {Path(row["path"]).stem.split("_")[0] for row in train}
            valid_tracks = {Path(row["path"]).stem.split("_")[0] for row in valid}
            self.assertFalse(train_tracks & valid_tracks)
            self.assertEqual(len(train) + len(valid), 8)
            self.assertEqual(len(valid_tracks), 1)
            self.assertTrue(any(0 < row["duration"] < 0.5 for row in train + valid))
            for row in train + valid:
                with wave.open(row["path"], "rb") as audio:
                    self.assertEqual(audio.getframerate(), 32000)
                    self.assertEqual(audio.getnchannels(), 1)
                    self.assertAlmostEqual(row["duration"], audio.getnframes() / 32000)

            with self.assertRaises(SystemExit) as error:
                run()
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(read_manifests(), first)

            stale = out_root / "train" / "old.wav"
            stale.write_bytes(b"stale")
            unrelated = out_root / "notes.txt"
            unrelated.write_text("keep")
            run(["--overwrite"])
            self.assertEqual(read_manifests(), first)
            self.assertFalse(stale.exists())
            self.assertEqual(unrelated.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
