# Training and checking a FART tokenizer

The tokenizer groups recurring sequences of codec IDs. It preserves their exact values and codebook order. Its effect on language-model generation must be evaluated separately from this lossless conversion.

## A small complete example

Use Python 3.10, `ffmpeg`, and an isolated environment. Install the experiment dependencies and run:

```sh
python -m pip install -r requirements-fart-smoke.txt
python scripts/fart_codec_smoke.py --output outputs/tokenizer-smoke
```

The command uses the first eight seconds of `dataset/example/electro_1.mp3` for training and `electro_2.mp3` for evaluation. It downloads Meta's public EnCodec 24 kHz codec, encodes four codebooks at 3 kbps, fits a SentencePiece BPE tokenizer, and checks exact codec reconstruction on the held-out clip. No language model is trained. The downloaded codec weights and all intermediate files stay inside the chosen output directory, which must not already exist.

The output contains `train/` and `heldout/` code arrays, `tokenizer/` with the model and manifest, `evaluation.json`, and `provenance.json`. The latter records source audio hashes, codec-weight hash, dependency versions and preprocessing. [Recorded results](../examples/fart-tokenizer-smoke/README.md) are included for inspection. The same experiment is available through **Actions → FART tokenizer audio experiment → Run workflow**, with downloadable results retained for 14 days.

This two-clip example checks integration. It is too small to establish general compression performance or music quality. It uses a different codec from MusicGen's 32 kHz model, so its fitted tokenizer is not compatible with a standard FART/MusicGen checkpoint.

## Train on your own codec arrays

Save one NumPy `.npy` file per example, containing integer IDs with shape `[codebooks, frames]`. Files must contain valid audio frames only, without batch padding. Use disjoint source tracks for training and evaluation, and obtain all arrays with the same codec, settings and codebook order. The loader disables pickle loading.

For MusicGen's four 2048-entry codebooks:

```sh
python scripts/fart_tokenizer.py train \
  --codes /path/to/train-codes \
  --output outputs/my-tokenizer \
  --codec-cardinality 2048 \
  --num-codebooks 4 \
  --vocab-size 32000

python scripts/fart_tokenizer.py evaluate \
  --codes /path/to/heldout-codes \
  --tokenizer outputs/my-tokenizer \
  --output outputs/my-evaluation.json
```

Arrays are flattened in codebook-major order and mapped to Unicode with offset `77844`. Training uses identity normalization, a complete codec alphabet and deterministic ordering. Every alphabet symbol receives an individual one-character coverage example, which introduces no artificial pairs. Long examples beyond the documented 1 MiB encoded-text limit are rejected instead of silently skipped. Existing outputs are protected.

The vocabulary can be smaller than requested when the corpus contains too few distinct sequences. Read `actual_vocab_size` from `manifest.json` and set `transformer_lm.card` to that value. Keep `transformer_lm.n_q=1`, and supply the generated `tokenizer.model` as `fart.tokenizer_path` alongside the checkpoint trained with it. The manifest stores hashes, codec layout, training options and software versions. Changing the codec or tokenizer requires a compatible language model.

Evaluation rejects exact training examples even if renamed or saved with a different integer dtype. This check does not detect overlapping excerpts from the same recording, so splitting by source track remains necessary.

## Reading the measurements

- `raw_codes_per_lm_token` divides the number of flattened codec IDs by the number of SentencePiece tokens. Values above one indicate a shorter token sequence relative to that flattened representation.
- `lm_tokens_per_codec_frame` compares the sequence with codec frames. MusicGen's delayed codebooks operate largely in parallel, so its autoregressive length is close to its frame count, with pattern overhead.
- `roundtrip_exact` verifies that tokenization and decoding recover every input codec ID in the same position.

These measurements describe representation size. They do not measure inference latency, GPU memory, waveform fidelity relative to the original recording, or generated music quality. A comparison of generative models needs matched data and hardware, actual decoding-step and latency measurements, and a separate audio evaluation.
