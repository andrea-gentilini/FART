# A small tokenizer experiment

This recorded experiment fits a FART-compatible SentencePiece tokenizer on codec IDs from the first eight seconds of `electro_1.mp3` and evaluates it on the first eight seconds of `electro_2.mp3`. Both source clips already belong to AudioCraft's example dataset. The extraction uses EnCodec 24 kHz at 3 kbps, with four 1024-entry codebooks. The fitted vocabulary contains 2048 pieces.

| Held-out measurement | Result |
| --- | ---: |
| Audio duration | 8 seconds |
| Codec frames | 600 |
| Flattened codec IDs | 2400 |
| SentencePiece tokens | 2395 |
| Raw-code reduction | 0.21% |
| Raw codec IDs per SentencePiece token | 1.0021 |
| SentencePiece tokens per codec frame | 3.9917 |
| Reconstructed codec IDs | Exact match |
| Exact training/evaluation array overlap | None |

The token reduction on this held-out clip is negligible. The learned merges rarely shorten this held-out sequence. The useful result here is a complete, inspectable training and evaluation path, including lossless reconstruction and a comparison that exposes the limited compression rather than assuming it.

This is an integration example with one training clip and one held-out clip. It does not establish generalization performance. No language model was trained, and neither music quality nor inference latency was measured. MusicGen processes its codebooks largely in parallel, so reducing the flattened code count does not by itself reduce its autoregressive steps. The 24 kHz example tokenizer is incompatible with standard MusicGen 32 kHz checkpoints.

## Check the saved result without downloading model weights

From the repository root, in the Python 3.10 test environment:

```sh
python -m pip install -r requirements-fart-tests.txt
python scripts/fart_tokenizer.py evaluate \
  --codes examples/fart-tokenizer-smoke/heldout \
  --tokenizer examples/fart-tokenizer-smoke/tokenizer \
  --output outputs/example-evaluation.json
```

The output path must be new. The [evaluation report](evaluation.json) records every count and checksum. The [tokenizer manifest](tokenizer/manifest.json) records the training inputs, options, software versions and exact implementation commit. The fitted model and both code arrays are included. The [provenance record](provenance.json) identifies the original audio, preprocessing and public codec weights. AudioCraft and EnCodec remain the work of their respective authors; the codec weights are not redistributed here.

## Reproduce the entire extraction

```sh
python -m pip install -r requirements-fart-smoke.txt
python scripts/fart_codec_smoke.py --output outputs/tokenizer-smoke
```

Install `ffmpeg` first. The command downloads about 89 MiB of codec weights into the output directory and writes a fresh tokenizer, code arrays and reports there. The published run used macOS arm64 and the versions recorded in `provenance.json`. Codec outputs may vary with the decoding library, platform or numerical kernels; compare hashes before treating a fresh run as an identical input. Evaluation of the saved arrays and fitted tokenizer avoids those extraction differences.
