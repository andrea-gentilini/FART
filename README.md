# FART: Fused Audio Representation Tokens

FART is Andrea Gentilini's research fork of [Meta AudioCraft](https://github.com/facebookresearch/audiocraft), exploring learned tokenization of discrete audio representations for music generation. It adapts MusicGen's training and generation solver to model SentencePiece tokens over EnCodec codes, with the aim of shortening the sequences processed by the autoregressive model.

This repository contains the implementation and experiment configuration. A small tokenizer experiment is included with its fitted tokenizer, held-out evaluation and reproduction commands. Trained FART music-generation checkpoints and a representative benchmark-results bundle are not included.

## Approach

```text
Audio -> EnCodec codes -> Unicode sequence -> SentencePiece tokens
                                                |
                                      Autoregressive language model
                                                |
Waveform <- EnCodec decoder <- Audio codes <- SentencePiece decode
```

The four EnCodec codebook streams are flattened into one sequence and mapped to Unicode characters. A fitted SentencePiece model groups recurring code sequences into variable-length tokens. The language model predicts this single token stream; generation maps its output back into EnCodec codes for waveform decoding. The tokenizer vocabulary and language-model output vocabulary must match.

FART training is selected explicitly with `solver=musicgen/fart`. The `MusicGen` inference API can load a compatible local FART checkpoint with its matching tokenizer; inherited pretrained-model examples use standard AudioCraft checkpoints.

## Try the tokenizer

The [tokenizer guide](docs/FART_TOKENIZER.md) covers training on codec arrays, checking exact reconstruction, and measuring token counts on held-out clips. A small CPU experiment uses two bundled audio excerpts and the public EnCodec 24 kHz model. It can run locally or through the manual [audio experiment workflow](.github/workflows/fart_smoke.yml).

```bash
python -m pip install -r requirements-fart-smoke.txt
python scripts/fart_codec_smoke.py --output outputs/tokenizer-smoke
```

Use Python 3.10 with `ffmpeg` installed. The command downloads only the codec weights and keeps them inside the output directory. The [recorded example](examples/fart-tokenizer-smoke/README.md) includes measurements and limitations. Its tokenizer is specific to this small 24 kHz example.

## Contributions in this fork

| Component | Implementation |
| --- | --- |
| Audio-token integration | SentencePiece encoding during training and validation, and decoding generated tokens back to EnCodec indices in [`MusicGenSolver`](audiocraft/solvers/musicgen.py). |
| Language-model configuration | A single output stream and 32,000-token vocabulary in the [FART solver configuration](config/solver/musicgen/fart.yaml); a [`smaller`](config/model/lm/model_scale/smaller.yaml) model with 8 layers, width 768, and 12 attention heads. |
| Dataset preparation | [`make_jamendo_subset.py`](make_jamendo_subset.py) selects MTG-Jamendo tracks, converts audio to 32 kHz mono clips, and writes metadata sidecars and compressed AudioCraft manifests. |
| Music metadata support | Optional metadata fields in [`music_dataset.py`](audiocraft/data/music_dataset.py) allow training on the available Jamendo annotations. |

These changes build on AudioCraft's existing model, training, generation, and evaluation infrastructure. The original project documentation is retained in [`docs/`](docs/), with its [upstream overview](docs/UPSTREAM_AUDIOCRAFT.md) archived separately.

## Setup

The inherited dependency stack targets Python 3.9 and PyTorch 2.1.0. Use a separate environment and install this checkout so that the FART changes are available:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install 'torch==2.1.0'
python -m pip install -e '.[dev]'
```

Install `ffmpeg` and `ffprobe` for dataset conversion. Model training requires a suitable GPU environment; the full dependency list is in [`requirements.txt`](requirements.txt).

### Prepare an MTG-Jamendo subset

Obtain the audio files and their matching metadata separately, then supply local paths:

```bash
python make_jamendo_subset.py \
  --jamendo_audio_root /path/to/jamendo/audio \
  --meta_tsv /path/to/raw_30s_cleantags.tsv \
  --genres 'rock pop jazz' \
  --num_per_genre 50 \
  --out_root /path/to/fart-dataset \
  --split_pct_valid 0.1 \
  --chunk 30 \
  --seed 42
```

The output contains `train.jsonl.gz`, `valid.jsonl.gz`, and audio with matching JSON metadata. Tracks are split before conversion, and each manifest records the actual clip duration. Existing exports are protected; pass `--overwrite` only to replace their train/validation outputs. `--genres all --num_per_genre -1` selects all available tagged tracks. Audio and source-dataset access are not bundled with this repository.

### Configure a FART experiment

Set `FART_DATASET_ROOT` to the absolute path containing the generated manifests. Choose persistent locations for experiment outputs and evaluation assets:

```bash
export FART_DATASET_ROOT=/path/to/fart-dataset
export AUDIOCRAFT_DORA_DIR=/path/to/fart-runs
export AUDIOCRAFT_REFERENCE_DIR=/path/to/evaluation-assets
```

With a fitted SentencePiece model and the required model/data assets available, the configuration is selected through AudioCraft's Dora/Hydra interface:

```bash
dora run \
  solver=musicgen/fart \
  dset=my_music \
  model/lm/model_scale=smaller \
  fart.tokenizer_path=/path/to/music.model
```

This is a configuration example, not a reproduced training run. The tokenizer must use the repository's codebook-major flattening and Unicode offset (`77844`); `transformer_lm.card` must equal its vocabulary size, which defaults to `32000` in the FART configuration. Use the [tokenizer training command](docs/FART_TOKENIZER.md) with code arrays from your own training split. The original research tokenizer and training corpus are not included. See the [training guide](docs/TRAINING.md) for experiment management.

FART generation is experimental. It generates a full training-segment layout, decodes the fixed codebook boundaries, then crops the waveform to the requested duration. The solver and inference API support generation without an audio continuation prompt; continuation and long-form generation are unsupported. A relocated tokenizer can be supplied to `MusicGen.get_pretrained(local_checkpoint, tokenizer_path='/path/to/music.model')`. Reserved and non-audio SentencePiece IDs are excluded during sampling. Generation still uses a fixed token budget, so token compression alone does not establish an inference speedup. Generation quality with a trained FART checkpoint has not been validated by the included CPU tests.

## Evaluation and tests

The solver reports cross-entropy and perplexity and inherits optional FAD, CLAP text-consistency, and other audio metrics from AudioCraft. These metrics require their own reference data and model assets; see [metrics setup](docs/METRICS.md). Perplexity values from different tokenizations are not directly comparable without accounting for the changed prediction units.

To reproduce an experiment, preserve the fitted tokenizer, dataset manifests and split, complete configuration, trained checkpoint, and evaluation assets alongside the reported metrics.

The focused CPU suite covers tokenization, configuration, and dataset preprocessing without downloading a pretrained model. In a Python 3.10 environment with `ffmpeg` and `ffprobe` installed:

```bash
# On Linux, use the CPU wheel, as in CI.
python -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-fart-tests.txt
python -m pytest tests/fart -q
```

On other platforms, omit the Linux CPU-wheel command and install the requirements directly. [FART CI](.github/workflows/fart_tests.yml) runs this suite and checks the modified Python code on pushes and pull requests to `main`. The heavier upstream AudioCraft test and lint workflows are available for manual runs; the full test suite can also be run with `python -m pytest tests` in a configured environment. Unit tests are separate from training and audio-quality validation; a passing suite does not reproduce the research results.

## Attribution and license

FART's project-specific modifications are by Andrea Gentilini. AudioCraft, MusicGen, EnCodec, and the inherited framework were developed by Meta and their respective contributors; original copyright notices are retained.

- Code: [MIT license](LICENSE).
- Upstream model weights: [CC BY-NC 4.0](LICENSE_weights); consult the relevant model card when using a checkpoint.
- Dataset audio and annotations retain their own licenses.

When using MusicGen or AudioCraft, cite the original work:

```bibtex
@inproceedings{copet2023simple,
    title={Simple and Controllable Music Generation},
    author={Jade Copet and Felix Kreuk and Itai Gat and Tal Remez and David Kant and Gabriel Synnaeve and Yossi Adi and Alexandre Défossez},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
}
```

See the [MusicGen documentation](docs/MUSICGEN.md) and the other model-specific documentation for additional credits and citations.
