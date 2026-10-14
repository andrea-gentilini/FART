"""Reproduce a small tokenizer experiment on AudioCraft's two bundled clips.

This uses the public EnCodec 24 kHz codec. It is a tokenizer integration smoke
experiment, not a MusicGen benchmark or a trained music-generation model.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='New directory for codes, tokenizer, reports and model cache.')
    parser.add_argument('--seconds', type=float, default=8.0)
    parser.add_argument('--vocab-size', type=int, default=2048)
    args = parser.parse_args()
    if not 0 < args.seconds <= 10:
        parser.error('--seconds must be greater than zero and at most 10 for this small experiment.')
    if args.vocab_size < 1025:
        parser.error('--vocab-size must cover the 1024 codec symbols and the unknown token.')
    output = args.output.resolve()
    if output.exists():
        parser.error('--output must be a new directory; existing results are never overwritten.')
    output.mkdir(parents=True)
    os.environ['TORCH_HOME'] = str(output / '.cache' / 'torch')
    import numpy as np
    import torch
    from encodec import EncodecModel
    from importlib.metadata import version

    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    codec = EncodecModel.encodec_model_24khz()
    codec.set_target_bandwidth(3.0)
    codec.eval()
    sources = []
    for split, filename in [('train', 'electro_1.mp3'), ('heldout', 'electro_2.mp3')]:
        source = ROOT / 'dataset' / 'example' / filename
        # ffmpeg trims and resamples without temporary waveform files.
        result = subprocess.run([
            'ffmpeg', '-v', 'error', '-i', str(source), '-t', str(args.seconds),
            '-ar', '24000', '-ac', '1', '-f', 'f32le', 'pipe:1',
        ], check=True, capture_output=True)
        waveform = np.frombuffer(result.stdout, dtype='<f4').copy()
        if len(waveform) != round(args.seconds * 24000):
            raise ValueError(f'{filename} did not yield the requested number of samples.')
        with torch.no_grad():
            frames = codec.encode(torch.from_numpy(waveform)[None, None])
        codes = torch.cat([item[0] for item in frames], dim=-1)[0].cpu().numpy()
        if codes.shape[0] != 4 or codes.min() < 0 or codes.max() >= 1024:
            raise ValueError('Unexpected EnCodec code layout.')
        directory = output / split
        directory.mkdir()
        path = directory / f'{source.stem}.npy'
        np.save(path, codes, allow_pickle=False)
        sources.append({
            'split': split, 'source': source.relative_to(ROOT).as_posix(),
            'source_sha256': sha256(source), 'seconds': args.seconds,
            'samples': len(waveform), 'frames': int(codes.shape[1]),
            'codes': path.relative_to(output).as_posix(), 'codes_sha256': sha256(path),
        })
    cli = ROOT / 'scripts' / 'fart_tokenizer.py'
    subprocess.run([
        sys.executable, str(cli), 'train', '--codes', str(output / 'train'),
        '--output', str(output / 'tokenizer'), '--codec-cardinality', '1024',
        '--num-codebooks', '4', '--vocab-size', str(args.vocab_size),
    ], check=True)
    subprocess.run([
        sys.executable, str(cli), 'evaluate', '--codes', str(output / 'heldout'),
        '--tokenizer', str(output / 'tokenizer'), '--output', str(output / 'evaluation.json'),
    ], check=True)
    weights = sorted((output / '.cache' / 'torch').rglob('*.th'))
    if len(weights) != 1:
        raise ValueError('Expected exactly one downloaded EnCodec checkpoint.')
    provenance = {
        'experiment': 'bundled-audio-tokenizer-smoke',
        'scope': 'Tokenization and exact code reconstruction on two short example clips.',
        'limitations': [
            'One training clip and one held-out clip are not a representative benchmark.',
            'EnCodec 24 kHz at 3 kbps differs from the MusicGen 32 kHz codec.',
            'This tokenizer cannot be paired with a standard MusicGen/FART 32 kHz checkpoint.',
            'No language model is trained and no audio-generation quality or speedup is measured.',
            'Raw flattened-code compression is distinct from MusicGen autoregressive step count.',
        ],
        'codec': {'name': 'encodec_model_24khz', 'sample_rate': 24000,
                  'bandwidth_kbps': 3.0, 'num_codebooks': 4, 'cardinality': 1024,
                  'checkpoint': weights[0].name, 'checkpoint_sha256': sha256(weights[0])},
        'sources': sources,
        'environment': {name: version(name) for name in ('torch', 'torchaudio', 'numpy', 'encodec', 'sentencepiece')},
        'platform': {'python': platform.python_version(), 'system': platform.system(),
                     'machine': platform.machine()},
        'ffmpeg': subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0],
        'evaluation_sha256': sha256(output / 'evaluation.json'),
    }
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Completed experiment: {output}')
    print('No language-model generation or end-to-end speedup was evaluated.')


if __name__ == '__main__':
    main()
