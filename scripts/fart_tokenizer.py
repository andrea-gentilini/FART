"""Train and audit a lossless FART tokenizer on integer NumPy codec arrays [K, T]."""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import subprocess
import tempfile

import numpy as np
import sentencepiece as spm
import torch


ROOT = Path(__file__).resolve().parents[1]
MAX_SENTENCE_BYTES = 1024 * 1024


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def source_provenance():
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    status = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True)
    files = ['scripts/fart_tokenizer.py', 'audiocraft/utils/audio_tokenizer.py']
    return {'source_commit': commit.stdout.strip() if commit.returncode == 0 else None,
            'source_tree_dirty': bool(status.stdout) if status.returncode == 0 else None,
            'source_files_sha256': {name: sha256((ROOT / name).read_bytes()) for name in files}}


def load_codes(path, cardinality, codebooks):
    codes = np.load(path, allow_pickle=False)
    if not isinstance(codes, np.ndarray) or codes.ndim != 2 or codes.shape[0] != codebooks:
        raise ValueError(f'{path.name}: expected [{codebooks}, T] NumPy array.')
    if codes.shape[1] == 0 or codes.dtype.kind not in 'iu':
        raise ValueError(f'{path.name}: expected nonempty integer codec IDs.')
    if int(codes.min()) < 0 or int(codes.max()) >= cardinality:
        raise ValueError(f'{path.name}: codec IDs outside [0, {cardinality}).')
    return np.asarray(codes, dtype='<i8', order='C')


def content_hash(codes):
    # Include shape, but normalize dtype and storage order to catch renamed copies.
    return sha256(json.dumps(list(codes.shape)).encode() + b'\0' + codes.tobytes(order='C'))


def inspect_inputs(directory, cardinality, codebooks):
    directory = Path(directory)
    paths = sorted(directory.rglob('*.npy')) if directory.is_dir() else []
    if not paths:
        raise ValueError(f'No .npy files found in {directory}.')
    records = []
    for path in paths:
        codes = load_codes(path, cardinality, codebooks)
        records.append({'file': path.relative_to(directory).as_posix(),
                        'sha256': sha256(path.read_bytes()), 'content_sha256': content_hash(codes),
                        'frames': codes.shape[1], 'raw_codes': codes.size})
    return paths, records


def audio_tokenizer(model_path, config, vocabulary_size):
    spec = importlib.util.spec_from_file_location(
        'fart_audio_tokenizer_cli', ROOT / 'audiocraft/utils/audio_tokenizer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AudioTokenTokenizer(str(model_path), config['codec_cardinality'],
                                      config['num_codebooks'], vocabulary_size,
                                      unicode_offset=config['unicode_offset'])


def validate_config(cardinality, codebooks, offset):
    if cardinality <= 0 or codebooks <= 0:
        raise ValueError('Codec cardinality and codebook count must be positive.')
    end = offset + cardinality
    if offset < 0 or end > 0x110000 or (offset < 0xE000 and end > 0xD800):
        raise ValueError('Codec alphabet must contain valid Unicode scalar values.')
    alphabet = ''.join(chr(offset + i) for i in range(cardinality))
    if any(c.isspace() or ord(c) < 32 for c in alphabet) or '\u2581' in alphabet:
        raise ValueError('Codec alphabet must avoid whitespace, control characters and U+2581.')
    return alphabet


def train(codes_dir, output, codec_cardinality, num_codebooks, vocab_size, unicode_offset=77844):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}.')
    alphabet = validate_config(codec_cardinality, num_codebooks, unicode_offset)
    if vocab_size < codec_cardinality + 1:
        raise ValueError('Vocabulary must fit the full codec alphabet plus the unknown token.')
    paths, records = inspect_inputs(codes_dir, codec_cardinality, num_codebooks)
    config = {'codec_cardinality': codec_cardinality, 'num_codebooks': num_codebooks,
              'unicode_offset': unicode_offset, 'layout': 'codebook-major'}
    options = dict(model_type='bpe', vocab_size=vocab_size, hard_vocab_limit=False,
                   character_coverage=1.0, required_chars=alphabet,
                   normalization_rule_name='identity', add_dummy_prefix=False,
                   remove_extra_whitespaces=False, split_by_unicode_script=False,
                   split_by_whitespace=False, split_by_number=False, split_digits=False,
                   bos_id=-1, eos_id=-1, pad_id=-1, unk_id=0,
                   input_sentence_size=0, shuffle_input_sentence=False, num_threads=1,
                   max_sentence_length=MAX_SENTENCE_BYTES, max_sentencepiece_length=16)

    def sentences():
        # BPE can drop required_chars absent from its input. Single-character
        # coverage sentences preserve every ID without introducing synthetic pairs.
        yield from alphabet
        for path, record in zip(paths, records):
            codes = load_codes(path, codec_cardinality, num_codebooks)
            if sha256(path.read_bytes()) != record['sha256'] or content_hash(codes) != record['content_sha256']:
                raise ValueError(f'Input changed while training: {record["file"]}.')
            text = ''.join(alphabet[int(i)] for i in codes.reshape(-1))
            if len(text.encode('utf-8')) >= MAX_SENTENCE_BYTES:
                raise ValueError(f'{record["file"]}: clip exceeds the 1 MiB training sentence limit.')
            yield text

    serialized = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=sentences(), model_writer=serialized, minloglevel=2, **options)
    model = serialized.getvalue()
    processor = spm.SentencePieceProcessor(model_proto=model)
    actual_size = processor.get_piece_size()
    # Prepare atomically: a failed alphabet/roundtrip check leaves no output directory.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.fart-tokenizer-', dir=output.parent) as directory:
        temporary = Path(directory)
        (temporary / 'tokenizer.model').write_bytes(model)
        tokenizer = audio_tokenizer(temporary / 'tokenizer.model', config, actual_size)
        for path in paths:
            codes = torch.from_numpy(load_codes(path, codec_cardinality, num_codebooks)).unsqueeze(0)
            ids, mask = tokenizer.encode_batch(codes)
            if not torch.equal(tokenizer.decode_sequence(ids[0, 0, mask[0, 0]]), codes):
                raise ValueError(f'Training roundtrip failed for {path.name}.')
        (temporary / 'tokenizer.vocab').write_text(
            ''.join(f'{processor.id_to_piece(i)}\t{processor.get_score(i)}\n' for i in range(actual_size)),
            encoding='utf-8')
        # Store the alphabet specification compactly instead of repeating every character.
        stored_options = {k: v for k, v in options.items() if k != 'required_chars'}
        stored_options['required_chars'] = 'complete alphabet defined by unicode_offset and codec_cardinality'
        manifest = {'schema_version': 1, 'config': config, 'requested_vocab_size': vocab_size,
                    'actual_vocab_size': actual_size, 'model_sha256': sha256(model),
                    'training_options': stored_options, 'inputs': records,
                    'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                                 'sentencepiece': spm.__version__, 'torch': torch.__version__},
                    **source_provenance(), 'alphabet_coverage_sentences': codec_cardinality,
                    'training_roundtrip_exact': True}
        write_json(temporary / 'manifest.json', manifest)
        if output.exists():
            raise FileExistsError(f'Refusing to overwrite {output}.')
        temporary.rename(output)
    return manifest


def evaluate(codes_dir, tokenizer_dir, output):
    output, tokenizer_dir = Path(output), Path(tokenizer_dir)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}.')
    manifest = json.loads((tokenizer_dir / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported tokenizer manifest schema.')
    config = manifest['config']
    validate_config(config['codec_cardinality'], config['num_codebooks'], config['unicode_offset'])
    if config.get('layout') != 'codebook-major':
        raise ValueError('Unsupported codec layout.')
    model_path = tokenizer_dir / 'tokenizer.model'
    if sha256(model_path.read_bytes()) != manifest['model_sha256']:
        raise ValueError('Tokenizer model checksum does not match manifest.')
    paths, records = inspect_inputs(codes_dir, config['codec_cardinality'], config['num_codebooks'])
    training_hashes = {record['content_sha256'] for record in manifest['inputs']}
    overlaps = [record['file'] for record in records if record['content_sha256'] in training_hashes]
    if overlaps:
        raise ValueError('Training/evaluation content overlap: ' + ', '.join(overlaps))
    tokenizer = audio_tokenizer(model_path, config, manifest['actual_vocab_size'])
    for path, record in zip(paths, records):
        array = load_codes(path, config['codec_cardinality'], config['num_codebooks'])
        if content_hash(array) != record['content_sha256'] or sha256(path.read_bytes()) != record['sha256']:
            raise ValueError(f'Input changed while evaluating: {record["file"]}.')
        codes = torch.from_numpy(array).unsqueeze(0)
        ids, mask = tokenizer.encode_batch(codes)
        record['lm_tokens'] = int(mask.sum())
        record['roundtrip_exact'] = torch.equal(tokenizer.decode_sequence(ids[0, 0, mask[0, 0]]), codes)
        if not record['roundtrip_exact']:
            raise ValueError(f'Lossless roundtrip failed for {record["file"]}.')
        record['raw_codes_per_lm_token'] = record['raw_codes'] / record['lm_tokens']
        record['lm_tokens_per_codec_frame'] = record['lm_tokens'] / record['frames']
    raw_codes = sum(record['raw_codes'] for record in records)
    frames = sum(record['frames'] for record in records)
    lm_tokens = sum(record['lm_tokens'] for record in records)
    report = {'schema_version': 1, 'config': config, 'model_sha256': manifest['model_sha256'],
              'actual_vocab_size': manifest['actual_vocab_size'], 'samples': records,
              'training_overlap_count': 0,
              'summary': {'sample_count': len(records), 'raw_codes': raw_codes, 'codec_frames': frames,
                          'lm_tokens': lm_tokens, 'roundtrip_exact': True,
                          'raw_codes_per_lm_token': raw_codes / lm_tokens,
                          'lm_tokens_per_codec_frame': lm_tokens / frames,
                          'raw_code_reduction_fraction': 1 - lm_tokens / raw_codes},
              'interpretation': 'Ratios describe token counts only. MusicGen processes codebooks in parallel; '
                                'codec frames are a separate baseline. No generation speedup or audio quality '
                                'improvement is established by this tokenizer evaluation.',
              'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                           'sentencepiece': spm.__version__, 'torch': torch.__version__},
              **source_provenance()}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    training = commands.add_parser('train', help='Train a BPE tokenizer and verify exact roundtrips.')
    training.add_argument('--codes', required=True, type=Path)
    training.add_argument('--output', required=True, type=Path)
    training.add_argument('--codec-cardinality', required=True, type=int)
    training.add_argument('--num-codebooks', required=True, type=int)
    training.add_argument('--vocab-size', required=True, type=int)
    training.add_argument('--unicode-offset', type=int, default=77844)
    evaluation = commands.add_parser('evaluate', help='Evaluate losslessness and token counts on held-out codes.')
    evaluation.add_argument('--codes', required=True, type=Path)
    evaluation.add_argument('--tokenizer', required=True, type=Path)
    evaluation.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == 'train':
            result = train(args.codes, args.output, args.codec_cardinality, args.num_codebooks,
                           args.vocab_size, args.unicode_offset)
            print(f'Trained {result["actual_vocab_size"]} pieces; exact training roundtrips verified.')
        else:
            result = evaluate(args.codes, args.tokenizer, args.output)
            print(json.dumps(result['summary'], indent=2))
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        parser.exit(2, f'error: {error}\n')


if __name__ == '__main__':
    main()
