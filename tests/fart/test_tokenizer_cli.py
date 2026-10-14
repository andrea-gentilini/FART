"""Public CLI regressions without downloading any audio or pretrained model."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import sentencepiece as spm


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/fart_tokenizer.py'
spec = importlib.util.spec_from_file_location('fart_tokenizer_cli', SCRIPT)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def save(directory, name, values, dtype=np.int64):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    np.save(path, np.asarray(values, dtype=dtype))
    return path


@pytest.fixture
def corpus(tmp_path):
    training, heldout = tmp_path / 'train', tmp_path / 'heldout'
    # ID 3 is intentionally absent from training, but belongs to the required alphabet.
    save(training, 'one.npy', [[0, 1, 2] * 12, [2, 1, 0] * 12])
    save(training, 'two.npy', [[1, 0, 2] * 10, [0, 2, 1] * 10])
    save(heldout, 'short.npy', [[3, 2, 0, 1], [1, 0, 3, 2]])
    save(heldout, 'long.npy', [[0, 1, 3] * 17, [3, 1, 0] * 17])
    return training, heldout, tmp_path / 'tokenizer'


def test_end_to_end_full_alphabet_roundtrip_and_weighted_ratios(corpus, tmp_path):
    training, heldout, output = corpus
    manifest = cli.train(training, output, 4, 2, 128)
    assert manifest['actual_vocab_size'] < manifest['requested_vocab_size']
    assert manifest['training_roundtrip_exact'] is True
    assert isinstance(manifest['source_tree_dirty'], bool)
    assert set(manifest['source_files_sha256']) == {
        'scripts/fart_tokenizer.py', 'audiocraft/utils/audio_tokenizer.py'}
    assert manifest['alphabet_coverage_sentences'] == 4
    assert manifest['versions']['sentencepiece'] == spm.__version__
    assert all('content_sha256' in record and 'sha256' in record for record in manifest['inputs'])
    processor = spm.SentencePieceProcessor(model_file=str(output / 'tokenizer.model'))
    assert processor.bos_id() == processor.eos_id() == processor.pad_id() == -1
    assert processor.unk_id() not in processor.encode(''.join(chr(77844 + i) for i in range(4)))
    report_path = tmp_path / 'report.json'
    report = cli.evaluate(heldout, output, report_path)
    assert json.loads(report_path.read_text()) == report
    summary = report['summary']
    assert summary['raw_codes'] == 110
    assert summary['codec_frames'] == 55
    assert summary['roundtrip_exact'] is True
    assert summary['sample_count'] == 2
    assert summary['lm_tokens'] == sum(record['lm_tokens'] for record in report['samples'])
    assert summary['raw_codes_per_lm_token'] == 110 / summary['lm_tokens']
    assert summary['lm_tokens_per_codec_frame'] == summary['lm_tokens'] / 55
    assert 'No generation speedup' in report['interpretation']


def test_training_is_deterministic_and_contains_no_machine_paths(corpus, tmp_path):
    training, _, first = corpus
    second = tmp_path / 'another-tokenizer'
    cli.train(training, first, 4, 2, 20)
    cli.train(training, second, 4, 2, 20)
    assert (first / 'tokenizer.model').read_bytes() == (second / 'tokenizer.model').read_bytes()
    assert str(tmp_path).encode() not in (first / 'tokenizer.model').read_bytes()
    assert str(tmp_path) not in (first / 'manifest.json').read_text()


def test_overlap_detected_after_rename_dtype_and_storage_order_change(corpus, tmp_path):
    training, _, output = corpus
    cli.train(training, output, 4, 2, 20)
    duplicate = tmp_path / 'duplicate'
    duplicate.mkdir()
    np.save(duplicate / 'renamed.npy', np.asfortranarray(np.load(training / 'one.npy').astype('uint8')))
    report_path = tmp_path / 'report.json'
    with pytest.raises(ValueError, match='overlap'):
        cli.evaluate(duplicate, output, report_path)
    assert not report_path.exists()


def test_model_checksum_prevents_evaluation_with_changed_tokenizer(corpus, tmp_path):
    training, heldout, output = corpus
    cli.train(training, output, 4, 2, 20)
    (output / 'tokenizer.model').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='checksum'):
        cli.evaluate(heldout, output, tmp_path / 'report.json')


def test_outputs_are_never_overwritten(corpus, tmp_path):
    training, heldout, output = corpus
    cli.train(training, output, 4, 2, 20)
    model = (output / 'tokenizer.model').read_bytes()
    with pytest.raises(FileExistsError, match='overwrite'):
        cli.train(training, output, 4, 2, 21)
    assert (output / 'tokenizer.model').read_bytes() == model
    report_path = tmp_path / 'report.json'
    report_path.write_text('keep me')
    with pytest.raises(FileExistsError, match='overwrite'):
        cli.evaluate(heldout, output, report_path)
    assert report_path.read_text() == 'keep me'


@pytest.mark.parametrize('values,dtype', [
    ([[0, 4], [1, 2]], np.int64),
    ([[0, -1], [1, 2]], np.int64),
    ([[0., 1.], [1., 2.]], np.float32),
    ([[True, False], [False, True]], np.bool_),
    ([[0, 1]], np.int64),
    ([[], []], np.int64),
    ([[0, 1], [1, 2]], object),
])
def test_bad_numpy_inputs_fail_without_partial_outputs(tmp_path, values, dtype):
    training, output = tmp_path / 'codes', tmp_path / 'tokenizer'
    save(training, 'bad.npy', values, dtype)
    with pytest.raises(ValueError):
        cli.train(training, output, 4, 2, 20)
    assert not output.exists()


def test_invalid_configuration_rejected(corpus):
    training, _, output = corpus
    for cardinality, codebooks, vocabulary, offset in [(4, 2, 4, 77844), (0, 2, 20, 77844),
                                                       (4, 2, 20, 0xD7FF), (4, 2, 20, 31)]:
        with pytest.raises(ValueError):
            cli.train(training, output, cardinality, codebooks, vocabulary, offset)
    assert not output.exists()


def test_cli_train_evaluate_and_failure_exit_codes(corpus, tmp_path):
    training, heldout, output = corpus
    result = subprocess.run([sys.executable, str(SCRIPT), 'train', '--codes', str(training),
                             '--output', str(output), '--codec-cardinality', '4',
                             '--num-codebooks', '2', '--vocab-size', '20'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = tmp_path / 'report.json'
    arguments = [sys.executable, str(SCRIPT), 'evaluate', '--codes', str(heldout),
                 '--tokenizer', str(output), '--output', str(report)]
    result = subprocess.run(arguments, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['roundtrip_exact'] is True
    result = subprocess.run(arguments, capture_output=True, text=True)
    assert result.returncode == 2
    assert 'Refusing to overwrite' in result.stderr
