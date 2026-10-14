"""Re-evaluate the published real-codec fixture without network or model weights."""
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / 'examples/fart-tokenizer-smoke'
spec = importlib.util.spec_from_file_location('fart_cli_recorded', ROOT / 'scripts/fart_tokenizer.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def test_recorded_real_codec_example_reproduces_counts_and_exact_roundtrip(tmp_path):
    saved = json.loads((EXAMPLE / 'evaluation.json').read_text())
    result = cli.evaluate(EXAMPLE / 'heldout', EXAMPLE / 'tokenizer', tmp_path / 'evaluation.json')
    assert result['summary'] == saved['summary']
    assert result['samples'] == saved['samples']
    assert result['model_sha256'] == saved['model_sha256']


def test_recorded_source_files_and_evaluation_match_provenance():
    provenance = json.loads((EXAMPLE / 'provenance.json').read_text())
    for source in provenance['sources']:
        assert hashlib.sha256((ROOT / source['source']).read_bytes()).hexdigest() == source['source_sha256']
        assert hashlib.sha256((EXAMPLE / source['codes']).read_bytes()).hexdigest() == source['codes_sha256']
    assert hashlib.sha256((EXAMPLE / 'evaluation.json').read_bytes()).hexdigest() == provenance['evaluation_sha256']
