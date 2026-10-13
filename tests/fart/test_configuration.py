"""Compose real training configurations without loading pretrained models."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError
import pytest


CONFIG = Path(__file__).resolve().parents[2] / 'config'


def training_config(*overrides):
    with initialize_config_dir(config_dir=str(CONFIG), version_base='1.3'):
        return compose(config_name='config', overrides=list(overrides))


def test_standard_musicgen_keeps_codec_vocabulary_and_pattern():
    cfg = training_config('solver=musicgen/musicgen_base_32khz', 'dset=audio/example')
    assert cfg.fart.tokenizer_path is None
    assert (cfg.transformer_lm.n_q, cfg.transformer_lm.card) == (4, 2048)
    assert cfg.codebooks_pattern.modeling == 'delay'
    assert cfg.generate.lm.prompted_samples
    assert cfg.datasource.train == 'egs/example'


def test_documented_fart_command_composes(monkeypatch, tmp_path):
    monkeypatch.setenv('FART_DATASET_ROOT', str(tmp_path))
    cfg = training_config(
        'solver=musicgen/fart', 'dset=my_music', 'model/lm/model_scale=smaller',
        f'fart.tokenizer_path={tmp_path}/music.model')
    assert (cfg.transformer_lm.n_q, cfg.transformer_lm.card) == (1, 32000)
    assert (cfg.transformer_lm.dim, cfg.transformer_lm.num_heads, cfg.transformer_lm.num_layers) == (768, 12, 8)
    assert cfg.codebooks_pattern.modeling == 'unroll'
    assert list(cfg.codebooks_pattern.unroll.flattening) == [0]
    assert list(cfg.codebooks_pattern.unroll.delays) == [0]
    assert not cfg.generate.lm.prompted_samples
    assert cfg.generate.lm.unprompted_samples
    assert cfg.fart.tokenizer_path == str(tmp_path / 'music.model')
    assert cfg.datasource.train == str(tmp_path / 'train.jsonl.gz')
    assert cfg.datasource.valid == str(tmp_path / 'valid.jsonl.gz')
    assert cfg.datasource.max_sample_rate == 32000
    assert cfg.sample_rate == 32000
    assert cfg.channels == 1


def test_fart_requires_explicit_tokenizer_and_dataset(monkeypatch):
    monkeypatch.delenv('FART_DATASET_ROOT', raising=False)
    monkeypatch.delenv('FART_TOKENIZER_PATH', raising=False)
    cfg = training_config('solver=musicgen/fart', 'dset=my_music')
    with pytest.raises(InterpolationResolutionError, match='FART_TOKENIZER_PATH'):
        _ = cfg.fart.tokenizer_path
    with pytest.raises(InterpolationResolutionError, match='FART_DATASET_ROOT'):
        _ = cfg.datasource.train


def test_team_defaults_do_not_require_a_specific_home_directory():
    cfg = OmegaConf.load(CONFIG / 'teams/default.yaml')
    for cluster in ('default', 'darwin'):
        assert cfg[cluster].dora_dir == './outputs'
        assert cfg[cluster].reference_dir == './reference'
