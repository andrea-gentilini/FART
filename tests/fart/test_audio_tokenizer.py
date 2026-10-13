"""CPU regressions for FART; no pretrained models or full AudioCraft imports."""
import importlib.util
import ast
import contextlib
import math
from pathlib import Path
import time
from types import SimpleNamespace

import pytest
import sentencepiece as spm
import torch


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('fart_audio_tokenizer', ROOT / 'audiocraft/utils/audio_tokenizer.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
AudioTokenTokenizer = module.AudioTokenTokenizer


@pytest.fixture(scope='module')
def tokenizer_model(tmp_path_factory):
    path = tmp_path_factory.mktemp('sentencepiece')
    alphabet = [chr(77844 + i) for i in range(4)]
    corpus = path / 'corpus.txt'
    corpus.write_text('\n'.join(''.join(alphabet[i:] + alphabet[:i]) * 12 for i in range(4)))
    prefix = path / 'codec'
    spm.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=str(prefix), vocab_size=20,
        model_type='bpe', hard_vocab_limit=False, character_coverage=1.0,
        normalization_rule_name='identity', add_dummy_prefix=False,
        split_by_unicode_script=False, split_by_whitespace=False,
        bos_id=-1, eos_id=-1, pad_id=-1, max_sentencepiece_length=4,
        minloglevel=2)
    model = str(prefix) + '.model'
    card = spm.SentencePieceProcessor(model_file=model).get_piece_size()
    return model, card


@pytest.fixture
def tokenizer(tokenizer_model):
    path, card = tokenizer_model
    return AudioTokenTokenizer(path, 4, 2, card)


def test_roundtrip_preserves_codebook_order(tokenizer):
    codes = torch.tensor([[[0, 1, 2, 3], [3, 2, 1, 0]]])
    ids, mask = tokenizer.encode_batch(codes)
    assert torch.equal(tokenizer.decode_sequence(ids[0, 0, mask[0, 0]]), codes)


def test_padding_is_not_tokenized_and_unequal_lengths_roundtrip(tokenizer):
    codes = torch.tensor([[[0, 1, 99, 99], [2, 3, 99, 99]],
                          [[3, 0, 2, 1], [1, 3, 0, 2]]])
    mask = torch.tensor([[[True, True, False, False]] * 2, [[True] * 4] * 2])
    ids, encoded_mask = tokenizer.encode_batch(codes, mask)
    assert ids.shape[1] == 1
    assert torch.all(ids[~encoded_mask] == tokenizer.cardinality)
    for i, length in enumerate([2, 4]):
        restored = tokenizer.decode_sequence(ids[i, 0, encoded_mask[i, 0]])
        assert torch.equal(restored, codes[i:i + 1, :, :length])


@pytest.mark.parametrize('mask', [
    [[[True, False], [False, True]]],
    [[[False, True], [False, True]]],
    [[[False, False], [False, False]]],
])
def test_invalid_frame_masks_rejected(tokenizer, mask):
    with pytest.raises(ValueError):
        tokenizer.encode_batch(torch.zeros(1, 2, 2, dtype=torch.long), torch.tensor(mask))


def test_invalid_codec_ids_rejected(tokenizer):
    with pytest.raises(ValueError, match='outside'):
        tokenizer.encode_batch(torch.tensor([[[0, 4], [1, 2]]]))


def test_invalid_and_reserved_sentencepiece_ids_rejected(tokenizer):
    for value in [-1, tokenizer.cardinality, tokenizer.processor.unk_id()]:
        with pytest.raises(ValueError):
            tokenizer.decode_sequence(torch.tensor([value]))


def test_partial_layout_requires_known_frame_count(tokenizer):
    ids = torch.tensor(tokenizer.processor.encode_as_ids(''.join(chr(77844 + i) for i in [0, 1, 2])))
    with pytest.raises(ValueError, match='complete codebook'):
        tokenizer.decode_sequence(ids)
    assert torch.equal(tokenizer.decode_sequence(ids, num_frames=1), torch.tensor([[[0], [1]]]))
    with pytest.raises(ValueError, match='too short'):
        tokenizer.decode_sequence(ids, num_frames=2)


def test_configuration_validation(tokenizer_model):
    path, card = tokenizer_model
    with pytest.raises(FileNotFoundError):
        AudioTokenTokenizer(path + '.missing', 4, 2, card)
    with pytest.raises(ValueError, match='vocabulary size'):
        AudioTokenTokenizer(path, 4, 2, card + 1)
    with pytest.raises(ValueError, match='n_q=1'):
        AudioTokenTokenizer(path, 4, 2, card, lm_num_codebooks=2)
    with pytest.raises(ValueError, match='losslessly'):
        AudioTokenTokenizer(path, 5, 2, card)


def test_standard_configuration_does_not_load_sentencepiece():
    codec = SimpleNamespace(cardinality=2048, num_codebooks=4)
    assert module.tokenizer_from_config({}, codec, 2048, 4) is None
    with pytest.raises(ValueError, match='tokenizer_path'):
        module.tokenizer_from_config({}, codec, 32000, 1)


def test_decode_batch_before_waveform_padding(tokenizer):
    # Equal LM lengths can decode to unequal codec lengths. Padding the flat
    # streams first would move the boundary between codebooks in the short item.
    pieces = []
    for i in range(tokenizer.cardinality):
        if tokenizer.processor.is_unknown(i):
            continue
        text = tokenizer.processor.decode_ids([i])
        if text and len(text) % 2 == 0:
            pieces.append((len(text), i))
    pieces.sort()
    assert pieces[0][0] < pieces[-1][0]
    tokens = torch.tensor([[[pieces[0][1]]], [[pieces[-1][1]]]])
    seen = []

    class Codec:
        sample_rate = 1

        def decode(self, codes, scale):
            seen.append(codes.clone())
            return codes[:, :1].float()

    audio = module.decode_generated_audio(tokenizer, Codec(), tokens, 1, layout_frames=1)
    assert audio.shape == (2, 1, 1)
    assert seen[0].shape == seen[1].shape == (1, 2, 1)
    for row, codes in zip(tokens[:, 0], seen):
        text = tokenizer.processor.decode_ids(row.tolist())
        expected = torch.tensor([ord(c) - tokenizer.unicode_offset for c in text[:2]]).reshape(1, 2, 1)
        assert torch.equal(codes, expected)


def test_extra_generated_suffix_cannot_move_codebook_boundaries(tokenizer):
    raw = [0, 1, 2, 3, 3, 2, 1, 0] + [3, 3, 2, 2, 1, 1]
    ids = torch.tensor(tokenizer.processor.encode_as_ids(''.join(chr(77844 + i) for i in raw)))
    actual = tokenizer.decode_sequence(ids, num_frames=4)
    assert torch.equal(actual, torch.tensor([[[0, 1, 2, 3], [3, 2, 1, 0]]]))


def isolated_method(filename, class_name, method_name):
    """Execute the real method body without importing unrelated GPU/audio dependencies."""
    source = ast.parse((ROOT / filename).read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    method.decorator_list = []
    tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                            method], type_ignores=[])
    ast.fix_missing_locations(tree)
    namespace = {'torch': torch, 'math': math, 'time': time,
                 'decode_generated_audio': module.decode_generated_audio}
    exec(compile(tree, filename, 'exec'), namespace)
    return namespace[method_name]


@pytest.mark.parametrize('fart', [False, True])
def test_solver_generation_uses_correct_token_domain(tokenizer, fart):
    method = isolated_method('audiocraft/solvers/musicgen.py', 'MusicGenSolver', 'run_generate_step')
    codes = torch.tensor([[[0, 1, 2, 3], [3, 2, 1, 0]]])
    tokens = tokenizer.encode_batch(codes)[0] if fart else codes
    calls, decoded = [], []

    class Codec:
        sample_rate = 4
        frame_rate = 4
        num_codebooks = 2

        def decode(self, actual, scale):
            decoded.append(actual)
            return torch.ones(1, 1, 4)

    def generate(prompt, attributes, **kwargs):
        calls.append(kwargs)
        return tokens

    solver = SimpleNamespace(audio_tokenizer=tokenizer if fart else None,
                             cfg=SimpleNamespace(dataset=SimpleNamespace(segment_duration=1)),
                             compression_model=Codec(), model=SimpleNamespace(generate=generate),
                             autocast=contextlib.nullcontext(), generation_params={})
    batch = (torch.ones(1, 1, 4), [SimpleNamespace(to_condition_attributes=lambda: object())])
    result = method(solver, batch, gen_duration=0.5)
    assert calls[0]['max_gen_len'] == (8 if fart else 2)
    assert torch.equal(decoded[0], codes)
    assert result['gen_audio'].shape[-1] == (2 if fart else 4)
    if fart:
        with pytest.raises(ValueError, match='continuation'):
            method(solver, batch, gen_duration=1, prompt_duration=0.5)
        with pytest.raises(ValueError, match='training segment'):
            method(solver, batch, gen_duration=2)


def test_public_api_uses_training_layout_and_rejects_continuation(tokenizer):
    method = isolated_method('audiocraft/models/musicgen.py', 'MusicGen', '_generate_tokens')
    calls = []

    def generate(prompt, attrs, **kwargs):
        calls.append(kwargs)
        return torch.ones(1, 1, 8, dtype=torch.long)
    model = SimpleNamespace(audio_tokenizer=tokenizer, duration=0.5, max_duration=1, frame_rate=4,
                            compression_model=SimpleNamespace(num_codebooks=2),
                            lm=SimpleNamespace(generate=generate), generation_params={},
                            autocast=contextlib.nullcontext(), _progress_callback=None)
    method(model, [object()], None)
    assert calls[0]['max_gen_len'] == 8
    with pytest.raises(ValueError, match='continuation'):
        method(model, [object()], torch.zeros(1, 2, 1))
    model.duration = 2
    with pytest.raises(ValueError, match='long-form'):
        method(model, [object()], None)


def test_solver_training_masks_sentencepiece_padding(tokenizer):
    method = isolated_method('audiocraft/solvers/musicgen.py', 'MusicGenSolver', 'run_step')
    codes = torch.tensor([[[0, 1, 99, 99], [2, 3, 99, 99]],
                          [[3, 0, 2, 1], [1, 3, 0, 2]]])
    mask = torch.tensor([[[True, True, False, False]] * 2, [[True] * 4] * 2])
    observed = []

    def predictions(tokens, attrs, conditions):
        return SimpleNamespace(logits=torch.empty(0), mask=torch.ones_like(tokens, dtype=torch.bool))

    def cross_entropy(logits, tokens, actual_mask):
        observed.append((tokens, actual_mask))
        return torch.tensor(1.0), [torch.tensor(1.0)]
    solver = SimpleNamespace(device='cpu', audio_tokenizer=tokenizer, is_training=False,
                             model=SimpleNamespace(compute_predictions=predictions,
                                                   condition_provider=SimpleNamespace(conditioners={})),
                             _prepare_tokens_and_attributes=lambda *args: ({}, codes, mask),
                             _compute_cross_entropy=cross_entropy, autocast=contextlib.nullcontext(),
                             deadlock_detect=SimpleNamespace(update=lambda name: None))
    metrics = method(solver, 0, None, {})
    assert metrics['ce'] == 1
    encoded, valid = observed[0]
    assert valid.sum(2)[0] < valid.sum(2)[1]
    assert torch.all(encoded[~valid] == tokenizer.cardinality)
    assert torch.equal(tokenizer.decode_sequence(encoded[0, 0, valid[0, 0]]), codes[:1, :, :2])
