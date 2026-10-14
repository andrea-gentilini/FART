"""Lossless SentencePiece adaptation of codebook-major EnCodec token streams.

This experimental representation is not prefix-preserving for audio continuation.
Its vocabulary and Unicode mapping must match the tokenizer training corpus.
"""

from pathlib import Path
import typing as tp

import torch


class AudioTokenTokenizer:
    def __init__(self, model_path: str, codec_cardinality: int, num_codebooks: int,
                 lm_cardinality: int, lm_num_codebooks: int = 1,
                 unicode_offset: int = 77844):
        if not model_path or not Path(model_path).expanduser().is_file():
            raise FileNotFoundError(f"SentencePiece model not found: {model_path!r}")
        if codec_cardinality <= 0 or num_codebooks <= 0:
            raise ValueError("Codec cardinality and codebook count must be positive.")
        if lm_num_codebooks != 1:
            raise ValueError("FART requires transformer_lm.n_q=1.")
        end = unicode_offset + codec_cardinality
        if unicode_offset < 0 or end > 0x110000 or (unicode_offset < 0xE000 and end > 0xD800):
            raise ValueError("Codec alphabet must contain valid Unicode scalar values.")
        import sentencepiece as spm
        self.processor = spm.SentencePieceProcessor(model_file=str(Path(model_path).expanduser()))
        self.cardinality = self.processor.get_piece_size()
        if lm_cardinality != self.cardinality:
            raise ValueError(
                f"transformer_lm.card={lm_cardinality} does not match "
                f"SentencePiece vocabulary size {self.cardinality}.")
        self.codec_cardinality = codec_cardinality
        self.num_codebooks = num_codebooks
        self.unicode_offset = unicode_offset
        # Check coverage and normalization before training, not after an OOV batch.
        self._encode_text(''.join(chr(unicode_offset + i) for i in range(codec_cardinality)))
        # Every sampled piece must expand to at least one valid codec symbol.
        # SentencePiece keeps reserved IDs (including <unk>) in its vocabulary.
        forbidden = []
        for piece_id in range(self.cardinality):
            reserved = (self.processor.is_unknown(piece_id) or self.processor.is_control(piece_id)
                        or self.processor.is_unused(piece_id))
            text = '' if reserved else self.processor.decode_ids([piece_id])
            if not text or any(not unicode_offset <= ord(c) < end for c in text):
                forbidden.append(piece_id)
        self.forbidden_token_ids = tuple(forbidden)

    def _encode_text(self, text: str) -> tp.List[int]:
        ids = self.processor.encode_as_ids(text)
        if self.processor.unk_id() in ids or self.processor.decode_ids(ids) != text:
            raise ValueError("SentencePiece model does not losslessly encode the codec alphabet.")
        return ids

    def encode_batch(self, codes: torch.Tensor,
                     padding_mask: tp.Optional[torch.Tensor] = None) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """Encode valid codec frames; return [B, 1, S] IDs and valid-position mask."""
        if codes.ndim != 3 or codes.shape[1] != self.num_codebooks or codes.shape[0] == 0:
            raise ValueError("Expected nonempty codec tokens of shape [B, num_codebooks, T].")
        if codes.dtype != torch.long:
            raise ValueError("Codec tokens must have torch.long dtype.")
        if padding_mask is None:
            padding_mask = torch.ones_like(codes, dtype=torch.bool)
        if padding_mask.shape != codes.shape or padding_mask.dtype != torch.bool:
            raise ValueError("Padding mask must be boolean and match codec tokens.")
        sequences = []
        for item, mask in zip(codes.cpu(), padding_mask.cpu()):
            valid = mask[0]
            length = int(valid.sum())
            expected = torch.arange(item.shape[-1]) < length
            if not torch.equal(mask, expected.expand_as(mask)):
                raise ValueError("All codebooks must share a contiguous prefix of valid frames.")
            if length == 0:
                raise ValueError("Cannot tokenize an example with no valid audio frames.")
            values = item[:, :length].reshape(-1).tolist()
            if any(i < 0 or i >= self.codec_cardinality for i in values):
                raise ValueError("Codec token outside the configured alphabet.")
            text = ''.join(chr(self.unicode_offset + i) for i in values)
            sequences.append(torch.tensor(self._encode_text(text), dtype=torch.long))
        padded = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=self.cardinality).to(codes.device)
        lengths = torch.tensor([len(s) for s in sequences], device=codes.device)
        mask = torch.arange(padded.shape[-1], device=codes.device)[None] < lengths[:, None]
        return padded[:, None], mask[:, None]

    def decode_sequence(self, ids: torch.Tensor, num_frames: tp.Optional[int] = None) -> torch.Tensor:
        """Decode one *unpadded* LM sequence to [1, K, T] codec IDs.

        Generation must specify its training layout length. Extra raw codes are
        discarded before reshaping, so expansion cannot move codebook boundaries.
        Without a known length, only exact, complete round trips are accepted.
        """
        if ids.ndim != 1 or ids.dtype != torch.long:
            raise ValueError("Expected one unpadded torch.long SentencePiece sequence.")
        values = ids.tolist()
        if any(i < 0 or i >= self.cardinality for i in values):
            raise ValueError("SentencePiece ID outside the vocabulary (possibly batch padding).")
        if any(self.processor.is_unknown(i) or self.processor.is_control(i)
               or self.processor.is_unused(i) for i in values):
            raise ValueError("Generated sequence contains reserved SentencePiece IDs.")
        text = self.processor.decode_ids(values)
        codes = [ord(c) - self.unicode_offset for c in text]
        if any(i < 0 or i >= self.codec_cardinality for i in codes):
            raise ValueError("Decoded text contains characters outside the codec alphabet.")
        if num_frames is not None:
            required = self.num_codebooks * num_frames
            if num_frames <= 0 or len(codes) < required:
                raise ValueError("Generated stream is too short for the required codebook layout.")
            codes = codes[:required]
        elif len(codes) % self.num_codebooks:
            raise ValueError("Decoded stream is not a complete codebook layout.")
        if not codes:
            raise ValueError("Decoded sequence contains no complete codec frames.")
        return torch.tensor(codes, dtype=torch.long, device=ids.device).reshape(1, self.num_codebooks, -1)


def tokenizer_from_config(cfg, compression_model, lm_cardinality: int, lm_num_codebooks: int,
                          model_path: tp.Optional[str] = None) -> tp.Optional[AudioTokenTokenizer]:
    options = cfg.get('fart', {}) if cfg is not None else {}
    path = model_path if model_path is not None else options.get('tokenizer_path')
    if path:
        return AudioTokenTokenizer(path, compression_model.cardinality, compression_model.num_codebooks,
                                   lm_cardinality, lm_num_codebooks, options.get('unicode_offset', 77844))
    if lm_cardinality != compression_model.cardinality or lm_num_codebooks != compression_model.num_codebooks:
        raise ValueError("LM/codec cardinality or codebook count mismatch. "
                         "For FART checkpoints, configure fart.tokenizer_path.")
    return None


def decode_generated_audio(tokenizer: AudioTokenTokenizer, compression_model,
                           tokens: torch.Tensor, duration: float, layout_frames: int) -> torch.Tensor:
    """Decode independently, then crop every waveform to the requested duration."""
    if tokens.ndim != 3 or tokens.shape[1] != 1:
        raise ValueError("FART generation must return [B, 1, S] tokens.")
    output = []
    num_samples = int(duration * compression_model.sample_rate)
    if num_samples <= 0:
        raise ValueError("Generation duration must be positive.")
    for sequence in tokens[:, 0]:
        codes = tokenizer.decode_sequence(sequence, num_frames=layout_frames)
        audio = compression_model.decode(codes, None)
        if audio.shape[-1] < num_samples:
            raise ValueError("Generated FART sequence is too short for the requested audio duration.")
        output.append(audio[..., :num_samples])
    return torch.cat(output, dim=0)
