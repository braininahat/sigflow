"""Phoneme recognizer process node — wav2vec2 ONNX.

Receives recorded audio, runs phoneme recognition, scores against
expected phonemes, and emits results as Event samples.  Uses the
shared inference module for model loading and recognition.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from sigflow.node import process_node, Param
from sigflow.types import Event, Port, Sample, TimeSeries1D

log = logging.getLogger(__name__)


@process_node(
    name="phoneme_recognizer",
    inputs=[Port("audio", TimeSeries1D)],
    outputs=[Port("phonemes", Event)],
    category="inference",
    params=[
        Param("model_path", "str", "weights/wav2vec2-phoneme/model.onnx", label="Model Path"),
        Param("vocab_path", "str", "weights/wav2vec2-phoneme/vocab.json", label="Vocab Path"),
    ],
)
def phoneme_recognizer_node(item, *, state, config):
    if "session" not in state:
        _load_model(state, config)

    audio = item.data
    if isinstance(audio, np.ndarray):
        audio = audio.astype(np.float32)
    else:
        return None

    word = item.metadata.get("word", "")
    expected = item.metadata.get("expected_phonemes", [])

    t0 = time.perf_counter()

    from ultraspeech.inference.phoneme_recognizer import (
        recognize_phonemes_detailed,
        score_phonemes_detailed,
    )
    from ultraspeech.inference.phoneme_vocab import score_phonemes

    from sigflow.paths import resolve_data_path
    detail = recognize_phonemes_detailed(audio, state=state,
                                          model_path=str(resolve_data_path(config.get("model_path", "weights/wav2vec2-phoneme/model.onnx"))),
                                          vocab_path=str(resolve_data_path(config.get("vocab_path", "weights/wav2vec2-phoneme/vocab.json"))))

    recognized = detail.phonemes
    latency_ms = (time.perf_counter() - t0) * 1000

    # Word-level Levenshtein score (backward compatible)
    score = score_phonemes(expected, recognized) if expected else 0.0

    # Per-phoneme confidence scoring
    per_phoneme = []
    word_score_1_5 = 3
    if expected and state.get("vocab_reverse"):
        phoneme_scores = score_phonemes_detailed(expected, detail, state["vocab_reverse"])
        per_phoneme = [
            {
                "phoneme": ps.phoneme,
                "score": ps.score_1_5,
                "confidence": round(ps.confidence, 3),
                "op": ps.op,
                "recognized_as": ps.recognized_as,
                "start_s": round(ps.start_s, 3),
                "end_s": round(ps.end_s, 3),
            }
            for ps in phoneme_scores
        ]
        if phoneme_scores:
            from ultraspeech.inference.phoneme_recognizer import confidence_to_score
            avg_confidence = sum(ps.confidence for ps in phoneme_scores) / len(phoneme_scores)
            word_score_1_5 = confidence_to_score(avg_confidence)

    duration_s = len(audio) / 16000
    log.info(
        "phoneme: '%s' expected=%s recognized=%s score=%.0f%% word_1_5=%d (%.0fms, %.1fs audio)",
        word, expected, recognized, score * 100, word_score_1_5, latency_ms, duration_s,
    )
    if per_phoneme:
        summary = " ".join(f"{p['phoneme']}({p['score']},{p['op']})" for p in per_phoneme)
        log.info("phoneme: '%s' per-phoneme: %s", word, summary)

    return {"phonemes": item.replace(
        data={
            "phonemes": recognized,
            "word": word,
            "expected": expected,
            "score": score,
            "latency_ms": latency_ms,
            "per_phoneme": per_phoneme,
            "word_score_1_5": word_score_1_5,
        },
        port_type=Event,
    )}


def _load_model(state, config):
    """Lazy-load ONNX session and vocab via the inference module."""
    from sigflow.paths import resolve_data_path
    from ultraspeech.inference.phoneme_recognizer import _load_model as _inf_load

    model_path = str(resolve_data_path(config.get("model_path", "weights/wav2vec2-phoneme/model.onnx")))
    vocab_path = str(resolve_data_path(config.get("vocab_path", "weights/wav2vec2-phoneme/vocab.json")))
    _inf_load(state, model_path, vocab_path)
