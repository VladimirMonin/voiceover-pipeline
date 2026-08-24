"""Transcript-free quality metrics for generated TTS audio."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def normalize_quality_text(text: str) -> str:
    """Return the stable, privacy-local normalization used by the quality gate."""
    return " ".join(_WORD_RE.findall(text.casefold()))


def _repeated_ngram_excess(expected: list[str], actual: list[str]) -> int:
    excess = 0
    for width in (2, 3):
        expected_counts = Counter(zip(*(expected[index:] for index in range(width))))
        actual_counts = Counter(zip(*(actual[index:] for index in range(width))))
        excess += sum(
            max(0, count - max(1, expected_counts[ngram])) for ngram, count in actual_counts.items()
        )
    return excess


@dataclass(frozen=True)
class TTSQualityResult:
    passed: bool
    similarity: float
    expected_word_count: int
    actual_word_count: int
    missing_word_count: int
    unexpected_word_count: int
    repeated_ngram_excess: int
    expected_text_sha256: str
    actual_transcript_sha256: str
    failure_reasons: tuple[str, ...]

    def public_receipt(
        self,
        *,
        audio_sha256: str,
        asr_provider: str,
        asr_model: str | None,
        asr_runtime: str,
        asr_model_revision: str | None,
    ) -> dict[str, Any]:
        """Return content-free evidence suitable for durable public receipts."""
        return {
            "artifact_type": "voiceover-tts-quality-receipt",
            "passed": self.passed,
            "audio_sha256": audio_sha256,
            "expected_text_sha256": self.expected_text_sha256,
            "actual_transcript_sha256": self.actual_transcript_sha256,
            "expected_word_count": self.expected_word_count,
            "actual_word_count": self.actual_word_count,
            "similarity": self.similarity,
            "missing_word_count": self.missing_word_count,
            "unexpected_word_count": self.unexpected_word_count,
            "repeated_ngram_excess": self.repeated_ngram_excess,
            "failure_reasons": list(self.failure_reasons),
            "asr": {
                "provider": asr_provider,
                "model": asr_model,
                "runtime": asr_runtime,
                "model_revision": asr_model_revision,
            },
            "human_listening_required": True,
        }


def evaluate_tts_transcript(
    *,
    expected_text: str,
    actual_transcript: str,
    minimum_similarity: float = 0.95,
    maximum_missing_ratio: float = 0.05,
    maximum_unexpected_ratio: float = 0.05,
    maximum_repeated_ngram_excess: int = 0,
) -> TTSQualityResult:
    """Detect major omissions, unexpected speech, and repetition fail-closed."""
    expected_normalized = normalize_quality_text(expected_text)
    actual_normalized = normalize_quality_text(actual_transcript)
    expected_words = expected_normalized.split()
    actual_words = actual_normalized.split()
    if not expected_words:
        raise ValueError("Expected TTS text must contain at least one word")

    matcher = SequenceMatcher(None, expected_words, actual_words, autojunk=False)
    missing = 0
    unexpected = 0
    for tag, expected_start, expected_end, actual_start, actual_end in matcher.get_opcodes():
        if tag in ("delete", "replace"):
            missing += expected_end - expected_start
        if tag in ("insert", "replace"):
            unexpected += actual_end - actual_start
    repeated = _repeated_ngram_excess(expected_words, actual_words)
    similarity = round(matcher.ratio(), 9)
    denominator = len(expected_words)

    reasons: list[str] = []
    if similarity < minimum_similarity:
        reasons.append("similarity")
    if missing / denominator > maximum_missing_ratio:
        reasons.append("missing_words")
    if unexpected / denominator > maximum_unexpected_ratio:
        reasons.append("unexpected_words")
    if repeated > maximum_repeated_ngram_excess:
        reasons.append("repeated_ngram")

    return TTSQualityResult(
        passed=not reasons,
        similarity=similarity,
        expected_word_count=len(expected_words),
        actual_word_count=len(actual_words),
        missing_word_count=missing,
        unexpected_word_count=unexpected,
        repeated_ngram_excess=repeated,
        expected_text_sha256=hashlib.sha256(expected_normalized.encode("utf-8")).hexdigest(),
        actual_transcript_sha256=hashlib.sha256(actual_normalized.encode("utf-8")).hexdigest(),
        failure_reasons=tuple(reasons),
    )
