"""GDPVal scorer for the EvoSkill run loop.

Scoring strategy
----------------
GDPVal tasks are graded by the file types the agent creates.  The official
OpenAI grader is not public, so we use two complementary approaches:

* **Manifest-based** (used in the run loop): parse file extensions from the
  agent's FINAL_ANSWER manifest and compare them against the expected
  extensions stored in *ground_truth*.  This avoids filesystem state issues
  that arise when the same workspace is reused across loop iterations.

* **Filesystem-based** (used in the standalone benchmark runner): scan the
  actual submission directory on disk.  See gdpval_runner._score().
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

_FINAL_ANSWER_RE = re.compile(
    r"<FINAL_ANSWER>\s*(.*?)\s*</FINAL_ANSWER>", re.DOTALL | re.IGNORECASE
)
_EXPECTED_EXT_RE = re.compile(r"Expected deliverable file types:\s*([^\n]+)")


def _parse_manifest_extensions(predicted: str) -> Counter:
    """Extract file extensions from the agent's FINAL_ANSWER manifest.

    Falls back to scanning all lines if FINAL_ANSWER tags are absent.
    """
    match = _FINAL_ANSWER_RE.search(predicted)
    content = match.group(1) if match else predicted
    exts: Counter = Counter()
    for line in content.splitlines():
        line = line.strip().lstrip("- ").strip()
        if not line or line.startswith("#"):
            continue
        suffix = Path(line).suffix.lower()
        if suffix:
            exts[suffix] += 1
    return exts


def _parse_expected_extensions(question: str, ground_truth: str) -> Counter:
    """Derive expected extensions from *ground_truth* or from the question prompt.

    *ground_truth* may be a comma-separated list of extensions such as
    ``".xlsx,.md"`` (as stored in loop train/val data).  If that fails we
    fall back to parsing the "Expected deliverable file types:" line baked
    into the prepared prompt.
    """
    if ground_truth and ground_truth.strip():
        parts = [e.strip() for e in ground_truth.split(",") if e.strip()]
        if parts and all(p.startswith(".") for p in parts):
            return Counter(parts)
    match = _EXPECTED_EXT_RE.search(question)
    if match:
        raw = match.group(1).strip()
        parts = [e.strip() for e in raw.split(",") if e.strip()]
        return Counter(p if p.startswith(".") else f".{p}" for p in parts)
    return Counter()


def score_gdpval(question: str, predicted: str, ground_truth: str) -> float:
    """Manifest-based GDPVal scorer for the run loop.

    Signature matches the loop's ``ScorerFn``:
        (question, predicted, ground_truth) -> float in [0, 1]

    Args:
        question:     The prepared task prompt (contains "Expected deliverable
                      file types:" line and submission directory path).
        predicted:    The agent's raw output — expected to contain a
                      ``<FINAL_ANSWER>…</FINAL_ANSWER>`` manifest.
        ground_truth: Comma-separated expected extensions, e.g. ``".xlsx,.md"``.
                      If empty, the extensions are extracted from *question*.

    Returns:
        Proportion of expected file types present in the manifest.
        Returns 0.0 when no files were listed and extensions were expected.
    """
    expected = _parse_expected_extensions(question, ground_truth)
    if not expected:
        # No expectations → score by whether agent produced any manifest
        return 1.0 if _parse_manifest_extensions(predicted) else 0.0

    generated = _parse_manifest_extensions(predicted)
    matched = sum(min(generated[ext], cnt) for ext, cnt in expected.items())
    total = sum(expected.values())
    return matched / total if total else 0.0
