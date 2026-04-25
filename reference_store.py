"""
reference_store.py — Clinician reference recording management for SpeakFlow
=============================================================================
A speech pathologist records themselves saying each exercise word once.
Those recordings become the acoustic targets patients are compared against,
replacing generic textbook averages with the clinician's own voice.

Storage layout
--------------
  references/
    index.json          ← acoustic features + metadata for all words
    cat.wav             ← 16 kHz mono WAV per word
    sun.wav
    ...
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REF_DIR  = os.path.join(BASE_DIR, "references")
INDEX    = os.path.join(REF_DIR, "index.json")

os.makedirs(REF_DIR, exist_ok=True)


# ── Index helpers ─────────────────────────────────────────────────────────────

def _load() -> dict:
    try:
        with open(INDEX) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save(data: dict):
    with open(INDEX, "w") as f:
        json.dump(data, f, indent=2)


# ── Public API ────────────────────────────────────────────────────────────────

def get(word: str) -> dict | None:
    """Return stored reference data for a word, or None."""
    return _load().get(word.lower())


def list_all() -> dict:
    """Return the full index {word: entry}."""
    return _load()


def delete(word: str) -> bool:
    """Remove a reference. Returns True if it existed."""
    idx  = _load()
    key  = word.lower()
    if key not in idx:
        return False
    wav = os.path.join(REF_DIR, f"{key}.wav")
    if os.path.exists(wav):
        os.unlink(wav)
    del idx[key]
    _save(idx)
    return True


def save(word: str, audio_path: str, ipa: str, phonemes: list[str]) -> dict:
    """
    Save a clinician reference recording and extract its acoustic features.

    Parameters
    ----------
    word       : e.g. "cat"
    audio_path : path to the uploaded audio (any ffmpeg-supported format)
    ipa        : Australian English IPA string for this word
    phonemes   : parsed phoneme list, e.g. ["k", "æ", "t"]

    Returns
    -------
    The stored index entry.
    """
    key      = word.lower()
    dest_wav = os.path.join(REF_DIR, f"{key}.wav")

    # Convert to 16 kHz mono WAV for consistency
    subprocess.run(
        ["ffmpeg", "-y", "-i", audio_path,
         "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", dest_wav],
        capture_output=True, check=True,
    )

    # Extract acoustic features from the reference
    vowel_refs, friction_refs = {}, {}
    try:
        from forced_align import align_word
        from acoustic_analysis import analyse_segment, VOWELS, SIBILANTS

        boundaries = align_word(dest_wav, key, phonemes, verbose=False)

        for b in boundaries:
            ph  = b["ipa"]
            seg = analyse_segment(dest_wav, ph, b["start_ms"], b["end_ms"])

            if seg.vowel and seg.vowel.confidence > 0.25:
                vowel_refs[ph] = {
                    "F1":    round(seg.vowel.F1_hz,  0),
                    "F2":    round(seg.vowel.F2_hz,  0),
                    "label": seg.vowel.label,
                }

            if seg.friction:
                friction_refs[ph] = {
                    "CoG":    round(seg.friction.CoG_hz,   0),
                    "energy": round(seg.friction.energy_db, 1),
                }

    except Exception as exc:
        print(f"  [reference_store] acoustic extraction failed: {exc}")

    entry = {
        "word":          key,
        "ipa":           ipa,
        "phonemes":      phonemes,
        "vowel_refs":    vowel_refs,
        "friction_refs": friction_refs,
        "recorded_at":   datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "has_audio":     True,
    }

    idx       = _load()
    idx[key]  = entry
    _save(idx)
    return entry


def wav_path(word: str) -> str | None:
    """Return the path to the reference WAV, or None if not recorded."""
    p = os.path.join(REF_DIR, f"{word.lower()}.wav")
    return p if os.path.exists(p) else None
