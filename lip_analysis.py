"""
lip_analysis.py — Lip landmark interpretation for SpeakFlow
============================================================
Takes averaged MediaPipe Face Mesh measurements from the browser
and cross-references them with expected lip posture for a target phoneme.

Measurements (from frontend, normalised by face height, 0-1 scale):
  jaw_aperture  — inner lip gap (landmarks 13→14) / face height
  lip_ratio     — lip height / lip width (high = rounded, low = spread)
  n_frames      — number of camera frames averaged

Clinical positioning:
  Reports observed vs expected posture.
  Never diagnoses — clinician interprets.
"""

from __future__ import annotations
from typing import Optional

# ── Expected lip postures per phoneme ────────────────────────────────────────
# jaw:     'open' | 'mid' | 'closed'
# rounding: 'spread' | 'neutral' | 'rounded'

_EXPECT: dict[str, dict] = {
    # monophthong vowels
    "iː": {"jaw": "closed", "rounding": "spread",  "label": "FLEECE"},
    "ɪ":  {"jaw": "closed", "rounding": "spread",  "label": "KIT"},
    "e":  {"jaw": "mid",    "rounding": "spread",  "label": "DRESS"},
    "æ":  {"jaw": "open",   "rounding": "spread",  "label": "TRAP"},
    "ɑː": {"jaw": "open",   "rounding": "neutral", "label": "PALM"},
    "ɒ":  {"jaw": "open",   "rounding": "rounded", "label": "LOT"},
    "ɔː": {"jaw": "mid",    "rounding": "rounded", "label": "THOUGHT"},
    "ʊ":  {"jaw": "closed", "rounding": "rounded", "label": "FOOT"},
    "uː": {"jaw": "closed", "rounding": "rounded", "label": "GOOSE"},
    "ʌ":  {"jaw": "mid",    "rounding": "neutral", "label": "STRUT"},
    "ɜː": {"jaw": "mid",    "rounding": "neutral", "label": "NURSE"},
    "ə":  {"jaw": "mid",    "rounding": "neutral", "label": "schwa"},
    # Australian diphthongs (starting position)
    "æɪ": {"jaw": "open",   "rounding": "spread",  "label": "FACE (AuE)"},
    "ɑɪ": {"jaw": "open",   "rounding": "neutral", "label": "PRICE (AuE)"},
    "æɔ": {"jaw": "open",   "rounding": "neutral", "label": "MOUTH (AuE)"},
    "əʊ": {"jaw": "mid",    "rounding": "rounded", "label": "GOAT (AuE)"},
    "ɔɪ": {"jaw": "mid",    "rounding": "rounded", "label": "CHOICE"},
}

# Calibrated thresholds (normalised by face height, espeak-validated)
_JAW_OPEN   = 0.045   # jaw_aperture above this → open
_JAW_CLOSED = 0.015   # jaw_aperture below this → closed
_ROUND_HIGH = 0.32    # lip_ratio above this → rounded
_ROUND_LOW  = 0.18    # lip_ratio below this → spread


def _jaw_label(v: float) -> str:
    if v > _JAW_OPEN:   return "open"
    if v < _JAW_CLOSED: return "closed"
    return "mid"


def _round_label(v: float) -> str:
    if v > _ROUND_HIGH: return "rounded"
    if v < _ROUND_LOW:  return "spread"
    return "neutral"


# ── Main analysis ─────────────────────────────────────────────────────────────

def analyse_lip(lip_data: dict, target_phoneme: str) -> dict:
    """
    Interpret lip measurements for a target phoneme.

    Parameters
    ----------
    lip_data        : { jaw_aperture, lip_ratio, n_frames }
    target_phoneme  : IPA symbol, e.g. "æ", "uː", "s"

    Returns
    -------
    {
      available, jaw_aperture, lip_ratio, n_frames,
      jaw_observed, jaw_expected,
      round_observed, round_expected,
      confirmations: [str], mismatches: [str]
    }
    """
    n = lip_data.get("n_frames", 0)
    if n < 3:
        return {"available": False, "reason": "Too few camera frames — face may not be visible"}

    jaw   = lip_data.get("jaw_aperture")
    ratio = lip_data.get("lip_ratio")
    if jaw is None or ratio is None:
        return {"available": False, "reason": "Landmark data missing from request"}

    jaw_obs   = _jaw_label(jaw)
    round_obs = _round_label(ratio)

    exp = _EXPECT.get(target_phoneme)
    if not exp:
        # Consonant or unknown — still report what was observed, no expectation check
        return {
            "available":      True,
            "jaw_aperture":   round(jaw, 4),
            "lip_ratio":      round(ratio, 4),
            "n_frames":       n,
            "jaw_observed":   jaw_obs,
            "round_observed": round_obs,
            "jaw_expected":   None,
            "round_expected": None,
            "confirmations":  [],
            "mismatches":     [],
            "note": f"No lip expectation defined for /{target_phoneme}/",
        }

    exp_jaw   = exp["jaw"]
    exp_round = exp["rounding"]
    confirmations: list[str] = []
    mismatches:    list[str] = []

    # ── Jaw opening ───────────────────────────────────────────────────────────
    if exp_jaw == "open":
        if jaw_obs == "open":
            confirmations.append("Jaw opening is correct — mouth open enough")
        else:
            mismatches.append(
                f"Jaw too {'mid-height' if jaw_obs=='mid' else 'closed'} "
                f"for /{target_phoneme}/ ({exp['label']}) — open your jaw wider"
            )
    elif exp_jaw == "closed":
        if jaw_obs in ("closed", "mid"):
            confirmations.append("Jaw height is appropriate — lips close together")
        else:
            mismatches.append(
                f"Jaw more open than expected for /{target_phoneme}/ "
                f"({exp['label']}) — bring your lips closer together"
            )
    else:  # mid
        if jaw_obs == "open":
            mismatches.append(f"Jaw slightly too open for /{target_phoneme}/")
        elif jaw_obs == "closed":
            mismatches.append(f"Jaw slightly too closed for /{target_phoneme}/")
        else:
            confirmations.append("Jaw height looks good")

    # ── Lip rounding ──────────────────────────────────────────────────────────
    if exp_round == "rounded":
        if round_obs == "rounded":
            confirmations.append("Lip rounding is correct — lips pursed well")
        else:
            mismatches.append(
                f"Lips not rounded enough for /{target_phoneme}/ — "
                f"pucker your lips forward"
            )
    elif exp_round == "spread":
        if round_obs in ("spread", "neutral"):
            confirmations.append("Lip spreading looks correct")
        else:
            mismatches.append(
                f"Lips too rounded for /{target_phoneme}/ — "
                f"spread your lips wide (like a smile)"
            )
    else:  # neutral — wide tolerance
        confirmations.append("Lip position is neutral — that's fine for this sound")

    return {
        "available":      True,
        "jaw_aperture":   round(jaw, 4),
        "lip_ratio":      round(ratio, 4),
        "n_frames":       n,
        "jaw_observed":   jaw_obs,
        "jaw_expected":   exp_jaw,
        "round_observed": round_obs,
        "round_expected": exp_round,
        "confirmations":  confirmations,
        "mismatches":     mismatches,
    }


# ── Cross-reference with acoustic signal ──────────────────────────────────────

def cross_reference(lip: dict, acoustic_segments: list[dict], target_phoneme: str) -> dict:
    """
    Combine lip and acoustic evidence into a single confidence verdict.

    Two signals agreeing = HIGH confidence.
    Signals disagreeing = MISMATCH (likely a deeper articulatory issue, e.g. tongue).

    Returns { confidence, notes: [str] }
    """
    if not lip.get("available"):
        return {"confidence": "acoustic_only", "notes": []}

    # Find the acoustic segment for this phoneme
    vowel = None
    for seg in acoustic_segments:
        if seg.get("phoneme") == target_phoneme and "vowel" in seg:
            vowel = seg["vowel"]
            break

    if not vowel:
        return {
            "confidence": "lip_only",
            "notes": lip.get("confirmations", []) + lip.get("mismatches", []),
        }

    F1_diff = vowel.get("F1_diff", 0)   # positive = jaw more open than ref
    F2_diff = vowel.get("F2_diff", 0)   # positive = more front/spread

    jaw_obs   = lip.get("jaw_observed", "mid")
    round_obs = lip.get("round_observed", "neutral")

    notes: list[str] = []

    # F1 vs jaw cross-check
    if F1_diff < -80 and jaw_obs == "closed":
        notes.append(
            "CONFIRMED: Both acoustic (F1 too low) and lip tracking (jaw closed) "
            "show the jaw is not open enough. Open your jaw wider."
        )
    elif F1_diff < -80 and jaw_obs in ("open", "mid"):
        notes.append(
            "MISMATCH: Jaw appears open (lip tracking) but vowel sounds closed (F1 low). "
            "The issue may be tongue position rather than jaw — check tongue height."
        )
    elif F1_diff > 80 and jaw_obs == "open":
        notes.append(
            "CONFIRMED: Jaw is open and vowel sounds open (F1 high) — jaw position is correct."
        )

    # F2 vs lip rounding cross-check
    if F2_diff < -120 and round_obs != "rounded":
        notes.append(
            "MISMATCH: Vowel sounds rounded/back (F2 low) but lips appear unrounded. "
            "Try pursing your lips forward for this vowel."
        )
    elif F2_diff > 120 and round_obs == "rounded":
        notes.append(
            "MISMATCH: Lips appear rounded but vowel sounds front (F2 high). "
            "Spread your lips for this sound."
        )
    elif F2_diff > 120 and round_obs in ("spread", "neutral"):
        notes.append(
            "CONFIRMED: Lips are spread and vowel sounds front (F2 high) — "
            "lip spreading is correct."
        )

    # Overall confidence
    lip_mismatch      = bool(lip.get("mismatches"))
    acoustic_mismatch = abs(F1_diff) > 80 or abs(F2_diff) > 120
    has_cross_confirm = any("CONFIRMED" in n for n in notes)
    has_cross_mismatch = any("MISMATCH" in n for n in notes)

    if not lip_mismatch and not acoustic_mismatch:
        confidence = "high"
    elif has_cross_confirm and not has_cross_mismatch:
        confidence = "high"
    elif has_cross_mismatch:
        confidence = "mismatch"
    elif lip_mismatch or acoustic_mismatch:
        confidence = "low"
    else:
        confidence = "moderate"

    return {"confidence": confidence, "notes": notes}
