"""
rule_engine.py — SpeakFlow Measurement → Error Code Mapper
===========================================================
Pure logic.  No strings.  No patient-facing text.

Takes structured acoustic + visual measurements and returns
a list of error codes that index into clinical_templates.py.

This file contains the clinical decision rules.
A speech pathologist should review the thresholds.

Thresholds
----------
  F1 deviation  > ±80 Hz   → mild;  > ±150 Hz → moderate;  > ±250 Hz → severe
  F2 deviation  > ±150 Hz  → mild;  > ±300 Hz → moderate;  > ±500 Hz → severe
  CoG /s/       < 5500 Hz  → mild lisp risk
                < 4800 Hz  → moderate (addental)
                < 3800 Hz  → high (interdental or lateral)
  Score         ≥ 0.95 → perfect;  0.80–0.95 → very good;  0.60 → good; etc.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ErrorCode:
    code:     str
    severity: str       # "mild" | "moderate" | "severe" | "info"
    # Optional diagnostic values for the clinician report
    f1_diff:  float | None = None
    f2_diff:  float | None = None
    cog_hz:   float | None = None
    energy_db:float | None = None


# ── Vowel error rules ─────────────────────────────────────────────────────────

def check_vowel(
    F1_hz: float, F2_hz: float,
    F1_ref: float, F2_ref: float,
    confidence: float = 1.0,
) -> list[ErrorCode]:
    """
    Compare measured F1/F2 to target values and return error codes.
    confidence < 0.4 → skip (measurement unreliable).
    """
    errors: list[ErrorCode] = []
    if confidence < 0.35:
        return errors

    f1d = F1_hz - F1_ref   # positive = tongue too low (F1 above target)
    f2d = F2_hz - F2_ref   # positive = tongue too front (F2 above target)

    # ── F1 / tongue height ────────────────────────────────────────────────────
    if f1d > 250:
        errors.append(ErrorCode("TONGUE_TOO_LOW",      "severe",   f1_diff=f1d))
    elif f1d > 150:
        errors.append(ErrorCode("TONGUE_TOO_LOW",      "moderate", f1_diff=f1d))
    elif f1d > 80:
        errors.append(ErrorCode("TONGUE_TOO_LOW_MILD", "mild",     f1_diff=f1d))

    if f1d < -250:
        errors.append(ErrorCode("TONGUE_TOO_HIGH",     "severe",   f1_diff=f1d))
    elif f1d < -80:
        errors.append(ErrorCode("TONGUE_TOO_HIGH",     "moderate", f1_diff=f1d))

    # ── F2 / tongue front-back ────────────────────────────────────────────────
    if f2d < -500:
        errors.append(ErrorCode("TONGUE_TOO_BACK",      "severe",  f2_diff=f2d))
    elif f2d < -300:
        errors.append(ErrorCode("TONGUE_TOO_BACK",      "moderate",f2_diff=f2d))
    elif f2d < -150:
        errors.append(ErrorCode("TONGUE_TOO_BACK_MILD", "mild",    f2_diff=f2d))

    if f2d > 500:
        errors.append(ErrorCode("TONGUE_TOO_FRONT",     "severe",  f2_diff=f2d))
    elif f2d > 150:
        errors.append(ErrorCode("TONGUE_TOO_FRONT",     "moderate",f2_diff=f2d))

    return errors


# ── Frication / lisp error rules ──────────────────────────────────────────────

def check_sibilant(
    phoneme:    str,
    cog_hz:     float,
    energy_db:  float,
    tilt_db_oct:float,
    spread_hz:  float | None = None,
) -> list[ErrorCode]:
    """
    Classify sibilant production quality.
    Only applies to /s/ and /z/.
    """
    if phoneme not in {"s", "z"}:
        return []

    errors: list[ErrorCode] = []

    if cog_hz < 3800:
        # Very low CoG → interdental lisp (/s/→/θ/)
        # OR lateral lisp (distinguish by energy + spread)
        if energy_db < -40 and (spread_hz is None or spread_hz > 2500):
            errors.append(ErrorCode("LISP_LATERAL",      "severe",
                                    cog_hz=cog_hz, energy_db=energy_db))
        else:
            errors.append(ErrorCode("LISP_INTERDENTAL",  "severe", cog_hz=cog_hz))

    elif cog_hz < 4800:
        errors.append(ErrorCode("LISP_ADDENTAL", "moderate", cog_hz=cog_hz))

    elif cog_hz < 5500:
        errors.append(ErrorCode("LISP_INTERDENTAL_MILD", "mild", cog_hz=cog_hz))

    # Weak sibilant (very low energy regardless of CoG)
    if energy_db < -55 and not errors:
        errors.append(ErrorCode("LISP_LATERAL", "moderate",
                                cog_hz=cog_hz, energy_db=energy_db))

    return errors


# ── Lip / jaw visual error rules ──────────────────────────────────────────────

def check_lip(
    jaw_observed:    str,   # "open" | "mid" | "closed"
    round_observed:  str,   # "rounded" | "neutral" | "spread"
    jaw_expected:    str,
    round_expected:  str,
) -> list[ErrorCode]:
    """
    Check camera-observed lip/jaw position against expected.
    Only returns codes when observations are clearly mismatched.
    """
    errors: list[ErrorCode] = []

    if jaw_expected == "open" and jaw_observed == "closed":
        errors.append(ErrorCode("JAW_TOO_CLOSED", "moderate"))
    elif jaw_expected == "open" and jaw_observed == "mid":
        errors.append(ErrorCode("JAW_TOO_CLOSED", "mild"))
    elif jaw_expected == "closed" and jaw_observed == "open":
        errors.append(ErrorCode("JAW_TOO_OPEN", "moderate"))

    if round_expected == "rounded" and round_observed == "spread":
        errors.append(ErrorCode("LIPS_NOT_ROUNDED", "moderate"))
    elif round_expected == "spread" and round_observed == "rounded":
        errors.append(ErrorCode("LIPS_NOT_SPREAD", "moderate"))

    return errors


# ── Phoneme substitution rules ────────────────────────────────────────────────

_SUBSTITUTIONS: dict[tuple, str] = {
    ("s",  "θ"):  "SUB_S_TO_TH",
    ("s",  "ʃ"):  "SUB_S_TO_SH",
    ("θ",  "s"):  "SUB_TH_TO_S",
    ("θ",  "f"):  "SUB_TH_TO_F",
    ("ð",  "z"):  "SUB_DH_TO_Z",
    ("ð",  "d"):  "SUB_DH_TO_D",
    ("ɹ",  "w"):  "SUB_R_TO_W",
    ("l",  "w"):  "SUB_L_TO_W",
    ("k",  "t"):  "SUB_K_TO_T",
    ("ŋ",  "n"):  "SUB_NG_MISSING",
    ("n",  "ŋ"):  "SUB_NG_MISSING",
    ("tʃ", "ʃ"):  "SUB_S_TO_SH",    # close enough for template reuse
    ("dʒ", "j"):  "SUB_R_TO_W",     # closest available template
}


def check_substitution(target_ph: str, heard_ph: str) -> list[ErrorCode]:
    """Return error code for a known phoneme substitution."""
    code = _SUBSTITUTIONS.get((target_ph, heard_ph))
    if code:
        return [ErrorCode(code, "moderate")]
    return []


# ── Position errors ───────────────────────────────────────────────────────────

def check_position_error(status: str, position: str) -> list[ErrorCode]:
    """
    status   : "missing" | "extra"
    position : "initial" | "medial" | "final"
    """
    if status == "missing":
        if position == "final":
            return [ErrorCode("SOUND_MISSING_FINAL", "mild")]
        if position == "initial":
            return [ErrorCode("SOUND_MISSING_INITIAL", "mild")]
    return []


# ── Vowel nearest-neighbour comparison ───────────────────────────────────────

_VOWEL_SPACE: dict[str, tuple[float, float, str, str]] = {
    # ipa: (F1, F2, label, TEMPLATE_CODE)
    "iː": (280,  2570, "FEET",    "VOWEL_SOUNDS_LIKE_FEET"),
    "ɪ":  (360,  2100, "BIT",     "VOWEL_SOUNDS_LIKE_BIT"),
    "e":  (560,  1970, "BED",     "VOWEL_SOUNDS_LIKE_BED"),
    "æ":  (820,  1660, "CAT",     "VOWEL_SOUNDS_LIKE_CAT"),
    "uː": (290,  1460, "FOOD",    "VOWEL_SOUNDS_LIKE_FOOD"),
    "ʌ":  (680,  1310, "CUP",     "VOWEL_SOUNDS_LIKE_CUP"),
}


def nearest_vowel_code(F1: float, F2: float, exclude: str = "") -> str | None:
    """Return the TEMPLATE_CODE for the closest vowel, or None if match is close."""
    import math
    best_code, best_d = None, math.inf
    for ph, (r1, r2, _, code) in _VOWEL_SPACE.items():
        if ph == exclude:
            continue
        d = ((F1 - r1) / 400) ** 2 + ((F2 - r2) / 800) ** 2
        if d < best_d:
            best_d, best_code = d, code
    return best_code if best_d > 0.25 else None   # skip if already close enough


# ── Headline selection ────────────────────────────────────────────────────────

def headline_code(score: float, word_correct: bool) -> str:
    if not word_correct:
        return "HEADLINE_WRONG_WORD"
    if score >= 0.95:
        return "HEADLINE_PERFECT"
    if score >= 0.80:
        return "HEADLINE_VERY_GOOD"
    if score >= 0.60:
        return "HEADLINE_GOOD"
    if score >= 0.40:
        return "HEADLINE_KEEP_GOING"
    return "HEADLINE_TRY_AGAIN"


# ── Acoustic / visual cross-reference ────────────────────────────────────────

def check_acoustic_visual_mismatch(
    vowel_errors: list[ErrorCode],
    lip_errors:   list[ErrorCode],
) -> list[ErrorCode]:
    """
    If acoustics flag a vowel error but lip camera shows correct position,
    the problem is likely internal (tongue body) rather than external (jaw/lips).
    """
    acoustic_has_f1 = any(e.code in ("TONGUE_TOO_LOW", "TONGUE_TOO_HIGH") for e in vowel_errors)
    lip_has_jaw     = any(e.code in ("JAW_TOO_CLOSED", "JAW_TOO_OPEN")    for e in lip_errors)

    if acoustic_has_f1 and not lip_has_jaw:
        return [ErrorCode("ACOUSTIC_MISMATCH", "info")]
    return []
