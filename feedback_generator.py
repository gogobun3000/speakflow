"""
feedback_generator.py — SpeakFlow Feedback Coordinator
=======================================================
Thin adapter layer. Calls:
  rule_engine          → converts measurements to error codes
  clinical_templates   → converts error codes to patient-facing strings
  report_engine        → (optional) uses Claude API for multi-error synthesis

No clinical text lives here. No clinical logic lives here.
"""

from __future__ import annotations

import rule_engine    as RE
import clinical_templates as T
import report_engine  as RE2

import math


# ── Vowel sound comparison table (for nearest-neighbour matching) ─────────────
_VOWEL_SPACE: dict[str, tuple[float, float, str]] = {
    "iː": (280,  2570, "FEET"),
    "ɪ":  (360,  2100, "BIT"),
    "e":  (560,  1970, "BED"),
    "æ":  (820,  1660, "CAT"),
    "ɑː": (800,  1200, "PALM"),
    "ɒ":  (720,  1000, "HOT"),
    "ɔː": (560,  800,  "THOUGHT"),
    "ʊ":  (430,  1020, "FOOT"),
    "uː": (290,  1460, "FOOD"),
    "ʌ":  (680,  1310, "CUP"),
    "ɜː": (490,  1460, "BIRD"),
    "ə":  (500,  1350, "about"),
}

def _nearest_vowel(F1, F2, exclude=""):
    best_ph, best_word, best_d = "", "", math.inf
    for ph, (r1, r2, word) in _VOWEL_SPACE.items():
        if ph == exclude: continue
        d = ((F1-r1)/400)**2 + ((F2-r2)/800)**2
        if d < best_d:
            best_d, best_ph, best_word = d, ph, word
    return best_ph, best_word


# ── Main entry point ──────────────────────────────────────────────────────────

def generate(
    target_word:     str,
    target_phonemes: list[str],
    alignment:       list[dict],
    acoustic:        dict,
    lip:             dict,
    score:           float,
    word_correct:    bool,
    patient_age:     int = 15,
) -> dict:
    """
    Coordinate rule engine + templates + (optionally) AI synthesis.

    Returns { headline, tips, praise, clinician_notes, source }
    """

    # ── 1. Collect error codes from rule engine ───────────────────────────────
    error_codes: list[RE.ErrorCode] = []
    praise_codes: list[str] = []

    segs   = (acoustic or {}).get("segments", [])
    errors = [a for a in alignment if a["status"] in ("wrong", "missing")]
    goods  = [a for a in alignment if a["status"] == "correct"]

    # Substitution errors
    for err in errors[:2]:
        tgt, hrd = err.get("target",""), err.get("heard","")
        subs = RE.check_substitution(tgt, hrd)
        error_codes.extend(subs)

        # If no substitution match, check acoustic data
        if not subs:
            seg = next((s for s in segs if s.get("phoneme") == tgt), None)
            if seg and "vowel" in seg:
                v = seg["vowel"]
                error_codes.extend(RE.check_vowel(
                    v.get("F1_hz",0), v.get("F2_hz",0),
                    v.get("F1_ref",0), v.get("F2_ref",0),
                    v.get("confidence", 1.0),
                ))
                # Nearest-vowel comparison code
                near_code = RE.nearest_vowel_code(v.get("F1_hz",0), v.get("F2_hz",0), exclude=tgt)
                if near_code:
                    error_codes.append(RE.ErrorCode(near_code, "moderate"))

            if seg and "friction" in seg:
                f = seg["friction"]
                error_codes.extend(RE.check_sibilant(
                    tgt,
                    f.get("CoG_hz", 9999),
                    f.get("energy_db", 0),
                    f.get("tilt_db_oct", 0),
                ))

        # Position errors (missing sounds)
        if err["status"] == "missing":
            # Rough position estimate from alignment index
            pos = "final" if alignment.index(err) == len(alignment)-1 else "initial"
            error_codes.extend(RE.check_position_error("missing", pos))

    # Lip / jaw errors
    if lip and lip.get("available"):
        error_codes.extend(RE.check_lip(
            lip.get("jaw_observed",  "mid"),
            lip.get("round_observed","neutral"),
            lip.get("jaw_expected",  "mid"),
            lip.get("round_expected","neutral"),
        ))

    # Cross-reference acoustic vs visual
    vowel_errs = [e for e in error_codes if "TONGUE" in e.code]
    lip_errs   = [e for e in error_codes if "JAW" in e.code or "LIPS" in e.code]
    error_codes.extend(RE.check_acoustic_visual_mismatch(vowel_errs, lip_errs))

    # ── 2. Praise for correct phonemes ────────────────────────────────────────
    for c in goods[:2]:
        ph = c.get("target","")
        seg = next((s for s in segs if s.get("phoneme")==ph), None)
        if seg and "vowel" in seg:
            praise_codes.append("CORRECT_VOWEL")
        elif seg and "friction" in seg and seg["friction"].get("lisp_risk","") == "low":
            praise_codes.append("CORRECT_SIBILANT")
        elif ph:
            praise_codes.append("CORRECT_SOUND")

    # ── 3. Synthesise feedback ────────────────────────────────────────────────
    result = RE2.synthesise_feedback(
        codes=error_codes,
        word=target_word,
        score=score,
        word_correct=word_correct,
        patient_age=patient_age,
    )

    # Add praise
    praise = list(dict.fromkeys(T.cue(c) for c in praise_codes if T.cue(c)))[:2]
    result["praise"] = praise

    return result
