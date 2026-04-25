"""
feedback_generator.py — Plain-English patient feedback for SpeakFlow
=====================================================================
Translates raw acoustic + IPA data into short, encouraging, actionable
sentences a teenager can immediately understand and act on.

No jargon. No Hz. No IPA symbols in the patient-facing output.
Numbers live in the clinician panels — not here.
"""

from __future__ import annotations
import math

# ── Vowel space reference (F1, F2, plain example word) ───────────────────────
# Used to name what a misproduced vowel SOUNDS like.
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

# Vowel articulation tips (jaw + tongue position, plain language)
_VOWEL_TIPS: dict[str, str] = {
    "iː": "Stretch your lips wide like a big smile and keep your jaw close.",
    "ɪ":  "Like the EE in FEET but more relaxed — don't stretch as wide.",
    "e":  "Spread your lips lightly and let your jaw drop just a little.",
    "æ":  "Drop your jaw wide open — this vowel needs a big open mouth.",
    "ɑː": "Open your mouth wide and let your tongue sit low and back.",
    "ɒ":  "Open mouth, lips very lightly rounded — like you're surprised.",
    "ɔː": "Round your lips and pull your tongue back — like saying 'or'.",
    "ʊ":  "Lightly round your lips — not as tight as in FOOD.",
    "uː": "Push your lips forward in a tight circle like you're blowing a kiss.",
    "ʌ":  "Relax everything — jaw slightly open, lips neutral.",
    "ɜː": "Tongue in the middle of your mouth, lips relaxed and slightly open.",
    "ə":  "Completely relax — this sound has no effort in it.",
    "æɪ": "Start with your jaw open wide (like CAT) then glide your lips into a smile.",
    "ɑɪ": "Start at the back of your mouth (like PALM) then glide forward.",
    "æɔ": "Start open like CAT then slowly round your lips.",
    "əʊ": "Start relaxed then round your lips as you finish the sound.",
}

# ── IPA substitution messages ─────────────────────────────────────────────────
# (target_ph, heard_ph) → (what happened, what to do instead)
_SUBST: dict[tuple, tuple] = {
    ("s",  "θ"): ("Your S sounds like TH.",
                  "Keep your tongue BEHIND your teeth — don't let it poke out."),
    ("s",  "ʃ"): ("Your S sounds like SH.",
                  "Bring your tongue a little closer to the ridge behind your top teeth."),
    ("θ",  "s"): ("Your TH sounds like S.",
                  "For TH, let your tongue tip sit gently between your teeth."),
    ("θ",  "f"): ("Your TH sounds like F.",
                  "Put your tongue between your teeth — not your teeth on your lip."),
    ("ð",  "z"): ("Your voiced TH sounds like Z.",
                  "Touch your tongue tip between your teeth and add your voice."),
    ("ð",  "s"): ("Your voiced TH sounds like S.",
                  "Put your tongue between your teeth AND add your voice."),
    ("ð",  "d"): ("Your voiced TH sounds like D.",
                  "Move your tongue forward — tip gently between your teeth."),
    ("ɹ",  "w"): ("Your R sounds like W.",
                  "Raise the back of your tongue without rounding your lips."),
    ("w",  "ɹ"): ("Your W sounds like R.",
                  "Round your lips into a tight circle before the vowel."),
    ("l",  "w"): ("Your L sounds like W.",
                  "Put your tongue tip on the ridge behind your top teeth."),
    ("n",  "ŋ"): ("Your N sounds like the NG in SING.",
                  "Touch your tongue tip to the ridge for N — not the back."),
    ("ŋ",  "n"): ("You dropped the -NG ending.",
                  "Let the back of your tongue touch the roof at the back for -ING."),
    ("tʃ", "ʃ"): ("Your CH sounds like SH.",
                  "Start with your tongue on the ridge like a T, then release."),
    ("dʒ", "j"): ("Your J sounds like Y.",
                  "Start with a D sound then release into a ZH."),
    ("v",  "b"): ("Your V sounds like B.",
                  "Rest your top teeth on your bottom lip and blow air through."),
    ("f",  "p"): ("Your F sounds like P.",
                  "Let your top teeth touch your bottom lip — don't close your lips."),
    ("p",  "b"): ("Your P sounds voiced — a little like B.",
                  "Release the air with a sharp pop and no voice."),
    ("t",  "d"): ("Your T sounds a bit like D.",
                  "Keep it crisp and voiceless — no buzz in the throat."),
    ("k",  "t"): ("Your K sounds like T.",
                  "Move the contact point to the back — back of tongue on the soft palate."),
}

# ── Lisp plain-language feedback ──────────────────────────────────────────────
_LISP: dict[str, dict] = {
    "high": {
        "what":   "Your S is coming from the wrong spot — it sounds a bit like TH.",
        "how":    "Keep your tongue BEHIND your teeth. If it pokes out even slightly, the S becomes TH.",
        "analogy": "Picture the air as a thin stream going straight out the front of your mouth, not around your tongue.",
    },
    "moderate": {
        "what":   "Your S sounds a little off — it might be going sideways.",
        "how":    "Direct the air through the very CENTRE of your mouth.",
        "analogy": "Imagine a tiny laser beam of air going perfectly straight forward.",
    },
}

# ── Helper functions ──────────────────────────────────────────────────────────

def _nearest_vowel(F1: float, F2: float, exclude: str = "") -> tuple[str, str]:
    """Return the IPA symbol and example word of the closest vowel to F1/F2."""
    best_ph, best_word, best_d = "", "", math.inf
    for ph, (r1, r2, word) in _VOWEL_SPACE.items():
        if ph == exclude:
            continue
        d = ((F1 - r1) / 400) ** 2 + ((F2 - r2) / 800) ** 2
        if d < best_d:
            best_d, best_ph, best_word = d, ph, word
    return best_ph, best_word


def _vowel_direction(diff: float, axis: str) -> str:
    """Human direction for a formant deviation."""
    if axis == "F1":
        if diff < -80:  return "too close"     # jaw too closed
        if diff >  80:  return "too open"
    if axis == "F2":
        if diff < -120: return "too back"
        if diff >  120: return "too front"
    return ""


def _find_seg(segments: list[dict], phoneme: str) -> dict | None:
    for s in segments:
        if s.get("phoneme") == phoneme:
            return s
    return None


# ── Main entry point ─────────────────────────────────────────────────────────

def generate(
    target_word:    str,
    target_phonemes: list[str],
    alignment:      list[dict],
    acoustic:       dict,
    lip:            dict,
    score:          float,
    word_correct:   bool,
) -> dict:
    """
    Return plain-English feedback for the patient.

    Returns
    -------
    {
      headline : str          — one sentence summary
      tips     : [str]        — up to 3 actionable sentences
      praise   : [str]        — what they did right (encouragement)
    }
    """
    tips:   list[str] = []
    praise: list[str] = []

    errors   = [a for a in alignment if a["status"] in ("wrong", "missing")]
    corrects = [a for a in alignment if a["status"] == "correct"]
    segs     = (acoustic or {}).get("segments", [])

    # ── Praise for correct phonemes ───────────────────────────────────────────
    if corrects and word_correct:
        names = []
        for c in corrects:
            ph = c.get("target", "")
            # Only name consonants/vowels the patient can understand
            name = _ph_plain_name(ph)
            if name and name not in names:
                names.append(name)
        if names:
            praise.append(f"Your {_join(names)} sound{'s' if len(names)>1 else ''} {'were' if len(names)>1 else 'was'} good.")

    # ── Tips for each error ───────────────────────────────────────────────────
    for err in errors[:2]:   # cap at 2 so we don't overwhelm
        tgt = err.get("target", "")
        hrd = err.get("heard",  "")

        # 1. Named substitution?
        subst_key = (tgt, hrd)
        if subst_key in _SUBST:
            what, how = _SUBST[subst_key]
            tips.append(what)
            tips.append(how)
            continue

        # 2. Vowel acoustic analysis
        seg = _find_seg(segs, tgt)
        if seg and "vowel" in seg:
            v = seg["vowel"]
            F1_d = v.get("F1_diff", 0)
            F2_d = v.get("F2_diff", 0)
            F1   = v.get("F1_hz", 0)
            F2   = v.get("F2_hz", 0)
            target_label = v.get("label", tgt)

            # What does it sound like?
            near_ph, near_word = _nearest_vowel(F1, F2, exclude=tgt)
            if near_ph and near_ph != tgt:
                tips.append(
                    f"Your {target_label} vowel sounds more like the vowel in {near_word} — "
                    f"{_vowel_direction(F1_d,'F1') or _vowel_direction(F2_d,'F2') or 'try adjusting your tongue position'}."
                )

            # Specific jaw/tongue tip
            if _vowel_direction(F1_d, "F1") == "too close":
                tips.append("Open your jaw a little wider for this vowel.")
            elif _vowel_direction(F1_d, "F1") == "too open":
                tips.append("Bring your jaw up slightly — not as wide open.")
            if _vowel_direction(F2_d, "F2") == "too back":
                tips.append("Move your tongue a bit further forward in your mouth.")
            elif _vowel_direction(F2_d, "F2") == "too front":
                tips.append("Pull your tongue back slightly.")

            # Fallback tip from table
            if not tips and tgt in _VOWEL_TIPS:
                tips.append(_VOWEL_TIPS[tgt])
            continue

        # 3. Lisp / frication
        if seg and "friction" in seg:
            f = seg["friction"]
            risk = f.get("lisp_risk", "low")
            if risk in _LISP:
                d = _LISP[risk]
                tips.append(d["what"])
                tips.append(d["how"])
                tips.append(d["analogy"])
            continue

        # 4. Generic missing phoneme
        if err["status"] == "missing":
            name = _ph_plain_name(tgt)
            if name:
                tips.append(f"The {name} sound at the end was a bit quiet — make sure it's clear.")
            continue

        # 5. Fallback from vowel tip table
        if tgt in _VOWEL_TIPS:
            tips.append(_VOWEL_TIPS[tgt])

    # ── Lip additional tip ────────────────────────────────────────────────────
    if lip and lip.get("available"):
        for m in (lip.get("mismatches") or [])[:1]:
            # Strip clinical language to keep it simple
            clean = m.replace("(aperture", "(").replace("need >0.", "need more than ").split("—")[-1].strip()
            if clean and clean not in tips:
                tips.append(clean.capitalize() if clean else "")

    # ── Cross-reference insight ───────────────────────────────────────────────
    xref = (lip or {}).get("cross_reference", {})
    for note in (xref.get("notes") or [])[:1]:
        if "MISMATCH" in note:
            # Tongue-position insight — plain version
            tips.append("Your mouth position looks right but the sound is still off — this might be about tongue position inside your mouth.")

    # ── Headline ─────────────────────────────────────────────────────────────
    word = target_word.upper()
    if not word_correct:
        headline = f"I couldn't quite catch {word} — have another go!"
    elif score >= 0.95:
        headline = f"Spot on — {word} sounded great! 🎉"
    elif score >= 0.80:
        headline = f"Really close! Just one small thing to fix on {word}."
    elif score >= 0.60:
        headline = f"Good try on {word}! A couple of sounds to work on."
    elif score >= 0.40:
        headline = f"Keep going with {word} — you're getting there."
    else:
        headline = f"This one takes practice — don't give up on {word}!"

    return {
        "headline": headline,
        "tips":     tips[:4],    # cap at 4 tips
        "praise":   praise[:2],
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

_PH_NAMES: dict[str, str] = {
    "p":"P","b":"B","t":"T","d":"D","k":"K","ɡ":"G","m":"M","n":"N","ŋ":"NG",
    "f":"F","v":"V","s":"S","z":"Z","ʃ":"SH","ʒ":"ZH","h":"H","θ":"TH","ð":"TH (voiced)",
    "tʃ":"CH","dʒ":"J","l":"L","ɹ":"R","w":"W","j":"Y",
    "iː":"EE","ɪ":"short I","e":"short E","æ":"A (as in CAT)","ɑː":"AH","ɒ":"short O",
    "ɔː":"AW","ʊ":"short OO","uː":"long OO","ʌ":"short U","ɜː":"ER","ə":"schwa",
    "æɪ":"AY (Australian)","ɑɪ":"long I","æɔ":"OW","əʊ":"long O",
}


def _ph_plain_name(ph: str) -> str:
    return _PH_NAMES.get(ph, "")


def _join(items: list[str]) -> str:
    if not items:   return ""
    if len(items) == 1: return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]
