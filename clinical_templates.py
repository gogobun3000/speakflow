"""
clinical_templates.py — SpeakFlow Clinical Content Library
===========================================================
ALL patient-facing strings live here and ONLY here.

This file is the clinical validation document.
A practising speech pathologist should review, edit, and sign off
every string before the app goes live with patients.

Structure per error code
-------------------------
  cue        str   — What to tell the patient (1–2 sentences, plain English)
  exercise   str   — The specific technique to use
  analogy    str   — A visual or sensory analogy to make it concrete
  clinician  str   — Technical note for the clinician report (may use jargon)

Guidelines for editors
-----------------------
  - Write for a 12–17 year old. Short sentences. Active voice.
  - Never use IPA symbols, Hz values, or technical terms in cue/exercise/analogy.
  - The clinician field may use standard SLP terminology.
  - One action per instruction. Do not combine two techniques in one string.
  - Validated by: [CLINICIAN NAME, CREDENTIALS, DATE]
"""

# ── Error code registry ───────────────────────────────────────────────────────
# Each key is an error code returned by rule_engine.py.
# Codes are grouped by category.

TEMPLATES: dict[str, dict] = {

    # ══════════════════════════════════════════════════════════════════════════
    # VOWEL ERRORS — tongue height (F1 axis)
    # ══════════════════════════════════════════════════════════════════════════

    "TONGUE_TOO_LOW": {
        "cue":       "Your tongue is sitting too low in your mouth for this sound.",
        "exercise":  "Lift the body of your tongue toward the roof of your mouth — like you're about to say EE.",
        "analogy":   "Think of your tongue as an elevator — this sound needs it on a higher floor.",
        "clinician": "F1 elevated above target. Tongue body depressed. Suggest minimal pair drilling with high vowel contrast.",
    },
    "TONGUE_TOO_LOW_MILD": {
        "cue":       "Your tongue is slightly low — just a small adjustment needed.",
        "exercise":  "Raise your tongue a little — you're almost there.",
        "analogy":   "Just a half-step up.",
        "clinician": "F1 mildly elevated. Tongue height slightly below target. May self-correct with practice.",
    },
    "TONGUE_TOO_HIGH": {
        "cue":       "Your tongue is pressing up too high for this sound — it needs more room.",
        "exercise":  "Let your tongue relax downward slightly and open your jaw a little more.",
        "analogy":   "Give your tongue some breathing room — it's sitting too close to the roof.",
        "clinician": "F1 below target. Tongue body elevated. Check for hypertension. Contrast with low vowel.",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # VOWEL ERRORS — tongue front/back (F2 axis)
    # ══════════════════════════════════════════════════════════════════════════

    "TONGUE_TOO_BACK": {
        "cue":       "Your tongue is pulled too far back in your mouth.",
        "exercise":  "Push the front of your tongue forward — the tip should almost touch the back of your lower front teeth.",
        "analogy":   "Imagine pushing your tongue toward the front of your mouth as if to touch your teeth — but just short of touching.",
        "clinician": "F2 below target. Tongue retracted. Check for compensatory backing. Contrast with front vowel.",
    },
    "TONGUE_TOO_BACK_MILD": {
        "cue":       "Your tongue is just a little too far back — nudge it forward.",
        "exercise":  "Shift your tongue slightly toward the front of your mouth.",
        "analogy":   "A small step forward is all it needs.",
        "clinician": "F2 mildly below target. Slight retraction. Minimal pair practice recommended.",
    },
    "TONGUE_TOO_FRONT": {
        "cue":       "Your tongue is too far forward for this sound.",
        "exercise":  "Pull your tongue back slightly — away from your teeth and toward the middle of your mouth.",
        "analogy":   "Let your tongue settle back — like sinking into a comfortable chair.",
        "clinician": "F2 above target. Tongue fronted. Common with front vowel overgeneralisation.",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # JAW / LIP ERRORS (confirmed by camera)
    # ══════════════════════════════════════════════════════════════════════════

    "JAW_TOO_CLOSED": {
        "cue":       "Your mouth isn't open wide enough for this sound.",
        "exercise":  "Drop your jaw — let it fall open. This sound needs more space inside your mouth.",
        "analogy":   "Open your mouth like you're at the dentist — wider than feels natural.",
        "clinician": "Jaw aperture below target (MediaPipe). Corroborated by elevated F1. Demonstrate exaggerated jaw drop.",
    },
    "JAW_TOO_OPEN": {
        "cue":       "Your mouth is a little too wide open for this sound.",
        "exercise":  "Close your jaw slightly — bring your teeth a little closer together.",
        "analogy":   "Think of a small, relaxed opening rather than a big yawn.",
        "clinician": "Jaw aperture above target (MediaPipe). Corroborated by reduced F1.",
    },
    "LIPS_NOT_ROUNDED": {
        "cue":       "Your lips need to be more rounded for this sound.",
        "exercise":  "Pucker your lips forward into a circle — like you're about to whistle or blow a kiss.",
        "analogy":   "Make a small tight O shape with your lips before you start the sound.",
        "clinician": "Lip rounding absent (lip_ratio below threshold). Target phoneme requires labial rounding.",
    },
    "LIPS_NOT_SPREAD": {
        "cue":       "Your lips need to be spread out wider for this sound.",
        "exercise":  "Stretch the corners of your mouth outward — like a big wide smile.",
        "analogy":   "Pull your lips wide as if you're showing all your teeth in a grin.",
        "clinician": "Lip spreading absent (high lip_ratio). Target phoneme requires retracted lip corners.",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # LISP ERRORS — sibilant /s/ and /z/
    # ══════════════════════════════════════════════════════════════════════════

    "LISP_INTERDENTAL": {
        "cue":       "Your S is coming out between your teeth — it sounds like the TH in THINK.",
        "exercise":  "Keep your tongue TIP behind your upper front teeth. The tip should touch the bumpy ridge just behind the teeth — not stick through the gap.",
        "analogy":   "Your tongue tip is like a dog that must stay behind a fence. The fence is your top teeth. Keep it there.",
        "clinician": "Spectral CoG below 3800 Hz. Pattern consistent with interdental sibilant substitution (/s/→/θ/). Recommend tongue-tip placement on alveolar ridge. Introduce /ts/ clusters to stabilise placement.",
    },
    "LISP_INTERDENTAL_MILD": {
        "cue":       "Your S is slightly forward — the tip of your tongue may be touching your teeth.",
        "exercise":  "Move your tongue tip back just a millimetre — away from the teeth, onto the ridge behind them.",
        "analogy":   "A tiny adjustment — like stepping one step back from the fence.",
        "clinician": "CoG 3800–4500 Hz. Mild addental or borderline interdental pattern. Monitor for consistency.",
    },
    "LISP_LATERAL": {
        "cue":       "Your S has a wet or splashy sound — air is going around the sides of your tongue instead of through the middle.",
        "exercise":  "Focus the air through the very centre of your mouth in a thin, straight stream. Sides of the tongue should be raised and touching the upper back teeth.",
        "analogy":   "Imagine shooting a laser beam of air perfectly straight forward through a tiny gap in the middle — nothing goes sideways.",
        "clinician": "CoG below target with reduced energy (energy_db < −40) and elevated spectral spread. Lateral airflow pattern. Recommend sustained /s/ production with straw visualisation. Tactile feedback may help.",
    },
    "LISP_ADDENTAL": {
        "cue":       "Your S sounds a bit dull — your tongue might be touching the upper teeth instead of hovering just behind them.",
        "exercise":  "Lift your tongue tip just slightly away from the teeth — create a tiny gap. The air should hiss sharply through that gap.",
        "analogy":   "A window cracked open just a centimetre lets the air hiss through. That gap between tongue tip and ridge is your window.",
        "clinician": "CoG 4500–5500 Hz. Dentalized sibilant — tongue contact on upper incisors rather than alveolar ridge. Increase blade-to-ridge distance.",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CONSONANT SUBSTITUTIONS
    # ══════════════════════════════════════════════════════════════════════════

    "SUB_S_TO_TH": {
        "cue":       "Your S sounds like TH — the tongue is coming forward.",
        "exercise":  "Keep your tongue behind your teeth for S. Pull it back just a little.",
        "analogy":   "For S, the tongue stays inside. For TH, it sneaks out. Keep it inside.",
        "clinician": "Phonemic substitution /s/→/θ/. Likely interdental tongue placement. Use mirror work.",
    },
    "SUB_S_TO_SH": {
        "cue":       "Your S sounds like SH.",
        "exercise":  "Bring your tongue a little further forward — closer to the ridge behind your top teeth.",
        "analogy":   "SH is made further back. Slide your tongue forward just a little.",
        "clinician": "Phonemic substitution /s/→/ʃ/. Tongue dorsum retracted. Increase anterior placement.",
    },
    "SUB_TH_TO_S": {
        "cue":       "Your TH sounds like S — your tongue needs to come forward between your teeth.",
        "exercise":  "Let your tongue tip rest gently between your upper and lower front teeth — just the tip, lightly.",
        "analogy":   "For TH, your tongue peeks out between the teeth like a shy visitor.",
        "clinician": "Phonemic substitution /θ/→/s/. Tongue tip not achieving interdental placement. Use mirror or tactile cue.",
    },
    "SUB_TH_TO_F": {
        "cue":       "Your TH sounds like F — you're using your teeth on your lip instead of your tongue between your teeth.",
        "exercise":  "Put your tongue between your teeth, not your teeth on your lip.",
        "analogy":   "F uses your bottom lip. TH uses your tongue. Completely different.",
        "clinician": "Phonemic substitution /θ/→/f/. Labiodental rather than interdental placement.",
    },
    "SUB_DH_TO_Z": {
        "cue":       "Your voiced TH sounds like Z.",
        "exercise":  "Put your tongue tip between your teeth and add your voice — feel the buzz.",
        "analogy":   "Touch your throat while making the sound. You should feel a buzz.",
        "clinician": "Phonemic substitution /ð/→/z/. Tongue not achieving interdental position. Add voicing cue.",
    },
    "SUB_DH_TO_D": {
        "cue":       "Your voiced TH sounds like D — the tongue needs to come forward.",
        "exercise":  "Move your tongue tip forward so it sits gently between your teeth.",
        "analogy":   "D is made with the tongue on the ridge. TH is made with it between the teeth — a step forward.",
        "clinician": "Phonemic substitution /ð/→/d/. Tongue tip at alveolar ridge rather than interdental.",
    },
    "SUB_R_TO_W": {
        "cue":       "Your R sounds like W — your lips are doing too much work.",
        "exercise":  "Raise the back of your tongue toward the roof of your mouth without rounding your lips.",
        "analogy":   "W needs round lips. R doesn't. Keep your lips relaxed and flat.",
        "clinician": "Phonemic substitution /ɹ/→/w/. Labial rounding substituting for tongue body elevation. Reduce lip rounding cue.",
    },
    "SUB_L_TO_W": {
        "cue":       "Your L sounds like W.",
        "exercise":  "Put your tongue tip on the bumpy ridge just behind your upper front teeth.",
        "analogy":   "Your tongue tip needs to tap that ridge for L — like a brief touch.",
        "clinician": "Phonemic substitution /l/→/w/. Tongue tip not achieving alveolar contact.",
    },
    "SUB_K_TO_T": {
        "cue":       "Your K sounds like T — the back of your tongue needs to do the work.",
        "exercise":  "Feel the back of your tongue touch the soft part at the very back of the roof of your mouth.",
        "analogy":   "T is made at the front with your tongue tip. K is made at the back — much further back.",
        "clinician": "Phonemic substitution /k/→/t/. Tongue body not achieving velar contact.",
    },
    "SUB_NG_MISSING": {
        "cue":       "You dropped the NG sound at the end.",
        "exercise":  "At the end of the word, let the back of your tongue touch the soft roof at the back and hum through your nose.",
        "analogy":   "NG is a hum in your nose. Feel the back of your tongue reach up for it.",
        "clinician": "Final /ŋ/ deleted. Likely velar nasal weakening in word-final position.",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # GENERAL / STRUCTURAL ERRORS
    # ══════════════════════════════════════════════════════════════════════════

    "SOUND_MISSING_FINAL": {
        "cue":       "The sound at the end of the word was cut off — finish it clearly.",
        "exercise":  "Hold the final sound for an extra half-second. Don't let the word trail off.",
        "analogy":   "Think of the last sound as putting a full stop at the end of a sentence.",
        "clinician": "Final consonant deletion. Coda position weakening. Extend duration of final phoneme.",
    },
    "SOUND_MISSING_INITIAL": {
        "cue":       "The sound at the start of the word was missing.",
        "exercise":  "Start the word cleanly — prepare your mouth before you begin.",
        "analogy":   "Take a breath, get your tongue in position, then start.",
        "clinician": "Initial consonant deletion or reduction. Pre-positioning may help.",
    },
    "WORD_WRONG": {
        "cue":       "I couldn't quite catch that word — try again.",
        "exercise":  "Say it slowly and clearly. One syllable at a time if needed.",
        "analogy":   None,
        "clinician": "Target word not recognised by ASR. May be intelligibility issue or background noise.",
    },
    "ACOUSTIC_MISMATCH": {
        "cue":       "Your mouth position looks right but the sound is still off.",
        "exercise":  "The issue may be inside your mouth — tongue position, not just jaw or lips. Try focusing on where your tongue tip is.",
        "analogy":   "The outside looks right but the engine inside needs adjusting.",
        "clinician": "Acoustic/visual discrepancy — lip geometry matches target but F1/F2 deviate. Suggests tongue body position error rather than jaw or lip.",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # VOWEL SOUND COMPARISONS (nearest-neighbour mismatch)
    # ══════════════════════════════════════════════════════════════════════════

    "VOWEL_SOUNDS_LIKE_BIT":     {"cue": "Your vowel sounds more like the short I in BIT.",     "exercise": "Raise your tongue a little more and spread your lips wider.", "analogy": None, "clinician": "F1/F2 close to /ɪ/ region."},
    "VOWEL_SOUNDS_LIKE_BED":     {"cue": "Your vowel sounds more like the E in BED.",            "exercise": "Drop your jaw slightly and spread your lips.", "analogy": None, "clinician": "F1/F2 close to /e/ region."},
    "VOWEL_SOUNDS_LIKE_CAT":     {"cue": "Your vowel sounds more like the A in CAT.",            "exercise": "Raise your tongue and close your jaw a little.", "analogy": None, "clinician": "F1/F2 close to /æ/ region."},
    "VOWEL_SOUNDS_LIKE_FEET":    {"cue": "Your vowel sounds more like the EE in FEET.",          "exercise": "Lower your tongue and relax your jaw open.", "analogy": None, "clinician": "F1/F2 close to /iː/ region."},
    "VOWEL_SOUNDS_LIKE_FOOD":    {"cue": "Your vowel sounds more like the OO in FOOD.",          "exercise": "Spread your lips and move your tongue forward.", "analogy": None, "clinician": "F1/F2 close to /uː/ region."},
    "VOWEL_SOUNDS_LIKE_CUP":     {"cue": "Your vowel sounds more like the U in CUP.",            "exercise": "Adjust your tongue — it needs to move to the right position.", "analogy": None, "clinician": "F1/F2 close to /ʌ/ region."},

    # ══════════════════════════════════════════════════════════════════════════
    # POSITIVE CONFIRMATIONS (used for praise)
    # ══════════════════════════════════════════════════════════════════════════

    "CORRECT_SOUND":    {"cue": "That sound was spot on.",           "exercise": None, "analogy": None, "clinician": "Target phoneme produced within acceptable acoustic range."},
    "CORRECT_VOWEL":    {"cue": "Your vowel sounded great.",         "exercise": None, "analogy": None, "clinician": "F1/F2 within target range. Confidence acceptable."},
    "CORRECT_SIBILANT": {"cue": "Your S sounded clear and sharp.",   "exercise": None, "analogy": None, "clinician": "CoG within normal range for /s/. No lisp indicators."},
    "CORRECT_WORD":     {"cue": "That word was said perfectly.",     "exercise": None, "analogy": None, "clinician": "All target phonemes produced correctly."},

    # ══════════════════════════════════════════════════════════════════════════
    # HEADLINES — overall session result (score-based)
    # ══════════════════════════════════════════════════════════════════════════

    "HEADLINE_PERFECT":    {"cue": "Spot on — that sounded great!",                   "exercise": None, "analogy": None, "clinician": "Score ≥ 0.95. All phonemes within acceptable range."},
    "HEADLINE_VERY_GOOD":  {"cue": "Really close — just one small thing to tighten.", "exercise": None, "analogy": None, "clinician": "Score 0.80–0.95."},
    "HEADLINE_GOOD":       {"cue": "Good try! A couple of sounds to work on.",        "exercise": None, "analogy": None, "clinician": "Score 0.60–0.80."},
    "HEADLINE_KEEP_GOING": {"cue": "Keep going — you're getting there.",              "exercise": None, "analogy": None, "clinician": "Score 0.40–0.60."},
    "HEADLINE_TRY_AGAIN":  {"cue": "This one takes practice — don't give up!",       "exercise": None, "analogy": None, "clinician": "Score < 0.40."},
    "HEADLINE_WRONG_WORD": {"cue": "I couldn't quite catch that — have another go!",  "exercise": None, "analogy": None, "clinician": "ASR word recognition failed."},
}


# ── Accessor helpers ──────────────────────────────────────────────────────────

def get(code: str) -> dict:
    """Return the template for an error code, or a safe fallback."""
    return TEMPLATES.get(code, {
        "cue":       "Try that again — focus on one sound at a time.",
        "exercise":  None,
        "analogy":   None,
        "clinician": f"Unknown code: {code}",
    })


def cue(code: str) -> str:
    """Return just the patient-facing cue string."""
    return get(code).get("cue", "")


def exercise(code: str) -> str | None:
    """Return the exercise instruction, or None if not applicable."""
    return get(code).get("exercise")


def analogy(code: str) -> str | None:
    """Return the analogy, or None if not applicable."""
    return get(code).get("analogy")


def clinician_note(code: str) -> str:
    """Return the technical clinician note."""
    return get(code).get("clinician", "")


def all_codes() -> list[str]:
    return list(TEMPLATES.keys())
