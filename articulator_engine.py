"""
articulator_engine.py — SpeakFlow Clinical Decision Engine
============================================================
Layer 2 + 3 of the clinical architecture:

  Acoustic measurements  →  Articulator identification  →  Intervention protocol  →  Stage

This file contains the clinical knowledge base.
It is NOT machine learning. It is structured clinical logic
derived from published speech pathology intervention literature.

Every rule and protocol references its source.
A speech pathologist should review and sign off on the thresholds and protocols.

Sources
-------
  Van Riper C (1978) Speech Correction: Principles and Methods. 6th ed. Prentice-Hall.
  Bernthal JE, Bankson NW, Flipsen P (2017) Articulation and Phonological Disorders. 8th ed. Pearson.
  McLeod S & Baker E (2017) Children's Speech: An Evidence-Based Approach. Pearson.
  Levelt WJM (1989) Speaking: From Intention to Articulation. MIT Press.
  Gibbon F (1999) Undifferentiated lingual gestures. J Speech Lang Hear Res.

Validated by: [SPEECH PATHOLOGIST NAME, CREDENTIALS, DATE]
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
import sqlite3
import os
from datetime import datetime, timedelta


# ── Output data class ─────────────────────────────────────────────────────────

@dataclass
class ArticulatorFinding:
    """What is wrong, which body part, how bad, what to do."""
    articulator:         str           # jaw | tongue_body | tongue_tip | tongue_sides | lips | velum | airflow
    problem:             str           # specific error descriptor
    severity:            int           # 0–100  (0=correct, 100=maximum error)
    disorder_name:       str           # plain-English disorder label
    description:         str           # one sentence describing what is happening
    intervention_stage:  int           # 1–6 based on Van Riper hierarchy
    stage_name:          str           # "Isolation" | "Syllable" | "Word" | etc.
    instruction:         str           # what to tell the patient at this stage
    cue_type:            str           # placement | auditory | tactile | visual | combined
    facilitating_words:  list[str]     # words that make this sound easier
    do_not:              list[str]     # what NOT to do in this session
    clinician_note:      str           # SLP technical note
    references:          list[str]     # published sources
    ai_prompt_context:   dict          # structured input for AI feedback layer


# ── Stage advancement query ───────────────────────────────────────────────────

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speakflow.db")

def _get_phoneme_stage(phoneme: str, disorder_key: str) -> int:
    """
    Query phoneme_contexts to determine which stage of the
    intervention hierarchy the patient is currently at.

    Advancement criteria (Van Riper criterion-referenced):
      Advance when ≥ 80% accuracy on 5+ attempts at current context level.
      Context levels map to hierarchy stages:
        isolation → stage 1–2
        word      → stage 3–5  (initial/medial/final by position)
        sentence  → stage 5–6

    Returns 1 if no history.
    """
    if not os.path.exists(DB_PATH):
        return 1

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        since = (datetime.utcnow() - timedelta(days=60)).strftime("%Y-%m-%dT%H:%M:%SZ")

        rows = conn.execute("""
            SELECT context_type, position, success, attempted_at
            FROM phoneme_contexts
            WHERE phoneme = ? AND attempted_at >= ?
            ORDER BY attempted_at DESC
            LIMIT 50
        """, (phoneme, since)).fetchall()
        conn.close()
    except Exception:
        return 1

    if not rows:
        return 1

    records = [dict(r) for r in rows]

    def accuracy(subset):
        if not subset:
            return None
        return sum(r["success"] for r in subset) / len(subset)

    # Group by context
    isolation = [r for r in records if r["context_type"] == "isolation"]
    words      = [r for r in records if r["context_type"] == "word"]
    sentences  = [r for r in records if r["context_type"] == "sentence"]

    word_init  = [r for r in words if r.get("position") == "initial"]
    word_med   = [r for r in words if r.get("position") == "medial"]
    word_fin   = [r for r in words if r.get("position") == "final"]

    THRESHOLD = 0.80
    MIN_ATTEMPTS = 5

    # Stage 1: isolation — not achieved or below threshold
    iso_acc = accuracy(isolation[:10])
    if iso_acc is None or (len(isolation) < MIN_ATTEMPTS and iso_acc < THRESHOLD):
        return 1

    # Stage 2: syllables — we don't track explicitly, advance after isolation mastered
    if iso_acc is not None and iso_acc >= THRESHOLD and len(isolation) >= MIN_ATTEMPTS:
        if not words:
            return 2   # ready to move to syllables/words
    else:
        return 1

    # Stage 3: word initial
    wi_acc = accuracy(word_init[:10])
    if wi_acc is None or len(word_init) < MIN_ATTEMPTS:
        return 3
    if wi_acc < THRESHOLD:
        return 3

    # Stage 4: word medial
    wm_acc = accuracy(word_med[:10])
    if wm_acc is None or len(word_med) < MIN_ATTEMPTS:
        return 4
    if wm_acc < THRESHOLD:
        return 4

    # Stage 5: word final + sentence start
    wf_acc = accuracy(word_fin[:10])
    if wf_acc is None or len(word_fin) < MIN_ATTEMPTS:
        return 5
    if wf_acc < THRESHOLD:
        return 5

    # Stage 6: sentence generalisation
    sent_acc = accuracy(sentences[:10])
    if sent_acc is None or len(sentences) < MIN_ATTEMPTS:
        return 6
    if sent_acc < THRESHOLD:
        return 6

    return 6   # maintenance — all stages mastered


# ── Intervention protocol library ─────────────────────────────────────────────
# Each key: "articulator__problem"
# Structure follows Van Riper (1978) traditional articulation therapy hierarchy.

PROTOCOLS: dict[str, dict] = {

    # ── Tongue tip — interdental lisp ─────────────────────────────────────────
    "tongue_tip__interdental_protrusion": {
        "disorder_name": "Interdental Lisp",
        "articulator":   "Tongue tip",
        "description":   "Tongue tip protruding between upper and lower teeth during /s/ or /z/, producing /θ/ substitution.",
        "do_not": [
            "Do not use the word 'lisp' with the patient without clinical consent.",
            "Do not model the error sound.",
            "Do not accept approximations at stages 1–3.",
            "Do not move to the next stage until criterion is met.",
        ],
        "facilitating_contexts": [
            "/s/ is more easily achieved following /t/ — use 'cats', 'its', 'bats' to establish alveolar placement via coarticulation.",
            "High front vowel /iː/ facilitates correct /s/ tongue position — start with 'see', 'sit', 'seal'.",
            "Whispering /s/ can help establish the correct airstream direction without voicing interference.",
        ],
        "references": [
            "Van Riper C (1978) Speech Correction. 6th ed. Prentice-Hall.",
            "Bernthal JE et al (2017) Articulation and Phonological Disorders. 8th ed. Pearson.",
        ],
        "hierarchy": [
            {
                "stage": 1, "name": "Isolation",
                "goal": "Produce a correct /s/ in isolation with tongue tip behind upper teeth.",
                "instruction": "Close your teeth gently together. Place your tongue tip on the bumpy ridge just behind your upper front teeth — not touching the teeth, and not poking through. Now breathe out steadily. Listen for the hiss. If the sound comes out as TH, your tongue has come forward — bring it back.",
                "cue_type": "placement",
                "facilitating_words": [],
                "clinician_note": "Use a tongue depressor to demonstrate the alveolar ridge position if the patient cannot find it independently. Mirror work essential at this stage. Criterion: 9/10 correct isolated /s/ productions.",
            },
            {
                "stage": 2, "name": "Syllable",
                "goal": "Produce /s/ correctly in CV and VC syllables.",
                "instruction": "Now add a vowel. Say SA — keep your tongue behind your teeth for the S. Then try SI, SO, SEE. The tongue must stay in position even as you move to the vowel.",
                "cue_type": "movement",
                "facilitating_words": ["see", "sea", "sigh", "say"],
                "clinician_note": "High front vowel /iː/ facilitates correct placement — start with 'see'. Criterion: 9/10 correct in 20 syllable productions across two sessions.",
            },
            {
                "stage": 3, "name": "Word — Initial position",
                "goal": "Produce /s/ correctly at the start of words.",
                "instruction": "Say the word slowly. Focus entirely on the S at the very start — get your tongue in position before you begin speaking.",
                "cue_type": "word",
                "facilitating_words": ["see", "sun", "sit", "sea", "six", "soft", "safe", "sing"],
                "clinician_note": "Begin with high-frequency, short words. Delayed auditory feedback can help if patient reverts to error in word context. Criterion: 80% across 20 target words.",
            },
            {
                "stage": 4, "name": "Word — Medial and Final position",
                "goal": "Produce /s/ correctly in all word positions.",
                "instruction": "Now try words where S is in the middle or at the end. These are harder because you can't prepare as easily.",
                "cue_type": "word",
                "facilitating_words": ["bus", "miss", "cats", "boats", "beside", "classic", "basket"],
                "clinician_note": "Final position is often harder — patient may drop the /s/ entirely. Extend duration of final /s/ as a compensatory strategy. Criterion: 80% across all positions.",
            },
            {
                "stage": 5, "name": "Phrase and Sentence",
                "goal": "Maintain correct /s/ in simple phrases and sentences.",
                "instruction": "Now use S words in a sentence. Say it at a normal speed — not too slow, not too fast. The goal is for the S to sound correct without you having to slow down.",
                "cue_type": "sentence",
                "facilitating_words": ["Sally sits by the sea", "Six soft sounds", "The sun is warm today"],
                "clinician_note": "Rate of speech affects placement accuracy. Slow rate initially, then gradually increase. Self-monitoring training begins here — patient should identify their own errors. Criterion: 80% in 10 sentence productions.",
            },
            {
                "stage": 6, "name": "Conversation and Generalisation",
                "goal": "Use correct /s/ spontaneously in connected speech.",
                "instruction": "Tell me about your day — or talk about anything you like. I'll be listening for your S sounds. You should be monitoring them too.",
                "cue_type": "conversation",
                "facilitating_words": [],
                "clinician_note": "Carryover is the hardest stage. Enlist family/friends to provide gentle reminders outside sessions. Structured home practice: 10 minutes per day of reading aloud with self-monitoring. Criterion: 80% correct in spontaneous speech samples across two sessions.",
            },
        ],
    },

    # ── Tongue sides — lateral lisp ───────────────────────────────────────────
    "tongue_sides__lateral_seal_failure": {
        "disorder_name": "Lateral Lisp",
        "articulator":   "Tongue sides",
        "description":   "Air escaping laterally around the sides of the tongue rather than through a central groove during sibilant production.",
        "do_not": [
            "Do not rush through the auditory discrimination stage — without it, placement work will not stick.",
            "Do not compare to interdental lisp treatment — lateral lisp requires a fundamentally different approach.",
            "Do not expect rapid progress — lateral lisps typically require 6–12 months of consistent intervention.",
            "Do not model the lateral sound.",
        ],
        "facilitating_contexts": [
            "Sustained /iː/ before /s/ helps establish lateral bracing of the tongue sides against upper molars.",
            "Voiceless production is easier first — establish the central airstream without voicing.",
            "Some patients respond to /ʃ/ as an intermediate target before attempting /s/.",
        ],
        "references": [
            "Gibbon F (1999) Undifferentiated lingual gestures. J Speech Lang Hear Res 42(2):382-397.",
            "McLeod S & Baker E (2017) Children's Speech. Pearson.",
            "Bauman-Waengler J (2020) Articulatory and Phonological Impairments. 5th ed.",
        ],
        "hierarchy": [
            {
                "stage": 1, "name": "Auditory Awareness",
                "goal": "Patient can reliably discriminate between their lateral /s/ and a correct /s/.",
                "instruction": "Listen carefully to these two sounds — one is how you say it, one is the target. Can you hear the difference? The correct S is sharp and thin. Yours sounds wetter. This awareness is the first step.",
                "cue_type": "auditory",
                "facilitating_words": [],
                "clinician_note": "Auditory discrimination MUST precede placement work for lateral lisps. Record patient's production and play back alongside modelled target. Without perceptual awareness, motor learning cannot proceed. Do not move to stage 2 until patient consistently identifies the difference. This may take multiple sessions.",
            },
            {
                "stage": 2, "name": "Central Airstream Establishment",
                "goal": "Produce a sustained central airstream without lateral escape.",
                "instruction": "Flatten your tongue and press the sides of it firmly up against your upper back teeth. Now blow air through the very centre — straight forward. Hold a piece of tissue in front of your mouth. It should flutter straight forward. If it blows sideways, the air is escaping sideways.",
                "cue_type": "tactile",
                "facilitating_words": [],
                "clinician_note": "Central airstream is the prerequisite for /s/ in lateral lisp. Use tissue paper or dental floss as visual airstream feedback. Some clinicians use a straw placed centrally to guide airstream direction. This stage may take several sessions before proceeding.",
            },
            {
                "stage": 3, "name": "Isolation with Central Groove",
                "goal": "Produce /s/ in isolation with no lateral escape.",
                "instruction": "Keep your tongue sides pressing against your upper back teeth. Now make the groove in the middle of your tongue — like a tiny channel. Direct the air through that channel. Add the hiss sound. The sides must stay pressed up.",
                "cue_type": "placement",
                "facilitating_words": [],
                "clinician_note": "Electropalatography (EPG) is the gold standard assessment for lateral lisp if available. Criterion: 5 consecutive correct isolated /s/ with no lateral air escape confirmed by tissue paper feedback.",
            },
            {
                "stage": 4, "name": "Syllable",
                "goal": "Maintain central /s/ in syllable context.",
                "instruction": "Now add a vowel, keeping everything in place. Start with SEE — the /iː/ helps keep your tongue sides up. Go slowly.",
                "cue_type": "movement",
                "facilitating_words": ["see", "sea", "sigh"],
                "clinician_note": "Rate control is critical here. Slow rate maintains correct placement. /iː/ facilitates lateral bracing. Criterion: 80% in 20 syllable productions.",
            },
            {
                "stage": 5, "name": "Word",
                "goal": "Correct /s/ in word context.",
                "instruction": "Use the sound in a short word. Go slowly — the goal is correct production, not speed.",
                "cue_type": "word",
                "facilitating_words": ["see", "sit", "sun", "six", "bus"],
                "clinician_note": "Expect regression when moving to word context — this is normal. Return to syllable level if more than 50% errors occur. Criterion: 80% across 20 words.",
            },
            {
                "stage": 6, "name": "Sentence and Generalisation",
                "goal": "Correct /s/ in sentences and connected speech.",
                "instruction": "Now use S words in a short sentence. Self-monitor each /s/ as you say it.",
                "cue_type": "sentence",
                "facilitating_words": ["Six soft sounds", "Sally sat beside the sea"],
                "clinician_note": "Carryover for lateral lisp requires intensive family involvement. Provide home programme. Note: some patients with lateral lisp have underlying phonological disorder — if no progress after 12+ sessions, refer for comprehensive assessment.",
            },
        ],
    },

    # ── Tongue tip — addental placement ──────────────────────────────────────
    "tongue_tip__dental_contact": {
        "disorder_name": "Addental /s/",
        "articulator":   "Tongue tip",
        "description":   "Tongue blade contacting upper incisors rather than maintaining appropriate distance from alveolar ridge, producing a muffled sibilant.",
        "do_not": [
            "Do not confuse with interdental lisp — the tongue is not protruding, just contacting teeth.",
            "Do not overcorrect by pulling tongue too far back.",
        ],
        "facilitating_contexts": [
            "Contrast /t/ (full contact) with /s/ (near-contact with airflow) to establish the difference.",
            "Slightly increased airflow pressure helps maintain the gap.",
        ],
        "references": ["Bernthal JE et al (2017) Articulation and Phonological Disorders. Pearson."],
        "hierarchy": [
            {
                "stage": 1, "name": "Isolation",
                "goal": "Produce /s/ with tongue tip near but not touching alveolar ridge.",
                "instruction": "Find the bumpy ridge just behind your top teeth. Place your tongue tip near it — but lift it just slightly, leaving a tiny gap. Breathe out. The air should hiss sharply through that gap. If it sounds dull, you are touching — add a millimetre of space.",
                "cue_type": "placement",
                "facilitating_words": [],
                "clinician_note": "Contrast /t/ (tongue touching ridge) with /s/ (tongue near ridge but not touching). Tactile awareness: patient should feel air against fingertip when held in front of mouth. Criterion: 9/10 correct with sharp hiss quality.",
            },
            {
                "stage": 2, "name": "Syllable",
                "goal": "Maintain non-contact /s/ in syllable context.",
                "instruction": "Now say SA, SI, SO — keep that tiny gap between your tongue tip and the ridge.",
                "cue_type": "movement",
                "facilitating_words": ["see", "sat", "so"],
                "clinician_note": "Criterion: 80% across 20 syllables.",
            },
            {
                "stage": 3, "name": "Word",
                "goal": "Correct /s/ in words.",
                "instruction": "Use the sound in a word. Listen for the sharp hiss on each S.",
                "cue_type": "word",
                "facilitating_words": ["sun", "sit", "six", "set", "bus"],
                "clinician_note": "Criterion: 80% across 20 words.",
            },
            {
                "stage": 4, "name": "Sentence and Generalisation",
                "goal": "Correct /s/ in connected speech.",
                "instruction": "Now use S words in a sentence with self-monitoring.",
                "cue_type": "sentence",
                "facilitating_words": ["Six small steps", "The sun is bright"],
                "clinician_note": "Addental /s/ typically responds well to treatment. Prognosis good. Carryover usually achieved within 8–12 sessions.",
            },
        ],
    },

    # ── Jaw — insufficient opening ────────────────────────────────────────────
    "jaw__insufficient_depression": {
        "disorder_name": "Vowel distortion — insufficient jaw opening",
        "articulator":   "Jaw / Mandible",
        "description":   "Insufficient mandibular depression causing first formant (F1) to be lower than target, distorting the perceived vowel quality.",
        "do_not": [
            "Do not force the jaw open mechanically.",
            "Do not exaggerate to the point of distorting the vowel.",
        ],
        "facilitating_contexts": [
            "/ɑː/ as an anchor vowel — it requires maximum jaw opening.",
            "Yawning before the exercise relaxes jaw musculature.",
        ],
        "references": ["Levelt WJM (1989) Speaking: From Intention to Articulation. MIT Press."],
        "hierarchy": [
            {
                "stage": 1, "name": "Jaw awareness and control",
                "goal": "Patient can feel and voluntarily control jaw aperture.",
                "instruction": "Place two fingers sideways between your front teeth — that is approximately the opening needed for this vowel. Remove them and try to recreate the same gap. Say AH and feel how far your jaw drops. Now hold that position.",
                "cue_type": "tactile",
                "facilitating_words": [],
                "clinician_note": "Proprioceptive awareness of jaw position is the foundation. Two-finger width is a useful tactile reference. Check for signs of jaw tension or temporomandibular joint (TMJ) involvement if jaw movement is restricted.",
            },
            {
                "stage": 2, "name": "Target vowel in isolation",
                "goal": "Produce the target vowel with correct jaw height.",
                "instruction": "Drop your jaw to the right position and hold it. Now say the target vowel. The jaw should not close before you finish.",
                "cue_type": "placement",
                "facilitating_words": [],
                "clinician_note": "Use mirror so patient can see jaw position. Criterion: consistent jaw opening matching target across 10 sustained vowel productions.",
            },
            {
                "stage": 3, "name": "Word context",
                "goal": "Correct vowel in word context.",
                "instruction": "Use the vowel in a word. Focus on keeping the jaw open for the entire vowel — don't let it close early.",
                "cue_type": "word",
                "facilitating_words": [],
                "clinician_note": "Choose words where the target vowel is stressed. Criterion: 80% correct across 20 words.",
            },
            {
                "stage": 4, "name": "Sentence and Generalisation",
                "goal": "Correct vowel in connected speech.",
                "instruction": "Use words with this vowel in sentences. The jaw should open naturally without you having to think about it.",
                "cue_type": "sentence",
                "facilitating_words": [],
                "clinician_note": "At this stage, motor automaticity should be developing. Reading aloud is a useful generalisation task.",
            },
        ],
    },

    # ── Tongue body — retraction ───────────────────────────────────────────────
    "tongue_body__retraction": {
        "disorder_name": "Vowel distortion — tongue retraction",
        "articulator":   "Tongue body",
        "description":   "Tongue body pulled posteriorly during production, lowering F2 and producing a back-shifted vowel quality.",
        "do_not": [
            "Do not push tongue too far forward — overcorrection causes a new error.",
        ],
        "facilitating_contexts": [
            "/iː/ as front-tongue anchor — the patient can feel the tongue in a forward position.",
            "Lip spreading facilitates tongue advancement.",
        ],
        "references": ["Bernthal JE et al (2017) Articulation and Phonological Disorders. Pearson."],
        "hierarchy": [
            {
                "stage": 1, "name": "Establishing front tongue position",
                "goal": "Patient can feel and produce a front tongue position.",
                "instruction": "Say EE slowly and notice where your tongue goes — it moves forward and high. Feel the front of your tongue near the bottom of your upper teeth. That forward position is what you need for this sound. Start from EE and then gradually move toward the target vowel.",
                "cue_type": "tactile",
                "facilitating_words": ["EE", "see", "tea"],
                "clinician_note": "/iː/ is the anchor vowel for establishing anterior tongue position. Criterion: patient can demonstrate front tongue position on request.",
            },
            {
                "stage": 2, "name": "Target vowel isolation",
                "goal": "Produce target vowel with forward tongue position.",
                "instruction": "Start with EE, hold the forward tongue position, and slide gradually into the target vowel. Keep the tongue moving forward the whole time.",
                "cue_type": "movement",
                "facilitating_words": [],
                "clinician_note": "Transition approach from known correct /iː/ to target vowel is effective for tongue advancement. Criterion: 80% of isolated vowel productions within F2 target range.",
            },
            {
                "stage": 3, "name": "Word and sentence",
                "goal": "Correct vowel in words and sentences.",
                "instruction": "Use the vowel in a word. Remember to keep the tongue forward.",
                "cue_type": "word",
                "facilitating_words": [],
                "clinician_note": "Minimal pair contrast with back vowel counterpart reinforces perceptual distinction. Criterion: 80% across 20 words.",
            },
        ],
    },

    # ── Lips — insufficient rounding ─────────────────────────────────────────
    "lips__insufficient_rounding": {
        "disorder_name": "Vowel distortion — lip rounding",
        "articulator":   "Lips",
        "description":   "Absence or insufficiency of labial protrusion and rounding required for rounded vowels, raising F2 above target.",
        "do_not": [
            "Do not over-exaggerate rounding to the point of tension.",
        ],
        "facilitating_contexts": [
            "Whistling posture approximates correct lip rounding.",
            "Contrast with spread-lip vowels to heighten awareness of the rounding distinction.",
        ],
        "references": ["Bernthal JE et al (2017) Articulation and Phonological Disorders. Pearson."],
        "hierarchy": [
            {
                "stage": 1, "name": "Lip rounding awareness",
                "goal": "Patient can produce consistent lip rounding on demand.",
                "instruction": "Pucker your lips as if you are about to whistle or blow out a candle. Hold that circle shape. Notice how your lips come forward and round. That is the shape you need before you make the sound.",
                "cue_type": "visual",
                "facilitating_words": [],
                "clinician_note": "Use mirror. Confirm protrusion as well as rounding — some patients round without protruding. Criterion: 9/10 correct lip rounding posture on request.",
            },
            {
                "stage": 2, "name": "Vowel in isolation",
                "goal": "Produce rounded vowel with correct lip posture.",
                "instruction": "Round your lips first. Hold the shape. Now make the vowel sound. The lip rounding must begin before the sound.",
                "cue_type": "placement",
                "facilitating_words": [],
                "clinician_note": "Lip rounding must precede phonation onset. Criterion: F2 within target range on 80% of productions.",
            },
            {
                "stage": 3, "name": "Word and sentence",
                "goal": "Automatic lip rounding in connected speech.",
                "instruction": "Use the vowel in a word. The rounding should happen automatically now.",
                "cue_type": "word",
                "facilitating_words": [],
                "clinician_note": "If automatic rounding does not develop, return to explicit pre-phonation rounding cue. Criterion: 80% across 20 words.",
            },
        ],
    },

    # ── /r/ — labial compensation (w/r substitution) ─────────────────────────
    "tongue_body__labial_compensation_r": {
        "disorder_name": "/r/ production difficulty — labial compensation",
        "articulator":   "Tongue body (with labial compensation)",
        "description":   "Patient producing /w/ in place of /ɹ/ by substituting labial rounding for the required tongue body elevation.",
        "do_not": [
            "Do not teach retroflex /r/ — Australian English /ɹ/ is a bunched rhotic.",
            "Do not allow lip rounding to persist — it maintains the /w/ error.",
            "Do not start with word-initial /r/ — medial position is neurologically easier.",
        ],
        "facilitating_contexts": [
            "/ɹ/ is easier in medial word position — begin with words like 'arrow', 'very', 'carry'.",
            "Prolonged /ɔː/ produces a tongue body position close to /ɹ/ — use it as a transition.",
            "Vowel /ɜː/ (as in BIRD) shares tongue body position with /ɹ/ — use as an anchor.",
        ],
        "references": [
            "McLeod S & Baker E (2017) Children's Speech. Pearson.",
            "Shriberg LD & Kwiatkowski J (1982) Phonological disorders III. J Speech Hear Disord.",
        ],
        "hierarchy": [
            {
                "stage": 1, "name": "Eliminate lip rounding",
                "goal": "Patient produces approximant without lip rounding.",
                "instruction": "Keep your lips completely flat and relaxed — no rounding at all. Watch yourself in the mirror. As soon as your lips start to round, the sound becomes W. Keep them flat.",
                "cue_type": "visual",
                "facilitating_words": [],
                "clinician_note": "Eliminating lip rounding is the prerequisite step. Use mirror. Some patients benefit from holding a tongue depressor horizontally between the lips to prevent rounding. Do not proceed until lip rounding is eliminated.",
            },
            {
                "stage": 2, "name": "Tongue body elevation — vowel context",
                "goal": "Produce /ɹ/ in vowel context (medial position).",
                "instruction": "With flat lips, bunch the back of your tongue upward toward the roof of your mouth. Say OR slowly — hold the OR sound. Your tongue should be high and back. Now say 'arrow' very slowly: AH-R-OH. Feel the tongue rise for the R in the middle.",
                "cue_type": "tactile",
                "facilitating_words": ["arrow", "very", "carry", "hurry", "sorry"],
                "clinician_note": "Medial /ɹ/ is the easiest position due to surrounding vowel context providing coarticulatory support. /ɜː/ (bird) provides the closest tongue body approximation. Criterion: 80% in VCV syllables with no lip rounding.",
            },
            {
                "stage": 3, "name": "Word — medial then final",
                "goal": "Correct /ɹ/ in word medial and final positions.",
                "instruction": "Use the sound in a word where R is in the middle. Go slowly. Lips stay flat.",
                "cue_type": "word",
                "facilitating_words": ["arrow", "very", "carry", "better", "butter", "letter"],
                "clinician_note": "Final /ɹ/ in Australian English is often vocalised — ensure expectations are consistent with AuE norms. Criterion: 80% across 20 words.",
            },
            {
                "stage": 4, "name": "Word — initial position",
                "goal": "Correct /ɹ/ at start of words.",
                "instruction": "Now try R at the start of a word. Prepare your tongue position before you begin — tongue up, lips flat.",
                "cue_type": "word",
                "facilitating_words": ["run", "red", "rain", "right", "road"],
                "clinician_note": "Initial /ɹ/ is harder because there is no preceding vowel to provide coarticulatory support. Criterion: 80% across 20 initial-/r/ words.",
            },
            {
                "stage": 5, "name": "Sentence and Generalisation",
                "goal": "Correct /ɹ/ in connected speech.",
                "instruction": "Use R words in sentences. Monitor yourself — especially at the start of words.",
                "cue_type": "sentence",
                "facilitating_words": ["Robert ran really fast", "The red rabbit ran around"],
                "clinician_note": "Carryover for /r/ is often the longest stage. Home practice with recording and playback recommended. Enlist parents/teachers. Criterion: 80% in spontaneous speech samples.",
            },
        ],
    },
}


# ── Articulator rule triggers ─────────────────────────────────────────────────
# Maps measured values to (articulator, problem, severity 0-100, protocol_key)

def _apply_rules(
    phoneme:          str,
    F1_hz:            Optional[float] = None,
    F1_ref:           Optional[float] = None,
    F2_hz:            Optional[float] = None,
    F2_ref:           Optional[float] = None,
    cog_hz:           Optional[float] = None,
    energy_db:        Optional[float] = None,
    tilt_db_oct:      Optional[float] = None,
    lisp_type:        Optional[str]   = None,
    lip_rounding:     Optional[float] = None,   # 0–1
    lip_spread:       Optional[float] = None,
    jaw_aperture:     Optional[float] = None,
    jaw_expected:     Optional[str]   = None,
    round_expected:   Optional[str]   = None,
    confidence:       float = 1.0,
) -> list[tuple[str, str, int, str]]:
    """
    Returns list of (articulator, problem, severity, protocol_key).
    Severity: 0 = correct, 100 = extreme error.
    """
    findings = []

    # ── Sibilant / lisp rules ─────────────────────────────────────────────────
    if phoneme in {"s", "z"} and cog_hz is not None:
        if lisp_type == "interdental" or cog_hz < 3800:
            severity = min(100, int((3800 - cog_hz) / 1800 * 100)) if cog_hz < 3800 else 60
            findings.append(("tongue_tip", "interdental_protrusion", severity,
                             "tongue_tip__interdental_protrusion"))

        elif lisp_type == "lateral":
            severity = min(100, int((5000 - cog_hz) / 3000 * 100)) if energy_db and energy_db < -40 else 65
            findings.append(("tongue_sides", "lateral_seal_failure", severity,
                             "tongue_sides__lateral_seal_failure"))

        elif lisp_type == "addental" or (3800 <= cog_hz < 5500):
            severity = min(100, int((5500 - cog_hz) / 1700 * 60))
            findings.append(("tongue_tip", "dental_contact", severity,
                             "tongue_tip__dental_contact"))

    # ── Vowel rules ───────────────────────────────────────────────────────────
    if F1_hz and F1_ref and confidence >= 0.35:
        f1d = F1_hz - F1_ref
        if f1d > 80:   # tongue too low / jaw too open
            sev = min(100, int(f1d / 300 * 100))
            findings.append(("jaw", "insufficient_depression", sev,
                             "jaw__insufficient_depression"))
        elif f1d < -80:  # tongue too high
            sev = min(100, int(abs(f1d) / 300 * 100))
            findings.append(("jaw", "excessive_depression", sev,
                             "jaw__insufficient_depression"))  # same protocol, different direction

    if F2_hz and F2_ref and confidence >= 0.35:
        f2d = F2_hz - F2_ref
        if f2d < -200:   # tongue too back
            sev = min(100, int(abs(f2d) / 1000 * 100))
            findings.append(("tongue_body", "retraction", sev,
                             "tongue_body__retraction"))
        elif f2d > 200:  # tongue too front
            sev = min(100, int(f2d / 1000 * 100))
            findings.append(("tongue_body", "advancement", sev,
                             "tongue_body__retraction"))  # same protocol, opposite direction

    # ── Lip / jaw visual rules ────────────────────────────────────────────────
    if round_expected == "rounded" and lip_rounding is not None and lip_rounding < 0.3:
        sev = int((0.3 - lip_rounding) / 0.3 * 70)
        findings.append(("lips", "insufficient_rounding", sev,
                         "lips__insufficient_rounding"))

    # Sort by severity — most severe first
    findings.sort(key=lambda x: x[2], reverse=True)
    return findings


# ── Main public function ──────────────────────────────────────────────────────

def analyse(
    phoneme:        str,
    frication:      Optional[dict] = None,   # FricationFeatures as dict
    vowel:          Optional[dict] = None,   # VowelFeatures as dict
    lip:            Optional[dict] = None,   # lip_analysis result
    confidence:     float = 1.0,
) -> Optional[ArticulatorFinding]:
    """
    Takes acoustic + visual measurements and returns a structured
    ArticulatorFinding with intervention protocol and current stage.

    Returns None if no error is detected.
    """

    # Unpack measurements
    f = frication or {}
    v = vowel     or {}
    l = lip       or {}

    findings = _apply_rules(
        phoneme       = phoneme,
        F1_hz         = v.get("F1_hz"),
        F1_ref        = v.get("F1_ref"),
        F2_hz         = v.get("F2_hz"),
        F2_ref        = v.get("F2_ref"),
        cog_hz        = f.get("CoG_hz"),
        energy_db     = f.get("energy_db"),
        tilt_db_oct   = f.get("tilt_db_oct"),
        lisp_type     = f.get("lisp_type"),
        lip_rounding  = l.get("lip_ratio"),
        lip_spread    = l.get("lip_spread"),
        jaw_aperture  = l.get("jaw_aperture"),
        jaw_expected  = l.get("jaw_expected"),
        round_expected= l.get("round_expected"),
        confidence    = confidence,
    )

    if not findings:
        return None

    # Take the most severe finding
    articulator, problem, severity, protocol_key = findings[0]

    proto = PROTOCOLS.get(protocol_key)
    if not proto:
        return None

    # Determine current stage from session history
    disorder_key = protocol_key
    stage_num    = _get_phoneme_stage(phoneme, disorder_key)
    stage_num    = max(1, min(stage_num, len(proto["hierarchy"])))
    stage        = proto["hierarchy"][stage_num - 1]

    return ArticulatorFinding(
        articulator        = articulator,
        problem            = problem,
        severity           = severity,
        disorder_name      = proto["disorder_name"],
        description        = proto["description"],
        intervention_stage = stage_num,
        stage_name         = stage["name"],
        instruction        = stage["instruction"],
        cue_type           = stage["cue_type"],
        facilitating_words = stage.get("facilitating_words", []),
        do_not             = proto.get("do_not", []),
        clinician_note     = stage.get("clinician_note", ""),
        references         = proto.get("references", []),
        ai_prompt_context  = {
            "articulator":   articulator,
            "problem":       problem,
            "severity":      severity,
            "disorder":      proto["disorder_name"],
            "stage":         stage_num,
            "stage_name":    stage["name"],
            "instruction":   stage["instruction"],
            "cue_type":      stage["cue_type"],
            "do_not":        proto.get("do_not", []),
        },
    )


def analyse_to_dict(phoneme: str, frication=None, vowel=None, lip=None, confidence=1.0) -> Optional[dict]:
    """Convenience wrapper returning dict or None."""
    result = analyse(phoneme, frication, vowel, lip, confidence)
    return asdict(result) if result else None
