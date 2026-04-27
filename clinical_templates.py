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
        "cue":       "Your tongue is too low for that vowel — the sound is coming out too open.",
        "exercise":  "Say EE slowly and feel where your tongue goes — high in your mouth. Now try to keep that same raised position as you move into the target vowel. Your jaw and tongue should both come up.",
        "analogy":   "Say EE and freeze. Feel how high your tongue is? That's the direction you need — bring it up toward the roof of your mouth.",
        "clinician": "F1 significantly elevated above target. Tongue body depressed, likely accompanied by excess jaw opening. Recommend contrast drilling with /iː/ as high-tongue anchor before transitioning to target vowel. Check jaw coupling.",
    },
    "TONGUE_TOO_LOW_MILD": {
        "cue":       "Your vowel is almost there — your tongue needs to come up just slightly.",
        "exercise":  "Close your jaw a small amount and feel your tongue rise with it. You only need a small change — the sound is close.",
        "analogy":   "You're one floor below where you need to be. A small lift is all it takes.",
        "clinician": "F1 mildly elevated above target. Minor tongue body depression. May self-correct with repeated practice. If persistent, introduce minimal pair contrast with higher vowel.",
    },
    "TONGUE_TOO_HIGH": {
        "cue":       "Your tongue is pressed too high and your mouth isn't open enough — the vowel sounds squashed.",
        "exercise":  "Let your jaw drop and allow your tongue to relax downward. Say AH to feel that open position, then try the target sound with the same relaxed feeling.",
        "analogy":   "Your tongue is gripping the roof. Let go and let it fall — the sound needs space.",
        "clinician": "F1 below target. Tongue body elevated above target height. Check for muscular hypertension. Introduce /ɑː/ as low-tongue anchor. Tactile jaw-drop prompt may assist if patient tenses.",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # VOWEL ERRORS — tongue front/back (F2 axis)
    # ══════════════════════════════════════════════════════════════════════════

    "TONGUE_TOO_BACK": {
        "cue":       "Your tongue is pulling too far back in your mouth — the vowel sounds darker than it should.",
        "exercise":  "Spread your lips slightly and push the front of your tongue toward your lower front teeth. The tip doesn't need to touch them — just move in that direction. Try EE first to find the front position, then bring some of that forward feeling into the target sound.",
        "analogy":   "Your tongue is retreating to the back of your mouth. Bring it to the front — toward your teeth.",
        "clinician": "F2 below target. Posterior tongue body placement. Check for compensatory velar retraction or habitual backing pattern. Use /iː/ as front-tongue anchor. Minimal pairs contrasting back and front vowels recommended.",
    },
    "TONGUE_TOO_BACK_MILD": {
        "cue":       "Your vowel sounds slightly dark — your tongue is a little too far back.",
        "exercise":  "Nudge your tongue forward slightly and spread your lips a little wider. The change is small.",
        "analogy":   "Take a small step forward — the tongue just needs to come forward a little.",
        "clinician": "F2 mildly below target. Slight posterior tongue placement. Minimal pair contrast practice recommended. Monitor across multiple attempts for consistency.",
    },
    "TONGUE_TOO_FRONT": {
        "cue":       "Your tongue is too far forward — the vowel sounds brighter than it should be.",
        "exercise":  "Pull your tongue back slightly from your teeth and round your lips gently. The tongue should sit more in the middle or back of your mouth for this sound.",
        "analogy":   "Your tongue is crowding the front of your mouth. Let it settle back and relax.",
        "clinician": "F2 above target. Anterior tongue placement. Common pattern with front vowel overgeneralisation. Introduce /uː/ or /ɔː/ as back-tongue anchors. Check for lip rounding which may assist retraction.",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # JAW / LIP ERRORS (confirmed by camera)
    # ══════════════════════════════════════════════════════════════════════════

    "JAW_TOO_CLOSED": {
        "cue":       "Your mouth isn't open enough — the vowel is being squeezed. Drop your jaw more.",
        "exercise":  "Place two fingers sideways between your front teeth as a guide, then remove them and try to keep that same opening as you say the sound. Let your jaw fall naturally — don't force it.",
        "analogy":   "Imagine your jaw is heavy — let gravity pull it down. This vowel needs room to breathe.",
        "clinician": "Jaw aperture below target confirmed by camera landmarks and corroborated by elevated F1. Introduce tactile jaw-drop cue (two-finger width). Exaggerated jaw movement in mirror work may assist. Rule out dental occlusion issues.",
    },
    "JAW_TOO_OPEN": {
        "cue":       "Your jaw is dropping too far — the vowel loses its shape when the mouth is too wide.",
        "exercise":  "Bring your jaw up slightly so your teeth are closer together. The vowel needs a more controlled, smaller opening.",
        "analogy":   "You're opening the door too wide. Close it a little — just enough for the sound to get through.",
        "clinician": "Jaw aperture above target confirmed by camera. Corroborated by reduced F1. Demonstrate controlled, relaxed jaw position. Avoid over-correction. Tactile chin support cue may help if jaw depression is habitual.",
    },
    "LIPS_NOT_ROUNDED": {
        "cue":       "Your lips need to come forward and round into a circle for this sound.",
        "exercise":  "Before you start, pucker your lips as if blowing out a candle. Hold that rounded shape and then make the sound. The lips stay forward and circular throughout.",
        "analogy":   "Make a small O with your lips — like the opening of a cup. Keep that shape.",
        "clinician": "Lip rounding absent confirmed by lip ratio measurement. Target phoneme requires active labial protrusion and rounding. Contrast with spread-lip vowel using mirror. Ensure rounding precedes phonation onset.",
    },
    "LIPS_NOT_SPREAD": {
        "cue":       "Your lips need to stretch outward at the corners for this sound — spread them wide.",
        "exercise":  "Pull the corners of your mouth outward and slightly back — like a wide, flat smile showing your teeth. Keep that spread as you say the sound.",
        "analogy":   "Stretch your lips wide like you're showing someone your teeth. That flat, wide shape is what this sound needs.",
        "clinician": "Lip spreading absent confirmed by lip spread ratio. Target phoneme requires retracted lip corners. Patient may be using rounded or neutral lip position. Mirror work recommended. Contrast with rounded vowel to establish the distinction.",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # LISP ERRORS — sibilant /s/ and /z/
    # ══════════════════════════════════════════════════════════════════════════

    "LISP_INTERDENTAL": {
        "cue":       "Your S is sounding like TH — your tongue is coming forward between your teeth when it needs to stay behind them.",
        "exercise":  "Step 1: Close your teeth lightly together and feel that barrier. Step 2: Place your tongue TIP on the bumpy ridge just behind your upper front teeth — that ridge is called the alveolar ridge. Step 3: Breathe out through that position and feel the hiss. Your tongue must not touch the teeth or poke through the gap. If needed, start with just the T sound to find the ridge, then add airflow for S.",
        "analogy":   "Your teeth are a gate. For S, your tongue stays behind the gate. For TH, it sneaks through. You need the tongue behind the gate every time.",
        "clinician": "Spectral CoG below 3800 Hz consistent with /s/→/θ/ substitution (interdental lisp). Tongue tip achieving interdental rather than alveolar placement. Recommended progression: (1) auditory discrimination /s/ vs /θ/ using minimal pairs; (2) tongue-tip placement on alveolar ridge in isolation; (3) /tsa/ syllables to stabilise alveolar placement via stop-fricative coarticulation; (4) initial /s/ in CV words; (5) medial then final position; (6) connected speech. Mirror work and tongue depressor as physical barrier during initial training may assist.",
    },
    "LISP_INTERDENTAL_MILD": {
        "cue":       "Your S is almost right — your tongue is just slightly too far forward and may be grazing your teeth.",
        "exercise":  "Feel the bumpy ridge behind your upper teeth with your tongue tip. That is where the tongue belongs for S — not on the teeth themselves, just behind them on the ridge. Pull back just a millimetre and listen for the sharper hiss.",
        "analogy":   "You're one small step too far forward. Pull back onto the ridge.",
        "clinician": "CoG 3800–4800 Hz. Borderline interdental or addental pattern — tongue may be contacting incisors rather than alveolar ridge. Monitor consistency across productions. If not self-correcting, introduce explicit alveolar placement cue.",
    },
    "LISP_LATERAL": {
        "cue":       "Your S has a wet, slushy quality — air is leaking out around the sides of your tongue instead of flowing through the middle in a stream.",
        "exercise":  "Step 1: Say EE and notice how your tongue fills the sides of your mouth, touching your upper back teeth. Step 2: While holding that side contact, try to direct all your breath through a narrow groove in the CENTRE of your tongue — like air through a straw. Step 3: Slowly add voiceless breath to get S. Hold a piece of tissue paper in front of your mouth — it should flutter straight forward, not sideways.",
        "analogy":   "Your air is taking the scenic route around the sides. It needs to go straight down the middle — like water through a narrow pipe.",
        "clinician": "CoG below target with reduced frication energy and elevated spectral spread — pattern consistent with lateral sibilant. Lateral airflow replacing central groove. Priority: establish central airstream before working on tongue tip placement. Recommend: (1) sustained /iː/ to establish lateral bracing; (2) central airflow production using straw or tissue feedback; (3) /ʃ/ as intermediate target if /s/ groove cannot be achieved directly; (4) gradual forward tongue movement from /ʃ/ to /s/ with central groove maintained. Note: lateral lisps are often resistant to change — patient and family counselling may be needed to set realistic expectations.",
    },
    "LISP_ADDENTAL": {
        "cue":       "Your S sounds slightly dull or soft — your tongue tip may be resting against your upper teeth instead of holding just behind them.",
        "exercise":  "Find the bumpy alveolar ridge behind your top teeth with your tongue tip. Now lift the tip just slightly off the teeth so there's a tiny gap. The air should hiss through that gap sharply. If it sounds dull, you're touching — if it sounds sharp, you've found it.",
        "analogy":   "There should be a thin crack of space — like a door open just a centimetre. The air hisses through that crack. If the tongue is touching the teeth, the door is shut.",
        "clinician": "CoG 4500–5500 Hz. Dentalized sibilant production — tongue blade contacting upper incisors rather than alveolar ridge. Increase anterior blade-to-ridge distance. Tactile cue using tongue depressor placed behind teeth may guide correct placement. Contrast with /t/ production (full contact) versus /s/ (near-contact with airflow) to establish distinction.",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CONSONANT SUBSTITUTIONS
    # ══════════════════════════════════════════════════════════════════════════

    "SUB_S_TO_TH": {
        "cue":       "Your S is coming out as TH — your tongue is sneaking through your teeth when it needs to stay behind them.",
        "exercise":  "Close your teeth lightly. Say T — feel your tongue tip on the ridge. Now keep it there and breathe out. That is S. The tongue never leaves the area behind your teeth.",
        "analogy":   "TH sends the tongue through the gate. S keeps it behind the gate. Every time.",
        "clinician": "Phonemic substitution /s/→/θ/. Interdental tongue placement confirmed acoustically (low CoG) and visually (tongue tip visible between incisors on camera). Introduce auditory discrimination of /s/ vs /θ/ using minimal pairs (sea/thee, sigh/thigh) before placement work. Use mirror to provide real-time visual feedback on tongue position.",
    },
    "SUB_S_TO_SH": {
        "cue":       "Your S is coming out as SH — your tongue is too far back in your mouth.",
        "exercise":  "Move your tongue forward slightly — the tip should be just behind your upper front teeth, not in the middle of your mouth. Try saying T first to find the right spot, then breathe out for S.",
        "analogy":   "SH is made in the middle of the mouth. S is made much closer to the front. Slide your tongue forward until you hear the change.",
        "clinician": "Phonemic substitution /s/→/ʃ/. Posterior tongue dorsum placement. Increase anterior placement — tongue tip target is alveolar ridge, not postalveolar region. Introduce minimal pairs (sea/she, sip/ship) for auditory discrimination. Tactile forward tongue movement cue may assist.",
    },
    "SUB_TH_TO_S": {
        "cue":       "Your TH sounds like S — your tongue needs to move forward between your teeth to make TH.",
        "exercise":  "Open your mouth slightly and put your tongue tip gently between your upper and lower front teeth — just the very tip, resting lightly. Now breathe out steadily. You should feel air flowing over the top of your tongue. That friction is the TH sound.",
        "analogy":   "For S, the tongue hides. For TH, it peeks out between the teeth — just a little peek.",
        "clinician": "Phonemic substitution /θ/→/s/. Tongue tip not achieving interdental placement — remaining alveolar. Use mirror for visual feedback. Tactile cue: patient can briefly touch tongue tip with finger to confirm protrusion. Auditory discrimination with minimal pairs (sin/thin, sink/think) recommended before placement work.",
    },
    "SUB_TH_TO_F": {
        "cue":       "Your TH is coming out as F — you're using your lip and teeth instead of your tongue between your teeth.",
        "exercise":  "Move your tongue tip forward between your upper and lower teeth. Don't let your bottom lip touch your top teeth. It's the TONGUE that does the work for TH — not the lip.",
        "analogy":   "F is made with teeth on lip. TH is made with tongue between teeth. Two completely different articulators.",
        "clinician": "Phonemic substitution /θ/→/f/. Labiodental placement instead of linguadental. Patient may be using the visually similar labiodental posture. Emphasise tongue protrusion — use mirror so patient can see the difference. Tactile awareness: patient should feel tongue between teeth, not teeth on lip.",
    },
    "SUB_DH_TO_Z": {
        "cue":       "Your voiced TH is coming out as Z — your tongue is behind your teeth when it needs to be between them.",
        "exercise":  "Put your tongue tip gently between your front teeth — the same position as for the S-to-TH exercise. Now add your voice. You should feel a buzz. If you feel it, that is voiced TH.",
        "analogy":   "Z hides the tongue. Voiced TH shows it. Let the tongue peek through and add your voice.",
        "clinician": "Phonemic substitution /ð/→/z/. Tongue remaining at alveolar position rather than achieving interdental. Establish voiceless /θ/ first (easier to monitor), then add voicing. Touch larynx to confirm voicing. Contrast /ð/ with /z/ using minimal pairs (then/zen, thy/Zion).",
    },
    "SUB_DH_TO_D": {
        "cue":       "Your voiced TH sounds like D — your tongue is on the ridge when it needs to come forward between your teeth.",
        "exercise":  "Say D first — feel your tongue tip on the ridge. Now slide your tongue tip forward, past the top teeth, until it sits lightly between your upper and lower teeth. Add your voice. That is voiced TH.",
        "analogy":   "D is one step back. Voiced TH is one step forward — with the tongue between the teeth.",
        "clinician": "Phonemic substitution /ð/→/d/. Alveolar stop replacing voiced interdental fricative — common developmental pattern. Establish interdental placement using voiceless /θ/ first, then introduce voicing. Mirror work to visually confirm tongue protrusion.",
    },
    "SUB_R_TO_W": {
        "cue":       "Your R is coming out as W — you're using your lips to make the sound instead of your tongue.",
        "exercise":  "Keep your lips flat and relaxed — no rounding. Then bunch the back and middle of your tongue upward toward the roof of your mouth. Hold that position and say the vowel that follows. Try 'ora' or 'arrow' — the R in the middle is easier to find. Once you feel it there, practise moving it to the front of words.",
        "analogy":   "W lives in your lips. R lives in your tongue. Keep your lips still and let the tongue do the work.",
        "clinician": "Phonemic substitution /ɹ/→/w/. Patient compensating with labial rounding rather than tongue body elevation. Australian English /ɹ/ is typically a bunched rhotic — tongue body bunched upward, no lip rounding. Intervention sequence: (1) eliminate lip rounding; (2) establish bunched tongue body position in vowel context (e.g. /ɔːɹ/ prolonged); (3) CV syllables with /ɹ/ in medial position; (4) initial /ɹ/ in words. Avoid retroflex cue — not typical for AuE.",
    },
    "SUB_L_TO_W": {
        "cue":       "Your L is coming out as W — your tongue tip needs to tap the ridge behind your top teeth.",
        "exercise":  "Find the bumpy ridge just behind your upper front teeth. Place your tongue tip there. Now say a vowel — the tongue should lift up to the ridge for L and then come down. Practice: say ah-la-la-la slowly, feeling the tongue tip tap the ridge each time.",
        "analogy":   "L needs a tap — your tongue tip briefly touches the ridge like pressing a button, then releases.",
        "clinician": "Phonemic substitution /l/→/w/. Tongue tip not achieving alveolar contact. Likely velarisation or labialisation substituting for lateral approximant. Establish tip-up position in isolation, then add vowel context. Syllable repetition (la, li, lo) recommended to build automaticity of alveolar contact.",
    },
    "SUB_K_TO_T": {
        "cue":       "Your K sounds like T — the front of your tongue is working when it should be the back.",
        "exercise":  "Press the back of your tongue against the soft part at the very back of the roof of your mouth — that soft area behind the hard part. Hold it there. Now release it with a pop. That is K. The tongue TIP stays down and out of the way.",
        "analogy":   "T is made at the front of your mouth with the tip. K is made way at the back with the back. Keep the tip down — let the back do the work.",
        "clinician": "Phonemic substitution /k/→/t/. Anterior tongue placement with tip contact substituting for posterior velar contact. Establish velar placement in isolation using tactile feedback (finger on back of tongue) if needed, then syllable context. Minimal pairs (cap/tap, key/tea) for auditory discrimination. Some patients respond to downward chin pressure cue which facilitates tongue body retraction.",
    },
    "SUB_NG_MISSING": {
        "cue":       "You're dropping the NG sound at the end — the word is ending too early.",
        "exercise":  "At the end of the word, press the back of your tongue against the soft roof at the back of your mouth and let the sound hum through your nose. Hold it for a moment before you stop. The sound should feel like it's coming out of your nose, not your mouth.",
        "analogy":   "NG is a nose hum with the back of your tongue up. Feel the vibration in your nose as you finish the word.",
        "clinician": "Final /ŋ/ deletion. Velar nasal weakening in coda position — common in connected speech. May reflect phonological process (final consonant deletion) or coarticulation difficulty. Establish /ŋ/ in isolation first (sustained nasal hum with velar closure confirmed by mirror), then CV+ŋ syllables, then final position in words. Tactile vibration cue on nose bridge may reinforce nasal resonance.",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # GENERAL / STRUCTURAL ERRORS
    # ══════════════════════════════════════════════════════════════════════════

    "SOUND_MISSING_FINAL": {
        "cue":       "The sound at the end of the word faded out — you need to finish the word completely.",
        "exercise":  "Say the word again, but this time hold the very last sound for an extra beat before you stop. The word isn't finished until that final sound is clear. Try tapping your finger on the table as you say the final sound to help you commit to it.",
        "analogy":   "Think of the final sound as landing a plane — you need to touch down fully, not hover above the runway.",
        "clinician": "Final consonant deletion or significant reduction. Coda position weakening — may reflect phonological simplification process or motor planning difficulty under connected speech conditions. Introduce final consonant in isolation, then VC syllables, then CVC words with extended final consonant duration. Rhythmic cueing (finger tap on final phoneme) assists awareness.",
    },
    "SOUND_MISSING_INITIAL": {
        "cue":       "The sound at the start of the word was missing — the word started mid-way through.",
        "exercise":  "Before you start the word, prepare your mouth in the right position for the first sound. Take a breath, get set, then begin. Don't start speaking until you feel your mouth is ready.",
        "analogy":   "A runner gets into position before the starting gun. Get your mouth into position before you start the word.",
        "clinician": "Initial consonant deletion or severe reduction. Onset position weakening — consider whether omission is consistent or variable. If consistent, may reflect phonological process. If variable, may reflect motor initiation difficulty. Introduce onset consonant in isolation, then in CV syllables before the full word. Slow rate with exaggerated onset may assist.",
    },
    "WORD_WRONG": {
        "cue":       "The word wasn't quite clear enough for me to catch — give it another go, slowly and clearly.",
        "exercise":  "If the word has more than one syllable, break it apart and say each syllable separately, then join them back together.",
        "analogy":   None,
        "clinician": "Target word not recognised by speech recogniser. May indicate reduced intelligibility, environmental noise, or recording quality issue. Check microphone placement. If recurring for same word, consider whether target is within patient's current phonological capacity.",
    },
    "ACOUSTIC_MISMATCH": {
        "cue":       "Your jaw and lips look right, but the sound is still off — the issue is likely inside your mouth with your tongue.",
        "exercise":  "Keep your jaw and lips exactly where they are. Now focus entirely on where the body of your tongue is sitting. Try moving the tongue slightly higher, lower, or further forward to find the right sound.",
        "analogy":   "The outside of the car looks fine — it's the engine inside that needs the adjustment.",
        "clinician": "Acoustic/visual discrepancy — camera confirms jaw aperture and lip geometry within acceptable range, but F1/F2 deviate from target. Suggests tongue body position error independent of jaw and lip movements. Isolate tongue body placement — provide explicit tactile or kinematic cues. Check for tongue humping, retraction, or excessive lateral bracing.",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # VOWEL SOUND COMPARISONS (nearest-neighbour mismatch)
    # ══════════════════════════════════════════════════════════════════════════

    "VOWEL_SOUNDS_LIKE_BIT": {
        "cue":       "Your vowel sounds more like the short I in BIT than the target sound.",
        "exercise":  "Your tongue is a little too high and too far forward. Drop your jaw slightly and let your tongue relax downward. Listen for the difference — BIT is higher and tighter than where you need to be.",
        "analogy":   None,
        "clinician": "Measured F1/F2 in /ɪ/ region. Tongue body elevated and fronted relative to target. Introduce target vowel in minimal pair contrast with /ɪ/.",
    },
    "VOWEL_SOUNDS_LIKE_BED": {
        "cue":       "Your vowel sounds more like the E in BED than the target sound.",
        "exercise":  "Listen to the difference between BED and the target word. Your tongue needs to move — either higher, lower, or further back, depending on the target sound.",
        "analogy":   None,
        "clinician": "Measured F1/F2 in /e/ region. Mid-front tongue placement. Use minimal pair contrast with target vowel to establish perceptual distinction before motor training.",
    },
    "VOWEL_SOUNDS_LIKE_CAT": {
        "cue":       "Your vowel sounds more like the A in CAT than the target sound — it's too open.",
        "exercise":  "Bring your jaw up slightly and raise your tongue. You're dropping too low. The target sound needs less jaw opening than CAT.",
        "analogy":   None,
        "clinician": "Measured F1/F2 in /æ/ region. Low front tongue placement. Jaw drop excessive relative to target. Introduce target vowel contrasted with /æ/ using minimal pairs.",
    },
    "VOWEL_SOUNDS_LIKE_FEET": {
        "cue":       "Your vowel sounds more like the EE in FEET — it's too high and too tight.",
        "exercise":  "Let your jaw drop a little and relax your tongue downward. EE is made with the tongue very high — your target sound needs more space.",
        "analogy":   None,
        "clinician": "Measured F1/F2 in /iː/ region. Tongue body excessively elevated. Jaw too closed. Contrast with low or mid vowel to establish appropriate openness.",
    },
    "VOWEL_SOUNDS_LIKE_FOOD": {
        "cue":       "Your vowel sounds more like the OO in FOOD — it's too rounded and too back.",
        "exercise":  "Spread your lips and move your tongue slightly forward. OO uses rounded lips and a back tongue — your target sound probably needs a different shape.",
        "analogy":   None,
        "clinician": "Measured F1/F2 in /uː/ region. High back tongue with lip rounding. Reduce rounding and advance tongue body toward target. Contrast with front vowel.",
    },
    "VOWEL_SOUNDS_LIKE_CUP": {
        "cue":       "Your vowel sounds more like the short U in CUP — it's too central and relaxed.",
        "exercise":  "Your tongue is sitting in a neutral resting position. Move it deliberately — higher, lower, or to the front or back — depending on which sound you're aiming for.",
        "analogy":   None,
        "clinician": "Measured F1/F2 close to /ʌ/ region. Tongue body in mid-central position. Schwa-like neutralisation — patient may be reducing vowel rather than targeting. Ensure patient is aware of the target vowel identity.",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # POSITIVE CONFIRMATIONS (used for praise)
    # ══════════════════════════════════════════════════════════════════════════

    "CORRECT_SOUND": {
        "cue":       "That sound was exactly right — well done.",
        "exercise":  None,
        "analogy":   None,
        "clinician": "Target phoneme produced within acceptable acoustic range.",
    },
    "CORRECT_VOWEL": {
        "cue":       "Your vowel sounded great — tongue position was right on target.",
        "exercise":  None,
        "analogy":   None,
        "clinician": "F1/F2 within target range. Formant confidence acceptable.",
    },
    "CORRECT_SIBILANT": {
        "cue":       "Your S was clear and sharp — that's exactly how it should sound.",
        "exercise":  None,
        "analogy":   None,
        "clinician": "Spectral CoG within normal range for /s/. No lisp indicators present.",
    },
    "CORRECT_WORD": {
        "cue":       "Every sound in that word was correct — that's a solid production.",
        "exercise":  None,
        "analogy":   None,
        "clinician": "All target phonemes produced within acceptable acoustic and phonemic range.",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # HEADLINES — overall session result (score-based)
    # ══════════════════════════════════════════════════════════════════════════

    "HEADLINE_PERFECT":    {"cue": "Excellent — every sound was right on target.",              "exercise": None, "analogy": None, "clinician": "Score ≥ 0.95. All phonemes within acceptable acoustic range."},
    "HEADLINE_VERY_GOOD":  {"cue": "Nearly there — just one small adjustment to make.",         "exercise": None, "analogy": None, "clinician": "Score 0.80–0.95."},
    "HEADLINE_GOOD":       {"cue": "Good work — there are a couple of sounds to focus on.",     "exercise": None, "analogy": None, "clinician": "Score 0.60–0.80."},
    "HEADLINE_KEEP_GOING": {"cue": "You're making progress — keep working on it.",              "exercise": None, "analogy": None, "clinician": "Score 0.40–0.60."},
    "HEADLINE_TRY_AGAIN":  {"cue": "This one takes time — have another go and focus on one sound at a time.", "exercise": None, "analogy": None, "clinician": "Score < 0.40."},
    "HEADLINE_WRONG_WORD": {"cue": "I didn't quite catch that — try again, speaking clearly.", "exercise": None, "analogy": None, "clinician": "ASR word recognition failed. Check recording quality and patient intelligibility."},
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
