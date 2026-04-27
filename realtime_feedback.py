"""
SpeakFlow — Flask backend
Run:  python3 realtime_feedback.py
Open: http://localhost:5001

Pipeline per request:
  audio → Whisper (what was said)
        → espeak-ng en-au (Australian IPA)
        → phoneme alignment (equal-division approximation)
        → parselmouth/Praat (F1/F2 formants, frication CoG, lisp detection)
        → clinical feedback strings
"""

import difflib
import os
import shutil
import subprocess
import tempfile
import traceback

from flask import Flask, request, jsonify, send_from_directory, Response, send_file
from flask_cors import CORS

# ── Acoustic analysis (parselmouth / Praat) ───────────────────────────────────
ACOUSTIC_OK = False
try:
    from acoustic_analysis import (
        analyse_segment, analyse_full_recording, result_to_dict,
        VOWELS, SIBILANTS, VOWEL_REFS,
    )
    ACOUSTIC_OK = True
    print("  ✓ Acoustic analysis (parselmouth) ready")
except Exception as _ae:
    print(f"  ✗ Acoustic analysis not available: {_ae}")

# ── Tongue position detector ─────────────────────────────────────────────────
try:
    import tongue_detector as _td
    TONGUE_OK = True
    print("  ✓ Tongue detector ready")
except Exception as _te:
    TONGUE_OK = False
    print(f"  ✗ Tongue detector: {_te}")

# ── Session history (SQLite) ──────────────────────────────────────────────────
try:
    import session_store as _ss
    _ss.init_db()
    SESSION_OK = True
    print("  ✓ Session store ready")
except Exception as _se:
    SESSION_OK = False
    print(f"  ✗ Session store: {_se}")

# helper used in /check to extract and persist phoneme contexts
def _save_phoneme_contexts(word, alignment, acoustic_segs, context_type):
    if not SESSION_OK:
        return
    try:
        recs = _ss.contexts_from_alignment(word, alignment, acoustic_segs, context_type)
        for r in recs:
            _ss.save_phoneme_context(r)
    except Exception as e:
        print(f"  [context save] {e}")

# ── Clinician reference store ─────────────────────────────────────────────────
CLINICIAN_PASSWORD = os.environ.get("SPEAKFLOW_PASSWORD", "speakflow2024")
REF_OK = False
try:
    import reference_store as _ref
    REF_OK = True
    print("  ✓ Reference store ready")
except Exception as _re:
    print(f"  ✗ Reference store: {_re}")

# ── Feedback + report engines ─────────────────────────────────────────────────
try:
    from feedback_generator import generate as _generate_feedback
    import report_engine as _report_engine
    FEEDBACK_OK = True
    _api_status = "with AI" if _report_engine.API_AVAILABLE else "template-only"
    print(f"  ✓ Feedback generator ready ({_api_status})")
except Exception as _fg:
    FEEDBACK_OK = False
    print(f"  ✗ Feedback generator: {_fg}")

# ── Articulator engine ────────────────────────────────────────────────────────
ARTICULATOR_OK = False
try:
    from articulator_engine import analyse_to_dict as _articulator_analyse
    ARTICULATOR_OK = True
    print("  ✓ Articulator engine ready")
except Exception as _ae2:
    print(f"  ✗ Articulator engine: {_ae2}")

# ── Lip analysis ─────────────────────────────────────────────────────────────
try:
    from lip_analysis import analyse_lip as _analyse_lip, cross_reference as _cross_ref
    LIP_ANALYSIS_OK = True
    print("  ✓ Lip analysis ready")
except Exception as _la:
    LIP_ANALYSIS_OK = False
    print(f"  ✗ Lip analysis: {_la}")

# ── Forced alignment ──────────────────────────────────────────────────────────
ALIGN_OK = False
try:
    from forced_align import align_word as _align_word
    ALIGN_OK = True
    print("  ✓ Forced alignment ready (WebMAUS → Praat → equal)")
except Exception as _fa:
    print(f"  ✗ Forced alignment not available: {_fa}")

# ── espeak-ng ─────────────────────────────────────────────────────────────────
_ESPEAK = (
    shutil.which("espeak-ng")
    or "/opt/homebrew/bin/espeak-ng"
    or "/usr/bin/espeak-ng"
)
ESPEAK_OK = bool(_ESPEAK and os.path.isfile(_ESPEAK))
print(f"  {'✓' if ESPEAK_OK else '✗'} espeak-ng: {_ESPEAK or 'not found'}")


def _raw_ipa(text: str) -> str:
    if not text.strip() or not ESPEAK_OK:
        return ""
    try:
        r = subprocess.run(
            [_ESPEAK, "--ipa", "-v", "en-au", "--", text],
            capture_output=True, text=True, timeout=15,
        )
        return r.stdout.strip()
    except Exception as e:
        print(f"  espeak error: {e}")
        return ""


def _australianise(ipa: str) -> str:
    """Post-process espeak IPA → General Australian English."""
    ipa = ipa.replace("eɪ", "æɪ")   # FACE:  RP eɪ  → AuE æɪ
    ipa = ipa.replace("aɪ", "ɑɪ")   # PRICE: RP aɪ  → AuE ɑɪ
    ipa = ipa.replace("aʊ", "æɔ")   # MOUTH: RP aʊ  → AuE æɔ
    return ipa


def _ipa(text: str) -> str:
    return _australianise(_raw_ipa(text))


# ── Whisper ───────────────────────────────────────────────────────────────────
WHISPER_OK = False
try:
    import whisper as _whisper
    print("Loading Whisper (small) …")
    _whisper_model = _whisper.load_model("small")
    WHISPER_OK = True
    print("  ✓ Whisper ready")
except Exception as _e:
    print(f"  ✗ Whisper: {_e}")

# ── IPA parser ────────────────────────────────────────────────────────────────
_DIGRAPHS = [
    "tʃ", "dʒ",
    "æɪ", "ɑɪ", "æɔ",                       # Australian diphthongs
    "aɪ", "aʊ", "eɪ", "ɔɪ", "əʊ", "oʊ",
    "ɪə", "eə", "ʊə",
    "uː", "iː", "ɑː", "ɔː", "ɜː",
]
_SKIP = set("ˈˌ|‖. \n\t")


def _parse_ipa(s: str) -> list:
    tokens, i = [], 0
    while i < len(s):
        if s[i] in _SKIP:
            i += 1
            continue
        matched = False
        for dg in _DIGRAPHS:
            if s[i: i + len(dg)] == dg:
                tokens.append(dg)
                i += len(dg)
                matched = True
                break
        if not matched:
            tokens.append(s[i])
            i += 1
    return tokens


# ── phoneme comparison ────────────────────────────────────────────────────────
def _compare(target: list, heard: list) -> tuple:
    """
    Align two phoneme lists with SequenceMatcher.
    Returns (alignment, score) where alignment is a list of
    { target, heard, status: 'correct'|'wrong'|'missing'|'extra' }.
    """
    sm = difflib.SequenceMatcher(None, target, heard, autojunk=False)
    alignment = []
    matched = 0

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for i in range(i1, i2):
                alignment.append({"target": target[i], "heard": target[i], "status": "correct"})
                matched += 1
        elif tag == "replace":
            t_seg = target[i1:i2]
            h_seg = heard[j1:j2]
            for t, h in zip(t_seg, h_seg):
                alignment.append({"target": t, "heard": h, "status": "wrong"})
            for t in t_seg[len(h_seg):]:
                alignment.append({"target": t, "heard": "", "status": "missing"})
            for h in h_seg[len(t_seg):]:
                alignment.append({"target": "", "heard": h, "status": "extra"})
        elif tag == "delete":
            for t in target[i1:i2]:
                alignment.append({"target": t, "heard": "", "status": "missing"})
        elif tag == "insert":
            for h in heard[j1:j2]:
                alignment.append({"target": "", "heard": h, "status": "extra"})

    total = max(len(target), len(heard), 1)
    return alignment, round(matched / total, 3)


# ── exercise catalogue ────────────────────────────────────────────────────────
_RAW = [
    # word            category              cat_id   emoji  hint
    ("mate",      "FACE /æɪ/",         "face",  "🇦🇺", "The most iconic Australian word! Starts more open than RP."),
    ("day",       "FACE /æɪ/",         "face",  "🇦🇺", "Short word, great for isolating the diphthong."),
    ("face",      "FACE /æɪ/",         "face",  "🇦🇺", ""),
    ("late",      "FACE /æɪ/",         "face",  "🇦🇺", ""),
    ("rain",      "FACE /æɪ/",         "face",  "🇦🇺", ""),
    ("brain",     "FACE /æɪ/",         "face",  "🇦🇺", ""),
    ("plane",     "FACE /æɪ/",         "face",  "🇦🇺", ""),
    ("game",      "FACE /æɪ/",         "face",  "🇦🇺", ""),
    ("train",     "FACE /æɪ/",         "face",  "🇦🇺", ""),
    ("same",      "FACE /æɪ/",         "face",  "🇦🇺", ""),

    ("my",        "PRICE /ɑɪ/",        "price", "💰", "Starts further back than British English."),
    ("time",      "PRICE /ɑɪ/",        "price", "💰", ""),
    ("night",     "PRICE /ɑɪ/",        "price", "💰", ""),
    ("right",     "PRICE /ɑɪ/",        "price", "💰", ""),
    ("smile",     "PRICE /ɑɪ/",        "price", "💰", ""),
    ("bright",    "PRICE /ɑɪ/",        "price", "💰", ""),
    ("drive",     "PRICE /ɑɪ/",        "price", "💰", ""),
    ("mind",      "PRICE /ɑɪ/",        "price", "💰", ""),

    ("now",       "MOUTH /æɔ/",        "mouth", "👄", "Starts with /æ/ (like 'cat'), rounds to /ɔ/."),
    ("how",       "MOUTH /æɔ/",        "mouth", "👄", ""),
    ("out",       "MOUTH /æɔ/",        "mouth", "👄", ""),
    ("loud",      "MOUTH /æɔ/",        "mouth", "👄", ""),
    ("house",     "MOUTH /æɔ/",        "mouth", "👄", ""),
    ("brown",     "MOUTH /æɔ/",        "mouth", "👄", ""),
    ("found",     "MOUTH /æɔ/",        "mouth", "👄", ""),
    ("around",    "MOUTH /æɔ/",        "mouth", "👄", ""),

    ("think",     "TH voiceless /θ/",  "theta", "🦷", "Tongue tip gently between teeth, breathe out."),
    ("three",     "TH voiceless /θ/",  "theta", "🦷", ""),
    ("bath",      "TH voiceless /θ/",  "theta", "🦷", "TH at the end — don't add voice."),
    ("tooth",     "TH voiceless /θ/",  "theta", "🦷", ""),
    ("throw",     "TH voiceless /θ/",  "theta", "🦷", ""),
    ("through",   "TH voiceless /θ/",  "theta", "🦷", ""),
    ("thumb",     "TH voiceless /θ/",  "theta", "🦷", ""),
    ("thousand",  "TH voiceless /θ/",  "theta", "🦷", ""),

    ("this",      "TH voiced /ð/",     "eth",   "🔊", "Same position as /θ/ — add your voice."),
    ("then",      "TH voiced /ð/",     "eth",   "🔊", ""),
    ("breathe",   "TH voiced /ð/",     "eth",   "🔊", "Voiced at the end — feel the buzz."),
    ("brother",   "TH voiced /ð/",     "eth",   "🔊", ""),
    ("mother",    "TH voiced /ð/",     "eth",   "🔊", ""),
    ("weather",   "TH voiced /ð/",     "eth",   "🔊", ""),
    ("together",  "TH voiced /ð/",     "eth",   "🔊", ""),
    ("smooth",    "TH voiced /ð/",     "eth",   "🔊", ""),

    ("run",       "R sound /ɹ/",       "r",     "🌀", "Raise the body of your tongue — don't curl the tip."),
    ("rain",      "R sound /ɹ/",       "r",     "🌀", ""),
    ("red",       "R sound /ɹ/",       "r",     "🌀", ""),
    ("street",    "R sound /ɹ/",       "r",     "🌀", ""),
    ("three",     "R sound /ɹ/",       "r",     "🌀", ""),
    ("draw",      "R sound /ɹ/",       "r",     "🌀", ""),
    ("proud",     "R sound /ɹ/",       "r",     "🌀", ""),
    ("drink",     "R sound /ɹ/",       "r",     "🌀", ""),

    ("arvo",      "Aussie Slang",       "slang", "🦘", "Afternoon — a classic Australian shortening."),
    ("barbie",    "Aussie Slang",       "slang", "🦘", "Barbecue — not the doll!"),
    ("brekkie",   "Aussie Slang",       "slang", "🦘", "Breakfast."),
    ("servo",     "Aussie Slang",       "slang", "🦘", "Service station / petrol station."),
    ("footie",    "Aussie Slang",       "slang", "🦘", "Australian Rules Football."),
    ("cuppa",     "Aussie Slang",       "slang", "🦘", "Cup of tea."),
    ("sanga",     "Aussie Slang",       "slang", "🦘", "Sandwich."),
    ("rego",      "Aussie Slang",       "slang", "🦘", "Vehicle registration."),
]

print("Pre-computing exercise IPA …")
EXERCISES = []
for word, category, cat_id, emoji, hint in _RAW:
    w_ipa      = _ipa(word)
    w_phonemes = _parse_ipa(w_ipa)
    EXERCISES.append({
        "word":      word,
        "category":  category,
        "cat_id":    cat_id,
        "emoji":     emoji,
        "hint":      hint,
        "ipa":       w_ipa,
        "phonemes":  w_phonemes,
    })
print(f"  ✓ {len(EXERCISES)} exercise words ready")

# ── Sentence exercises ────────────────────────────────────────────────────────
# Each entry: (sentence, category, cat_id, emoji, hint, focus_phoneme)
_SENTENCES = [
    # /s/ — lisp practice
    ("Sally sees six silly seals",            "Sentences /s/",  "sent_s", "💬", "Keep tongue BEHIND teeth for every S sound", "s"),
    ("Sue sells seashells by the seashore",   "Sentences /s/",  "sent_s", "💬", "Classic tongue-tip practice sentence",       "s"),
    ("Sam and Sara sat by the sunny sea",     "Sentences /s/",  "sent_s", "💬", "Nice slow pace — feel each S clearly",       "s"),
    ("Suzie saw some small stars in the sky", "Sentences /s/",  "sent_s", "💬", "Focus on making each S crisp and forward",   "s"),
    ("The bees and roses buzz in summer",     "Sentences /z/",  "sent_z", "💬", "Voiced S — add buzz to your S sounds",       "z"),
    ("The lazy lizards dozed on the rocks",   "Sentences /z/",  "sent_z", "💬", "Feel the vibration in your throat for Z",    "z"),
    # /r/
    ("Robert rode the red racing car",        "Sentences /r/",  "sent_r", "🌀", "Raise the back of your tongue for R",        "ɹ"),
    ("Rain poured on the river road",         "Sentences /r/",  "sent_r", "🌀", "No lip rounding — just tongue body up",      "ɹ"),
    ("Red roses grew around the railing",     "Sentences /r/",  "sent_r", "🌀", "Every R: tongue up, no curl",                "ɹ"),
    ("The brown rabbit ran really fast",      "Sentences /r/",  "sent_r", "🌀", "Cluster practice: br, tr, fr",               "ɹ"),
    # /θ/ voiceless TH
    ("Think about three things thoroughly",   "Sentences /θ/",  "sent_th","🦷", "Tongue tip between teeth — breathe out",     "θ"),
    ("Three thick thorns pierced the cloth",  "Sentences /θ/",  "sent_th","🦷", "Feel the air flowing over your tongue tip",  "θ"),
    ("Matthew thought about the path north",  "Sentences /θ/",  "sent_th","🦷", "TH at the end of words — don't drop it",    "θ"),
    # /ð/ voiced TH
    ("The other brother breathed deeply",     "Sentences /ð/",  "sent_dh","🔊", "Same position as TH — add your voice",       "ð"),
    ("Mother and father walked together",     "Sentences /ð/",  "sent_dh","🔊", "Feel the buzz between your teeth",           "ð"),
    ("Those three brothers think together",   "Sentences /ð/",  "sent_dh","🔊", "Mix of voiced and voiceless TH",             "ð"),
    # /l/
    ("The little lamb liked lollipops",       "Sentences /l/",  "sent_l", "💡", "Tongue tip on the ridge — not between teeth","l"),
    ("Laura loves long lazy lunches",         "Sentences /l/",  "sent_l", "💡", "Feel tongue tip bounce for each L",          "l"),
    ("Billy fell off the yellow hill",        "Sentences /l/",  "sent_l", "💡", "Final L — tongue up at the end",             "l"),
    # FACE /æɪ/
    ("Today is a great day to play",          "Sentences FACE /æɪ/", "sent_face","🇦🇺","Start open, glide to EE — Australian FACE","æɪ"),
    ("The train came late on a rainy day",    "Sentences FACE /æɪ/", "sent_face","🇦🇺","Every AY word: jaw open, then glide",      "æɪ"),
    ("James waited at the gate all day",      "Sentences FACE /æɪ/", "sent_face","🇦🇺","AY in words and at sentence end",          "æɪ"),
    # General fluency
    ("The quick brown fox jumps over the lazy dog", "Sentences: Fluency","sent_flu","🗣️","Every letter of the alphabet — great warm-up","s"),
    ("How much wood would a woodchuck chuck", "Sentences: Fluency","sent_flu","🗣️","Classic tongue twister — go slow first",    "ʌ"),
    ("Red lorry yellow lorry",                "Sentences: Fluency","sent_flu","🗣️","Repeat 5 times — speed up gradually",       "l"),
]

print("Pre-computing sentence IPA …")
for sentence, category, cat_id, emoji, hint, focus_ph in _SENTENCES:
    sent_ipa = _ipa(sentence)
    EXERCISES.append({
        "word":          sentence,       # repurposed as the sentence text
        "sentence":      sentence,       # explicit marker
        "type":          "sentence",
        "category":      category,
        "cat_id":        cat_id,
        "emoji":         emoji,
        "hint":          hint,
        "focus_phoneme": focus_ph,
        "ipa":           sent_ipa,
        "phonemes":      _parse_ipa(sent_ipa),
    })
print(f"  ✓ {len(_SENTENCES)} sentence exercises ready  ({len(EXERCISES)} total)")

# ── Flask ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "realtime_feedback.html")


@app.route("/health")
def health():
    return jsonify({
        "status":   "ok",
        "whisper":  WHISPER_OK,
        "espeak":   ESPEAK_OK,
        "acoustic": ACOUSTIC_OK,
        "aligner":  ALIGN_OK,
    })


@app.route("/exercises")
def exercises():
    return jsonify(EXERCISES)


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio field"}), 400
    if not WHISPER_OK:
        return jsonify({"error": "Whisper not loaded"}), 503

    audio_file = request.files["audio"]
    ext = os.path.splitext(audio_file.filename or "")[1] or ".webm"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = _whisper_model.transcribe(tmp_path, language="en")
        text = result["text"].strip()
        if not text:
            return jsonify({"text": "", "ipa": "", "phonemes": [], "words": []})

        full_ipa  = _ipa(text)
        word_data = []
        for raw_word in text.split():
            clean = "".join(c for c in raw_word if c.isalpha() or c == "'")
            if not clean:
                continue
            w_ipa = _ipa(clean)
            word_data.append({"word": raw_word, "ipa": w_ipa, "phonemes": _parse_ipa(w_ipa)})

        return jsonify({"text": text, "ipa": full_ipa,
                        "phonemes": _parse_ipa(full_ipa), "words": word_data})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.route("/check", methods=["POST"])
def check():
    """
    POST multipart: 'audio' file + 'target' form field (the word to practise).

    Returns:
      heard, heard_ipa, target_ipa, alignment, score, word_correct
      acoustic: { segments: [...], clinical_summary: [...] }   ← new
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio"}), 400
    if not WHISPER_OK:
        return jsonify({"error": "Whisper not loaded"}), 503

    target_word = request.form.get("target", "").strip().lower()
    if not target_word:
        return jsonify({"error": "No target word"}), 400

    audio_file = request.files["audio"]
    ext = os.path.splitext(audio_file.filename or "")[1] or ".webm"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # ── 1. Whisper transcription ──────────────────────────────────────────
        result = _whisper_model.transcribe(tmp_path, language="en")
        heard  = result["text"].strip()

        heard_words = [w.strip(".,!?\"'").lower() for w in heard.split() if w.strip(".,!?\"'")]
        best_heard, best_ratio = "", 0.0
        for w in heard_words:
            r = difflib.SequenceMatcher(None, target_word, w).ratio()
            if r > best_ratio:
                best_ratio, best_heard = r, w

        # ── 2. IPA + phoneme alignment ────────────────────────────────────────
        target_ipa = _ipa(target_word)
        heard_ipa  = _ipa(best_heard) if best_heard else ""
        target_ph  = _parse_ipa(target_ipa)
        heard_ph   = _parse_ipa(heard_ipa)
        alignment, score = _compare(target_ph, heard_ph)
        word_ok    = best_ratio >= 0.75

        # Persist phoneme-level context (populated before acoustic so segs may be empty here;
        # we enrich after acoustic analysis below)
        _ctx_word = target_word
        _ctx_type = "word"

        # ── 3. Forced alignment + acoustic analysis ───────────────────────────
        acoustic = None
        if ACOUSTIC_OK and target_ph:
            try:
                # Get phoneme boundaries (WebMAUS → Praat → equal fallback)
                align_method = "equal"
                if ALIGN_OK:
                    boundaries = _align_word(tmp_path, target_word, target_ph, verbose=False)
                    align_method = boundaries[0]["method"] if boundaries else "equal"
                else:
                    # No aligner — equal division via duration
                    import parselmouth as _pm
                    _dur = _pm.Sound(tmp_path).duration * 1000
                    _step = _dur / len(target_ph) if target_ph else 0
                    boundaries = [
                        {"ipa": ph, "start_ms": i * _step,
                         "end_ms": (i + 1) * _step, "method": "equal"}
                        for i, ph in enumerate(target_ph)
                    ]

                # Load clinician reference if one exists for this word
                ref_entry    = _ref.get(target_word) if REF_OK else None
                ref_vowels   = (ref_entry or {}).get("vowel_refs",    {})
                ref_frictions= (ref_entry or {}).get("friction_refs", {})

                # Patient calibration — scale textbook refs to match vocal tract
                import json as _json2
                _calib_raw = request.form.get("calibration", "{}")
                _calib     = _json2.loads(_calib_raw) if _calib_raw else {}
                _scale_F1  = float(_calib.get("scaleF1", 1.0))
                _scale_F2  = float(_calib.get("scaleF2", 1.0))

                # Run Praat feature extraction on each aligned segment
                raw_segs = []
                for b in boundaries:
                    ph = b["ipa"]
                    # Priority: clinician ref > calibrated textbook > raw textbook
                    if ph in ref_vowels:
                        v_ref = ref_vowels[ph]   # clinician's voice — no extra scaling
                    elif (_scale_F1 != 1.0 or _scale_F2 != 1.0) and ph in VOWEL_REFS:
                        base  = VOWEL_REFS[ph]
                        v_ref = {"F1": base["F1"] * _scale_F1,
                                 "F2": base["F2"] * _scale_F2,
                                 "label": base["label"]}
                    else:
                        v_ref = None

                    seg = analyse_segment(
                        tmp_path, ph,
                        start_ms=b["start_ms"],
                        end_ms=b["end_ms"],
                        vowel_ref_override=v_ref,
                    )
                    d = result_to_dict(seg)
                    d["align_method"] = b["method"]
                    raw_segs.append(d)

                # Clinical summary (only phonemes with interesting data)
                summary = []
                for b, d in zip(boundaries, raw_segs):
                    ph = b["ipa"]
                    if "vowel" in d:
                        v = d["vowel"]
                        line = (
                            f"/{ph}/ ({v['label']}): "
                            f"F1={v['F1_hz']:.0f} Hz (ref {v['F1_ref']:.0f}), "
                            f"F2={v['F2_hz']:.0f} Hz (ref {v['F2_ref']:.0f})"
                        )
                        if v["jaw_deviation"] != "on target":
                            line += f" — {v['jaw_deviation']}"
                        if v["back_deviation"] != "on target":
                            line += f", {v['back_deviation']}"
                        summary.append(line)
                    if "friction" in d:
                        f = d["friction"]
                        line = (
                            f"/{ph}/ ({f['label']}): "
                            f"CoG={f['CoG_hz']:.0f} Hz (ref {f['CoG_ref']:.0f})"
                        )
                        if f["lisp_risk"] != "low":
                            line += f" ⚠️ Lisp risk [{f['lisp_risk'].upper()}]"
                        summary.append(line)

                acoustic = {
                    "segments":          raw_segs,
                    "clinical_summary":  summary,
                    "align_method":      align_method,
                    "using_reference":   bool(ref_entry),
                    "reference_date":    (ref_entry or {}).get("recorded_at", ""),
                    "using_calibration": (_scale_F1 != 1.0 or _scale_F2 != 1.0),
                    "scale_F1":          round(_scale_F1, 3),
                    "scale_F2":          round(_scale_F2, 3),
                    "available":         True,
                }
            except Exception as ae:
                print(f"  acoustic analysis error: {ae}")
                acoustic = {"available": False, "error": str(ae)}

        # ── 4. Lip confirmatory signal ────────────────────────────────────────
        lip_result = None
        if LIP_ANALYSIS_OK:
            try:
                import json as _json
                raw_lip = request.form.get("lip_data", "")
                if raw_lip:
                    lip_data = _json.loads(raw_lip)
                    # Focus on the first vowel in the target word
                    _vowel_set = {"iː","ɪ","e","æ","ɑː","ɒ","ɔː","ʊ","uː","ʌ","ɜː","ə",
                                  "æɪ","ɑɪ","æɔ","əʊ","ɔɪ","a","ɑ","ɔ","ɐ","ɛ"}
                    vowel_phs = [p for p in target_ph if p in _vowel_set]
                    focus_ph  = vowel_phs[0] if vowel_phs else (target_ph[0] if target_ph else "")
                    if focus_ph:
                        lip_result = _analyse_lip(lip_data, focus_ph)
                        lip_result["focus_phoneme"] = focus_ph
                        if acoustic and acoustic.get("available") and lip_result.get("available"):
                            xref = _cross_ref(lip_result, acoustic.get("segments", []), focus_ph)
                            lip_result["cross_reference"] = xref
            except Exception as le:
                print(f"  lip analysis error: {le}")

        payload = {
            "heard":           heard,
            "heard_word":      best_heard,
            "heard_ipa":       heard_ipa,
            "target_ipa":      target_ipa,
            "target_phonemes": target_ph,
            "heard_phonemes":  heard_ph,
            "alignment":       alignment,
            "score":           score,
            "word_correct":    word_ok,
        }
        if acoustic:    payload["acoustic"] = acoustic
        if lip_result:  payload["lip"]      = lip_result

        # ── Articulator engine — structured clinical finding ──────────────────
        if ARTICULATOR_OK and acoustic and acoustic.get("available"):
            try:
                for seg in acoustic.get("segments", []):
                    ph   = seg.get("phoneme", "")
                    fric = seg.get("friction")
                    vow  = seg.get("vowel")
                    if fric or vow:
                        art = _articulator_analyse(
                            phoneme    = ph,
                            frication  = fric,
                            vowel      = vow,
                            lip        = lip_result,
                            confidence = (vow or {}).get("confidence", 1.0),
                        )
                        if art:
                            payload["articulator"] = art
                            break
            except Exception as _ae3:
                print(f"  articulator engine: {_ae3}")

        # Save phoneme co-occurrence data (with acoustic enrichment)
        _save_phoneme_contexts(
            _ctx_word, alignment,
            acoustic.get("segments", []) if acoustic else [],
            _ctx_type,
        )

        # ── 5. Plain-English patient feedback ─────────────────────────────────
        if FEEDBACK_OK:
            try:
                payload["feedback"] = _generate_feedback(
                    target_word          = target_word,
                    target_phonemes      = target_ph,
                    alignment            = alignment,
                    acoustic             = acoustic or {},
                    lip                  = lip_result or {},
                    score                = score,
                    word_correct         = word_ok,
                    articulator_finding  = payload.get("articulator"),
                )
            except Exception as fe:
                print(f"  feedback generator error: {fe}")

        return jsonify(payload)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.route("/acoustic", methods=["POST"])
def acoustic_endpoint():
    """
    Standalone acoustic analysis — used by the Phoneme tab.
    POST multipart: 'audio' + 'phoneme' (IPA symbol).
    Returns full parselmouth features for that phoneme.
    """
    if not ACOUSTIC_OK:
        return jsonify({"error": "Acoustic analysis not available", "available": False}), 503
    if "audio" not in request.files:
        return jsonify({"error": "No audio field"}), 400

    phoneme = request.form.get("phoneme", "").strip()
    if not phoneme:
        return jsonify({"error": "No phoneme specified"}), 400

    audio_file = request.files["audio"]
    ext = os.path.splitext(audio_file.filename or "")[1] or ".webm"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        seg = analyse_full_recording(tmp_path, phoneme)
        return jsonify({"available": True, **result_to_dict(seg)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.route("/play")
def play_audio():
    """
    GET /play?text=<word or sentence>
    Returns WAV audio.  Priority: clinician reference recording > espeak-ng TTS.
    Sets X-Audio-Source header: 'reference' | 'espeak'
    """
    text = request.args.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text"}), 400

    # 1 — clinician reference (single words only)
    if REF_OK and " " not in text:
        wav = _ref.wav_path(text.lower())
        if wav:
            resp = send_file(wav, mimetype="audio/wav")
            resp.headers["X-Audio-Source"] = "reference"
            resp.headers["Cache-Control"]   = "max-age=3600"
            return resp

    # 2 — espeak-ng synthesis
    if not ESPEAK_OK:
        return jsonify({"error": "No audio source available"}), 503

    # Slightly slower for sentences so they're easier to follow
    speed = "140" if " " in text else "155"
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    try:
        subprocess.run(
            [_ESPEAK, "-v", "en-au", "-s", speed, "-w", tmp, "--", text],
            capture_output=True, check=True, timeout=10,
        )
        with open(tmp, "rb") as f:
            audio_bytes = f.read()
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass

    resp = Response(audio_bytes, mimetype="audio/wav")
    resp.headers["X-Audio-Source"] = "espeak"
    resp.headers["Cache-Control"]  = "max-age=3600"
    return resp


@app.route("/calibrate_word", methods=["POST"])
def calibrate_word():
    """
    Record one calibration word and extract its formants.
    POST: 'audio' file + 'phoneme' IPA symbol of the target vowel.
    Returns: { F1, F2, confidence, F1_ref, F2_ref, phoneme }
    The frontend uses the ratio (F1/F1_ref) to build the scale factor.
    """
    if not ACOUSTIC_OK:
        return jsonify({"error": "Acoustic analysis unavailable"}), 503
    if "audio" not in request.files:
        return jsonify({"error": "No audio"}), 400

    phoneme = request.form.get("phoneme", "").strip()
    if not phoneme or phoneme not in VOWEL_REFS:
        return jsonify({"error": f"Unknown phoneme '{phoneme}'"}), 400

    audio_file = request.files["audio"]
    ext = os.path.splitext(audio_file.filename or "")[1] or ".webm"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        seg = analyse_full_recording(tmp_path, phoneme)
        if not seg.vowel:
            return jsonify({"error": "No vowel detected — speak closer to the mic and hold the vowel sound for 1–2 seconds"}), 422

        ref = VOWEL_REFS[phoneme]
        return jsonify({
            "phoneme":    phoneme,
            "F1":         round(seg.vowel.F1_hz, 0),
            "F2":         round(seg.vowel.F2_hz, 0),
            "confidence": round(seg.vowel.confidence, 2),
            "F1_ref":     ref["F1"],
            "F2_ref":     ref["F2"],
            "label":      ref["label"],
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.route("/attempt", methods=["POST"])
def save_attempt():
    """
    Frontend calls this after every exercise attempt to persist the result.
    Accepts JSON body matching the /check response shape.
    """
    if not SESSION_OK:
        return jsonify({"ok": False, "reason": "session store unavailable"})
    try:
        data = request.get_json(force=True, silent=True) or {}
        if not data.get("word"):
            return jsonify({"ok": False, "reason": "no word"}), 400

        # Extract first vowel segment for acoustic columns
        segs     = (data.get("acoustic") or {}).get("segments", [])
        v_seg    = next((s for s in segs if "vowel"    in s), None)
        f_seg    = next((s for s in segs if "friction" in s), None)

        row = {
            "word":        data.get("word", ""),
            "target_ipa":  data.get("target_ipa", ""),
            "heard":       data.get("heard", ""),
            "heard_ipa":   data.get("heard_ipa", ""),
            "score":       data.get("score"),
            "word_correct":data.get("word_correct"),
            "using_ref":   (data.get("acoustic") or {}).get("using_reference", False),
            "align_method":(data.get("acoustic") or {}).get("align_method", ""),
        }
        if v_seg:
            v = v_seg["vowel"]
            row.update({
                "phoneme": v_seg["phoneme"],
                "f1_hz":   v.get("F1_hz"),
                "f2_hz":   v.get("F2_hz"),
                "f1_ref":  v.get("F1_ref"),
                "f2_ref":  v.get("F2_ref"),
                "f1_diff": v.get("F1_diff"),
                "f2_diff": v.get("F2_diff"),
            })
        if f_seg:
            f = f_seg["friction"]
            row.update({
                "phoneme":   row.get("phoneme") or f_seg["phoneme"],
                "cog_hz":    f.get("CoG_hz"),
                "lisp_risk": f.get("lisp_risk"),
            })

        new_id = _ss.save(row)
        return jsonify({"ok": True, "id": new_id})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "reason": str(e)}), 500


@app.route("/history")
def history():
    if not SESSION_OK:
        return jsonify({"error": "session store unavailable"}), 503
    days  = int(request.args.get("days", 7))
    return jsonify(_ss.summary(days))


@app.route("/history/recent")
def history_recent():
    if not SESSION_OK:
        return jsonify([])
    limit = int(request.args.get("limit", 60))
    return jsonify(_ss.recent(limit))


@app.route("/report.html")
def report_html():
    """Printable weekly progress report — generated by report_engine."""
    if not SESSION_OK:
        return "<h2>Session store unavailable</h2>", 503
    days     = int(request.args.get("days", 7))
    patient  = request.args.get("patient", "Patient")
    s        = _ss.summary(days)

    if FEEDBACK_OK:
        md = _report_engine.generate_report(s, patient_id=patient)
    else:
        md = f"# SpeakFlow Report\n\nTotal attempts: {s['total']}\nAccuracy: {s['pct']}%"

    # Convert Markdown to simple HTML (no external library needed)
    import html as _html
    lines, html_lines = md.split("\n"), []
    for line in lines:
        if line.startswith("# "):
            html_lines.append(f"<h1>{_html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{_html.escape(line[3:])}</h2>")
        elif line.startswith("- **"):
            # Bold item
            content = line[2:]
            content = content.replace("**", "<strong>", 1).replace("**", "</strong>", 1)
            html_lines.append(f"<li>{content}</li>")
        elif line.startswith("- "):
            html_lines.append(f"<li>{_html.escape(line[2:])}</li>")
        elif line.startswith("*") and line.endswith("*"):
            html_lines.append(f"<p style='color:#94a3b8;font-size:.8rem'>{_html.escape(line.strip('*'))}</p>")
        elif line.strip():
            html_lines.append(f"<p>{_html.escape(line)}</p>")

    body = "\n".join(html_lines)

    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>SpeakFlow Report — {_html.escape(patient)}</title>
<style>
  body{{font-family:'Segoe UI',sans-serif;margin:32px;color:#1a1a2e;max-width:800px;}}
  h1{{font-size:1.5rem;margin-bottom:4px;}}
  h2{{font-size:1.1rem;margin-top:24px;margin-bottom:8px;color:#4b5563;border-bottom:1px solid #e5e7eb;padding-bottom:4px;}}
  p{{margin:6px 0;font-size:.9rem;line-height:1.6;}}
  li{{font-size:.88rem;line-height:1.7;margin-left:20px;}}
  @media print{{body{{margin:16px;}} .no-print{{display:none;}}}}
</style></head><body>
{body}
<div class="no-print" style="margin-top:28px">
  <button onclick="window.print()"
    style="padding:10px 20px;border-radius:8px;border:none;background:#7c3aed;color:#fff;font-size:.9rem;cursor:pointer;">
    🖨 Print / Save as PDF
  </button>
</div>
</body></html>""", 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/check_sentence", methods=["POST"])
def check_sentence():
    """
    Compare a patient's sentence recording to the target sentence.
    POST: 'audio' + 'target' (sentence text) + 'focus_phoneme' (IPA)
    Returns word-level alignment + acoustic analysis on focus phoneme.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio"}), 400
    if not WHISPER_OK:
        return jsonify({"error": "Whisper not loaded"}), 503

    target_sentence = request.form.get("target", "").strip()
    focus_phoneme   = request.form.get("focus_phoneme", "").strip()
    if not target_sentence:
        return jsonify({"error": "No target sentence"}), 400

    audio_file = request.files["audio"]
    ext = os.path.splitext(audio_file.filename or "")[1] or ".webm"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # ── 1. Transcribe ─────────────────────────────────────────────────
        result = _whisper_model.transcribe(tmp_path, language="en")
        heard  = result["text"].strip()

        # ── 2. Word-level alignment ────────────────────────────────────────
        def _clean(w):
            return w.strip(".,!?\"';:-").lower()

        t_words = [_clean(w) for w in target_sentence.split() if _clean(w)]
        h_words = [_clean(w) for w in heard.split()           if _clean(w)]

        sm        = difflib.SequenceMatcher(None, t_words, h_words, autojunk=False)
        alignment = []
        matched   = 0

        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for w in t_words[i1:i2]:
                    alignment.append({"target": w, "heard": w, "status": "correct"})
                    matched += 1
            elif tag == "replace":
                t_seg, h_seg = t_words[i1:i2], h_words[j1:j2]
                for tw, hw in zip(t_seg, h_seg):
                    alignment.append({"target": tw, "heard": hw, "status": "wrong"})
                for tw in t_seg[len(h_seg):]:
                    alignment.append({"target": tw, "heard": "",  "status": "missing"})
                for hw in h_seg[len(t_seg):]:
                    alignment.append({"target": "",  "heard": hw, "status": "extra"})
            elif tag == "delete":
                for w in t_words[i1:i2]:
                    alignment.append({"target": w, "heard": "",  "status": "missing"})
            elif tag == "insert":
                for w in h_words[j1:j2]:
                    alignment.append({"target": "",  "heard": w,  "status": "extra"})

        word_score = round(matched / len(t_words), 3) if t_words else 0

        # ── 3. Acoustic analysis on focus phoneme (whole-recording) ───────
        focus_acoustic = None
        if ACOUSTIC_OK and focus_phoneme:
            try:
                seg = analyse_full_recording(tmp_path, focus_phoneme)
                focus_acoustic = result_to_dict(seg)
            except Exception as ae:
                print(f"  sentence acoustic error: {ae}")

        # ── 4. Plain-English sentence feedback ────────────────────────────
        wrong   = [a for a in alignment if a["status"] in ("wrong", "missing")]
        correct = [a for a in alignment if a["status"] == "correct"]
        pct     = round(word_score * 100)

        if pct == 100:
            headline = "Perfect sentence! Every word was clear. 🎉"
        elif pct >= 80:
            headline = f"Great — {pct}% of words were clear. Just a couple to tighten up."
        elif pct >= 60:
            headline = f"Good effort — {pct}% correct. Focus on the highlighted words."
        else:
            headline = f"Keep trying — {pct}% correct. Try saying it more slowly."

        tips = []
        if wrong:
            missed = [a["target"] for a in wrong[:3] if a["target"]]
            if missed:
                tips.append(f"Work on: {', '.join(missed)}")
        if focus_acoustic and focus_acoustic.get("friction"):
            f = focus_acoustic["friction"]
            if f.get("lisp_risk") in ("high", "moderate"):
                tips.append(f"Your /{focus_phoneme}/ sounds: {f['lisp_note']}")
        if focus_acoustic and focus_acoustic.get("vowel"):
            v = focus_acoustic["vowel"]
            if abs(v.get("F1_diff", 0)) > 100:
                tips.append(f"Vowel /{focus_phoneme}/: {v.get('jaw_deviation','')}")

        return jsonify({
            "heard":          heard,
            "target":         target_sentence,
            "alignment":      alignment,
            "word_score":     word_score,
            "words_correct":  matched,
            "words_total":    len(t_words),
            "focus_acoustic": focus_acoustic,
            "feedback": {
                "headline": headline,
                "tips":     tips,
            },
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.route("/tongue/predict", methods=["POST"])
def tongue_predict():
    """
    POST JSON: { crop_b64: "<base64 PNG of mouth interior>" }
    Returns: { label, confidence, method, proba }
    """
    if not TONGUE_OK:
        return jsonify({"error": "tongue detector unavailable"}), 503
    data    = request.get_json(force=True, silent=True) or {}
    crop_b64 = data.get("crop_b64", "")
    if not crop_b64:
        return jsonify({"error": "no crop_b64"}), 400
    try:
        return jsonify(_td.predict_b64(crop_b64))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/tongue/data", methods=["POST"])
def tongue_data():
    """
    POST JSON: { crop_b64, label: 0-3, phoneme }
    Adds one labelled crop to the training dataset.
    label: 0=tongue-low  1=tongue-mid  2=tongue-high  3=tongue-tip
    """
    if not TONGUE_OK:
        return jsonify({"error": "tongue detector unavailable"}), 503
    data     = request.get_json(force=True, silent=True) or {}
    crop_b64 = data.get("crop_b64", "")
    label    = data.get("label")
    if not crop_b64 or label is None:
        return jsonify({"error": "need crop_b64 and label"}), 400
    try:
        result = _td.save_crop(crop_b64, int(label))
        return jsonify({"ok": True, **(result or {})})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/tongue/train", methods=["POST"])
def tongue_train():
    """Train the tongue position classifier on collected data."""
    if not TONGUE_OK:
        return jsonify({"error": "tongue detector unavailable"}), 503
    try:
        return jsonify(_td.train())
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/tongue/status")
def tongue_status():
    if not TONGUE_OK:
        return jsonify({"available": False})
    return jsonify({**_td.status(), "available": True})


@app.route("/lisp_assess", methods=["POST"])
def lisp_assess():
    """
    Analyses a recording for lisp characteristics and returns:
      - CoG measurement for each /s/ or /z/ in the word
      - Lisp type classification
      - Severity score
      - Specific feedback and exercise progression
    POST: 'audio' + 'word' (the target word, e.g. 'sun')
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio"}), 400
    if not (WHISPER_OK and ACOUSTIC_OK):
        return jsonify({"error": "Whisper or acoustic analysis unavailable"}), 503

    word       = request.form.get("word", "").strip().lower()
    audio_file = request.files["audio"]
    ext        = os.path.splitext(audio_file.filename or "")[1] or ".webm"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # Transcribe
        result = _whisper_model.transcribe(tmp_path, language="en")
        heard  = result["text"].strip()

        # Analyse the full recording for /s/ frication
        seg = analyse_full_recording(tmp_path, "s")
        fric = seg.friction

        if not fric:
            return jsonify({
                "heard":      heard,
                "word":       word,
                "analysed":   False,
                "reason":     "No sibilant detected in recording",
            })

        # Build detailed lisp assessment
        cog        = fric.CoG_hz
        lisp_type  = fric.lisp_type
        lisp_risk  = fric.lisp_risk
        energy     = fric.energy_db
        tilt       = fric.tilt_db_oct

        # Severity score 0-100 (100 = perfectly healthy /s/)
        severity = max(0, min(100, int((cog - 2000) / (8000 - 2000) * 100)))

        # Type-specific exercises and cues
        exercises = _lisp_exercises(lisp_type)

        return jsonify({
            "heard":       heard,
            "word":        word,
            "analysed":    True,
            "cog_hz":      cog,
            "cog_ref":     fric.CoG_ref,
            "energy_db":   energy,
            "tilt":        tilt,
            "lisp_type":   lisp_type,
            "lisp_risk":   lisp_risk,
            "lisp_note":   fric.lisp_note,
            "severity":    severity,
            "exercises":   exercises,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _lisp_exercises(lisp_type: str) -> dict:
    """Return type-specific exercise progression for a lisp."""
    base = {
        "none": {
            "headline": "Your /s/ sounds healthy!",
            "description": "Continue practising to maintain consistency.",
            "steps": [
                {"level": "Maintenance", "words": ["sun", "sea", "six", "sit", "set"],
                 "cue": "Keep doing what you're doing — your /s/ is clear."},
            ],
        },
        "interdental": {
            "headline": "Interdental lisp detected",
            "description": "Your tongue tip is coming forward between your teeth, turning /s/ into /θ/ (the TH sound).",
            "what_to_do": "Keep the tongue tip BEHIND the upper teeth. The tip should touch the ridge just behind the teeth, NOT poke through.",
            "visual_cue": "Imagine your tongue is a dog that must stay behind a fence — the fence is your top teeth.",
            "steps": [
                {"level": "Step 1 — Feel the difference",
                 "words": ["tee", "dee"],
                 "cue": "Say T and D — your tongue tip touches the ridge. That ridge is where it should go for S too."},
                {"level": "Step 2 — /ts/ cluster",
                 "words": ["ts", "its", "ats", "cats", "bats"],
                 "cue": "The /t/ before /s/ forces your tongue to the right place. Hold it there for the /s/."},
                {"level": "Step 3 — Initial /s/",
                 "words": ["sun", "sea", "six", "set", "soft", "safe"],
                 "cue": "Tongue tip behind the ridge. Air hisses through the narrow gap — no tongue between the teeth."},
                {"level": "Step 4 — Final /s/",
                 "words": ["bus", "miss", "yes", "this", "plus"],
                 "cue": "End each word cleanly — tongue stays behind teeth at the end too."},
                {"level": "Step 5 — Sentences",
                 "words": ["Sally sees six silly seals", "Sam sat by the sea", "The sun shines in the sky"],
                 "cue": "Keep every /s/ consistent. Slow down and feel each one."},
            ],
        },
        "lateral": {
            "headline": "Lateral lisp detected",
            "description": "Air is escaping around the sides of your tongue instead of through the centre, causing a wet or slushy /s/ sound.",
            "what_to_do": "Direct all the air through the CENTRE of your mouth in a narrow stream.",
            "visual_cue": "Imagine the air as a laser beam going perfectly straight forward through a tiny gap.",
            "steps": [
                {"level": "Step 1 — Find the airstream",
                 "words": ["hiss", "ssss"],
                 "cue": "Sustain the /s/ sound and feel the air. It should hit the back of your upper front teeth in the middle."},
                {"level": "Step 2 — Straw exercise",
                 "words": ["ssss", "ssun", "ssea"],
                 "cue": "Imagine you're blowing through a very thin straw. All air goes through the centre — none goes sideways."},
                {"level": "Step 3 — Short words",
                 "words": ["sun", "sea", "sit", "sad", "sip"],
                 "cue": "Groove in the middle of your tongue. Feel the narrow airstream."},
                {"level": "Step 4 — Connected speech",
                 "words": ["bus stop", "this seat", "last summer", "best place"],
                 "cue": "Maintain the central airstream even when /s/ is not at the start of the word."},
            ],
        },
        "addental": {
            "headline": "Addental /s/ detected",
            "description": "Your tongue tip is touching the upper teeth rather than the alveolar ridge, producing a dull /s/.",
            "what_to_do": "Lift the tongue tip slightly away from the teeth to create a small gap. The air should hiss through that gap.",
            "visual_cue": "A tiny gap — like a window open just a crack — between your tongue tip and the ridge.",
            "steps": [
                {"level": "Step 1 — Find the ridge",
                 "words": ["tee", "dee", "nee"],
                 "cue": "Say T, D, N — feel the bumpy ridge just behind your top teeth. That's where /s/ should be."},
                {"level": "Step 2 — Lift slightly",
                 "words": ["sss", "ssun", "ssea"],
                 "cue": "Tongue tip near the ridge but NOT touching it. A small gap lets the air hiss through sharply."},
                {"level": "Step 3 — Short words",
                 "words": ["sun", "sit", "set", "six", "sap"],
                 "cue": "Feel the sharp hiss. If it sounds dull, lift your tongue tip a little more."},
                {"level": "Step 4 — Sentences",
                 "words": ["six small steps", "stop and sit", "soft sweet sounds"],
                 "cue": "Sharp crisp /s/ on every word."},
            ],
        },
    }
    return base.get(lisp_type, base["none"])


@app.route("/phoneme_map")
def phoneme_map_endpoint():
    """Aggregated co-occurrence data for the heat map."""
    if not SESSION_OK:
        return jsonify({"error": "session store unavailable"}), 503
    days = int(request.args.get("days", 30))
    return jsonify(_ss.phoneme_map(days))


@app.route("/phoneme_attempt", methods=["POST"])
def phoneme_attempt():
    """
    Save a phoneme-tab (isolation) attempt sent from the frontend.
    POST JSON: { phoneme, word, success, cog_hz, f1_hz, f2_hz }
    """
    if not SESSION_OK:
        return jsonify({"ok": False})
    data = request.get_json(force=True, silent=True) or {}
    ph = data.get("phoneme", "").strip()
    if not ph:
        return jsonify({"ok": False, "reason": "no phoneme"})
    _ss.save_phoneme_context({
        "phoneme":      ph,
        "context_type": "isolation",
        "word":         data.get("word", ""),
        "position":     "any",
        "prev_ph":      None,
        "next_ph":      None,
        "cluster":      None,
        "success":      data.get("success", False),
        "cog_hz":       data.get("cog_hz"),
        "f1_hz":        data.get("f1_hz"),
        "f2_hz":        data.get("f2_hz"),
    })
    return jsonify({"ok": True})


def _check_password() -> bool:
    return request.headers.get("X-Clinician-Password") == CLINICIAN_PASSWORD


@app.route("/clinician/references")
def clinician_references():
    if not _check_password():
        return jsonify({"error": "Unauthorised"}), 401
    if not REF_OK:
        return jsonify({"error": "Reference store unavailable"}), 503
    return jsonify(_ref.list_all())


@app.route("/clinician/reference", methods=["POST"])
def clinician_save_reference():
    if not _check_password():
        return jsonify({"error": "Unauthorised"}), 401
    if not REF_OK:
        return jsonify({"error": "Reference store unavailable"}), 503
    if "audio" not in request.files:
        return jsonify({"error": "No audio"}), 400

    word = request.form.get("word", "").strip().lower()
    if not word:
        return jsonify({"error": "No word specified"}), 400

    audio_file = request.files["audio"]
    ext = os.path.splitext(audio_file.filename or "")[1] or ".webm"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        ipa      = _ipa(word)
        phonemes = _parse_ipa(ipa)
        entry    = _ref.save(word, tmp_path, ipa, phonemes)
        return jsonify({"ok": True, "entry": entry})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.route("/clinician/reference/<word>", methods=["DELETE"])
def clinician_delete_reference(word):
    if not _check_password():
        return jsonify({"error": "Unauthorised"}), 401
    if not REF_OK:
        return jsonify({"error": "Reference store unavailable"}), 503
    deleted = _ref.delete(word)
    return jsonify({"ok": deleted})


if __name__ == "__main__":
    print("\n" + "=" * 52)
    print("  Speech Therapy — Australian English IPA")
    print("  Open  http://localhost:5001  in your browser")
    print("=" * 52 + "\n")
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
