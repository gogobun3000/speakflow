"""
acoustic_analysis.py — SpeakFlow acoustic feature extractor
============================================================
Extracts clinically relevant acoustic features from a single audio segment
using parselmouth (Python wrapper for Praat).

Features extracted
------------------
  Vowels  : F1, F2 formants (Hz) — indicate tongue height/backness
  Sibilants: Frication energy, spectral CoG, spectral tilt — catches lisps
  Stops   : Voice Onset Time proxy (pre-voicing energy)
  General : Pitch (F0), HNR (voice quality), duration

Usage (standalone test)
-----------------------
  python3 acoustic_analysis.py path/to/audio.wav [phoneme_label]

Usage (as a module)
-------------------
  from acoustic_analysis import analyse_segment, analyse_full_recording
"""

from __future__ import annotations

import os
import sys
import json
import tempfile
import subprocess
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import parselmouth
from parselmouth.praat import call


# ── Clinical reference ranges (General Australian English, adult) ─────────────
# Sourced from Harrington et al. (1997), Cox (2006), and Peterson & Barney (1952)
# Values are approximate population means — clinicians adjust per patient.

VOWEL_REFS: dict[str, dict] = {
    # IPA: { F1_hz, F2_hz, label, jaw_height, tongue_backness }
    "iː": {"F1": 280,  "F2": 2570, "label": "FLEECE",  "jaw": "close",      "back": "front"},
    "ɪ":  {"F1": 360,  "F2": 2100, "label": "KIT",     "jaw": "close",      "back": "front"},
    "e":  {"F1": 560,  "F2": 1970, "label": "DRESS",   "jaw": "mid",        "back": "front"},
    "æ":  {"F1": 820,  "F2": 1660, "label": "TRAP",    "jaw": "open",       "back": "front"},
    "ɐ":  {"F1": 760,  "F2": 1320, "label": "AuE schwa","jaw":"mid-open",   "back": "central"},
    "ɑː": {"F1": 800,  "F2": 1200, "label": "PALM",    "jaw": "open",       "back": "back"},
    "ɒ":  {"F1": 720,  "F2": 1000, "label": "LOT",     "jaw": "open",       "back": "back"},
    "ɔː": {"F1": 560,  "F2": 800,  "label": "THOUGHT", "jaw": "mid",        "back": "back"},
    "ʊ":  {"F1": 430,  "F2": 1020, "label": "FOOT",    "jaw": "close",      "back": "back"},
    "uː": {"F1": 290,  "F2": 1460, "label": "GOOSE",   "jaw": "close",      "back": "central"},  # AuE fronted
    "ʌ":  {"F1": 680,  "F2": 1310, "label": "STRUT",   "jaw": "mid-open",   "back": "central"},
    "ɜː": {"F1": 490,  "F2": 1460, "label": "NURSE",   "jaw": "mid",        "back": "central"},
    "ə":  {"F1": 500,  "F2": 1350, "label": "schwa",   "jaw": "mid",        "back": "central"},
}

# Sibilant reference: healthy /s/ has CoG above 4000 Hz; lisped /s/ may sound
# like /θ/ with CoG around 2500–3500 Hz and lower frication energy.
SIBILANT_REFS = {
    "s":  {"CoG_hz": 7000, "energy_min_db": -30, "label": "alveolar sibilant"},
    "z":  {"CoG_hz": 6500, "energy_min_db": -32, "label": "voiced alveolar sibilant"},
    "ʃ":  {"CoG_hz": 4000, "energy_min_db": -32, "label": "postalveolar sibilant"},
    "ʒ":  {"CoG_hz": 3800, "energy_min_db": -34, "label": "voiced postalveolar sibilant"},
    "θ":  {"CoG_hz": 3000, "energy_min_db": -40, "label": "dental fricative"},
    "ð":  {"CoG_hz": 2500, "energy_min_db": -42, "label": "voiced dental fricative"},
    "f":  {"CoG_hz": 6000, "energy_min_db": -38, "label": "labiodental fricative"},
    "v":  {"CoG_hz": 5500, "energy_min_db": -40, "label": "voiced labiodental fricative"},
}

SIBILANTS     = set(SIBILANT_REFS.keys())
VOWELS        = set(VOWEL_REFS.keys())
STOP_VOICED   = {"b", "d", "ɡ"}
STOP_VOICELESS= {"p", "t", "k"}
STOPS         = STOP_VOICED | STOP_VOICELESS


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class VowelFeatures:
    F1_hz:        float
    F2_hz:        float
    F1_ref:       float
    F2_ref:       float
    F1_diff:      float          # positive = patient higher than reference
    F2_diff:      float
    phoneme:      str
    label:        str
    jaw_deviation: str           # "too open" / "too close" / "on target"
    back_deviation: str          # "too front" / "too back" / "on target"
    confidence:   float          # 0-1 based on formant stability

@dataclass
class FricationFeatures:
    phoneme:      str
    label:        str
    CoG_hz:       float          # spectral centre of gravity
    CoG_ref:      float
    CoG_diff:     float
    energy_db:    float          # RMS frication energy
    tilt_db_oct:  float          # spectral tilt (slope) — lisps show flatter spectrum
    lisp_risk:    str            # "high" / "moderate" / "low"
    lisp_note:    str

@dataclass
class GeneralFeatures:
    duration_ms:  float
    pitch_mean_hz: Optional[float]
    pitch_sd_hz:  Optional[float]
    HNR_db:       Optional[float]   # harmonics-to-noise — voice quality
    voiced:       bool

@dataclass
class SegmentResult:
    phoneme:  str
    start_ms: float
    end_ms:   float
    general:  GeneralFeatures
    vowel:    Optional[VowelFeatures]   = None
    friction: Optional[FricationFeatures] = None
    feedback: list[str]                = None   # human-readable clinical notes

    def __post_init__(self):
        if self.feedback is None:
            self.feedback = []


# ── Core Praat extraction ──────────────────────────────────────────────────────

def _load_sound(path: str) -> parselmouth.Sound:
    """Load any audio file parselmouth can handle (wav, flac, mp3 via ffmpeg)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".wav", ".aiff", ".aif"):
        return parselmouth.Sound(path)
    # Convert non-wav to wav via ffmpeg then load
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-ar", "16000", "-ac", "1", wav_path],
            capture_output=True, check=True
        )
        return parselmouth.Sound(wav_path)
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


def _extract_general(snd: parselmouth.Sound) -> GeneralFeatures:
    dur_ms = snd.duration * 1000.0

    # Pitch
    try:
        pitch = snd.to_pitch()
        f0_vals = pitch.selected_array["frequency"]
        f0_voiced = f0_vals[f0_vals > 0]
        if len(f0_voiced) > 3:
            pitch_mean = float(np.mean(f0_voiced))
            pitch_sd   = float(np.std(f0_voiced))
            voiced = True
        else:
            pitch_mean = pitch_sd = None
            voiced = False
    except Exception:
        pitch_mean = pitch_sd = None
        voiced = False

    # HNR
    try:
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        if hnr <= -200:
            hnr = None
    except Exception:
        hnr = None

    return GeneralFeatures(
        duration_ms=round(dur_ms, 1),
        pitch_mean_hz=round(pitch_mean, 1) if pitch_mean else None,
        pitch_sd_hz=round(pitch_sd, 1) if pitch_sd else None,
        HNR_db=round(hnr, 1) if hnr is not None else None,
        voiced=voiced,
    )


def _extract_formants(
    snd: parselmouth.Sound,
    phoneme: str,
    ref_override: dict | None = None,
) -> Optional[VowelFeatures]:
    """Extract F1/F2 using Burg's method with appropriate settings.

    ref_override: if provided, use these F1/F2 values as targets instead of
                  the textbook VOWEL_REFS table.  Keys: F1, F2, label.
    """
    if ref_override:
        ref = ref_override
    elif phoneme in VOWEL_REFS:
        ref = VOWEL_REFS[phoneme]
    else:
        return None

    # LPC settings: 5 formants, 5500 Hz ceiling (female/child: 5500, male: 5000)
    try:
        formants = call(snd, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
        dur = snd.duration

        # Sample at 20%-80% of segment to avoid edge effects
        t_start = dur * 0.20
        t_end   = dur * 0.80
        step    = 0.01

        F1s, F2s = [], []
        t = t_start
        while t <= t_end:
            f1 = call(formants, "Get value at time", 1, t, "Hertz", "Linear")
            f2 = call(formants, "Get value at time", 2, t, "Hertz", "Linear")
            if f1 and f1 > 100 and f2 and f2 > 500:
                F1s.append(f1)
                F2s.append(f2)
            t += step

        if len(F1s) < 3:
            return None

        F1 = float(np.median(F1s))
        F2 = float(np.median(F2s))

        # Formant stability as confidence proxy: low CV = stable
        F1_cv = np.std(F1s) / np.mean(F1s) if np.mean(F1s) > 0 else 1.0
        confidence = max(0.0, min(1.0, 1.0 - F1_cv * 3))

        F1_diff = F1 - ref["F1"]
        F2_diff = F2 - ref["F2"]
        tol = 100  # Hz tolerance before flagging deviation

        jaw_dev = "on target"
        if F1_diff > tol:
            jaw_dev = "jaw more open than target"
        elif F1_diff < -tol:
            jaw_dev = "jaw more closed than target"

        back_dev = "on target"
        if F2_diff > tol:
            back_dev = "tongue further forward than target"
        elif F2_diff < -tol:
            back_dev = "tongue further back than target"

        return VowelFeatures(
            F1_hz=round(F1, 0),
            F2_hz=round(F2, 0),
            F1_ref=ref["F1"],
            F2_ref=ref["F2"],
            F1_diff=round(F1_diff, 0),
            F2_diff=round(F2_diff, 0),
            phoneme=phoneme,
            label=ref["label"],
            jaw_deviation=jaw_dev,
            back_deviation=back_dev,
            confidence=round(confidence, 2),
        )
    except Exception as e:
        print(f"  [formant error] {e}")
        return None


def _extract_frication(snd: parselmouth.Sound, phoneme: str) -> Optional[FricationFeatures]:
    """Spectral analysis for fricatives — CoG and energy fingerprint."""
    if phoneme not in SIBILANT_REFS:
        return None

    ref = SIBILANT_REFS[phoneme]

    try:
        # Spectrum of full segment
        spectrum = snd.to_spectrum()

        # Centre of gravity (Hz) — weighted mean frequency
        cog = call(spectrum, "Get centre of gravity", 2)

        # Spectral tilt — slope from 1 kHz to Nyquist in dB/octave
        # Approximated by comparing energy in low vs high bands
        freqs = np.array(spectrum.xs())
        amps  = np.array(spectrum.values[0] ** 2 + spectrum.values[1] ** 2)  # power
        low_mask  = (freqs >= 1000) & (freqs < 4000)
        high_mask = (freqs >= 4000) & (freqs < 10000)
        low_energy  = np.sum(amps[low_mask])  if low_mask.any()  else 1e-10
        high_energy = np.sum(amps[high_mask]) if high_mask.any() else 1e-10
        tilt_db_oct = 10 * np.log10(high_energy / low_energy) if low_energy > 0 else 0.0

        # RMS energy in the frication band (2–10 kHz)
        fric_mask = (freqs >= 2000) & (freqs < 10000)
        rms_fric  = np.sqrt(np.mean(amps[fric_mask])) if fric_mask.any() else 0.0
        energy_db = 20 * np.log10(rms_fric + 1e-10)

        cog_diff = cog - ref["CoG_hz"]

        # Lisp risk: for /s/ and /z/, low CoG suggests /θ/-like production
        lisp_risk = "low"
        lisp_note = ""
        if phoneme in {"s", "z"}:
            if cog < 4500:
                lisp_risk = "high"
                lisp_note = (
                    f"CoG {cog:.0f} Hz is well below the target {ref['CoG_hz']:.0f} Hz. "
                    "This pattern matches a dental /θ/ substitution (interdental lisp). "
                    "Clinician should assess tongue placement."
                )
            elif cog < 6000:
                lisp_risk = "moderate"
                lisp_note = (
                    f"CoG {cog:.0f} Hz is below target {ref['CoG_hz']:.0f} Hz. "
                    "Possible lateral or addental lisp — check tongue lateralisation."
                )
            else:
                lisp_note = f"Frication CoG {cog:.0f} Hz is within typical range."

        return FricationFeatures(
            phoneme=phoneme,
            label=ref["label"],
            CoG_hz=round(cog, 0),
            CoG_ref=ref["CoG_hz"],
            CoG_diff=round(cog_diff, 0),
            energy_db=round(energy_db, 1),
            tilt_db_oct=round(tilt_db_oct, 1),
            lisp_risk=lisp_risk,
            lisp_note=lisp_note,
        )
    except Exception as e:
        print(f"  [frication error] {e}")
        return None


# ── Human-readable feedback generator ────────────────────────────────────────

def _generate_feedback(result: SegmentResult) -> list[str]:
    notes = []
    ph = result.phoneme

    # Duration
    d = result.general.duration_ms
    if d < 30:
        notes.append(f"/{ph}/ was very short ({d:.0f} ms) — it may have been reduced or deleted.")

    # Voice quality
    if result.general.HNR_db is not None:
        if result.general.HNR_db < 7:
            notes.append(
                f"Voice quality for /{ph}/ shows high noise (HNR {result.general.HNR_db:.1f} dB). "
                "May indicate breathiness or tension."
            )

    # Vowel feedback
    if result.vowel:
        v = result.vowel
        notes.append(
            f"/{ph}/ ({v.label}): F1 = {v.F1_hz:.0f} Hz (target {v.F1_ref:.0f} Hz, "
            f"diff {v.F1_diff:+.0f} Hz), F2 = {v.F2_hz:.0f} Hz (target {v.F2_ref:.0f} Hz, "
            f"diff {v.F2_diff:+.0f} Hz). Confidence: {v.confidence:.0%}."
        )
        if v.jaw_deviation != "on target":
            notes.append(f"Jaw position: {v.jaw_deviation}.")
        if v.back_deviation != "on target":
            notes.append(f"Tongue backness: {v.back_deviation}.")

    # Fricative / lisp feedback
    if result.friction:
        f = result.friction
        notes.append(
            f"/{ph}/ frication: CoG = {f.CoG_hz:.0f} Hz (target {f.CoG_ref:.0f} Hz, "
            f"diff {f.CoG_diff:+.0f} Hz). Energy = {f.energy_db:.1f} dB. "
            f"Spectral tilt = {f.tilt_db_oct:.1f} dB/oct."
        )
        if f.lisp_risk != "low":
            notes.append(f"⚠️  Lisp risk [{f.lisp_risk.upper()}]: {f.lisp_note}")
        else:
            notes.append(f.lisp_note)

    return notes


# ── Public API ─────────────────────────────────────────────────────────────────

def analyse_segment(
    audio_path: str,
    phoneme: str,
    start_ms: float = 0.0,
    end_ms: float   = None,
    vowel_ref_override: dict | None = None,
) -> SegmentResult:
    """
    Analyse one phoneme segment from an audio file.

    Parameters
    ----------
    audio_path : path to wav/mp3/webm/flac
    phoneme    : IPA symbol, e.g. "s", "æ", "tʃ"
    start_ms   : segment start in milliseconds (default: 0)
    end_ms     : segment end in milliseconds (default: end of file)

    Returns
    -------
    SegmentResult with all extracted features and feedback strings.
    """
    snd = _load_sound(audio_path)

    # Trim to segment
    start_s = start_ms / 1000.0
    end_s   = (end_ms / 1000.0) if end_ms is not None else snd.duration
    end_s   = min(end_s, snd.duration)

    if end_s - start_s < 0.02:  # < 20 ms → skip
        return SegmentResult(
            phoneme=phoneme, start_ms=start_ms, end_ms=end_ms or snd.duration * 1000,
            general=GeneralFeatures(duration_ms=0, pitch_mean_hz=None, pitch_sd_hz=None,
                                    HNR_db=None, voiced=False),
            feedback=["Segment too short to analyse."]
        )

    seg = snd.extract_part(from_time=start_s, to_time=end_s,
                           window_shape=parselmouth.WindowShape.RECTANGULAR,
                           relative_width=1.0, preserve_times=False)

    general  = _extract_general(seg)
    vowel    = _extract_formants(seg, phoneme, ref_override=vowel_ref_override) if phoneme in VOWELS else None
    friction = _extract_frication(seg, phoneme) if phoneme in SIBILANTS else None

    result = SegmentResult(
        phoneme=phoneme,
        start_ms=round(start_ms, 1),
        end_ms=round(end_s * 1000, 1),
        general=general,
        vowel=vowel,
        friction=friction,
    )
    result.feedback = _generate_feedback(result)
    return result


def analyse_full_recording(
    audio_path: str,
    phoneme: str,
) -> SegmentResult:
    """
    Analyse the entire recording as one phoneme segment.
    Use this when phoneme boundaries are not yet known.
    """
    return analyse_segment(audio_path, phoneme, start_ms=0.0, end_ms=None)


def analyse_word(
    audio_path: str,
    target_word: str,
    target_phonemes: list[str],
) -> list[SegmentResult]:
    """
    Analyse a full word recording without forced alignment.
    Divides the audio equally across the given phoneme list as a rough estimate.
    A forced aligner (MFA / WebMAUS) will give proper boundaries when available.

    Returns one SegmentResult per phoneme.
    """
    snd = _load_sound(audio_path)
    dur_ms = snd.duration * 1000.0
    n = len(target_phonemes)
    if n == 0:
        return []

    seg_ms = dur_ms / n
    results = []
    for i, ph in enumerate(target_phonemes):
        start = i * seg_ms
        end   = (i + 1) * seg_ms
        results.append(analyse_segment(audio_path, ph, start_ms=start, end_ms=end))

    return results


def result_to_dict(r: SegmentResult) -> dict:
    d = {
        "phoneme":  r.phoneme,
        "start_ms": r.start_ms,
        "end_ms":   r.end_ms,
        "general":  asdict(r.general),
        "feedback": r.feedback,
    }
    if r.vowel:
        d["vowel"] = asdict(r.vowel)
    if r.friction:
        d["friction"] = asdict(r.friction)
    return d


# ── Standalone test ────────────────────────────────────────────────────────────

def _generate_test_audio(phoneme: str) -> str:
    """Generate a short synthetic test WAV using espeak-ng."""
    word_map = {
        "s": "sun",   "z": "zoo",  "ʃ": "shop", "θ": "think",
        "æ": "cat",   "iː": "feet","ɑː": "palm", "ʌ": "cup",
        "t": "top",   "d": "dog",  "p": "pen",   "b": "ball",
    }
    word = word_map.get(phoneme, "hello")
    espeak = "/opt/homebrew/bin/espeak-ng"
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_wav = tmp.name
    try:
        subprocess.run(
            [espeak, "-v", "en-au", "-w", out_wav, word],
            capture_output=True, check=True
        )
        return out_wav
    except Exception:
        return None


if __name__ == "__main__":
    audio_path = sys.argv[1] if len(sys.argv) > 1 else None
    phoneme    = sys.argv[2] if len(sys.argv) > 2 else "s"

    # If no file given, generate a test tone using espeak-ng
    cleanup = False
    if not audio_path:
        print(f"No audio file provided — generating test audio for /{phoneme}/ …")
        audio_path = _generate_test_audio(phoneme)
        if not audio_path:
            print("Could not generate test audio. Provide a wav file as argument.")
            sys.exit(1)
        cleanup = True

    print(f"\n{'='*55}")
    print(f"  SpeakFlow Acoustic Analysis")
    print(f"  File   : {os.path.basename(audio_path)}")
    print(f"  Phoneme: /{phoneme}/")
    print(f"{'='*55}\n")

    result = analyse_full_recording(audio_path, phoneme)

    print(f"Duration  : {result.general.duration_ms:.1f} ms")
    print(f"Voiced    : {result.general.voiced}")
    if result.general.pitch_mean_hz:
        print(f"Pitch     : {result.general.pitch_mean_hz:.1f} ± {result.general.pitch_sd_hz:.1f} Hz")
    if result.general.HNR_db is not None:
        print(f"HNR       : {result.general.HNR_db:.1f} dB")

    if result.vowel:
        v = result.vowel
        print(f"\nFormants ({v.label}):")
        print(f"  F1 = {v.F1_hz:.0f} Hz  (ref {v.F1_ref:.0f}, diff {v.F1_diff:+.0f})")
        print(f"  F2 = {v.F2_hz:.0f} Hz  (ref {v.F2_ref:.0f}, diff {v.F2_diff:+.0f})")
        print(f"  Jaw : {v.jaw_deviation}")
        print(f"  Back: {v.back_deviation}")
        print(f"  Confidence: {v.confidence:.0%}")

    if result.friction:
        f = result.friction
        print(f"\nFrication ({f.label}):")
        print(f"  CoG    = {f.CoG_hz:.0f} Hz  (ref {f.CoG_ref:.0f}, diff {f.CoG_diff:+.0f})")
        print(f"  Energy = {f.energy_db:.1f} dB")
        print(f"  Tilt   = {f.tilt_db_oct:.1f} dB/oct")
        print(f"  Lisp risk: {f.lisp_risk.upper()}")

    print(f"\nClinical notes:")
    for note in result.feedback:
        print(f"  • {note}")

    print(f"\nFull JSON:\n{json.dumps(result_to_dict(result), indent=2)}")

    if cleanup and audio_path:
        try:
            os.unlink(audio_path)
        except OSError:
            pass
