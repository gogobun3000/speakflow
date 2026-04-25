"""
forced_align.py — Phoneme boundary detection for SpeakFlow
==========================================================

Strategy (tried in order):
  1. WebMAUS  — BAS WebServices forced aligner (needs internet + working server)
  2. Praat    — local voicing + intensity segmentation (no internet needed)
  3. Equal    — equal-duration fallback (always works)

Each result includes a 'method' key so the frontend can show
how confident the boundaries are.

Usage
-----
  from forced_align import align_word

  segs = align_word(
      audio_path  = "recording.wav",
      transcript  = "cat",
      phonemes    = ["k", "æ", "t"],   # from espeak IPA
  )
  # segs → [
  #   { ipa:"k", start_ms:0,   end_ms:80,  method:"praat" },
  #   { ipa:"æ", start_ms:80,  end_ms:250, method:"praat" },
  #   { ipa:"t", start_ms:250, end_ms:340, method:"praat" },
  # ]
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from typing import Optional
from xml.etree import ElementTree

import numpy as np

# optional — only needed for WebMAUS and Praat methods
try:
    import requests as _requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

try:
    import parselmouth
    from parselmouth.praat import call as _pcall
    _PRAAT_OK = True
except ImportError:
    _PRAAT_OK = False

# ── SAMPA → IPA table (BAS WebServices output) ───────────────────────────────
_SAMPA = {
    # monophthong vowels
    "i:": "iː", "I": "ɪ", "e": "e", "{": "æ",
    "A:": "ɑː", "Q": "ɒ", "O:": "ɔː",
    "U": "ʊ", "u:": "uː", "V": "ʌ",
    "3:": "ɜː", "@": "ə",
    # Australian diphthongs
    "eI": "æɪ", "aI": "ɑɪ", "{O": "æɔ", "{U": "æɔ",
    "@U": "əʊ", "OI": "ɔɪ", "I@": "ɪə", "e@": "eə", "U@": "ʊə",
    # consonants
    "p": "p", "b": "b", "t": "t", "d": "d", "k": "k", "g": "ɡ",
    "f": "f", "v": "v", "T": "θ", "D": "ð",
    "s": "s", "z": "z", "S": "ʃ", "Z": "ʒ", "h": "h",
    "tS": "tʃ", "dZ": "dʒ",
    "m": "m", "n": "n", "N": "ŋ",
    "l": "l", "r": "ɹ", "w": "w", "j": "j",
}

# Phoneme type sets (same as acoustic_analysis.py — kept independent to avoid circular import)
_VOWELS    = {"iː","ɪ","e","æ","ɑː","ɒ","ɔː","ʊ","uː","ʌ","ɜː","ə",
              "æɪ","ɑɪ","æɔ","əʊ","ɔɪ","ɪə","eə","ʊə","ɐ","ʉː",
              "a","ɑ","ɔ","o","i","u",    # bare vowels from some espeak outputs
              "ɛ","œ","ø","y","ɯ","ɐ"}  # additional IPA vowels
_SIBILANTS = {"s","z","ʃ","ʒ","θ","ð","f","v"}


# ═════════════════════════════════════════════════════════════════════════════
#  Method 1 — WebMAUS (BAS Web Services)
# ═════════════════════════════════════════════════════════════════════════════

_WEBMAUS_URL = (
    "https://clarin.phonetik.uni-muenchen.de"
    "/BASWebServices/services/runMAUSBasic"
)


def _to_wav16k(path: str) -> tuple[str, bool]:
    """Convert any audio to 16 kHz 16-bit mono WAV. Returns (path, needs_cleanup)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".wav", ".wave"):
        # Still re-encode to guarantee 16 kHz mono s16
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out = f.name
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-ar", "16000", "-ac", "1",
             "-sample_fmt", "s16", out],
            capture_output=True, check=True,
        )
        return out, True
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        out = f.name
    subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-ar", "16000", "-ac", "1",
         "-sample_fmt", "s16", out],
        capture_output=True, check=True,
    )
    return out, True


def _parse_textgrid(tg_text: str) -> list[dict]:
    """Extract non-empty intervals from the MAU tier of a Praat TextGrid."""
    tier_re = re.compile(
        r'name = "MAU".*?intervals: size = \d+(.*?)(?=item \[|\Z)', re.DOTALL
    )
    m = tier_re.search(tg_text)
    if not m:
        return []
    block = m.group(1)
    iv_re = re.compile(
        r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "([^"]*)"'
    )
    result = []
    for iv in iv_re.finditer(block):
        sampa = iv.group(3).strip()
        if sampa and sampa not in ("<p:>", "sil", ""):
            result.append({
                "ipa":      _SAMPA.get(sampa, sampa),
                "sampa":    sampa,
                "start_ms": round(float(iv.group(1)) * 1000, 1),
                "end_ms":   round(float(iv.group(2)) * 1000, 1),
                "method":   "webmaus",
            })
    return result


def _align_webmaus(wav_path: str, transcript: str, timeout: int = 15) -> list[dict]:
    """Try WebMAUS. Raises RuntimeError on any failure."""
    if not _REQUESTS_OK:
        raise RuntimeError("requests library not installed")

    with open(wav_path, "rb") as f:
        audio = f.read()

    # Try the two language codes that actually work on the BAS server
    last_err = "unknown error"
    for lang in ("eng-AU", "eng", "aus"):
        resp = _requests.post(
            _WEBMAUS_URL,
            files={"SIGNAL": ("audio.wav", audio, "audio/wav")},
            data={
                "TEXT":      transcript,
                "LANGUAGE":  lang,
                "OUTFORMAT": "TextGrid",
                "MODUS":     "standard",
                "OUTSYMBOL": "sampa",
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        root = ElementTree.fromstring(resp.text)
        if root.findtext("success", "").strip().lower() == "true":
            link = root.findtext("downloadLink", "").strip()
            tg_text = ""
            if link:
                dl = _requests.get(link, timeout=10)
                dl.raise_for_status()
                tg_text = dl.text
            else:
                tg_text = root.findtext("output", "").strip()
            segs = _parse_textgrid(tg_text)
            if segs:
                return segs
            raise RuntimeError("WebMAUS returned empty TextGrid")
        last_err = root.findtext("output", "")[:120]

    raise RuntimeError(f"WebMAUS failed: {last_err}")


# ═════════════════════════════════════════════════════════════════════════════
#  Method 2 — Praat-based smart segmentation (local, no internet)
# ═════════════════════════════════════════════════════════════════════════════

def _align_praat(wav_path: str, phonemes: list[str]) -> list[dict]:
    """
    Use Praat voicing + intensity to estimate phoneme boundaries.

    Algorithm:
      1. Find the voiced region (approximate vowel zone) using pitch tracking.
      2. Assign leading consonants to [0 → voiced_start].
      3. Assign vowel(s) to [voiced_start → voiced_end].
      4. Assign trailing consonants to [voiced_end → duration].
      5. Within each zone, subdivide equally if there are multiple phonemes.

    This is substantially better than equal division for CVC, CCVC, CVCC words.
    """
    if not _PRAAT_OK:
        raise RuntimeError("parselmouth not installed")

    n = len(phonemes)
    if n == 0:
        return []

    snd = parselmouth.Sound(wav_path)
    dur_ms = snd.duration * 1000.0

    if n == 1:
        return [{"ipa": phonemes[0], "start_ms": 0.0, "end_ms": dur_ms, "method": "praat"}]

    # ── 1. Voiced region via pitch tracking ───────────────────────────────────
    try:
        pitch = snd.to_pitch(time_step=0.005, pitch_floor=60.0)
        times  = pitch.xs()
        freqs  = pitch.selected_array["frequency"]
        voiced = times[freqs > 0]
        if len(voiced) >= 2:
            v_start_ms = float(voiced[0])  * 1000.0
            v_end_ms   = float(voiced[-1]) * 1000.0
        else:
            raise ValueError("No voiced region found")
    except Exception:
        # Fallback: assume middle 50% is voiced
        v_start_ms = dur_ms * 0.25
        v_end_ms   = dur_ms * 0.75

    # Clip to audio bounds with small margins
    v_start_ms = max(v_start_ms, 0.0)
    v_end_ms   = min(v_end_ms, dur_ms)
    if v_end_ms - v_start_ms < 20:          # voiced region too short → equal
        raise ValueError("Voiced region too narrow for Praat segmentation")

    # ── 2. Classify phonemes into zones ───────────────────────────────────────
    # Find first and last vowel indices
    vowel_indices = [i for i, p in enumerate(phonemes) if p in _VOWELS]

    if not vowel_indices:
        # No vowels → equal division
        raise ValueError("No vowels in phoneme list — Praat method not applicable")

    first_v = vowel_indices[0]
    last_v  = vowel_indices[-1]

    n_lead  = first_v                   # consonants before first vowel
    n_trail = n - last_v - 1            # consonants after last vowel
    n_vowel = last_v - first_v + 1      # vowels (including diphthongs)

    segs: list[dict] = []

    def _divide(start, end, items):
        if not items:
            return
        step = (end - start) / len(items)
        for j, ph in enumerate(items):
            segs.append({
                "ipa":      ph,
                "start_ms": round(start + j * step, 1),
                "end_ms":   round(start + (j + 1) * step, 1),
                "method":   "praat",
            })

    # Leading consonants: 0 → voiced_start
    _divide(0.0, v_start_ms, phonemes[:first_v])

    # Vowel(s): voiced_start → voiced_end
    _divide(v_start_ms, v_end_ms, phonemes[first_v: last_v + 1])

    # Trailing consonants: voiced_end → dur_ms
    _divide(v_end_ms, dur_ms, phonemes[last_v + 1:])

    return segs


# ═════════════════════════════════════════════════════════════════════════════
#  Method 3 — Equal division (last resort)
# ═════════════════════════════════════════════════════════════════════════════

def _align_equal(duration_ms: float, phonemes: list[str]) -> list[dict]:
    n = len(phonemes)
    if n == 0:
        return []
    step = duration_ms / n
    return [
        {
            "ipa":      ph,
            "start_ms": round(i * step, 1),
            "end_ms":   round((i + 1) * step, 1),
            "method":   "equal",
        }
        for i, ph in enumerate(phonemes)
    ]


def _get_duration_ms(wav_path: str) -> float:
    """Return audio duration in milliseconds."""
    try:
        snd = parselmouth.Sound(wav_path)
        return snd.duration * 1000.0
    except Exception:
        return 500.0  # safe guess


# ═════════════════════════════════════════════════════════════════════════════
#  Public API
# ═════════════════════════════════════════════════════════════════════════════

def align_word(
    audio_path: str,
    transcript: str,
    phonemes: list[str],
    webmaus_timeout: int = 12,
    verbose: bool = True,
) -> list[dict]:
    """
    Align phonemes to audio using the best available method.

    Parameters
    ----------
    audio_path      : path to any audio file (ffmpeg handles conversion)
    transcript      : orthographic word, e.g. "cat"
    phonemes        : IPA phoneme list, e.g. ["k", "æ", "t"]
    webmaus_timeout : seconds to wait for WebMAUS before giving up

    Returns
    -------
    List of segments: [{ ipa, start_ms, end_ms, method }, ...]
    method is one of: "webmaus" | "praat" | "equal"
    """
    if not phonemes:
        return []

    # Convert to WAV once — all methods need it
    wav_path, wav_cleanup = _to_wav16k(audio_path)

    try:
        # ── Method 1: WebMAUS ─────────────────────────────────────────────────
        try:
            segs = _align_webmaus(wav_path, transcript, timeout=webmaus_timeout)
            if verbose:
                print(f"  [align] WebMAUS ✓  ({len(segs)} segments)")
            return segs
        except Exception as e:
            if verbose:
                print(f"  [align] WebMAUS failed ({e.__class__.__name__}: {str(e)[:60]}) — trying Praat")

        # ── Method 2: Praat smart segmentation ───────────────────────────────
        try:
            segs = _align_praat(wav_path, phonemes)
            if verbose:
                print(f"  [align] Praat ✓  ({len(segs)} segments, method=praat)")
            return segs
        except Exception as e:
            if verbose:
                print(f"  [align] Praat failed ({e}) — falling back to equal division")

        # ── Method 3: Equal division ──────────────────────────────────────────
        dur = _get_duration_ms(wav_path)
        segs = _align_equal(dur, phonemes)
        if verbose:
            print(f"  [align] Equal division ({len(segs)} segments)")
        return segs

    finally:
        if wav_cleanup:
            try:
                os.unlink(wav_path)
            except OSError:
                pass


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    espeak = "/opt/homebrew/bin/espeak-ng"
    word   = sys.argv[1] if len(sys.argv) > 1 else "cat"

    # Generate test audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav = f.name
    subprocess.run([espeak, "-v", "en-au", "-w", wav, word], capture_output=True, check=True)

    # Get phonemes from espeak
    r = subprocess.run([espeak, "--ipa", "-v", "en-au", "--", word],
                       capture_output=True, text=True)
    ipa_str = r.stdout.strip()
    for old, new in [("eɪ","æɪ"),("aɪ","ɑɪ"),("aʊ","æɔ")]:
        ipa_str = ipa_str.replace(old, new)

    # Simple parse
    skip = set("ˈˌ|‖. \n\t")
    dgs  = ["tʃ","dʒ","æɪ","ɑɪ","æɔ","əʊ","uː","iː","ɑː","ɔː","ɜː"]
    phonemes, i = [], 0
    while i < len(ipa_str):
        if ipa_str[i] in skip: i+=1; continue
        matched = False
        for d in dgs:
            if ipa_str[i:i+len(d)] == d:
                phonemes.append(d); i+=len(d); matched=True; break
        if not matched:
            phonemes.append(ipa_str[i]); i+=1

    print(f"\nWord: {word!r}   IPA: {ipa_str}   Phonemes: {phonemes}\n")

    segs = align_word(wav, word, phonemes)
    print()
    for s in segs:
        bar = "█" * int((s["end_ms"] - s["start_ms"]) / 5)
        print(f"  /{s['ipa']:<4}  {s['start_ms']:6.1f} – {s['end_ms']:6.1f} ms  {bar}")
    print(f"\nMethod: {segs[0]['method'] if segs else 'none'}")

    os.unlink(wav)
