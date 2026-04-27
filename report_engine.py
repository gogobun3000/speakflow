"""
report_engine.py — SpeakFlow AI Layer
======================================
The ONLY place Claude API is used.

What it does:
  1. Combines multiple rule-engine error codes into coherent single-paragraph
     patient feedback (multi-error synthesis)
  2. Generates weekly clinician reports from session data
  3. Answers patient questions in session-appropriate language

What it does NOT do:
  - Generate real-time phoneme feedback  (that is rule_engine + clinical_templates)
  - Diagnose disorders
  - Override the measurement layer

Requires: ANTHROPIC_API_KEY environment variable
Falls back gracefully to template-only output if API unavailable.
"""

from __future__ import annotations

import os
import json
from typing import Optional

import clinical_templates as T
from rule_engine import ErrorCode

# ── API setup ─────────────────────────────────────────────────────────────────

_client = None

def _get_client():
    global _client
    if _client is not None:
        return _client
    try:
        import anthropic
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            return None
        _client = anthropic.Anthropic(api_key=key)
        return _client
    except ImportError:
        return None


API_AVAILABLE = bool(os.environ.get("ANTHROPIC_API_KEY"))


# ── Template-only fallback ────────────────────────────────────────────────────

def _template_feedback(codes: list[ErrorCode], word: str, score: float, word_correct: bool) -> dict:
    """
    Pure template output — no AI.
    Used when API is unavailable or for single-error cases.
    """
    headline = T.cue(T.get_headline(score, word_correct) if hasattr(T, 'get_headline')
                     else _headline_code(score, word_correct))

    tips, clinician_notes = [], []
    for ec in codes[:3]:   # cap at 3 to avoid overwhelming
        tmpl = T.get(ec.code)
        cue  = tmpl.get("cue")
        ex   = tmpl.get("exercise")
        ana  = tmpl.get("analogy")
        note = tmpl.get("clinician", "")

        if cue:   tips.append(cue)
        if ex:    tips.append(ex)
        if ana:   tips.append(ana)
        if note:  clinician_notes.append(f"[{ec.code}] {note}")

    return {
        "headline":        headline,
        "tips":            tips[:4],
        "clinician_notes": clinician_notes,
        "source":          "templates",
    }


def _headline_code(score: float, word_correct: bool) -> str:
    if not word_correct: return "HEADLINE_WRONG_WORD"
    if score >= 0.95:    return "HEADLINE_PERFECT"
    if score >= 0.80:    return "HEADLINE_VERY_GOOD"
    if score >= 0.60:    return "HEADLINE_GOOD"
    if score >= 0.40:    return "HEADLINE_KEEP_GOING"
    return "HEADLINE_TRY_AGAIN"


# ── Multi-error synthesis (AI) ────────────────────────────────────────────────

def synthesise_from_finding(
    finding:     dict,
    word:        str,
    score:       float,
    word_correct:bool,
    attempt_num: int  = 1,
    improving:   bool = False,
    patient_age: int  = 15,
) -> dict:
    """
    Generate AI feedback from a structured ArticulatorFinding dict.
    This is the preferred path when articulator_engine has produced a finding.

    The AI receives the clinical instruction and only handles tone/naturalness.
    """
    headline = T.cue(_headline_code(score, word_correct))
    clinician_note = (
        f"[{finding.get('articulator')}/{finding.get('problem')}] "
        f"Severity {finding.get('severity')}/100. "
        f"Stage {finding.get('intervention_stage')}: {finding.get('stage_name')}. "
        f"{finding.get('clinician_note','')}"
    )

    client = _get_client()
    if not client or not API_AVAILABLE:
        # Template-only: use the protocol instruction directly
        tips = [finding.get("instruction", "")]
        if finding.get("facilitating_words"):
            words = ", ".join(finding["facilitating_words"][:4])
            tips.append(f"Good practice words: {words}.")
        return {
            "headline": headline,
            "tips": tips,
            "clinician_notes": [clinician_note],
            "source": "articulator_protocol",
        }

    # AI polish: make the clinical instruction natural and age-appropriate
    ctx = finding.get("ai_prompt_context", {})
    attempt_note = ""
    if attempt_num > 5 and not improving:
        attempt_note = f"The patient has attempted this {attempt_num} times without improvement. Acknowledge the difficulty warmly and suggest they take a short break before trying again."

    prompt = f"""You are a warm, encouraging speech therapy assistant for a {patient_age}-year-old.

The clinical system has identified:
- Articulator: {ctx.get('articulator', '')}
- Problem: {ctx.get('problem', '')}
- Disorder: {ctx.get('disorder', '')}
- Intervention stage: {ctx.get('stage_num', 1)} — {ctx.get('stage_name', '')}

The clinician-approved instruction to deliver is:
"{ctx.get('instruction', '')}"

{attempt_note}

Write a single paragraph (2–3 sentences) that:
- Delivers the instruction above naturally and warmly
- Uses the exact technique described — do not change or add clinical content
- Speaks directly to a teenager without clinical jargon
- Is encouraging but specific — praise what was right if the score is above 70%
- Score was {int(score*100)}% on the word "{word}"

Output only the paragraph."""

    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=180,
            messages=[{"role": "user", "content": prompt}],
        )
        combined = msg.content[0].text.strip()
    except Exception as e:
        print(f"  [report_engine] AI polish failed: {e}")
        combined = finding.get("instruction", "")

    tips = [combined]
    if finding.get("facilitating_words"):
        words = ", ".join(finding["facilitating_words"][:4])
        tips.append(f"Good practice words to try: {words}.")

    return {
        "headline":        headline,
        "tips":            tips,
        "clinician_notes": [clinician_note],
        "source":          "articulator_ai",
    }


def synthesise_feedback(
    codes:       list[ErrorCode],
    word:        str,
    score:       float,
    word_correct:bool,
    patient_age: int = 15,
) -> dict:
    """
    Combine multiple error codes into a single coherent feedback paragraph.

    Single error → template only (fast, consistent).
    Multiple errors → AI synthesis for coherent paragraph,
                      with template fallback if API unavailable.
    """
    if len(codes) <= 1 or not API_AVAILABLE:
        return _template_feedback(codes, word, score, word_correct)

    client = _get_client()
    if not client:
        return _template_feedback(codes, word, score, word_correct)

    # Build the prompt from template strings so the AI never invents clinical content
    template_cues = []
    for ec in codes[:3]:
        tmpl = T.get(ec.code)
        parts = [tmpl.get("cue", ""), tmpl.get("exercise", ""), tmpl.get("analogy", "")]
        template_cues.append({
            "code":     ec.code,
            "severity": ec.severity,
            "cue":      tmpl.get("cue", ""),
            "exercise": tmpl.get("exercise"),
            "analogy":  tmpl.get("analogy"),
        })

    prompt = f"""You are helping a {patient_age}-year-old practise the word "{word}".
Their speech score was {int(score*100)}%.

The rule engine identified these issues (DO NOT invent any additional issues):
{json.dumps(template_cues, indent=2)}

Write a single short paragraph of feedback for the patient.
Rules:
- Use ONLY the cue/exercise/analogy text provided above — do not add clinical content
- Combine the points naturally so they flow as one piece of advice
- Keep total length under 80 words
- Use encouraging, clear language for a teenager
- Do not mention Hz, formants, or clinical terminology
- Do not mention scores or percentages
- Start with what went wrong, then give ONE clear action
Output only the paragraph, nothing else."""

    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        combined = msg.content[0].text.strip()
    except Exception as e:
        print(f"  [report_engine] AI synthesis failed: {e}")
        return _template_feedback(codes, word, score, word_correct)

    headline = T.cue(_headline_code(score, word_correct))
    clinician_notes = [f"[{ec.code}/{ec.severity}] {T.clinician_note(ec.code)}" for ec in codes]

    return {
        "headline":        headline,
        "tips":            [combined],
        "clinician_notes": clinician_notes,
        "source":          "ai_synthesised",
    }


# ── Weekly clinician report (AI) ─────────────────────────────────────────────

def generate_report(
    summary:    dict,
    patient_id: str = "Patient",
) -> str:
    """
    Generate a plain-English weekly report for a clinician from session_store summary data.

    summary: the dict returned by session_store.summary(days=7)
    Returns: a Markdown-formatted report string.
    """
    client = _get_client()

    # Build the raw data section from measurements (never from AI)
    words = summary.get("words", [])
    data_lines = []
    for w in words:
        lisp = w.get("lisp_risk", "—")
        f1   = f"{w['f1_avg']:+.0f}Hz" if w.get("f1_avg") is not None else "—"
        cog  = f"{w['cog_avg']:.0f}Hz" if w.get("cog_avg") is not None else "—"
        data_lines.append(
            f"- **{w['word']}**: {w['pct']}% accuracy ({w['n']} attempts), "
            f"trend: {w['trend']}, F1 avg diff: {f1}, /s/ CoG: {cog}, lisp risk: {lisp}"
        )

    data_section = "\n".join(data_lines) if data_lines else "No attempts recorded."

    # Template-only report (no AI needed for structure)
    report = f"""# SpeakFlow — Weekly Progress Report
**{patient_id}**  |  Period: last {summary.get('days', 7)} days

## Summary
- Total attempts: {summary.get('total', 0)}
- Overall accuracy: {summary.get('pct', 0)}%
- Words practised: {summary.get('n_words', 0)}

## Word-by-word breakdown
{data_section}

## Acoustic notes (auto-generated)
"""
    # Add auto-generated acoustic notes from templates
    acoustic_notes = []
    for w in words:
        if w.get("cog_avg") and w["cog_avg"] < 4800:
            note = T.clinician_note("LISP_INTERDENTAL" if w["cog_avg"] < 3800 else "LISP_ADDENTAL")
            acoustic_notes.append(f"- **{w['word']}**: {note}")
        if w.get("f1_avg") and abs(w["f1_avg"]) > 150:
            code = "TONGUE_TOO_LOW" if w["f1_avg"] > 0 else "TONGUE_TOO_HIGH"
            acoustic_notes.append(f"- **{w['word']}** (vowel): {T.clinician_note(code)}")

    report += "\n".join(acoustic_notes) if acoustic_notes else "No significant acoustic deviations detected.\n"

    if not client or not API_AVAILABLE:
        report += "\n\n*Report generated by rule engine (AI synthesis unavailable)*"
        return report

    # AI: write a one-paragraph clinical interpretation
    try:
        prompt = f"""You are a speech pathologist writing a brief clinical summary.
Data: {json.dumps({'summary': summary, 'word_data': words[:8]}, default=str)}

Write 2-3 sentences of clinical interpretation for a colleague.
- Reference specific words and sounds by name
- Note any trends (improving, declining, consistent errors)
- Suggest one clinical priority for next session
- Use standard SLP terminology
- Do not mention the app name or technology
- Do not make diagnostic statements
Output only the paragraph."""

        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=250,
            messages=[{"role": "user", "content": prompt}],
        )
        report += f"\n\n## Clinical interpretation\n{msg.content[0].text.strip()}"
        report += "\n\n*Interpretation generated by AI — clinician should verify*"
    except Exception as e:
        report += f"\n\n*AI interpretation unavailable: {e}*"

    return report


# ── Patient Q&A (AI) ─────────────────────────────────────────────────────────

def answer_question(
    question:    str,
    session_ctx: dict | None = None,
    patient_age: int = 15,
) -> str:
    """
    Answer a patient question in age-appropriate language.
    Stays strictly within the scope of the session data — no diagnosis.
    Falls back to a safe template if API unavailable.
    """
    safe_fallback = (
        "That's a great question! Ask your speech pathologist at your next session — "
        "they'll be able to give you the best answer based on how you're going."
    )

    client = _get_client()
    if not client:
        return safe_fallback

    ctx_str = json.dumps(session_ctx, default=str) if session_ctx else "No session data available."

    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            system=(
                f"You are a friendly speech therapy app assistant helping a {patient_age}-year-old. "
                "Answer questions about their practice only. "
                "Never diagnose, never give medical advice. "
                "If unsure, say to ask their speech pathologist. "
                "Keep answers under 50 words."
            ),
            messages=[
                {"role": "user", "content": f"Session data: {ctx_str}\n\nQuestion: {question}"}
            ],
        )
        return msg.content[0].text.strip()
    except Exception:
        return safe_fallback
