"""
session_store.py — SQLite persistence for SpeakFlow
=====================================================
Saves every exercise attempt so clinicians can track patient progress.
Database lives next to this file as speakflow.db.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speakflow.db")


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS attempts (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                attempted_at  TEXT    DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
                word          TEXT    NOT NULL,
                target_ipa    TEXT,
                heard         TEXT,
                heard_ipa     TEXT,
                score         REAL,
                word_correct  INTEGER,
                -- acoustic: first vowel phoneme
                phoneme       TEXT,
                f1_hz         REAL,
                f2_hz         REAL,
                f1_ref        REAL,
                f2_ref        REAL,
                f1_diff       REAL,
                f2_diff       REAL,
                -- acoustic: frication
                cog_hz        REAL,
                lisp_risk     TEXT,
                -- metadata
                using_ref     INTEGER DEFAULT 0,
                align_method  TEXT
            )
        """)


# ── Write ─────────────────────────────────────────────────────────────────────

def save(data: dict) -> int:
    """Insert one attempt. Returns the new row id."""
    init_db()
    with _conn() as c:
        cur = c.execute("""
            INSERT INTO attempts
                (word, target_ipa, heard, heard_ipa, score, word_correct,
                 phoneme, f1_hz, f2_hz, f1_ref, f2_ref, f1_diff, f2_diff,
                 cog_hz, lisp_risk, using_ref, align_method)
            VALUES
                (:word,:target_ipa,:heard,:heard_ipa,:score,:word_correct,
                 :phoneme,:f1_hz,:f2_hz,:f1_ref,:f2_ref,:f1_diff,:f2_diff,
                 :cog_hz,:lisp_risk,:using_ref,:align_method)
        """, {
            "word":        data.get("word", ""),
            "target_ipa":  data.get("target_ipa", ""),
            "heard":       data.get("heard", ""),
            "heard_ipa":   data.get("heard_ipa", ""),
            "score":       data.get("score"),
            "word_correct":int(bool(data.get("word_correct"))),
            "phoneme":     data.get("phoneme", ""),
            "f1_hz":       data.get("f1_hz"),
            "f2_hz":       data.get("f2_hz"),
            "f1_ref":      data.get("f1_ref"),
            "f2_ref":      data.get("f2_ref"),
            "f1_diff":     data.get("f1_diff"),
            "f2_diff":     data.get("f2_diff"),
            "cog_hz":      data.get("cog_hz"),
            "lisp_risk":   data.get("lisp_risk", ""),
            "using_ref":   int(bool(data.get("using_ref"))),
            "align_method":data.get("align_method", ""),
        })
        return cur.lastrowid


# ── Read ──────────────────────────────────────────────────────────────────────

def recent(limit: int = 60) -> list[dict]:
    init_db()
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM attempts ORDER BY attempted_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def word_history(word: str, limit: int = 30) -> list[dict]:
    init_db()
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM attempts WHERE word=? ORDER BY attempted_at LIMIT ?",
            (word.lower(), limit),
        ).fetchall()
    return [dict(r) for r in rows]


def summary(days: int = 7) -> dict:
    """Aggregate stats for the last N days — used for the weekly report."""
    init_db()
    since = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")

    with _conn() as c:
        total   = c.execute("SELECT COUNT(*) FROM attempts WHERE attempted_at>=?", (since,)).fetchone()[0]
        correct = c.execute("SELECT COUNT(*) FROM attempts WHERE attempted_at>=? AND word_correct=1", (since,)).fetchone()[0]
        words   = [r[0] for r in c.execute("SELECT DISTINCT word FROM attempts WHERE attempted_at>=?", (since,)).fetchall()]

        word_stats = []
        for w in words:
            rows = c.execute(
                """SELECT score, word_correct, f1_diff, f2_diff,
                          cog_hz, lisp_risk, attempted_at
                   FROM attempts WHERE word=? AND attempted_at>=?
                   ORDER BY attempted_at""",
                (w, since),
            ).fetchall()
            attempts = [dict(r) for r in rows]
            n         = len(attempts)
            n_correct = sum(1 for a in attempts if a["word_correct"])
            avg_score = sum(a["score"] or 0 for a in attempts) / n if n else 0

            # Score trend: compare first vs second half
            trend = "—"
            if n >= 4:
                mid  = n // 2
                fst  = sum(a["score"] or 0 for a in attempts[:mid]) / mid
                snd  = sum(a["score"] or 0 for a in attempts[mid:]) / (n - mid)
                trend = "↑ improving" if snd > fst + 0.05 else \
                        "↓ declining" if snd < fst - 0.05 else "→ stable"

            # Average F1 deviation (jaw position)
            f1s = [a["f1_diff"] for a in attempts if a["f1_diff"] is not None]
            f1_avg = round(sum(f1s) / len(f1s)) if f1s else None

            # Average CoG (lisp indicator)
            cogs = [a["cog_hz"] for a in attempts if a["cog_hz"] is not None]
            cog_avg = round(sum(cogs) / len(cogs)) if cogs else None

            # Latest lisp risk
            lisp_latest = next((a["lisp_risk"] for a in reversed(attempts) if a["lisp_risk"]), None)

            # Consecutive correct streak from most recent
            streak = 0
            for a in reversed(attempts):
                if a["word_correct"]:
                    streak += 1
                else:
                    break

            word_stats.append({
                "word":      w,
                "n":         n,
                "correct":   n_correct,
                "pct":       round(n_correct / n * 100) if n else 0,
                "avg_score": round(avg_score, 2),
                "trend":     trend,
                "f1_avg":    f1_avg,
                "cog_avg":   cog_avg,
                "lisp_risk": lisp_latest or "—",
                "streak":    streak,
            })

        word_stats.sort(key=lambda x: x["n"], reverse=True)

    return {
        "days":    days,
        "since":   since,
        "total":   total,
        "correct": correct,
        "pct":     round(correct / total * 100) if total else 0,
        "n_words": len(words),
        "words":   word_stats,
    }
