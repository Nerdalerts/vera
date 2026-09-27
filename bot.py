"""Vera — magicpin AI Challenge bot (single-file submission).

    uvicorn bot:app --host 0.0.0.0 --port 8080 --workers 1

Endpoints (challenge-testing-brief §2): /v1/context, /v1/tick, /v1/reply,
/v1/healthz, /v1/metadata, plus optional /v1/teardown.

Also exposes the brief's plain-Python contract, compose() (§7.1), backed by
the same pipeline as the HTTP endpoints.

Design in one line: deterministic code decides WHETHER and HOW to act
(routing, suppression, auto-reply/intent/hostility detection, timing); the
LLM only writes words, and a deterministic validator decides whether those
words may leave the process.

This file is a flattened, standalone build of a modular FastAPI project
(originally vera/config.py, llm.py, facts.py, classifier.py, validator.py,
brief.py, store.py, composer.py + prompts/composer_system_prompt.md) —
merged into one script to match the challenge's single-bot.py submission
format. Every regex, constant and code path below is unchanged from the
modular version; only cross-module imports were inlined. Optional
conversation_handlers.py (from vera-bot) still works unmodified against
this file, since it only does `from bot import ReplyBody, handle_reply`.

Solo developer: Adarsh Rawat. LLM provider: OpenAI (LLM_PROVIDER=openai by
default; see the "LLM provider adapters" section for how OPENAI_API_KEY is
read and used).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Awaitable, Callable, Iterable, Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

try:  # optional: local .env support for `uvicorn bot:app` during dev
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("vera.bot")


# =============================================================================
# Configuration (was vera/config.py) — read from the environment once at
# import time so a misconfigured deploy fails loudly on boot, not mid-test.
# =============================================================================

def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


# --- LLM ---------------------------------------------------------------------
# openai | anthropic  ("openai" covers any OpenAI-compatible endpoint too:
# OpenAI, Groq, Together, OpenRouter, Gemini's /openai endpoint, vLLM, ...)
#
# This submission uses the OpenAI API: LLM_PROVIDER defaults to "openai" and
# OPENAI_API_KEY is read directly from the environment (see is_configured()
# and _openai() below). The Anthropic adapter is kept in the file as a
# zero-cost, opt-in fallback (set LLM_PROVIDER=anthropic + ANTHROPIC_API_KEY
# to use it) — it is inert unless explicitly selected.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_MAX_TOKENS = int(_f("LLM_MAX_TOKENS", 900))
LLM_TEMPERATURE = _f("LLM_TEMPERATURE", 0.0)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# --- Time budgets (judge hard-times-out at 30s) ------------------------------
# The real judge waits 30s; magicpin's local judge_simulator.py waits only 15s
# for /v1/tick and /v1/reply. Defaults are sized for the stricter of the two.
TICK_BUDGET_SECONDS = _f("TICK_BUDGET_SECONDS", 13.0)
REPLY_BUDGET_SECONDS = _f("REPLY_BUDGET_SECONDS", 13.0)
LLM_CALL_TIMEOUT = _f("LLM_CALL_TIMEOUT", 10.0)  # per attempt
BACKGROUND_BUDGET_SECONDS = _f("BACKGROUND_BUDGET_SECONDS", 40.0)  # a composition may outlive one tick
LLM_CONCURRENCY = int(_f("LLM_CONCURRENCY", 6))  # global cap on simultaneous LLM calls
MIN_ATTEMPT_SECONDS = 2.0  # don't start an LLM call with less runway than this
MAX_ACTIONS_PER_TICK = 20
MAX_COMPOSE_ATTEMPTS = 3  # first try + up to two *informed* repairs (time permitting)

# --- Message shape ------------------------------------------------------------
MAX_BODY_CHARS = int(_f("MAX_BODY_CHARS", 640))
MIN_BODY_CHARS = 25

# --- Metadata (solo developer) ------------------------------------------------
# Adarsh Rawat is a solo developer on this submission. team_members intentionally
# has a single entry; env vars can still override these for a different deploy.
TEAM_NAME = os.getenv("TEAM_NAME", "Adarsh Rawat")
TEAM_MEMBERS = [m.strip() for m in os.getenv("TEAM_MEMBERS", "Adarsh Rawat").split(",") if m.strip()]
# NOTE: set your real contact email via the CONTACT_EMAIL env var (or edit the
# default below) before submitting — it is not filled in automatically.
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "REPLACE_WITH_YOUR_EMAIL@example.com")
BOT_VERSION = "2.0.0"


# =============================================================================
# System prompt (was prompts/composer_system_prompt.md) — the LLM's only job
# is to write words inside the rules below; everything else in this file is
# deterministic code that decides whether/how to act.
# =============================================================================

SYSTEM_PROMPT = """You are the writing engine behind **Vera**, magicpin's WhatsApp assistant for Indian local merchants (dentists, salons, restaurants, gyms, pharmacies). Sometimes you write as Vera to the merchant owner; sometimes you draft a message the merchant sends to one of their own customers.

A deterministic system around you has already decided THAT a message should be sent and in WHICH mode. Your only job is to write the best possible WhatsApp message for this one moment, and return it as JSON.

## What you receive (one JSON object)

- `mode` — `first_touch` (proactive message), or a reply mode: `action`, `answer`, `deescalate`, `continue`. See "Reply modes".
- `brief` — the authoritative, pre-digested evidence. Who you are writing to (`audience`, `address_as`), the `language` to write in, the `voice`, the resolved `why_now` (trigger facts, the resolved digest item, day-counts already computed against today's date), the merchant's numbers (already converted to % and compared with category averages), their offers, and a `playbook` for this kind of moment. **Prefer the brief over the raw contexts** — it has done the lookups and arithmetic correctly.
- `contexts` — the raw category / merchant / trigger / customer JSON, for anything the brief left out.
- `conversation` — for replies: the transcript so far and a read of the merchant's latest message.
- `must_not_repeat` — earlier messages in this conversation. Say something new.
- `repair` — present only if your previous draft was rejected. It lists the exact violations. Fix every one; change nothing else that was working.

## Non-negotiable rules

1. **Only facts from the input.** Every number, name, date, price, source, statistic and claim must be copied from `brief` or `contexts`. Never estimate, round up a vague claim, or bring in outside knowledge ("most patients…", "studies show…"). Never name a journal, body, brand or competitor that is not in the input. A number you want but don't have → leave it out, or ask the merchant for it.
2. **No arithmetic of your own.** Day-counts, % changes and comparisons are in the brief. Use those exact values. Fractions in raw data are rates: 0.021 is "2.1%".
3. **Thin events stay thin.** If `brief.why_now.thin_trigger_warning` is present, the event has no details. Do not invent them (no fake counts, names, times, amounts). Build the message on the strongest real number in the brief instead.
4. **One call to action, and it is the last sentence.** One binary ask ("Reply YES and I'll…", "Reply YES / STOP"), or one open question, or none for pure information. Booking messages to customers may offer at most 2 numbered slots taken from the input.
5. **No generic marketing copy.** Never write "X% off" unless it is literally one of the merchant's own active offers. Prefer service@price from `merchant_offers_active`, then `catalog_offers` ("Dental Cleaning @ ₹299"). Banned: "grow your business", "increase your sales", "amazing deal", "hurry", "limited time", "Dear Sir/Madam", "hope you are doing well", and any greeting preamble.
6. **Never say a word in `brief.voice.never_say`.** No medical or result promises.
7. **No system words in the message.** Never write field names (anything_with_underscores), or "trigger", "signal", "payload", "placeholder". Say "your calls dropped 50% this week", not "perf_dip".
8. **Don't re-introduce yourself** if the conversation or `recent_history` shows Vera has already spoken to this merchant.
9. **If the input really cannot support a specific, honest message, return** `{"skip": true, "reason": "..."}`. Silence beats a generic or invented message.

## How a great Vera message is built

Keep it to 2–4 short sentences, usually 250–450 characters. WhatsApp, not email.

1. **Hook (first ~12 words):** address them the way `brief.address_as` says (dentists are "Dr. <first name>"), then go straight to the why-now fact, e.g. "Dr. Meera, JIDA's Oct issue has one for your high-risk adults —". No "Hi, hope you're well".
2. **So-what for THIS merchant:** connect the fact to their own number, offer, cohort, locality or past conversation. This is what makes the message unmistakably theirs, not a broadcast.
3. **Effort externalization:** say what Vera has ready or will do ("I've drafted…", "I can pull the list in 2 minutes"). Only promise things Vera can plausibly do: draft posts, messages, offers, checklists, pull lists from their data, update the Google profile.
4. **CTA:** one low-friction ask, as the final sentence.

Pull the levers the `playbook` names. Two levers are under-used by today's Vera and win engagement when the input supports them:
- **Social proof**, only from real input: category averages ("most metro solo practices average 62 reviews; you're at …"), trend or digest data. Never invent "3 salons near you did X".
- **Asking the merchant**: a sharp, easy question only they can answer ("What's the most-asked service this week?"), plus what you'll do with the answer.

**Language:** follow `brief.language` exactly. For Hindi-English code-mix, write natural Roman-script Hinglish the way a Delhi or Mumbai professional texts ("Aapke 124 high-risk adult patients ke liye relevant hai"), keeping technical words in English. Don't use Devanagari unless the merchant wrote in it. In replies, mirror the language of the merchant's latest message.

**Voice by category** (refine with `brief.voice.tone` and `tone_examples`):
- dentists: clinical peer, precise, cites sources; technical terms welcome; never salesy.
- salons: warm, practical, stylist-to-owner.
- restaurants: fellow operator, busy and practical (covers, AOV, delivery times).
- gyms: coach energy, disciplined, numbers-first.
- pharmacies: trustworthy and precise; batch numbers, molecules, dates exact; compliance first.

**Customer-facing** (`send_as` = `merchant_on_behalf`): write as the merchant's own team ("Dr. Meera's clinic here"). Use the customer's first name, reference their real history (last visit, services), use a real price and real slots, and honor their preferred time. Never mention the merchant's analytics, magicpin internals, or anything outside the customer's consent scope. No health claims.

## Reply modes (when `mode` is not `first_touch`)

- `action` — the merchant just said yes / "let's do it" / "go ahead". **Switch to doing.** Confirm and show the first concrete deliverable right in the message (e.g. the actual post draft, the offer line, the checklist items, or the exact next step with a timeframe). Do **not** ask another qualifying question: no "would you", "do you", "what if", "how about". The only question allowed is a final binary confirm ("Reply YES to publish"). If a detail is missing (a price, a date), put a clear blank for them to fill (₹___) rather than asking a discovery question.
- `answer` — the merchant asked something about their business, their customers, magicpin or the thread. Answer from the input in one or two lines. (If the question is outside Vera's job, e.g. taxes, legal or loans, handle it as `deescalate`.) If the input doesn't hold the answer, say plainly that you'll check and confirm, and don't guess. Then move the thread one step forward.
- `deescalate` — the merchant was rude, or asked something off-mission (e.g. GST filing, unrelated tech help). Open with a short, genuine apology if they were upset ("Sorry for the extra messages."). For off-mission asks, say in one line that it's outside what you can help with here; give no advice on it. Then offer exactly one relevant thing from the brief, or offer to stop ("Reply STOP and I won't message again"). No pitch, no defensiveness.
- `continue` — a normal engaged reply. Acknowledge in a few words, then advance to the next concrete step. Don't repeat anything from `must_not_repeat`.

## Output

Return ONLY this JSON object, with no prose and no code fences:

{"body": "...", "cta": "binary" | "open_ended" | "none", "send_as": "vera" | "merchant_on_behalf", "suppression_key": "...", "template_params": ["values substituted into the approved template, e.g. name, key fact, offer"], "facts_used": ["exact values copied from the input that appear in body"], "rationale": "1-2 sentences: which input facts you used (by field) and which lever, and why this message now for this merchant"}

or {"skip": true, "reason": "..."}.

The `rationale` is read by the judge. Make it specific to this merchant and moment. A rationale that would fit any merchant means the message is too generic, so rewrite the message."""

PROMPT_HASH = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:10]


# =============================================================================
# LLM provider adapters (was vera/llm.py) — plain HTTPS via httpx, no vendor
# SDKs. This is where OPENAI_API_KEY is read and the OpenAI-compatible async
# call is made.
# =============================================================================

_client: Optional[httpx.AsyncClient] = None


def _http_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(LLM_CALL_TIMEOUT + 2, connect=5.0))
    return _client


class LLMError(RuntimeError):
    pass


async def _anthropic(system: str, user: str, timeout: float) -> str:
    if not ANTHROPIC_API_KEY:
        raise LLMError("ANTHROPIC_API_KEY not set")
    body = {
        "model": LLM_MODEL,
        "max_tokens": LLM_MAX_TOKENS,
        "temperature": LLM_TEMPERATURE,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    url = f"{ANTHROPIC_BASE_URL.rstrip('/')}/v1/messages"
    r = await _http_client().post(url, json=body, headers=headers, timeout=timeout)
    if r.status_code == 400 and "temperature" in r.text:
        body.pop("temperature")  # some models fix sampling params server-side
        r = await _http_client().post(url, json=body, headers=headers, timeout=timeout)
    if r.status_code != 200:
        raise LLMError(f"anthropic {r.status_code}: {r.text[:300]}")
    data = r.json()
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


async def _openai(system: str, user: str, timeout: float) -> str:
    if not OPENAI_API_KEY:
        raise LLMError("OPENAI_API_KEY not set")
    body = {
        "model": LLM_MODEL,
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
        "seed": 7,  # best-effort determinism where supported
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    headers = {"authorization": f"Bearer {OPENAI_API_KEY}", "content-type": "application/json"}
    url = f"{OPENAI_BASE_URL.rstrip('/')}/chat/completions"
    r = await _http_client().post(url, json=body, headers=headers, timeout=timeout)
    if r.status_code == 400:
        # compatible servers vary: drop optional params they reject, retry once
        for k in ("response_format", "seed", "temperature"):
            body.pop(k, None)
        body["max_completion_tokens"] = body.pop("max_tokens")
        r = await _http_client().post(url, json=body, headers=headers, timeout=timeout)
    if r.status_code != 200:
        raise LLMError(f"openai-compatible {r.status_code}: {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"] or ""


PROVIDERS: dict[str, Callable[[str, str, float], Awaitable[str]]] = {
    "anthropic": _anthropic,
    "openai": _openai,
}

# Tests can swap this for a scripted fake; production resolves from env.
_override: Optional[Callable[[str, str, float], Awaitable[str]]] = None


def set_override(fn: Optional[Callable[[str, str, float], Awaitable[str]]]) -> None:
    global _override
    _override = fn


async def complete(system: str, user: str, timeout: float) -> str:
    fn = _override or PROVIDERS.get(LLM_PROVIDER)
    if fn is None:
        raise LLMError(f"unknown LLM_PROVIDER {LLM_PROVIDER!r}")
    return await fn(system, user, timeout)


def is_configured() -> bool:
    if _override:
        return True
    if LLM_PROVIDER == "openai":
        return bool(OPENAI_API_KEY)
    return False


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)


def parse_json_object(text: str) -> dict:
    """Strict-but-forgiving: strips code fences / leading prose, then decodes
    the first complete JSON object. Anything else is a ValueError."""
    if not text:
        raise ValueError("empty completion")
    t = _FENCE.sub("", text.strip())
    start = t.find("{")
    if start < 0:
        raise ValueError("no JSON object in completion")
    obj, _ = json.JSONDecoder().raw_decode(t[start:])
    if not isinstance(obj, dict):
        raise ValueError("completion JSON is not an object")
    return obj


# =============================================================================
# Grounding substrate (was vera/facts.py)
#
# Builds, from the raw context JSON, the two things the validator needs:
#   1. a lowercase text blob of every groundable string, and
#   2. an index of every groundable *number* (as floats, with the common
#      re-expressions a copywriter legitimately uses: 0.021 -> 2.1%,
#      1499 -> "1.5k" is NOT added on purpose, see below).
#
# Derived numbers (day counts, % gaps vs peers) are computed in build_brief()
# and indexed here too, so the model never has to do arithmetic to be specific.
# =============================================================================

NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def dig(obj: Any, *paths: str, default: Any = None) -> Any:
    """dig(m, "identity.name", "name") -> first path that resolves."""
    for path in paths:
        cur = obj
        ok = True
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok and cur not in (None, "", [], {}):
            return cur
    return default


def as_list(x: Any) -> list:
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def iter_leaves(obj: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from iter_leaves(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from iter_leaves(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def parse_number(tok: str) -> Optional[float]:
    try:
        return float(tok.replace(",", ""))
    except ValueError:
        return None


ISO_TIME_RE = re.compile(r"T(\d{2}):(\d{2})")


class GroundingIndex:
    """Everything a message may state as a number or name. Built from the raw
    contexts PLUS the evidence brief (which only holds values copied from, or
    computed deterministically out of, those contexts)."""

    def __init__(self, *payloads: Optional[dict], extra_text: str = ""):
        parts = [json.dumps(p, ensure_ascii=False) for p in payloads if p]
        parts.append(extra_text)
        self.blob = " ".join(parts).lower()

        nums: set[float] = set()
        for tok in NUM_RE.findall(" ".join(parts)):
            v = parse_number(tok)
            if v is None:
                continue
            nums.add(abs(v))
            if 0 < abs(v) < 1:  # rates stored as fractions -> copy says "2.1%"
                nums.add(round(abs(v) * 100, 4))
        # Typed sub-indexes: a "45%" in the copy must match something that is a
        # percentage in context (not just any 45, e.g. 45 direction requests),
        # and "₹299" must match a money value, not a count.
        text = " ".join(parts)
        self.pct_numbers: set[float] = set()
        for tok in re.findall(r"([+-]?\d[\d,]*(?:\.\d+)?)\s*%", text):
            v = parse_number(tok); v is not None and self.pct_numbers.add(abs(v))
        for tok in re.findall(r"[_\s(]([+-]\d+(?:\.\d+)?)\b", text):  # "ORS_demand_+40", "(+18%)"
            v = parse_number(tok); v is not None and self.pct_numbers.add(abs(v))
        for v in nums:
            if 0 < v < 1:
                self.pct_numbers.add(round(v * 100, 4))
        self.money_numbers: set[float] = set()
        for tok in re.findall(r"(?:₹|rs\.?\s?|inr\s?)\s?(\d[\d,]*(?:\.\d+)?)", text, re.I):
            v = parse_number(tok); v is not None and self.money_numbers.add(v)
        for p in payloads:
            for path, val in iter_leaves(p or {}):
                if isinstance(val, (int, float, str)) and re.search(r"(value|price|amount|fee|mrp|cost|lifetime_value|aov)$", path.split(".")[-1].split("[")[0], re.I):
                    v = parse_number(str(val)) if isinstance(val, str) else float(val)
                    if v is not None:
                        self.money_numbers.add(abs(v))
        # "2026-04-26T19:30" may be written as "7:30 PM"
        for h, m in ISO_TIME_RE.findall(" ".join(parts)):
            nums.add(float(int(h) % 12 or 12))
            nums.add(float(int(m)))
        self.numbers = nums

    def has_number(self, value: float, decimals: int, kind: str = "plain") -> bool:
        pool = self.pct_numbers if kind == "pct" else self.money_numbers if kind == "money" else self.numbers
        for n in pool:
            if abs(n - value) < 1e-9:
                return True
            # a copywriter may round 4.63 -> 4.6; allowed only when the body
            # itself shows decimals (so "5" can never "match" 4.63)
            if decimals > 0 and abs(round(n, decimals) - value) < 1e-9:
                return True
        return False

    def has_text(self, s: str) -> bool:
        return s.lower().strip() in self.blob


# =============================================================================
# Deterministic reply classifier (was vera/classifier.py) — reads an inbound
# reply BEFORE any LLM call. Order of checks matters (see classify()).
# =============================================================================

OPT_OUT_RE = re.compile(
    r"^\s*stop\s*[.!]*\s*$|\b(unsubscribe|opt[\s-]?out|band karo|mat bhejo|message mat|msg mat|"
    r"don'?t (message|text|msg|contact) me|do not (message|text|contact)|remove me|no more (messages|msgs)|"
    r"stop (messaging|sending|texting|this))\b",
    re.I,
)

AUTO_REPLY_RE = re.compile(
    r"(thank(s| you) for (contacting|reaching out|your message|messaging)|we (will|'ll) get back|"
    r"will (revert|respond|get back) (to you )?(shortly|soon|asap)|currently (unavailable|closed|away|busy|out)|"
    r"out of (the )?office|business hours|working hours are|this is an auto(mated)?[- ]?(reply|response|message)|"
    r"auto[- ]?reply|our (team|executive) will (contact|call|reach)|aapka (sandesh|message) (mil gaya|prapt)|"
    r"hum (jald|jaldi) (hi )?(sampark|contact)|we are closed|shop (is )?closed (now|today)|"
    r"automated assistant|virtual assistant|jaankari ke liye .{0,20}shukriya|team tak pahuncha|"
    r"sandesh ke liye dhanyavaad|message ke liye (dhanyavaad|shukriya)|respond (shortly|soon)|"
    r"away from (the )?(phone|desk)|reply (you )?(shortly|as soon as possible))",
    re.I,
)

INTENT_RE = re.compile(
    r"(let'?s do (it|this)|lets do|go ahead|please proceed|\bproceed\b|sign me up|i'?m in\b|count me in|book it|"
    r"\bconfirmed\b|confirm (it|karo|kar do)|ok(ay)? do it|do it\b|start (it|karo|kar do|now)|chalo (start|shuru|karte)|shuru karo|"
    r"kar do\b|haan (chalo|karo|kar do|theek|ji)|i want to (join|start|sign up|try|activate)|"
    r"mujhe join karna|activate (it|karo|kar do)|set it up|send (me )?(the )?(link|details)|how do i start|"
    r"sounds good,? (let'?s|go)|yes,? (please|start|go|do it|let'?s))",
    re.I,
)
BARE_YES_RE = re.compile(r"^\s*(yes|y|yep|yeah|haan|haa|ha|ji|ji haan|ok|okay|sure|done|theek hai|thik hai|👍)\s*[.!]*\s*$", re.I)

DECLINE_RE = re.compile(r"\b(not interested|no thanks|no thank you|nahi chahiye|interest nahi|zarurat nahi|no need)\b|^\s*(no|nahi|nope)\s*[.!]*\s*$", re.I)
LATER_RE = re.compile(r"\b(later|baad mein|abhi (nahi|busy)|busy (now|hu|hoon|hun)|call (you )?later|kal baat|tomorrow)\b", re.I)

HOSTILE_RE = re.compile(
    r"\b(stupid|idiot|shut up|spam(ming)?|scam|fraud|bakwas|pagal|chup|bewakoof|nonsense|harass(ing|ment)?|"
    r"useless|worst|irritat(e|ing)|pestering|f+u+c+k+|wtf|bloody|block(ing)? you|report(ing)? you)\b",
    re.I,
)
OFF_MISSION_RE = re.compile(
    r"\b(gst|income tax|itr|tax return|file my|loan|lawyer|legal notice|court|visa|passport|"
    r"electricity bill|laptop|computer repair|cricket score|horoscope|recipe|homework|stock tips?|crypto|bitcoin)\b",
    re.I,
)
QUESTION_RE = re.compile(r"\?|^\s*(what|how|why|when|where|which|who|kya|kaise|kitna|kitne|kab|kyun|kaun|kahan)\b", re.I)


@dataclass
class ReplyRead:
    label: str                  # opt_out|auto_reply|intent_transition|hostile|decline|later|off_topic|question|general|empty
    confidence: str             # high|medium
    notes: list[str] = field(default_factory=list)


def classify(message: str, *, repeat_count: int, last_bot_cta: str | None) -> ReplyRead:
    msg = (message or "").strip()
    if not msg:
        return ReplyRead("empty", "high")
    if OPT_OUT_RE.search(msg):
        return ReplyRead("opt_out", "high", ["explicit opt-out phrase"])
    if AUTO_REPLY_RE.search(msg):
        return ReplyRead("auto_reply", "high", ["canned auto-reply phrasing"])
    if repeat_count >= 3:
        return ReplyRead("auto_reply", "high", [f"identical message received {repeat_count}x"])
    if HOSTILE_RE.search(msg):
        return ReplyRead("hostile", "high", ["abusive / spam-accusation language"])
    if INTENT_RE.search(msg):
        return ReplyRead("intent_transition", "high", ["explicit go-ahead phrase"])
    if BARE_YES_RE.match(msg) and last_bot_cta == "binary":
        return ReplyRead("intent_transition", "high", ["bare 'yes' to a binary CTA"])
    if DECLINE_RE.search(msg):
        return ReplyRead("decline", "high", ["polite decline"])
    if LATER_RE.search(msg):
        return ReplyRead("later", "medium", ["merchant is busy / asks for later"])
    if OFF_MISSION_RE.search(msg):
        return ReplyRead("off_topic", "medium", ["request outside Vera's remit"])
    if repeat_count == 2:
        return ReplyRead("auto_reply", "medium", ["same message received twice"])
    if QUESTION_RE.search(msg):
        return ReplyRead("question", "medium")
    return ReplyRead("general", "medium")


# =============================================================================
# Deterministic guardrail layer (was vera/validator.py). No LLM in here —
# that is the point.
#
# validate() returns (errors, warnings):
#   errors   -> the message may NOT be sent. The composer feeds them back to
#               the model for one informed repair, then falls back to a no-op.
#   warnings -> quality issues worth one repair attempt, but acceptable on the
#               final attempt (e.g. English to a Hindi-preferring merchant is
#               still valid WhatsApp copy, just weaker).
# =============================================================================

GENERIC_PATTERNS = [
    r"\bflat\s*\d+\s*%\s*off\b",
    r"\bup\s*to\s*\d+\s*%\s*off\b",
    r"\bincrease your (sales|revenue|footfall|business)\b",
    r"\bgrow your business\b",
    r"\bboost your (sales|business|revenue|growth)\b",
    r"\btake your business to the next level\b",
    r"\bamazing (deal|offer|discount)s?\b",
    r"\bhope (you('| a)re|this (message )?finds you) (doing )?well\b",
    r"\bdear (sir|madam|customer|valued|merchant|partner)\b",
    r"\bvalued (customer|partner|merchant)\b",
    r"\blimited[- ]time offer\b",
    r"\bdon'?t miss (out|this)\b",
    r"\bhurry\b",
    r"\bact now\b",
    r"\bunlock (your|the) (full )?potential\b",
    r"\bwe are (excited|thrilled|delighted) to\b",
    r"\bbest in (town|class|the city)\b",
    r"\bgreetings of the day\b",
]
_GENERIC_RE = [re.compile(p, re.IGNORECASE) for p in GENERIC_PATTERNS]

# ---- citation words: a "study" is only allowed if context carries a source ----
CITATION_RE = re.compile(r"\b(study|studies|research(ers)?|paper|journal|survey|published|according to|report(ed)? by)\b", re.I)
SOURCE_MARKERS = ("digest", "source", "study", "research", "paper", "journal", "survey", "report")

# ---- acronyms: "JIDA paper", "IMA says" -> must exist in context ---------------
ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,}\b")
ACRONYM_ALLOW = {
    "YES", "NO", "STOP", "OK", "OKAY", "UPI", "AM", "PM", "QR", "ID", "GST", "EMI", "SMS",
    "FAQ", "DM", "FYI", "ASAP", "BTW", "HI", "PS", "NA", "N/A", "TV", "CTA", "WA", "OTP",
    "Y", "N", "INR", "RS",
}

# internal field names / system words leaking into merchant-facing copy
JARGON_RE = re.compile(r"\b[a-z]+(?:_[a-z0-9]+)+\b|\b(placeholder|payload|suppression[ _]key|trigger|json|null)\b", re.I)
# phrases that mean "still qualifying" — banned once the merchant has said go
QUALIFYING_RE = re.compile(r"\b(would you|do you|can you tell|what if|how about|are you sure|kya aap chahenge|should we)\b", re.I)

# a reply *instruction* ("Reply YES", "YES reply kijiye"), not the noun in "review-reply drafts"
CTA_INSTR_RE = re.compile(r"(?<![-\w])reply\s+(?:with\s+)?[\"']?(?:yes|no|stop|y|n|\d)\b|\b(?:yes|stop|\d)\s+reply\b", re.I)

REINTRO_RE = re.compile(r"\b(i('| a)?m vera|this is vera|main vera|vera (here|this side)|meet vera)\b", re.I)

HINGLISH_MARKERS = re.compile(
    r"\b(hai|hain|aap|aapka|aapke|aapki|kya|ji|karein|kariye|karo|nahi|nahin|haan|chahiye|abhi|"
    r"bhi|mein|se|ka|ki|ke|ko|hoga|raha|rahe|wala|wale|kaise|kitna|thoda|bas|chalo|dekhiye)\b",
    re.I,
)
DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")

# number with optional currency prefix and scale suffix
BODY_NUM_RE = re.compile(
    r"(?P<cur>₹|rs\.?\s?|inr\s?)?(?P<num>\d[\d,]*(?:\.\d+)?)\s?(?P<suf>k\b|lakh\b|lac\b|l\b|cr\b|crore\b|%)?",
    re.I,
)
SCALE = {"k": 1e3, "lakh": 1e5, "lac": 1e5, "l": 1e5, "cr": 1e7, "crore": 1e7}
# digits that are CTA option labels ("Reply 1 for Wed, 2 for Thu"), not facts
CTA_DIGIT_RE = re.compile(r"(?:reply|type|send|press|option)\s+(\d)\b|\b(\d)\s*(?:for|=|->|→|:)\s", re.I)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = re.sub(r"[^\w\s₹%]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _cta_digit_spans(body: str) -> set[tuple[int, int]]:
    spans = set()
    for m in CTA_DIGIT_RE.finditer(body):
        g = 1 if m.group(1) else 2
        spans.add(m.span(g))
    return spans


def ungrounded_numbers(body: str, idx: GroundingIndex) -> list[str]:
    bad = []
    cta_spans = _cta_digit_spans(body)
    for m in BODY_NUM_RE.finditer(body):
        if m.span("num") in cta_spans:
            continue
        raw = m.group("num")
        # skip digits glued to letters (e.g. "4pm" handled below, "B2B", "24x7")
        start, end = m.span("num")
        if start > 0 and body[start - 1].isalpha():
            continue
        val = parse_number(raw)
        if val is None:
            continue
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        suf = (m.group("suf") or "").lower()
        if suf in SCALE:
            target = val * SCALE[suf]
            # "1.5k" legitimately rounds 1,480; allow 5% for scaled shorthand only
            if not any(abs(n - target) <= 0.05 * max(n, 1) for n in idx.numbers):
                bad.append(m.group(0).strip())
            continue
        kind = "pct" if suf == "%" else "money" if m.group("cur") else "plain"
        if not idx.has_number(val, decimals, kind):
            bad.append(m.group(0).strip())
    return bad


def validate(body: str, cta: str, idx: GroundingIndex, *, prior_bodies: list[str],
             taboo: list[str], prefers_hindi: bool, has_prior_bot_turns: bool,
             facts_used: Optional[list] = None, mode: str = "first_touch") -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(body, str) or not body.strip():
        return ["empty_body: body is empty"], []
    b = body.strip()

    # --- shape ---
    if len(b) > MAX_BODY_CHARS:
        errors.append(f"too_long: {len(b)} chars, limit {MAX_BODY_CHARS}; cut to the single strongest fact + CTA")
    if len(b) < MIN_BODY_CHARS:
        errors.append("too_short: body carries no usable content")
    if cta not in ("binary", "open_ended", "none"):
        errors.append(f"bad_cta_field: cta must be binary|open_ended|none, got {cta!r}")
    if len(CTA_INSTR_RE.findall(b)) > 1:
        errors.append("multiple_cta: more than one 'Reply ...' instruction; keep exactly one")
    if len(set(re.findall(r"\b(\d)\s*(?:for|=|->|→)\s", b, re.I))) >= 3:
        errors.append("too_many_options: 3+ numbered options; offer at most two")
    if b.count("?") > 2:
        errors.append("question_stack: more than two questions; ask one thing")
    if re.search(r"https?://|www\.", b, re.I) and not idx.has_text(re.search(r"(https?://\S+|www\.\S+)", b).group(0).rstrip(".,)")):
        errors.append("ungrounded_link: URL not present in context")

    # --- generic copy ---
    for rx in _GENERIC_RE:
        m = rx.search(b)
        if m:
            errors.append(f"generic_copy: '{m.group(0)}' is a generic framing; replace with a concrete fact from context")

    # --- taboo vocabulary from category.voice ---
    for word in taboo:
        if word and re.search(rf"\b{re.escape(word)}\b", b, re.I):
            errors.append(f"taboo_word: '{word}' is in category.voice.vocab_taboo")

    # --- grounding: numbers ---
    for n in ungrounded_numbers(b, idx):
        errors.append(f"ungrounded_number: '{n}' does not appear in context or derived_facts; remove it or use an exact value from context")

    # --- grounding: named bodies / acronyms ---
    for ac in sorted(set(ACRONYM_RE.findall(b))):
        if ac not in ACRONYM_ALLOW and ac.lower() not in idx.blob:
            errors.append(f"ungrounded_entity: '{ac}' is not in context; do not name sources/organisations that are not given")

    # --- grounding: citations ---
    if CITATION_RE.search(b) and not any(mk in idx.blob for mk in SOURCE_MARKERS):
        errors.append("ungrounded_citation: message cites a study/report but context has no source")

    # --- internal jargon (judge docks "exposing internal jargon") ---
    for m in JARGON_RE.finditer(b):
        errors.append(f"internal_jargon: '{m.group(0)}' is a system/field name; say it in plain words")
        break

    # --- intent transition: act, don't re-qualify ---
    if mode == "action":
        m = QUALIFYING_RE.search(b)
        if m:
            errors.append(f"requalifying: merchant already said go; remove '{m.group(0)}' and state what you are doing now")

    # --- conversation hygiene ---
    nb = normalize(b)
    for prev in prior_bodies:
        np_ = normalize(prev)
        if nb == np_:
            errors.append("verbatim_repeat: identical to an earlier message in this conversation")
            break
        if np_ and SequenceMatcher(None, nb, np_).ratio() >= 0.9:
            errors.append("near_repeat: >=90% similar to an earlier message; say something new")
            break
    if has_prior_bot_turns and REINTRO_RE.search(b):
        errors.append("reintroduction: conversation already started; do not re-introduce Vera")

    # --- soft quality signals ---
    if prefers_hindi and not (DEVANAGARI_RE.search(b) or len(HINGLISH_MARKERS.findall(b)) >= 2):
        warnings.append("language: merchant prefers Hindi; use natural Hindi-English code-mix")
    for f in facts_used or []:
        if isinstance(f, (str, int, float)) and str(f).strip() and not idx.has_text(str(f)):
            warnings.append(f"facts_used_mismatch: '{f}' is not a literal value in context; list exact values only")
            break

    return errors, warnings


# =============================================================================
# The evidence brief (was vera/brief.py) — deterministic pre-digestion of the
# 4 contexts (category / merchant / trigger / customer) into a short,
# authoritative brief the LLM is handed instead of raw JSON.
# =============================================================================

PLAYBOOK: dict[str, dict[str, str]] = {
    "research_digest": {
        "goal": "Share the one digest finding that matters to THIS practice and offer to do the follow-up work.",
        "levers": "specificity (trial size, effect, source+page) · reciprocity · effort externalization",
        "cta": "open_ended or binary offer to pull the abstract / draft a patient message",
    },
    "regulation_change": {
        "goal": "Flag the rule change, the exact deadline and what it means for their setup; offer a checklist.",
        "levers": "loss aversion (deadline) · specificity (old vs new limit) · effort externalization",
        "cta": "binary: want the audit checklist?",
    },
    "cde_opportunity": {
        "goal": "Surface the CDE event with date, credits and fee; make registering effortless.",
        "levers": "specificity (date/time, credits) · curiosity (topic) · single binary",
        "cta": "binary: want me to share the registration details?",
    },
    "competitor_opened": {
        "goal": "Tell them a named competitor opened nearby and how their offer compares; propose a counter-move built on the merchant's own strengths.",
        "levers": "loss aversion · specificity (distance, their price) · effort externalization",
        "cta": "binary: want me to draft the counter-offer post?",
    },
    "perf_dip": {
        "goal": "Name the exact metric drop and window, one likely lever from context, and a concrete fix Vera can start.",
        "levers": "loss aversion · specificity · effort externalization",
        "cta": "binary",
    },
    "seasonal_perf_dip": {
        "goal": "Reassure: the dip is seasonal (say so, with the season note); suggest using the quiet window productively.",
        "levers": "reciprocity (reassurance) · asking the merchant · effort externalization",
        "cta": "open_ended or binary",
    },
    "perf_spike": {
        "goal": "Celebrate the specific spike and its likely driver, then compound it with one next step.",
        "levers": "specificity · curiosity · effort externalization",
        "cta": "binary: want me to repeat what worked?",
    },
    "milestone_reached": {
        "goal": "Mark the milestone (or how close it is) and give a low-effort push to cross it.",
        "levers": "specificity · social proof (vs peer average if given) · effort externalization",
        "cta": "binary",
    },
    "review_theme_emerged": {
        "goal": "Quote the emerging review theme with its count, and offer a concrete fix + a reply draft.",
        "levers": "specificity (count, quote) · loss aversion · effort externalization",
        "cta": "binary: want me to draft the review replies?",
    },
    "renewal_due": {
        "goal": "State days left and plan/amount plainly; tie renewal to one number they would lose.",
        "levers": "loss aversion · specificity · single binary",
        "cta": "binary: Reply YES to renew",
    },
    "winback_eligible": {
        "goal": "Show what changed since the subscription lapsed (their own numbers) without guilt-tripping; offer an easy restart.",
        "levers": "loss aversion · reciprocity · effort externalization",
        "cta": "binary",
    },
    "dormant_with_vera": {
        "goal": "Re-open the thread with something genuinely new and useful for them, not 'just checking in'.",
        "levers": "curiosity · reciprocity · asking the merchant",
        "cta": "open_ended, one easy question",
    },
    "curious_ask_due": {
        "goal": "Ask the merchant one sharp, easy question about their business this week, and say what you'll do with the answer.",
        "levers": "asking the merchant · reciprocity · effort externalization",
        "cta": "open_ended",
    },
    "festival_upcoming": {
        "goal": "Name the festival, the exact days left, and one festival-specific service@price idea from the catalog.",
        "levers": "loss aversion (window) · specificity · effort externalization",
        "cta": "binary: want me to draft the festival post/offer?",
    },
    "category_seasonal": {
        "goal": "Translate the seasonal demand shifts into one shelf/stock action for this store.",
        "levers": "specificity (the % shifts) · effort externalization",
        "cta": "binary",
    },
    "ipl_match_today": {
        "goal": "Tie tonight's match (teams, time) to one concrete order/footfall move using their own offer.",
        "levers": "timeliness · specificity · effort externalization",
        "cta": "binary",
    },
    "active_planning_intent": {
        "goal": "The merchant already asked for this. Skip pitching: give a first concrete draft of the plan right away, with any number you cannot ground left as a blank for them to fill.",
        "levers": "effort externalization · asking the merchant (for missing numbers)",
        "cta": "binary: approve / tweak",
    },
    "gbp_unverified": {
        "goal": "Explain the verification gap in one line, the path, and the upside number given; offer to walk them through it.",
        "levers": "loss aversion · specificity · effort externalization",
        "cta": "binary",
    },
    "supply_alert": {
        "goal": "Urgent, precise: molecule, batch numbers, manufacturer, what to do; offer the affected-customer list.",
        "levers": "specificity · loss aversion (compliance) · effort externalization",
        "cta": "binary",
    },
    # customer-facing
    "recall_due": {
        "goal": "Friendly recall reminder from the merchant: months since last visit, the due service, real open slots, real price.",
        "levers": "specificity (slots, price) · low-friction choice",
        "cta": "slot choice (max 2 numbered options) or binary",
    },
    "customer_lapsed_soft": {
        "goal": "Warm nudge back from the merchant, referencing their history; one catalog service@price; easy booking.",
        "levers": "reciprocity · specificity · single binary",
        "cta": "binary",
    },
    "customer_lapsed_hard": {
        "goal": "Win-back from the merchant: acknowledge the gap without guilt, reference their previous goal, offer an easy restart.",
        "levers": "reciprocity · specificity · single binary",
        "cta": "binary",
    },
    "appointment_tomorrow": {
        "goal": "Clear reminder of tomorrow's appointment at this merchant with an easy confirm/reschedule.",
        "levers": "clarity · single binary",
        "cta": "binary: Reply YES to confirm",
    },
    "chronic_refill_due": {
        "goal": "Refill reminder: which medicines, when stock runs out, delivery option if saved; one-tap confirm.",
        "levers": "specificity · effort externalization",
        "cta": "binary: Reply YES to repeat the order",
    },
    "trial_followup": {
        "goal": "Follow up the trial with the next real session option and an easy yes.",
        "levers": "specificity (session slot) · single binary",
        "cta": "binary",
    },
    "wedding_package_followup": {
        "goal": "Bridal follow-up: days to the wedding, the next program window, one clear next step.",
        "levers": "timeliness · specificity · single binary",
        "cta": "binary",
    },
}
DEFAULT_PLAY = {
    "goal": "Explain in one line why this is worth their attention now, anchored on one verifiable number, and offer one concrete next step.",
    "levers": "specificity · effort externalization · single binary",
    "cta": "binary or open_ended",
}

# which digest item kind to fall back to when a trigger has no explicit item id
DIGEST_KIND_FOR = {
    "research_digest": ("research",), "regulation_change": ("compliance",),
    "cde_opportunity": ("cde",), "supply_alert": ("alert", "supply"),
    "category_trend_movement": ("trend",), "category_seasonal": ("trend", "supply"),
}
ITEM_ID_KEYS = ("top_item_id", "digest_item_id", "alert_id", "item_id")

HUMAN_METRIC = {"views": "profile views", "calls": "calls", "ctr": "CTR", "directions": "direction requests", "leads": "leads"}


def _parse_dt(s: Any) -> Optional[datetime]:
    if not isinstance(s, str) or not re.match(r"^\d{4}-\d{2}-\d{2}", s):
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _pct(x: Any) -> Optional[str]:
    """x is a fraction (0.18 -> '+18%'), as every *_pct field in the dataset is."""
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        v = x * 100
        s = f"{abs(v):.1f}".rstrip("0").rstrip(".")
        return ("+" if v > 0 else "-" if v < 0 else "") + s + "%"
    return None


def language_directive(merchant: dict, customer: Optional[dict]) -> tuple[str, bool]:
    """Returns (directive, prefers_hindi)."""
    if customer:
        pref = str(dig(customer, "identity.language_pref", default="en")).lower()
        if "hi" in pref and ("en" in pref or "mix" in pref):
            return "Hindi-English code-mix in Roman script (customer pref: hi-en mix)", True
        if pref in ("hi", "hindi"):
            return "Hindi in Roman script, simple words (customer pref: hi)", True
        return "English (customer pref: English)", False
    langs = [str(l).lower() for l in as_list(dig(merchant, "identity.languages", default=["en"]))]
    if "hi" in langs or "hindi" in langs:
        return "natural Hindi-English code-mix in Roman script (merchant languages include hi)", True
    return "English", False


def salutation(category_slug: str, merchant: dict, customer: Optional[dict]) -> str:
    if customer:
        parts = str(dig(customer, "identity.name", default="")).split()
        if len(parts) >= 2 and parts[0].rstrip(".").lower() in ("mr", "mrs", "ms", "dr", "shri", "smt"):
            return " ".join(parts[:2])  # "Mr. Sharma", not "Mr."
        return parts[0] if parts else "there"
    first = dig(merchant, "identity.owner_first_name")
    if first:
        first = re.sub(r"^dr\.?\s*", "", str(first), flags=re.I).strip()
        return f"Dr. {first}" if category_slug == "dentists" else first
    return str(dig(merchant, "identity.name", default="there"))


def resolve_digest_item(category: Optional[dict], trigger: dict) -> Optional[dict]:
    items = as_list(dig(category or {}, "digest", default=[]))
    payload = trigger.get("payload") or {}
    for k in ITEM_ID_KEYS:
        want = payload.get(k)
        if want:
            for it in items:
                if isinstance(it, dict) and it.get("id") == want:
                    return it
    # thin trigger: pick the first digest item of the matching kind
    kinds = DIGEST_KIND_FOR.get(trigger.get("kind", ""), ())
    for it in items:
        if isinstance(it, dict) and it.get("kind") in kinds:
            return it
    return None


def active_offers(merchant: dict) -> list[str]:
    return [o.get("title") for o in as_list(merchant.get("offers")) if isinstance(o, dict)
            and o.get("title") and str(o.get("status", "active")).lower() == "active"]


def catalog_offers(category: Optional[dict], audience: Optional[str] = None, n: int = 4) -> list[str]:
    out = []
    for o in as_list(dig(category or {}, "offer_catalog", default=[])):
        if not isinstance(o, dict) or not o.get("title"):
            continue
        if audience and o.get("audience") and o["audience"] != audience:
            continue
        out.append(o["title"])
    return out[:n]


def perf_lines(merchant: dict, category: Optional[dict]) -> list[str]:
    perf = merchant.get("performance") or {}
    peer = dig(category or {}, "peer_stats", default={}) or {}
    lines = []
    win = perf.get("window_days", 30)
    for k in ("views", "calls", "directions", "leads"):
        if isinstance(perf.get(k), (int, float)):
            peer_v = peer.get(f"avg_{k}_{win}d") or peer.get(f"avg_{k}")
            s = f"{HUMAN_METRIC[k]} last {win}d: {perf[k]}"
            if isinstance(peer_v, (int, float)) and peer_v:
                s += f" (category avg {peer_v}; {_pct((perf[k] - peer_v) / peer_v)} vs avg)"
            lines.append(s)
    if isinstance(perf.get("ctr"), (int, float)):
        s = f"CTR: {_pct(perf['ctr']).lstrip('+')}"
        if isinstance(peer.get("avg_ctr"), (int, float)):
            s += f" (category avg {_pct(peer['avg_ctr']).lstrip('+')})"
        lines.append(s)
    for k, v in (perf.get("delta_7d") or {}).items():
        p = _pct(v)
        if p:
            lines.append(f"7-day change in {HUMAN_METRIC.get(k.replace('_pct', ''), k.replace('_pct', ''))}: {p}")
    return lines


def time_lines(trigger: dict, customer: Optional[dict], now: Optional[str]) -> list[str]:
    nd = _parse_dt(now)
    if not nd:
        return []
    lines = [f"today: {nd.strftime('%a %d %b %Y')}"]
    payload = trigger.get("payload") or {}
    for k, v in list(payload.items()) + [("expires_at", trigger.get("expires_at"))]:
        d = _parse_dt(v)
        if not d:
            continue
        days = (d.date() - nd.date()).days
        label = k.replace("_iso", "").replace("_", " ")
        if k == "expires_at" and days < 0:
            continue  # the judge listed it as active; an expiry in the past is noise, not a fact to message
        if days > 0:
            lines.append(f"{label} ({d.strftime('%d %b %Y')}) is {days} days from today")
        elif days == 0:
            lines.append(f"{label} is today ({d.strftime('%d %b')}, {d.strftime('%I:%M %p').lstrip('0')})")
        else:
            lines.append(f"{label} ({d.strftime('%d %b %Y')}) was {-days} days ago")
    if customer:
        lv = _parse_dt(dig(customer, "relationship.last_visit"))
        if lv:
            days = (nd.date() - lv.date()).days
            if days >= 0:
                lines.append(f"customer's last visit was {days} days ago (~{round(days / 30.4)} months)")
    return lines


def seasonal_now(category: Optional[dict], now: Optional[str]) -> list[str]:
    nd = _parse_dt(now)
    if not nd:
        return []
    mon = nd.strftime("%b")
    order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    out = []
    for b in as_list(dig(category or {}, "seasonal_beats", default=[])):
        rng = str(b.get("month_range", ""))
        parts = [p.strip()[:3] for p in rng.split("-")]
        try:
            a = order.index(parts[0]); z = order.index(parts[-1])
        except ValueError:
            continue
        i = order.index(mon)
        inside = a <= i <= z if a <= z else (i >= a or i <= z)
        if inside:
            out.append(f"{rng}: {b.get('note')}")
    return out


def is_thin(trigger: dict) -> bool:
    p = trigger.get("payload") or {}
    return bool(p.get("placeholder")) or not [k for k in p if k not in ("placeholder", "metric_or_topic", "category")]


def build_brief(category: Optional[dict], merchant: dict, trigger: dict,
                customer: Optional[dict], now: Optional[str]) -> dict:
    slug = merchant.get("category_slug") or dig(category or {}, "slug", default="")
    kind = trigger.get("kind", "unknown")
    play = PLAYBOOK.get(kind, DEFAULT_PLAY)
    lang, _ = language_directive(merchant, customer)
    customer_facing = bool(customer) or trigger.get("scope") == "customer"
    thin = is_thin(trigger)
    item = resolve_digest_item(category, trigger)

    b: dict[str, Any] = {
        "audience": "customer (you write AS the merchant, send_as=merchant_on_behalf)" if customer_facing
                    else "merchant owner (you are Vera, send_as=vera)",
        "address_as": salutation(slug, merchant, customer if customer_facing else None),
        "merchant_name": dig(merchant, "identity.name"),
        "locality": ", ".join(x for x in [dig(merchant, "identity.locality"), dig(merchant, "identity.city")] if x),
        "category": slug,
        "voice": {
            "tone": dig(category or {}, "voice.tone"),
            "register": dig(category or {}, "voice.register"),
            "allowed_vocab": as_list(dig(category or {}, "voice.vocab_allowed"))[:10],
            "never_say": taboo_list(category),
            "tone_examples": as_list(dig(category or {}, "voice.tone_examples"))[:3],
        },
        "language": lang,
        "why_now": {
            "trigger_kind": kind,
            "urgency_1_to_5": trigger.get("urgency"),
            "facts": {k: v for k, v in (trigger.get("payload") or {}).items() if k not in ("placeholder", "metric_or_topic")},
            "resolved_digest_item": item,
            "dates": time_lines(trigger, customer, now),
        },
        "playbook": play,
        "merchant_numbers": perf_lines(merchant, category),
        "merchant_offers_active": active_offers(merchant),
        "catalog_offers": catalog_offers(category, "repeat_user" if customer and dig(customer, "relationship.visits_total", default=0) > 1 else None),
        "seasonal_now": seasonal_now(category, now),
        "subscription": merchant.get("subscription"),
        "customer_aggregate": merchant.get("customer_aggregate"),
        "review_themes": merchant.get("review_themes") or [],
        "recent_history": [
            {"from": h.get("from"), "body": h.get("body"), "engagement": h.get("engagement")}
            for h in as_list(merchant.get("conversation_history"))[-4:] if isinstance(h, dict)
        ],
    }
    if customer_facing:
        # never leak the merchant's analytics/billing into a customer message
        for k in ("merchant_numbers", "subscription", "customer_aggregate", "review_themes", "recent_history"):
            b.pop(k, None)
    if customer_facing and customer:
        b["customer"] = {
            "name": dig(customer, "identity.name"),
            "state": customer.get("state"),
            "relationship": customer.get("relationship"),
            "preferences": customer.get("preferences"),
            "consent_scope": dig(customer, "consent.scope"),
        }
    if thin:
        b["why_now"]["thin_trigger_warning"] = (
            f"This '{kind}' event arrived with NO specifics (no counts, names, dates, times or amounts). "
            "You may say the event happened only in the generic sense its kind implies. Do not invent any of its details. "
            "Anchor the message on the strongest verifiable number from merchant_numbers / offers / catalog instead."
        )
    return b


def taboo_list(category: Optional[dict]) -> list[str]:
    raw = as_list(dig(category or {}, "voice.vocab_taboo", "voice.taboos", default=[]))
    out = []
    for t in raw:
        t = re.sub(r"\(.*?\)", "", str(t)).strip()
        if t:
            out.append(t)
    return out


def send_as_for(trigger: dict, customer: Optional[dict]) -> str:
    return "merchant_on_behalf" if (customer or trigger.get("scope") == "customer") else "vera"


# =============================================================================
# In-memory state (was vera/store.py). The challenge contract guarantees no
# restarts during a test run, so process memory is the fastest and simplest
# correct store. IMPORTANT: run a single worker process (uvicorn --workers 1)
# — multiple workers would each hold a different copy of this state.
# =============================================================================

VALID_SCOPES = ("category", "merchant", "customer", "trigger")


class ContextStore:
    def __init__(self) -> None:
        self._data: dict[tuple[str, str], dict] = {}
        self._alias: dict[tuple[str, str], str] = {}  # (scope, alt_id) -> context_id

    def put(self, scope: str, context_id: str, version: int, payload: dict) -> tuple[str, Optional[int]]:
        """Returns ("stored"|"duplicate"|"stale", current_version)."""
        key = (scope, context_id)
        cur = self._data.get(key)
        if cur and cur["version"] == version:
            return "duplicate", version  # idempotent re-post: accepted, no side effects
        if cur and cur["version"] > version:
            return "stale", cur["version"]
        self._data[key] = {"version": version, "payload": payload, "stored_at": time.time()}
        # the payload's own id / slug may differ from context_id; index both so a
        # trigger that says merchant_id="m_42" finds context_id="merchant:m_42"
        for f in ("id", f"{scope}_id", "slug", "category_slug" if scope == "category" else None):
            if f and isinstance(payload.get(f), str) and payload[f] != context_id:
                self._alias[(scope, payload[f])] = context_id
        return "stored", version

    def get(self, scope: str, ident: Optional[str]) -> Optional[dict]:
        if not ident:
            return None
        e = self._data.get((scope, ident))
        if e is None and (scope, ident) in self._alias:
            e = self._data.get((scope, self._alias[(scope, ident)]))
        return e["payload"] if e else None

    def counts(self) -> dict[str, int]:
        c = {s: 0 for s in VALID_SCOPES}
        for s, _ in self._data:
            c[s] = c.get(s, 0) + 1
        return c

    def clear(self) -> None:
        self._data.clear()
        self._alias.clear()


@dataclass
class Conversation:
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    trigger_id: Optional[str] = None
    turns: list[dict] = field(default_factory=list)       # {"from", "body", "ts"}
    inbound_counts: Counter = field(default_factory=Counter)
    hostile_count: int = 0
    auto_reply_count: int = 0
    last_bot_cta: Optional[str] = None
    ended: bool = False

    def bot_bodies(self) -> list[str]:
        return [t["body"] for t in self.turns if t["from"] == "bot"]

    def add(self, who: str, body: str) -> None:
        self.turns.append({"from": who, "body": body, "ts": time.time()})


class State:
    def __init__(self) -> None:
        self.contexts = ContextStore()
        self.conversations: dict[str, Conversation] = {}
        self.sent_suppression_keys: set[str] = set()
        self.opted_out: set[str] = set()                   # merchant_id (or merchant_id:customer_id)
        self.compose_cache: dict[str, dict] = {}
        self.inflight: dict[str, "asyncio.Task"] = {}
        self.merchant_inbound: Counter = Counter()          # (merchant_id, normalized msg) across conversations
        self.merchant_auto_replies: Counter = Counter()     # merchant_id -> auto-replies seen
        self.stats: Counter = Counter()

    def conv(self, conversation_id: str, **meta: Any) -> Conversation:
        c = self.conversations.get(conversation_id)
        if c is None:
            c = Conversation(conversation_id)
            self.conversations[conversation_id] = c
        for k, v in meta.items():
            if v and not getattr(c, k, None):
                setattr(c, k, v)
        return c

    def reset(self) -> None:
        for t in list(self.inflight.values()):
            t.cancel()
        self.__init__()


STATE = State()


# =============================================================================
# compose(): contexts -> brief -> LLM -> validator -> (informed repair) ->
# message (was vera/composer.py).
#
# Contract: returns a validated message dict or None. Never raises, never
# returns unvalidated model text, never exceeds the caller's deadline.
#
# NOTE ON NAMING: the async pipeline function below is named `_compose` (it
# was imported into bot.py under that alias in the original modular repo, to
# avoid clashing with the synchronous plain-Python-contract `compose()`
# defined near the bottom of this file — that naming is preserved here).
# =============================================================================

@dataclass
class ComposeResult:
    message: Optional[dict] = None
    skipped_reason: Optional[str] = None
    attempts: int = 0
    violations: list[list[str]] = field(default_factory=list)


def _cache_key(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256((PROMPT_HASH + LLM_MODEL + raw).encode()).hexdigest()


def _normalise(raw: dict, trigger: dict, customer: Optional[dict], brief: dict) -> dict:
    cta = str(raw.get("cta", "none")).strip().lower().replace("-", "_")
    if cta not in ("binary", "open_ended", "none"):
        cta = "open_ended" if "open" in cta else "binary" if cta in ("yes_no", "yes/stop", "yes_stop") else cta
    params = raw.get("template_params")
    if not (isinstance(params, list) and all(isinstance(p, (str, int, float)) for p in params) and params):
        params = [brief.get("address_as", ""), brief.get("merchant_name", "")]
    return {
        "body": str(raw.get("body", "")).strip(),
        "cta": cta,
        # routing facts are decided by code, not by the model
        "send_as": send_as_for(trigger, customer),
        "suppression_key": trigger.get("suppression_key") or f"{trigger.get('kind', 'msg')}:{trigger.get('id', '')}",
        "template_params": [str(p) for p in params][:6],
        "facts_used": as_list(raw.get("facts_used"))[:12],
        "rationale": str(raw.get("rationale", "")).strip()[:600],
    }


_llm_sem: dict[int, asyncio.Semaphore] = {}


def _sem() -> asyncio.Semaphore:
    """One semaphore per event loop (the sync compose() wrapper may run several loops)."""
    loop_id = id(asyncio.get_running_loop())
    if loop_id not in _llm_sem:
        _llm_sem.clear()
        _llm_sem[loop_id] = asyncio.Semaphore(LLM_CONCURRENCY)
    return _llm_sem[loop_id]


async def _compose(category: Optional[dict], merchant: dict, trigger: dict, customer: Optional[dict], *,
                   now: Optional[str], deadline: float, mode: str = "first_touch",
                   conversation: Optional[dict] = None, prior_bodies: Optional[list[str]] = None,
                   allow_skip: bool = True, max_attempts: int = MAX_COMPOSE_ATTEMPTS) -> ComposeResult:
    """Cache -> in-flight dedupe -> fresh composition.

    The fresh composition runs as its own task with its own budget. The caller
    waits only until ITS deadline; if the work outlives the caller (slow LLM),
    it keeps running and lands in the cache, so the next tick is instant
    instead of paying the same latency again."""
    prior_bodies = prior_bodies or []
    try:
        brief = build_brief(category, merchant, trigger, customer, now)
    except Exception as e:  # malformed context must not crash a tick
        log.exception("brief build failed")
        return ComposeResult(skipped_reason=f"brief_error:{e}")

    payload: dict[str, Any] = {
        "mode": mode,
        "brief": brief,
        "contexts": {"category": category, "merchant": merchant, "trigger": trigger, "customer": customer},
        "conversation": conversation,
        "must_not_repeat": prior_bodies[-5:],
    }
    key = _cache_key(payload)
    cached = STATE.compose_cache.get(key)
    if cached:  # determinism: identical inputs -> identical output
        STATE.stats["cache_hits"] += 1
        return ComposeResult(message=dict(cached))

    task = STATE.inflight.get(key)
    if task is None:
        task = asyncio.create_task(_compose_fresh(
            key, payload, brief, category, merchant, trigger, customer, conversation, prior_bodies,
            mode, allow_skip, max(max_attempts, MAX_COMPOSE_ATTEMPTS)))
        STATE.inflight[key] = task
        task.add_done_callback(lambda _t, k=key: STATE.inflight.pop(k, None))
    else:
        STATE.stats["inflight_joins"] += 1

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return ComposeResult(skipped_reason="out_of_time")
    try:
        res = await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
    except asyncio.TimeoutError:
        STATE.stats["compose_deferred"] += 1
        return ComposeResult(skipped_reason="still_composing")
    except Exception as e:  # pragma: no cover - _compose_fresh never raises
        return ComposeResult(skipped_reason=f"compose_error:{e}")
    if res.message:
        res = ComposeResult(message=dict(res.message), attempts=res.attempts, violations=res.violations)
    return res


async def precompose(category: Optional[dict], merchant: dict, trigger: dict, customer: Optional[dict],
                     now: Optional[str]) -> None:
    """Fire-and-forget warm-up when a trigger context is pushed, so the tick
    that follows is served from cache inside a few milliseconds."""
    try:
        await _compose(category, merchant, trigger, customer, now=now,
                       deadline=time.monotonic() + BACKGROUND_BUDGET_SECONDS)
    except Exception:  # pragma: no cover
        log.exception("precompose failed")


async def _compose_fresh(key: str, payload: dict, brief: dict, category: Optional[dict], merchant: dict,
                         trigger: dict, customer: Optional[dict], conversation: Optional[dict],
                         prior_bodies: list[str], mode: str, allow_skip: bool, max_attempts: int) -> ComposeResult:
    res = ComposeResult()
    deadline = time.monotonic() + BACKGROUND_BUDGET_SECONDS
    idx = GroundingIndex(category, merchant, trigger, customer, extra_text=json.dumps(brief, ensure_ascii=False, default=str))
    _, prefers_hindi = language_directive(merchant, customer)
    if conversation and conversation.get("latest_message_language") == "en":
        prefers_hindi = False  # mirror the merchant's own switch to English
    has_prior = bool(prior_bodies) or any(
        isinstance(h, dict) and h.get("from") == "vera" for h in as_list(merchant.get("conversation_history")))

    best_soft: Optional[dict] = None
    repair: Optional[dict] = None
    for _attempt in range(max_attempts):
        remaining = deadline - time.monotonic()
        if remaining < MIN_ATTEMPT_SECONDS:
            res.skipped_reason = res.skipped_reason or "out_of_time"
            break
        if best_soft is not None and not repair:
            break
        res.attempts += 1
        call_payload = dict(payload, repair=repair) if repair else payload
        timeout = min(LLM_CALL_TIMEOUT, remaining - 0.5)
        try:
            async with _sem():
                text = await asyncio.wait_for(
                    complete(SYSTEM_PROMPT, json.dumps(call_payload, ensure_ascii=False, default=str), timeout),
                    timeout=timeout,
                )
            raw = parse_json_object(text)
        except asyncio.TimeoutError:
            STATE.stats["llm_timeouts"] += 1
            res.skipped_reason = "llm_timeout"
            continue
        except Exception as e:  # provider error / unparsable output
            STATE.stats["llm_errors"] += 1
            res.skipped_reason = f"llm_error:{type(e).__name__}"
            log.warning("llm call failed: %s", e)
            if isinstance(e, LLMError) and "not set" in str(e):
                break  # misconfiguration: retrying cannot help
            repair = {"previous_output": None, "violations": ["output was not a single valid JSON object"]}
            continue

        if raw.get("skip"):
            if allow_skip:
                res.skipped_reason = f"model_skip:{raw.get('reason', '')}"
                return res
            repair = {"previous_output": raw, "violations": [
                "skip is not allowed for this request: write the best message the input honestly supports, "
                "anchored on the strongest real number in the brief"]}
            continue

        msg = _normalise(raw, trigger, customer, brief)
        errors, warnings = validate(
            msg["body"], msg["cta"], idx, prior_bodies=prior_bodies, taboo=taboo_list(category),
            prefers_hindi=prefers_hindi, has_prior_bot_turns=has_prior, facts_used=msg["facts_used"], mode=mode,
        )
        res.violations.append(errors + warnings)
        if not errors and not warnings:
            best_soft = msg
            break
        if not errors:
            best_soft = msg  # already sendable; one polish attempt if time allows
        repair = {"previous_output": {"body": msg["body"], "cta": msg["cta"]}, "violations": errors + warnings}
        if best_soft is not None and deadline - time.monotonic() < LLM_CALL_TIMEOUT:
            break  # don't gamble a sendable message on a polish we can't finish

    if best_soft:
        STATE.compose_cache[key] = best_soft
        res.message = dict(best_soft)
        res.skipped_reason = None
        return res
    STATE.stats["compose_rejected"] += 1
    res.skipped_reason = res.skipped_reason or "failed_validation"
    log.info("compose rejected (%s): %s", trigger.get("id"), res.violations[-1:])
    return res


# =============================================================================
# HTTP surface (was bot.py's own top-level code). No CORS middleware: the
# judge's harness calls this API server-to-server, so cross-origin browser
# access is not needed for evaluation.
# =============================================================================

APP_START = time.time()
app = FastAPI(title="Vera — magicpin merchant assistant", version=BOT_VERSION)
_background: set[asyncio.Task] = set()


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _spawn(coro) -> None:
    t = asyncio.create_task(coro)
    _background.add(t)
    t.add_done_callback(_background.discard)


# ---------------------------------------------------------------------------
# Never emit a non-JSON error: a malformed response costs -2 per call.
# ---------------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def _bad_request(request: Request, exc: RequestValidationError):
    path = request.url.path
    if path.endswith("/tick"):
        return JSONResponse(status_code=200, content={"actions": []})
    if path.endswith("/reply"):
        return JSONResponse(status_code=200, content={"action": "wait", "wait_seconds": 300,
                                                      "rationale": "Could not parse the reply payload; backing off."})
    return JSONResponse(status_code=400, content={"accepted": False, "reason": "malformed_request",
                                                  "details": str(exc.errors())[:500]})


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    log.exception("unhandled error on %s", request.url.path)
    path = request.url.path
    if path.endswith("/tick"):
        return JSONResponse(status_code=200, content={"actions": []})
    if path.endswith("/reply"):
        return JSONResponse(status_code=200, content={"action": "wait", "wait_seconds": 600,
                                                      "rationale": "Internal error while composing; backing off instead of sending something unvalidated."})
    return JSONResponse(status_code=500, content={"accepted": False, "reason": "internal_error"})


# ---------------------------------------------------------------------------
# /v1/context
# ---------------------------------------------------------------------------

class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: Optional[str] = None


@app.post("/v1/context")
async def push_context(body: CtxBody, response: Response):
    if body.scope not in VALID_SCOPES:
        response.status_code = 400
        return {"accepted": False, "reason": "invalid_scope", "details": f"scope must be one of {list(VALID_SCOPES)}"}
    outcome, current = STATE.contexts.put(body.scope, body.context_id, body.version, body.payload)
    if outcome == "stale":
        response.status_code = 409
        return {"accepted": False, "reason": "stale_version", "current_version": current}
    if outcome == "duplicate":  # brief §2.1: re-posting the same version is a no-op
        return {"accepted": True, "ack_id": f"ack_{body.scope}_{body.context_id}_v{body.version}",
                "stored_at": _utcnow(), "duplicate": True}

    if body.scope == "trigger" and is_configured():
        # warm the composition now, so the tick that lists this trigger is instant
        trig = body.payload
        merchant = STATE.contexts.get("merchant", trig.get("merchant_id"))
        if merchant:
            category = STATE.contexts.get("category", merchant.get("category_slug"))
            customer = STATE.contexts.get("customer", trig.get("customer_id"))
            if not trig.get("customer_id") or customer:
                _spawn(precompose(category, merchant, trig, customer, body.delivered_at or _utcnow()))

    return {"accepted": True, "ack_id": f"ack_{body.scope}_{body.context_id}_v{body.version}", "stored_at": _utcnow()}


# ---------------------------------------------------------------------------
# /v1/tick
# ---------------------------------------------------------------------------

class TickBody(BaseModel):
    now: Optional[str] = None
    available_triggers: list[str] = Field(default_factory=list)


def _conv_id(trigger: dict, trg_id: str) -> str:
    who = trigger.get("customer_id") or trigger.get("merchant_id") or "x"
    return f"conv_{who}_{trg_id}"


def _select_candidates(trigger_ids: list[str]) -> tuple[list[dict], list[dict]]:
    """Cheap, deterministic filtering before any LLM work. Returns (chosen, skipped)."""
    chosen: dict[tuple[str, str], dict] = {}
    skipped: list[dict] = []
    for order, trg_id in enumerate(dict.fromkeys(trigger_ids)):
        trigger = STATE.contexts.get("trigger", trg_id)
        if not trigger:
            skipped.append({"trigger_id": trg_id, "why": "unknown_trigger"}); continue
        mid = trigger.get("merchant_id") or (trigger.get("payload") or {}).get("merchant_id")
        merchant = STATE.contexts.get("merchant", mid)
        if not merchant:
            skipped.append({"trigger_id": trg_id, "why": "no_merchant_context"}); continue
        cid = trigger.get("customer_id")
        customer = STATE.contexts.get("customer", cid) if cid else None
        if cid and not customer:
            skipped.append({"trigger_id": trg_id, "why": "no_customer_context"}); continue
        skey = trigger.get("suppression_key") or trg_id
        if skey in STATE.sent_suppression_keys:
            skipped.append({"trigger_id": trg_id, "why": "suppressed_already_sent"}); continue
        if mid in STATE.opted_out and not cid:
            skipped.append({"trigger_id": trg_id, "why": "merchant_opted_out"}); continue
        if cid and (f"{mid}:{cid}" in STATE.opted_out or (customer.get("preferences") or {}).get("reminder_opt_in") is False):
            skipped.append({"trigger_id": trg_id, "why": "customer_not_opted_in"}); continue
        conv_id = _conv_id(trigger, trg_id)
        if conv_id in STATE.conversations:
            skipped.append({"trigger_id": trg_id, "why": "conversation_already_open"}); continue
        cand = {"order": order, "trg_id": trg_id, "trigger": trigger, "merchant": merchant, "merchant_id": mid,
                "customer": customer, "customer_id": cid, "conv_id": conv_id,
                "category": STATE.contexts.get("category", merchant.get("category_slug"))}
        # cadence: one proactive message per recipient per tick — the most urgent one.
        # The others stay un-suppressed and are reconsidered next tick.
        recipient = (mid, cid or "-")
        cur = chosen.get(recipient)
        rank = (-(trigger.get("urgency") or 0), order)
        if cur is None or rank < (-(cur["trigger"].get("urgency") or 0), cur["order"]):
            if cur:
                skipped.append({"trigger_id": cur["trg_id"], "why": "deferred_same_recipient"})
            chosen[recipient] = cand
        else:
            skipped.append({"trigger_id": trg_id, "why": "deferred_same_recipient"})
    ranked = sorted(chosen.values(), key=lambda c: (-(c["trigger"].get("urgency") or 0), c["order"]))
    return ranked[: MAX_ACTIONS_PER_TICK], skipped


@app.post("/v1/tick")
async def tick(body: TickBody):
    deadline = time.monotonic() + TICK_BUDGET_SECONDS
    now = body.now or _utcnow()
    candidates, _skipped = _select_candidates(body.available_triggers)
    if not candidates or not is_configured():
        return {"actions": []}

    async def one(c: dict):
        r = await _compose(c["category"], c["merchant"], c["trigger"], c["customer"], now=now, deadline=deadline)
        return c, r

    results = await asyncio.gather(*(one(c) for c in candidates), return_exceptions=True)
    actions = []
    for item in results:
        if isinstance(item, Exception):
            log.warning("compose task failed: %s", item); continue
        c, r = item
        if not r.message:
            continue  # restraint: nothing validated for this trigger this tick
        m = r.message
        if time.monotonic() > deadline + 1:  # belt and braces; compose already honours the deadline
            break
        STATE.sent_suppression_keys.add(m["suppression_key"])
        conv = STATE.conv(c["conv_id"], merchant_id=c["merchant_id"], customer_id=c["customer_id"], trigger_id=c["trg_id"])
        conv.add("bot", m["body"])
        conv.last_bot_cta = m["cta"]
        actions.append({
            "conversation_id": c["conv_id"],
            "merchant_id": c["merchant_id"],
            "customer_id": c["customer_id"],
            "send_as": m["send_as"],
            "trigger_id": c["trg_id"],
            "template_name": f"{'merchant' if m['send_as'] == 'merchant_on_behalf' else 'vera'}_{c['trigger'].get('kind', 'update')}_v1",
            "template_params": m["template_params"],
            "body": m["body"],
            "cta": m["cta"],
            "suppression_key": m["suppression_key"],
            "rationale": m["rationale"] or f"{c['trigger'].get('kind')} for {c['merchant_id']}",
        })
    STATE.stats["actions_sent"] += len(actions)
    return {"actions": actions}


# ---------------------------------------------------------------------------
# /v1/reply
# ---------------------------------------------------------------------------

class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str = "merchant"
    message: str = ""
    received_at: Optional[str] = None
    turn_number: int = 0


def _message_language(msg: str) -> str:
    return "hi" if DEVANAGARI_RE.search(msg) or len(HINGLISH_MARKERS.findall(msg)) >= 2 else "en"


def _wait(seconds: int, why: str) -> dict:
    return {"action": "wait", "wait_seconds": seconds, "rationale": why}


def _end(why: str) -> dict:
    return {"action": "end", "rationale": why}


def _action_fallback(lang: str) -> str:
    """Only used if the LLM is down at the exact moment a merchant says 'go'.
    Contains no facts, so it can't be wrong; keeps momentum instead of
    going silent, which is the costliest failure at an intent hand-off."""
    if lang == "hi":
        return "Done, main abhi shuru kar rahi hoon. Pehla draft next 10 minute mein yahin bhej dungi — aap bas final OK kar dena."
    return "Done, starting on it now. I'll send the first draft right here in the next 10 minutes; you just give the final OK."


async def handle_reply(body: ReplyBody) -> dict:
    deadline = time.monotonic() + REPLY_BUDGET_SECONDS
    conv = STATE.conv(body.conversation_id, merchant_id=body.merchant_id, customer_id=body.customer_id)
    msg = (body.message or "").strip()
    conv.add(body.from_role or "merchant", msg)
    mid = conv.merchant_id or body.merchant_id or ""
    cid = conv.customer_id or body.customer_id

    # repeat counting is per conversation AND per merchant: a WA Business
    # auto-responder fires the same text into every thread we open
    norm = normalize(msg)
    conv.inbound_counts[norm] += 1
    STATE.merchant_inbound[(mid, norm)] += 1
    repeats = max(conv.inbound_counts[norm], STATE.merchant_inbound[(mid, norm)])
    read = classify(msg, repeat_count=repeats, last_bot_cta=conv.last_bot_cta)
    STATE.stats[f"reply_{read.label}"] += 1
    lang = _message_language(msg)

    if conv.ended and read.label in ("auto_reply", "opt_out", "hostile", "decline", "empty"):
        return _end(f"Conversation already closed; inbound read as {read.label}, so staying closed.")

    if read.label == "empty":
        return _wait(900, "Empty inbound message; nothing to respond to yet.")

    if read.label == "opt_out":
        conv.ended = True
        STATE.opted_out.add(f"{mid}:{cid}" if cid else mid)
        return _end("Explicit opt-out ('stop'/unsubscribe). Exiting immediately and suppressing all future proactive sends to this recipient.")

    if read.label == "auto_reply":
        conv.auto_reply_count += 1
        STATE.merchant_auto_replies[mid] += 1
        if conv.auto_reply_count >= 2 or STATE.merchant_auto_replies[mid] >= 2 or repeats >= 2:
            conv.ended = True
            return _end(f"WhatsApp Business auto-reply detected again ({'; '.join(read.notes)}). "
                        "Exiting now instead of burning more turns on a bot; will re-engage on the next real trigger.")
        return _wait(14400, f"Auto-reply detected ({'; '.join(read.notes)}): owner is not on the phone. "
                            "Not replying to a machine; backing off 4h so the next message lands when a human is likely to read it.")

    if read.label == "decline":
        conv.ended = True
        return _end("Merchant declined ('not interested'). Closing politely without a counter-pitch; this thread is not re-opened proactively.")

    if read.label == "later":
        secs = 86400 if re.search(r"\b(tomorrow|kal)\b", msg, re.I) else 10800
        return _wait(secs, f"Merchant asked for time; backing off {secs // 3600}h and resuming the same thread then.")

    if read.label == "hostile":
        conv.hostile_count += 1
        if conv.hostile_count >= 2:
            conv.ended = True
            STATE.opted_out.add(f"{mid}:{cid}" if cid else mid)
            return _end("Second hostile message in this thread. Exiting and suppressing further outreach; persisting would only damage the relationship.")

    mode = {"intent_transition": "action", "question": "answer", "hostile": "deescalate",
            "off_topic": "deescalate"}.get(read.label, "continue")
    conv.ended = False  # the merchant re-engaged on their own

    merchant = STATE.contexts.get("merchant", mid) or {"merchant_id": mid}
    category = STATE.contexts.get("category", merchant.get("category_slug"))
    customer = STATE.contexts.get("customer", cid) if cid else None
    trigger = (STATE.contexts.get("trigger", conv.trigger_id) if conv.trigger_id else None) or {
        "id": f"reply:{body.conversation_id}",
        "scope": "customer" if cid else "merchant",
        "kind": "merchant_reply",
        "source": "internal",
        "merchant_id": mid,
        "customer_id": cid,
        "payload": {"merchant_message": msg},
        "suppression_key": f"reply:{body.conversation_id}:{body.turn_number}",
    }
    conversation = {
        "transcript": [{"from": t["from"], "body": t["body"]} for t in conv.turns[-10:]],
        "latest_message": msg,
        "latest_message_read": {"label": read.label, "notes": read.notes},
        "latest_message_language": lang,
        "turn_number": body.turn_number,
    }
    r = await _compose(category, merchant, trigger, customer, now=body.received_at or _utcnow(),
                       deadline=deadline, mode=mode, conversation=conversation, prior_bodies=conv.bot_bodies())

    if r.message:
        m = r.message
        conv.add("bot", m["body"])
        conv.last_bot_cta = m["cta"]
        return {"action": "send", "body": m["body"], "cta": m["cta"],
                "rationale": f"[{read.label} -> {mode}] " + (m["rationale"] or "")}

    if mode == "action":
        text = _action_fallback(lang)
        if normalize(text) not in {normalize(b) for b in conv.bot_bodies()}:
            conv.add("bot", text)
            conv.last_bot_cta = "none"
            return {"action": "send", "body": text, "cta": "none",
                    "rationale": "Merchant committed; composition unavailable this turn, so confirming action immediately (no facts asserted) rather than going silent."}
    if mode == "deescalate":
        conv.ended = True
        return _end("Merchant upset/off-mission and no validated de-escalation copy this turn; exiting respectfully rather than risk a bad send.")
    return _wait(1800, f"No validated reply this turn ({r.skipped_reason}); backing off 30 min instead of sending filler.")


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    try:
        return await asyncio.wait_for(handle_reply(body), timeout=REPLY_BUDGET_SECONDS + 1)
    except asyncio.TimeoutError:
        return _wait(300, "Composition exceeded the time budget; retrying shortly rather than risking a late or bad send.")


# ---------------------------------------------------------------------------
# /v1/healthz, /v1/metadata, /v1/teardown
# ---------------------------------------------------------------------------

@app.get("/v1/healthz")
async def healthz():
    return {"status": "ok", "uptime_seconds": int(time.time() - APP_START),
            "contexts_loaded": STATE.contexts.counts()}


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": TEAM_NAME,
        "team_members": TEAM_MEMBERS,
        "model": LLM_MODEL,
        "approach": ("Solo-built by Adarsh Rawat. Deterministic router (auto-reply / intent / hostility / opt-out "
                     "detection, suppression, cadence) + evidence brief built in code (digest lookup, date math, "
                     "peer deltas, language, salutation) + LLM composer (OpenAI, temperature 0) with per-trigger-kind "
                     "playbooks + deterministic validator (number/entity/citation grounding, generic-copy, taboo, "
                     "CTA shape, repeat) with informed repair; background pre-composition on context push."),
        "contact_email": CONTACT_EMAIL,
        "version": BOT_VERSION,
        "submitted_at": _utcnow(),
    }


@app.post("/v1/teardown")
async def teardown():
    STATE.reset()
    return {"status": "wiped"}


@app.get("/")
async def root():
    return {"service": "vera", "docs": "/docs", "health": "/v1/healthz"}


# ---------------------------------------------------------------------------
# Plain-Python contract from challenge-brief §7.1 / §7.4
# ---------------------------------------------------------------------------

def compose(category: dict, merchant: dict, trigger: dict, customer: Optional[dict] = None,
            now: Optional[str] = None) -> dict:
    """Synchronous compose() as specified in the brief. Returns
    {body, cta, send_as, suppression_key, rationale}; body is "" only if no
    grounded message could be validated."""
    async def run():
        r = await _compose(category, merchant, trigger, customer, now=now or _utcnow(),
                           deadline=time.monotonic() + 28, allow_skip=False)
        return r
    r = asyncio.run(run())
    m = r.message or {}
    return {"body": m.get("body", ""), "cta": m.get("cta", "none"),
            "send_as": m.get("send_as", "merchant_on_behalf" if customer else "vera"),
            "suppression_key": m.get("suppression_key", trigger.get("suppression_key", "")),
            "rationale": m.get("rationale", f"no validated message: {r.skipped_reason}")}
