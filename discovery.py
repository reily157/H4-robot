"""
HIP-4 market discovery — parse outcomeMeta to identify cycle targets.

Scope:
    At cycle start, fetch /info outcomeMeta and identify:
      - The active priceBucket BTC question + its 3 namedOutcomes
      - The priceBinary BTC outcome with matching expiry
    Build a CycleSpec describing the cycle for the recorder to subscribe to.

Async-first: uses aiohttp to align with the recorder event loop.

Resilience strategy:
    - Missing bucket → log warning, return CycleSpec(bucket=None, binary=...)
    - Missing binary → log warning, return CycleSpec(bucket=..., binary=None)
    - Both missing → raise DiscoveryError (caller decides whether to retry)
    - Malformed description string → skip that entry, do not crash discovery
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

import aiohttp


log = logging.getLogger(__name__)


# ─── Constants ─────────────────────────────────────────────────────────────────

INFO_URL = "https://api.hyperliquid.xyz/info"
DEFAULT_TIMEOUT_S = 10


# ─── Exceptions ────────────────────────────────────────────────────────────────

class DiscoveryError(Exception):
    """Raised when no target markets can be identified at all."""


# ─── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PriceBinarySpec:
    outcome_id: int
    underlying: str           # "BTC"
    target_price: float
    expiry: datetime          # UTC
    period: str               # "1d"


@dataclass(frozen=True)
class PriceBucketSpec:
    question_id: int
    underlying: str
    expiry: datetime
    period: str
    thresholds: list[float]   # e.g. [75348.0, 78423.0]
    named_outcome_ids: list[int]   # e.g. [67, 68, 69]
    fallback_outcome_id: int       # e.g. 66


@dataclass(frozen=True)
class CycleSpec:
    """
    Full description of a single observation cycle.

    bucket: the multi-outcome bucket question (may be None if not found)
    binary: the matching priceBinary outcome (may be None if not found)
    raw_meta: original outcomeMeta JSON for reproducibility / debugging
    """
    bucket: PriceBucketSpec | None
    binary: PriceBinarySpec | None
    raw_meta: dict = field(repr=False)

    @property
    def has_bucket(self) -> bool:
        return self.bucket is not None

    @property
    def has_binary(self) -> bool:
        return self.binary is not None

    @property
    def is_complete(self) -> bool:
        """Both bucket and binary are present and aligned on expiry."""
        if not (self.has_bucket and self.has_binary):
            return False
        return self.bucket.expiry == self.binary.expiry


# ─── Description parsing ───────────────────────────────────────────────────────

# Examples from the brief:
#   "class:priceBinary|underlying:BTC|expiry:20260520-0600|targetPrice:76886|period:1d"
#   "class:priceBucket|underlying:BTC|expiry:20260520-0600|priceThresholds:75348,78423|period:1d"

_KV_PATTERN = re.compile(r"([a-zA-Z]+):([^|]+)")


def _parse_kv_description(desc: str) -> dict[str, str]:
    """Parse a HL pipe-delimited key:value description into a dict."""
    if not isinstance(desc, str):
        return {}
    out: dict[str, str] = {}
    for match in _KV_PATTERN.finditer(desc):
        key, val = match.group(1), match.group(2).strip()
        out[key] = val
    return out


def _parse_expiry(expiry_str: str) -> datetime:
    """
    Parse 'YYYYMMDD-HHMM' (UTC) into a datetime.

    Examples:
        '20260520-0600' → 2026-05-20 06:00:00 UTC
    """
    if not isinstance(expiry_str, str):
        raise ValueError(f"expiry must be str, got {type(expiry_str).__name__}")
    if len(expiry_str) != 13 or expiry_str[8] != "-":
        raise ValueError(f"expiry must be 'YYYYMMDD-HHMM', got {expiry_str!r}")
    try:
        dt = datetime.strptime(expiry_str, "%Y%m%d-%H%M")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError as e:
        raise ValueError(f"could not parse expiry {expiry_str!r}: {e}")


def parse_price_binary(outcome: dict) -> PriceBinarySpec | None:
    """
    Parse a single outcome dict into a PriceBinarySpec.
    Returns None if the description does not match priceBinary class.
    """
    if not isinstance(outcome, dict):
        return None
    outcome_id = outcome.get("outcome")
    desc = outcome.get("description", "")
    if outcome_id is None or not isinstance(outcome_id, int):
        return None

    kv = _parse_kv_description(desc)
    if kv.get("class") != "priceBinary":
        return None

    try:
        target = float(kv["targetPrice"])
        expiry = _parse_expiry(kv["expiry"])
        return PriceBinarySpec(
            outcome_id=outcome_id,
            underlying=kv.get("underlying", ""),
            target_price=target,
            expiry=expiry,
            period=kv.get("period", ""),
        )
    except (KeyError, ValueError) as e:
        log.warning(f"failed to parse priceBinary outcome {outcome_id}: {e}")
        return None


def parse_price_bucket(question: dict) -> PriceBucketSpec | None:
    """
    Parse a single question dict into a PriceBucketSpec.
    Returns None if the description does not match priceBucket class.
    """
    if not isinstance(question, dict):
        return None
    question_id = question.get("question")
    desc = question.get("description", "")
    if question_id is None or not isinstance(question_id, int):
        return None

    kv = _parse_kv_description(desc)
    if kv.get("class") != "priceBucket":
        return None

    try:
        thresholds_str = kv["priceThresholds"]
        thresholds = [float(t.strip()) for t in thresholds_str.split(",")]
        expiry = _parse_expiry(kv["expiry"])
        named_outcomes = question.get("namedOutcomes", [])
        if not isinstance(named_outcomes, list):
            named_outcomes = []
        fallback = question.get("fallbackOutcome")
        if not isinstance(fallback, int):
            log.warning(f"question {question_id} missing fallbackOutcome")
            return None
        return PriceBucketSpec(
            question_id=question_id,
            underlying=kv.get("underlying", ""),
            expiry=expiry,
            period=kv.get("period", ""),
            thresholds=thresholds,
            named_outcome_ids=[int(o) for o in named_outcomes if isinstance(o, int)],
            fallback_outcome_id=fallback,
        )
    except (KeyError, ValueError) as e:
        log.warning(f"failed to parse priceBucket question {question_id}: {e}")
        return None


# ─── Discovery logic ───────────────────────────────────────────────────────────

def identify_targets(meta: dict, underlying: str = "BTC") -> CycleSpec:
    """
    From a full outcomeMeta payload, identify the cycle targets for `underlying`.

    Selection rule:
        - Pick the priceBucket BTC question with the *latest* expiry.
        - Pick the priceBinary BTC outcome with matching expiry (or latest
          if no bucket found).
        - If multiple candidates, prefer the one with the most recent expiry.

    Returns:
        CycleSpec — may have bucket=None or binary=None if not found.

    Raises:
        DiscoveryError: if both bucket and binary are missing.
        ValueError: if meta is malformed (not a dict, missing keys).
    """
    if not isinstance(meta, dict):
        raise ValueError(f"meta must be dict, got {type(meta).__name__}")

    outcomes = meta.get("outcomes", [])
    questions = meta.get("questions", [])
    if not isinstance(outcomes, list):
        outcomes = []
    if not isinstance(questions, list):
        questions = []

    # ── Find candidate buckets for the underlying ──────────────────────────
    bucket_candidates: list[PriceBucketSpec] = []
    for q in questions:
        spec = parse_price_bucket(q)
        if spec is None:
            continue
        if spec.underlying != underlying:
            continue
        bucket_candidates.append(spec)

    bucket = max(bucket_candidates, key=lambda b: b.expiry) if bucket_candidates else None

    # ── Find candidate binaries for the underlying ─────────────────────────
    binary_candidates: list[PriceBinarySpec] = []
    for o in outcomes:
        spec = parse_price_binary(o)
        if spec is None:
            continue
        if spec.underlying != underlying:
            continue
        binary_candidates.append(spec)

    # Prefer binary aligned with bucket expiry; fall back to most recent.
    binary: PriceBinarySpec | None = None
    if bucket is not None:
        aligned = [b for b in binary_candidates if b.expiry == bucket.expiry]
        if aligned:
            binary = aligned[0]
    if binary is None and binary_candidates:
        binary = max(binary_candidates, key=lambda b: b.expiry)

    # ── Build CycleSpec ────────────────────────────────────────────────────
    if bucket is None and binary is None:
        raise DiscoveryError(
            f"no priceBucket or priceBinary found for underlying={underlying!r}"
        )
    if bucket is None:
        log.warning(f"no priceBucket found for {underlying} — binary-only cycle")
    if binary is None:
        log.warning(f"no priceBinary found for {underlying} — bucket-only cycle")
    if bucket and binary and bucket.expiry != binary.expiry:
        log.warning(
            f"bucket expiry {bucket.expiry} != binary expiry {binary.expiry} "
            "— misaligned cycle, downstream H5 will be unusable"
        )

    return CycleSpec(bucket=bucket, binary=binary, raw_meta=meta)


# ─── Async REST fetch ──────────────────────────────────────────────────────────

async def fetch_outcome_meta(
    session: aiohttp.ClientSession,
    url: str = INFO_URL,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict:
    """
    Fetch the current outcomeMeta from Hyperliquid info endpoint.

    Raises:
        aiohttp.ClientError on network failure.
        ValueError if response is not a dict.
    """
    payload = {"type": "outcomeMeta"}
    async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
        resp.raise_for_status()
        data = await resp.json()
    if not isinstance(data, dict):
        raise ValueError(f"outcomeMeta response is not a dict: {type(data).__name__}")
    return data


async def discover(
    session: aiohttp.ClientSession,
    underlying: str = "BTC",
) -> CycleSpec:
    """
    Convenience: fetch + identify in one call.
    """
    meta = await fetch_outcome_meta(session)
    return identify_targets(meta, underlying=underlying)
