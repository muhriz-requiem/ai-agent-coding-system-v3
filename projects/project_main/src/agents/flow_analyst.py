"""Deterministic flow-synthesis analyst (advisory, zero LLM cost).

Synthesizes the whale/flow fetch envelopes (insider Form 4/144, 13F
institutional holdings, unusual options activity) into ONE extra
``AnalystReport`` under the ``flow`` key, so the bull/bear debate sees a
coherent cross-source flow stance instead of three isolated ones.

Design contract (mirrors the whale/flow data philosophy):

* **Deterministic** -- a fixed, transparent scoring rubric; no LLM call,
  no model risk.  The thesis states the arithmetic.
* **Advisory** -- the report feeds the debate prompt only.  The risk
  manager's vetoes (rules 4-6) read ``flow_data`` directly and are
  unaffected; the portfolio manager's directional flow-conviction blend
  filters on the three per-source analyst names, so the ``flow`` report
  can never double-count into sizing.
* **Conditional** -- returns ``None`` unless at least one source has
  ``status == "ok"``; with no usable flow data there is no report, no
  debate contribution, and no LLM cost (the graph simply skips it).
* **Fail-open** -- never raises; an unexpected internal error degrades
  to ``None``.
* **Citation-gated** -- citations quote the envelopes' own evidence
  lines; when a source carries no citation lines, a factual line built
  from that envelope's numbers is emitted instead, so the
  ``citations min_length=1`` contract always holds with real evidence.
"""

from __future__ import annotations

from typing import Any

from src.agents.models import AnalystReport

__all__ = ["FLOW_ANALYST_NAME", "build_flow_report"]

#: Key this report is registered under in the reports mapping.
FLOW_ANALYST_NAME = "flow"

#: Scoring rubric (signed contributions toward bullish [+]).
INSIDER_CLUSTER_BUY_SCORE = 0.15     # >= 2 distinct insider buyers / 30d
INSIDER_CLUSTER_SELL_SCORE = -0.15   # >= 2 distinct insider sellers / 30d
INSIDER_NET_FLOW_SCORE = 0.05        # sign of net dollar flow
INSTITUTIONAL_BALANCE_SCORE = 0.10   # increased vs decreased positions
INSTITUTIONAL_WHALE_SCORE = 0.05     # any whale 'added' alert (+) / all trimmed (-)
OPTIONS_CALL_SKEW_SCORE = 0.10       # call/put premium ratio > 1.5
OPTIONS_PUT_SKEW_SCORE = -0.15       # call/put premium ratio < 0.5

#: Stance thresholds on the net score.
BULLISH_THRESHOLD = 0.10
BEARISH_THRESHOLD = -0.10

#: Confidence = 0.5 + |net|, clamped to the schema's maximum.
MAX_CONFIDENCE = 0.95


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def _insider_contribution(envelope: dict) -> tuple[float, list[str]]:
    """Score the insider envelope; returns (score, evidence lines)."""
    score = 0.0
    evidence: list[str] = []
    cluster_buys = envelope.get("cluster_buys") or 0
    cluster_sells = envelope.get("cluster_sells") or 0
    net_flow = _finite(envelope.get("net_insider_flow")) or 0.0
    evidence.append(
        f"insider: cluster_buys={cluster_buys}, cluster_sells={cluster_sells}, "
        f"net_insider_flow={net_flow:+,.0f}"
    )
    if cluster_buys >= 2:
        score += INSIDER_CLUSTER_BUY_SCORE
    if cluster_sells >= 2:
        score += INSIDER_CLUSTER_SELL_SCORE
    if net_flow > 0:
        score += INSIDER_NET_FLOW_SCORE
    elif net_flow < 0:
        score -= INSIDER_NET_FLOW_SCORE
    return score, evidence


def _institutional_contribution(envelope: dict) -> tuple[float, list[str]]:
    """Score the institutional (13F) envelope; returns (score, evidence)."""
    score = 0.0
    evidence: list[str] = []
    increased = envelope.get("increased_positions") or 0
    decreased = envelope.get("decreased_positions") or 0
    alerts = [a for a in (envelope.get("whale_alerts") or [])
              if isinstance(a, dict) and a.get("action") in ("added", "trimmed")]
    evidence.append(
        f"13F: increased_positions={increased}, decreased_positions={decreased}, "
        f"whale_alerts={[a.get('action') for a in alerts]}"
    )
    if increased > decreased:
        score += INSTITUTIONAL_BALANCE_SCORE
    elif decreased > increased:
        score -= INSTITUTIONAL_BALANCE_SCORE
    if alerts and all(a.get("action") == "added" for a in alerts):
        score += INSTITUTIONAL_WHALE_SCORE
    elif alerts and all(a.get("action") == "trimmed" for a in alerts):
        score -= INSTITUTIONAL_WHALE_SCORE
    return score, evidence


def _options_contribution(envelope: dict) -> tuple[float, list[str]]:
    """Score the unusual-options envelope; returns (score, evidence)."""
    score = 0.0
    evidence: list[str] = []
    ratio = _finite(envelope.get("call_put_premium_ratio"))
    premium = _finite(envelope.get("total_unusual_premium")) or 0.0
    if ratio is not None:
        evidence.append(
            f"UOA: call_put_premium_ratio={ratio:.2f}, "
            f"total_unusual_premium=${premium:,.0f}"
        )
        if ratio > 1.5:
            score += OPTIONS_CALL_SKEW_SCORE
        elif ratio < 0.5:
            score += OPTIONS_PUT_SKEW_SCORE
    return score, evidence


_CONTRIBUTORS = {
    "insider": _insider_contribution,
    "institutional": _institutional_contribution,
    "options_flow": _options_contribution,
}


def _fallback_citation(name: str, envelope: dict) -> list[str]:
    """Factual evidence line for a source with no citation lines."""
    return [f"{name}: status=ok (no citation lines; see thesis numbers)"]


def build_flow_report(ticker: str, flow_data: Any) -> AnalystReport | None:
    """Synthesize the whale/flow envelopes into one ``flow`` report.

    Args:
        ticker: Instrument symbol the report is about.
        flow_data: Mapping of analyst name (``insider``,
            ``institutional``, ``options_flow``) to its fetch envelope.
            Non-dict entries and non-``ok`` envelopes are ignored.

    Returns:
        An ``AnalystReport`` (stance/confidence from the deterministic
        rubric, thesis stating the arithmetic, citations quoting the
        envelopes' evidence) or ``None`` when no source has
        ``status == "ok"`` -- or on any internal error (fail-open).
    """
    try:
        if not isinstance(flow_data, dict):
            return None
        usable = {
            name: envelope
            for name, envelope in flow_data.items()
            if name in _CONTRIBUTORS
            and isinstance(envelope, dict)
            and envelope.get("status") == "ok"
        }
        if not usable:
            return None

        net = 0.0
        thesis_parts: list[str] = []
        citations: list[str] = []
        for name in ("insider", "institutional", "options_flow"):
            envelope = usable.get(name)
            if envelope is None:
                continue
            score, evidence = _CONTRIBUTORS[name](envelope)
            net += score
            thesis_parts.append(f"{name} {'+' if score >= 0 else ''}{score:.2f}")
            citations.extend(
                [str(c) for c in (envelope.get("citations") or []) if str(c).strip()]
                or _fallback_citation(name, envelope)
            )
            citations.extend(evidence)

        if net >= BULLISH_THRESHOLD:
            stance = "bullish"
        elif net <= BEARISH_THRESHOLD:
            stance = "bearish"
        else:
            stance = "neutral"
        confidence = min(MAX_CONFIDENCE, 0.5 + abs(net))

        thesis = (
            f"flow synthesis (deterministic rubric, net={net:+.2f}): "
            + ", ".join(thesis_parts)
            + f"; stance {stance}"
        )
        return AnalystReport.model_validate({
            "agent": FLOW_ANALYST_NAME,
            "ticker": ticker,
            "stance": stance,
            "confidence": round(confidence, 4),
            "thesis": thesis,
            "citations": citations,
        })
    except Exception:
        return None
