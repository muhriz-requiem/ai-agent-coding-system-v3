"""Deterministic risk manager veto gate (TASK-7007B).

Pure rule-based evaluation -- no LLM calls, no network, no async.

Rules are evaluated in a fixed order and every rule that fires appends
its veto reason, so ``veto_reasons`` is deterministic for a given input:

1. Citation gate: any analyst report or the debate record with empty
   citations vetoes with ``missing citations: <agent>`` (reports) or
   ``missing citations: debate:<ticker>`` (debate record).
2. Quant override: ``quant['signal'] == 'SELL'`` with confidence
   strictly greater than 0.80 sets ``quant_override=True`` and vetoes
   with ``quant override: SELL confidence <x> (z-score <z>)``.
3. Position size: ``proposed_position_pct`` strictly greater than the
   account limit ``account.get('max_position_pct', 0.10)`` vetoes with
   ``position size exceeds limit``.
4. Insider cluster sell: ``flow_data['insider']['cluster_sells']`` at or
   above ``account.get('cluster_insider_threshold', 3)`` vetoes with
   ``insider cluster sell: <n> insiders selling``.
5. Institutional exodus: with the sample-size guard
   ``flow_data['institutional']['total_holders'] >=
   account.get('flow_veto_min_holders', 10)``, more closed than twice
   the new positions vetoes with ``institutional exodus: more closed
   than new positions``.
6. Extreme put skew: ``flow_data['options_flow']['call_put_premium_ratio']``
   strictly below ``account.get('uoa_put_skew_veto', 0.30)`` vetoes with
   ``extreme put skew: call/put premium ratio <x>``.

Rules 4-6 encode bearish evidence, so they are BUY-gated: they fire only
when the quant signal is a BUY (risk-adding).  SELLs (exits) always pass
-- no gate ever traps an exit (concerns.txt #5, mirroring the PDT and
panic gates).  All three fail open: absent/invalid flow data produces no
veto, and thresholds come from ``account`` (config), not literals.

The decision is approved if and only if no veto reasons were produced.
"""

from __future__ import annotations

from typing import Any

from src.agents.models import AnalystReport, DebateRecord, RiskDecision

DEFAULT_MAX_POSITION_PCT = 0.10
SELL_CONFIDENCE_THRESHOLD = 0.80
DEFAULT_CLUSTER_INSIDER_THRESHOLD = 3
DEFAULT_FLOW_VETO_MIN_HOLDERS = 10
DEFAULT_UOA_PUT_SKEW_VETO = 0.30


def _debate_label(debate: DebateRecord) -> str:
    """Identifier used for the debate record in citation veto reasons."""
    ticker = getattr(debate, "ticker", None)
    if isinstance(ticker, str) and ticker:
        return f"debate:{ticker}"
    return "debate"


def _threshold(account: Any, key: str, default: float) -> float:
    """Read a numeric threshold from ``account`` (fail to the default)."""
    value = account.get(key, default) if isinstance(account, dict) else default
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _int_or(value: Any, default: int) -> int:
    """Coerce to int (fail to the default; accepts numeric strings)."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def evaluate_risk(
    ticker: str,
    reports: dict[str, AnalystReport],
    debate: DebateRecord,
    quant: dict,
    account: dict,
    proposed_position_pct: float | None = None,
    flow_data: dict | None = None,
) -> RiskDecision:
    """Apply the deterministic veto rules and return a ``RiskDecision``.

    Rules run in order: citation gate, quant override, position size,
    then the BUY-gated whale/flow rules (insider cluster sell,
    institutional exodus, extreme put skew).  ``approved`` is True iff
    no veto reason was produced. This function performs no model calls
    of any kind.  ``flow_data`` maps analyst name to its fetch envelope
    (``{"insider": ..., "institutional": ..., "options_flow": ...}``);
    ``None`` or empty produces no new vetoes.
    """
    veto_reasons: list[str] = []
    quant_override = False

    # Rule 1: citation gate -- every report (dict insertion order), then
    # the debate record.
    for report in reports.values():
        if not report.citations:
            veto_reasons.append(f"missing citations: {report.agent}")
    if not debate.citations:
        veto_reasons.append(f"missing citations: {_debate_label(debate)}")

    # Rule 2: quant override on high-confidence SELL signals. Strictly
    # greater than the threshold, so confidence == 0.80 does NOT veto.
    signal = quant.get("signal")
    confidence = quant.get("confidence")
    if (
        signal == "SELL"
        and confidence is not None
        and confidence > SELL_CONFIDENCE_THRESHOLD
    ):
        quant_override = True
        z_score = quant.get("z_score")
        z_text = f"{z_score:.2f}" if z_score is not None else "n/a"
        veto_reasons.append(
            f"quant override: SELL confidence {confidence:.2f} (z-score {z_text})"
        )

    # Rule 3: position size ceiling (default 0.10 when absent). Strictly
    # greater than the limit, so proposed == limit does NOT veto.
    if proposed_position_pct is not None:
        max_position_pct = account.get("max_position_pct", DEFAULT_MAX_POSITION_PCT)
        if proposed_position_pct > max_position_pct:
            veto_reasons.append("position size exceeds limit")

    # Rules 4-6: whale/flow vetoes. Bearish evidence must only ever block
    # risk-adding BUYs -- never an exit -- so all three are gated on the
    # quant signal being a BUY (case-insensitive).  Absent data fails open.
    signal_text = ""
    if isinstance(quant, dict):
        signal_text = str(quant.get("signal") or "").strip().lower()
    if signal_text == "buy" and flow_data:
        insider = flow_data.get("insider") or {}
        cluster_sells = _int_or(insider.get("cluster_sells"), 0)
        if cluster_sells >= _threshold(
            account, "cluster_insider_threshold", DEFAULT_CLUSTER_INSIDER_THRESHOLD
        ):
            veto_reasons.append(
                f"insider cluster sell: {cluster_sells} insiders selling"
            )

        inst = flow_data.get("institutional") or {}
        total_holders = _int_or(inst.get("total_holders"), 0)
        closed_positions = _int_or(inst.get("closed_positions"), 0)
        new_positions = _int_or(inst.get("new_positions"), 0)
        if (
            total_holders
            >= _threshold(account, "flow_veto_min_holders", DEFAULT_FLOW_VETO_MIN_HOLDERS)
            and closed_positions > new_positions * 2
        ):
            veto_reasons.append("institutional exodus: more closed than new positions")

        opts = flow_data.get("options_flow") or {}
        ratio = opts.get("call_put_premium_ratio")
        try:
            ratio_value = float(ratio) if ratio is not None else None
        except (TypeError, ValueError):
            ratio_value = None
        if ratio_value is not None and ratio_value < _threshold(
            account, "uoa_put_skew_veto", DEFAULT_UOA_PUT_SKEW_VETO
        ):
            veto_reasons.append(
                f"extreme put skew: call/put premium ratio {ratio_value:.2f}"
            )

    return RiskDecision(
        ticker=ticker,
        approved=len(veto_reasons) == 0,
        veto_reasons=veto_reasons,
        quant_override=quant_override,
    )
