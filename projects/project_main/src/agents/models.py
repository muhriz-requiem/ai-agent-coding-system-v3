"""Pydantic models for agent message types with evidence citations."""

import json
from typing import Any, List, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


_CITATION_TEXT_KEYS = ("claim", "text", "quote", "summary", "evidence")
_CITATION_SOURCE_KEYS = ("source", "source_url", "url")


def _coerce_string_item(item: Any) -> Any:
    """Flatten one LLM-provided citation/point into a plain string."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in _CITATION_TEXT_KEYS:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                text = value.strip()
                for source_key in _CITATION_SOURCE_KEYS:
                    source = item.get(source_key)
                    if isinstance(source, str) and source.strip():
                        return f"{text} [{source.strip()}]"
                return text
        return json.dumps(item, default=str)
    if isinstance(item, (list, tuple)):
        return json.dumps(item, default=str)
    return str(item)


def _coerce_string_list(value: Any) -> Any:
    """Coerce common LLM list-field slips to a list of plain strings.

    Accepts a bare string, dict-shaped entries (``{"claim": ...}``), and
    non-string scalars; the ``List[str]`` / ``min_length=1`` contract is
    still enforced by the model afterwards.
    """
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [_coerce_string_item(item) for item in value]
    return value


class AnalystReport(BaseModel):
    """Analyst report with stance and confidence.

    ``agent``/``thesis`` also accept the common LLM spellings ``analyst``/
    ``summary`` at validation time (the canonical names are unchanged for
    construction and serialization); ``citations`` stays strictly
    non-empty.
    """

    model_config = ConfigDict(populate_by_name=True)

    agent: str = Field(validation_alias=AliasChoices("agent", "analyst"))
    ticker: str
    stance: Literal["bullish", "bearish", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    thesis: str = Field(validation_alias=AliasChoices("thesis", "summary"))
    citations: List[str] = Field(min_length=1)

    @field_validator("citations", mode="before")
    @classmethod
    def _coerce_citations(cls, value: Any) -> Any:
        return _coerce_string_list(value)


class DebateRecord(BaseModel):
    """Record of a debate with bull and bear points."""

    ticker: str
    bull_points: List[str]
    bear_points: List[str]
    winner: Literal["bull", "bear", "undecided"]
    citations: List[str] = Field(min_length=1)

    @field_validator("citations", "bull_points", "bear_points", mode="before")
    @classmethod
    def _coerce_lists(cls, value: Any) -> Any:
        return _coerce_string_list(value)


class RiskDecision(BaseModel):
    """Risk decision with approval status and optional veto reasons."""
    
    ticker: str
    approved: bool
    veto_reasons: List[str] = Field(default_factory=list)
    quant_override: bool = False
    narrative: str = ""


class OrderPlan(BaseModel):
    """Order plan with side, quantity, and limit price."""
    
    ticker: str
    side: Literal["buy", "sell"]
    quantity: float = Field(gt=0)
    limit_price: float = Field(gt=0.0)
    rationale: str
