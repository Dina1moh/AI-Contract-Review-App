"""
Pydantic response models for the contract classification endpoint.
Location: app/models/schemas.py
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClauseResult(BaseModel):
    """One classified clause from an uploaded contract."""

    clause_index: int = Field(..., description="Order this clause appeared in the document")
    clause_text: str = Field(..., description="The extracted clause text")
    predicted_label: str = Field(..., description="Predicted CUAD clause type")
    clause_type: str | None = Field(
        default=None,
        description="Alias for predicted_label, kept for risk-review UI compatibility",
    )
    retrieved_labels: list[str] = Field(
        default_factory=list,
        description="Clause types of the top-K retrieved TRAIN examples used as evidence",
    )
    retrieved_scores: list[float] = Field(
        default_factory=list,
        description="Qdrant similarity scores for the retrieved examples, same order as retrieved_labels",
    )
    risk: bool = Field(default=False, description="Potential risk indicator")
    risk_type: str | None = Field(default=None, description="Potential risk type")
    reason: str = Field(
        default="Risk detection was not run for this clause.",
        description="Risk detector explanation",
    )
    evidence: str = Field(default="", description="Exact contract evidence for the indicator")
    source: dict = Field(
        default_factory=lambda: {"name": None, "url": None, "title": None},
        description="Legal guidance source used by the risk detector",
    )
    legal_knowledge: list[dict] = Field(
        default_factory=list,
        description="Retrieved legal guidance metadata and concise snippets",
    )


class ContractClassificationResponse(BaseModel):
    """Response for POST /classify-contract."""

    filename: str
    total_clauses: int
    clauses: list[ClauseResult]


class ErrorResponse(BaseModel):
    detail: str
