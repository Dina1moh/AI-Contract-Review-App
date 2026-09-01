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
    retrieved_labels: list[str] = Field(
        default_factory=list,
        description="Clause types of the top-K retrieved TRAIN examples used as evidence",
    )
    retrieved_scores: list[float] = Field(
        default_factory=list,
        description="Qdrant similarity scores for the retrieved examples, same order as retrieved_labels",
    )


class ContractClassificationResponse(BaseModel):
    """Response for POST /classify-contract."""

    filename: str
    total_clauses: int
    clauses: list[ClauseResult]


class ErrorResponse(BaseModel):
    detail: str