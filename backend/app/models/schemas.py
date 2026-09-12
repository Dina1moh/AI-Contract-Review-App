"""
Pydantic response models for the contract classification endpoint.
Location: app/models/schemas.py
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ContractRiskStatus = Literal[
    "COMPLETED",
    "RISK_ANALYSIS_UNAVAILABLE",
]
OverallRisk = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL", "NO_SIGNIFICANT_RISK"]


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
        description="Clause types of the compressed retrieved TRAIN evidence",
    )
    retrieved_scores: list[float] = Field(
        default_factory=list,
        description="Similarity scores for compressed retrieved evidence, same order as retrieved_labels",
    )
    candidate_labels: list[str] = Field(
        default_factory=list,
        description="Unique CUAD labels offered to the final candidate-constrained classifier",
    )
    retrieved_examples: list[dict] = Field(
        default_factory=list,
        description="At most two deduplicated CUAD evidence items with exact text and score",
    )
    classification_status: str | None = Field(
        default=None,
        description="Validated classifier outcome, such as VALID_CANDIDATE or FALLBACK_RETRIEVAL",
    )
    classification_source: str = Field(default="ollama")
    fallback_used: bool = Field(default=False)
    fallback_reason: str | None = Field(default=None)
    classification_confidence: float | None = Field(default=None)
    retrieval_status: str = Field(default="ok")
    retrieval_error: str | None = Field(default=None)
class ContractRiskAssessment(BaseModel):
    """Strict contract-level risk report; no risk fields exist on ClauseResult."""

    status: ContractRiskStatus = "RISK_ANALYSIS_UNAVAILABLE"
    overall_risk: OverallRisk | None = None
    summary: str = ""
    risk_domains: list[dict] = Field(default_factory=list)
    positive_protections: list[str] = Field(default_factory=list)
    missing_protections: list[str] = Field(default_factory=list)
    cross_clause_findings: list[dict] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    supporting_legal_guidance: list[dict] = Field(default_factory=list)


class ContractClassificationResponse(BaseModel):
    """Response for POST /classify-contract."""

    filename: str
    total_clauses: int
    clauses: list[ClauseResult]
    contract_risk_assessment: ContractRiskAssessment = Field(
        default_factory=dict,
        description="Evidence-constrained contract-level risk assessment",
    )


class ErrorResponse(BaseModel):
    detail: str
