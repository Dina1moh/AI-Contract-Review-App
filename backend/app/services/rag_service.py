"""
Application service layer for contract classification.

Ties together the RAG building blocks in core/rag/ into the full pipeline:

    uploaded contract (PDF/.txt)
        -> structured hybrid parser + contract-aware clause segmenter
        -> embedder.embed_queries          (Cohere, batched)
        -> per clause: generator.classify_clause
               (internally calls retriever.retrieve_similar)
        -> list of labeled clauses, in document order

Route handlers (api/routes/documents.py) call classify_contract() and
should not talk to core/rag/ directly -- that keeps the HTTP layer free of
pipeline details and makes this logic reusable (e.g. from a CLI script or
a background job later) without depending on FastAPI.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Callable

from ..core.config import load_labels, TOP_K
from ..core.rag import (
    make_cohere_client,
    embed_queries,
    make_qdrant_client,
    classify_clause,
    load_label_definitions,
)
from ..core.rag.clause_segmenter import segment_document
from ..core.rag.segmenter import is_definition
from ..core.rag.validator import is_real_clause
from ..core.risk import detect_clause_risk

ProgressCallback = Callable[[dict], None]
_CLASSIFICATION_LOCK = Lock()


def _emit(
    callback: ProgressCallback | None,
    stage: str,
    message: str,
    **details,
) -> None:
    if callback:
        callback({"type": "progress", "stage": stage, "message": message, **details})


def _default_risk(reason: str) -> dict:
    return {
        "risk": False,
        "risk_type": None,
        "reason": reason,
        "evidence": "",
        "source": {"name": None, "url": None, "title": None},
        "legal_knowledge": [],
    }


def _with_risk_fields(result: dict, risk_result: dict) -> dict:
    return {
        **result,
        "clause_type": result["predicted_label"],
        "risk": bool(risk_result.get("risk", False)),
        "risk_type": risk_result.get("risk_type"),
        "reason": risk_result.get("reason") or "",
        "evidence": risk_result.get("evidence") or "",
        "source": risk_result.get("source") or {"name": None, "url": None, "title": None},
        "legal_knowledge": risk_result.get("legal_knowledge") or [],
    }


def classify_contract(
    file_path: Path,
    top_k: int = TOP_K,
    progress_callback: ProgressCallback | None = None,
) -> list[dict]:
    # Local Qdrant storage is single-owner. Serialize complete analyses so a
    # second upload cannot open the same local collection while the first one
    # is still embedding, searching, or generating results.
    with _CLASSIFICATION_LOCK:
        return _classify_contract(
            file_path,
            top_k=top_k,
            progress_callback=progress_callback,
        )


def _classify_contract(
    file_path: Path,
    top_k: int = TOP_K,
    progress_callback: ProgressCallback | None = None,
) -> list[dict]:
    """Run the full pipeline on one uploaded contract file. Returns a list
    of per-clause classification results, in document order.

    Raises:
        ValueError: no clauses could be extracted from the document.
        RuntimeError: an upstream dependency (Cohere, Qdrant, Ollama) failed.
    """
    _emit(progress_callback, "extract", "Extracting document structure")
    segments = segment_document(file_path)
    _emit(
        progress_callback,
        "extract",
        "Document structure extracted",
        parser=segments[0].parser if segments else None,
        total_blocks=len(segments),
    )

    _emit(progress_callback, "segment", "Detecting contract clause boundaries")
    clauses = [segment.text for segment in segments]
    _emit(progress_callback, "segment", "Clause segmentation complete", total=len(clauses))

    if not clauses:
        raise ValueError(
            "No clauses were detected in this document. It may be empty, "
            "scanned/image-only (needs OCR), or use a layout the "
            "segmentation heuristic doesn't handle."
        )

    _emit(progress_callback, "setup", "Loading CUAD labels and model configuration")
    labels = load_labels()
    label_definitions = load_label_definitions(labels)

    cohere_client = make_cohere_client()
    qdrant_client = make_qdrant_client()

    # Lightweight validation pass: catch boundary errors (stray fragments,
    # footnote-like remnants the pattern-based filter missed) BEFORE
    # spending a Cohere embedding call + Qdrant lookup + full classification
    # prompt on something that isn't actually a clause. See validator.py
    # for why this doesn't let the LLM do segmentation itself.
    #
    # Original document position (clause_index) is preserved through
    # filtering so the final output stays in document order regardless of
    # what got validated vs. rejected.
    results: list[dict] = []
    validated: list[tuple[int, str]] = []  # (original_index, clause_text)

    _emit(progress_callback, "validate", "Validating extracted clauses", total=len(clauses))
    for index, clause_text in enumerate(clauses):
        if segments[index].metadata.get("is_signature_metadata"):
            results.append(
                _with_risk_fields(
                    {
                        "clause_index": index,
                        "clause_text": clause_text,
                        "predicted_label": "DOCUMENT_METADATA",
                        "retrieved_labels": [],
                        "retrieved_scores": [],
                    },
                    _default_risk("Signature or document metadata; classification skipped."),
                )
            )
            continue
        if is_real_clause(clause_text):
            validated.append((index, clause_text))
        else:
            results.append(
                _with_risk_fields(
                    {
                        "clause_index": index,
                        "clause_text": clause_text,
                        "predicted_label": "REJECTED_FRAGMENT",
                        "retrieved_labels": [],
                        "retrieved_scores": [],
                    },
                    _default_risk("Rejected fragment; risk detection was skipped."),
                )
            )

    if not validated:
        raise ValueError(
            "All extracted fragments were rejected by the clause validator. "
            "The document may not contain recognizable contract clauses."
        )

    validated_texts = [text for _, text in validated]

    # Batch-embed all clauses in one call instead of one request per clause.
    _emit(
        progress_callback,
        "embedding",
        "Creating Cohere embeddings for validated clauses",
        current=0,
        total=len(validated_texts),
    )
    query_vectors = embed_queries(cohere_client, validated_texts)
    _emit(
        progress_callback,
        "embedding",
        "Clause embeddings complete",
        current=len(validated_texts),
        total=len(validated_texts),
    )

    total_validated = len(validated)
    for clause_number, ((original_index, clause_text), query_vector) in enumerate(
        zip(validated, query_vectors), start=1
    ):
        _emit(
            progress_callback,
            "clause",
            f"Processing clause {clause_number} of {total_validated}",
            current=clause_number,
            total=total_validated,
            clause_index=original_index,
        )
        # Defined-term entries (e.g. inside a DEFINITIONS section) have no
        # matching CUAD label -- classifying them always produces a wrong
        # guess. Skip the retrieval + LLM call entirely and tag them
        # directly; cheaper and more accurate than forcing a guess.
        if is_definition(clause_text):
            _emit(
                progress_callback,
                "clause",
                f"Clause {clause_number} is a definition; model calls skipped",
                current=clause_number,
                total=total_validated,
                clause_index=original_index,
            )
            results.append(
                _with_risk_fields(
                    {
                        "clause_index": original_index,
                        "clause_text": clause_text,
                        "predicted_label": "DEFINITION",
                        "retrieved_labels": [],
                        "retrieved_scores": [],
                    },
                    _default_risk("Definition entry; risk detection was skipped."),
                )
            )
            continue

        result = classify_clause(
            clause_text=clause_text,
            query_vector=query_vector,
            qdrant_client=qdrant_client,
            labels=labels,
            label_definitions=label_definitions,
            top_k=top_k,
            progress_callback=progress_callback,
        )
        classification_result = {
            "clause_index": original_index,
            "clause_text": clause_text,
            "predicted_label": result["predicted_label"],
            "retrieved_labels": result["retrieved_labels"],
            "retrieved_scores": result["retrieved_scores"],
        }
        try:
            risk_result = detect_clause_risk(
                clause_text=clause_text,
                clause_type=result["predicted_label"],
                cohere_client=cohere_client,
                qdrant_client=qdrant_client,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            risk_result = _default_risk(
                "Risk detection could not run; classification result is still available. "
                f"Reason: {exc}"
            )
        results.append(
            _with_risk_fields(classification_result, risk_result)
        )
        _emit(
            progress_callback,
            "clause",
            f"Clause {clause_number} complete",
            current=clause_number,
            total=total_validated,
            clause_index=original_index,
        )

    results.sort(key=lambda r: r["clause_index"])
    _emit(progress_callback, "complete", "Contract analysis complete", total=len(results))
    return results
