"""
Application service layer for contract classification.

Ties together the RAG building blocks in core/rag/ into the full pipeline:

    uploaded contract (PDF/.txt)
        -> segmenter.extract_text + split_into_clauses
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

from ..core.config import load_labels, TOP_K
from ..core.rag import (
    make_cohere_client,
    embed_queries,
    make_qdrant_client,
    classify_clause,
    load_label_definitions,
    extract_text,
    split_into_clauses,
)
from ..core.rag.segmenter import is_definition
from ..core.rag.validator import is_real_clause


def classify_contract(file_path: Path, top_k: int = TOP_K) -> list[dict]:
    """Run the full pipeline on one uploaded contract file. Returns a list
    of per-clause classification results, in document order.

    Raises:
        ValueError: no clauses could be extracted from the document.
        RuntimeError: an upstream dependency (Cohere, Qdrant, Ollama) failed.
    """
    raw_text = extract_text(file_path)
    clauses = split_into_clauses(raw_text)

    if not clauses:
        raise ValueError(
            "No clauses were detected in this document. It may be empty, "
            "scanned/image-only (needs OCR), or use a layout the "
            "segmentation heuristic doesn't handle."
        )

    labels = load_labels()
    label_definitions = load_label_definitions(labels)

    cohere_client = make_cohere_client()
    qdrant_client = make_qdrant_client()

    # Lightweight validation pass: catch regex mis-splits (stray fragments,
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

    for index, clause_text in enumerate(clauses):
        if is_real_clause(clause_text):
            validated.append((index, clause_text))
        else:
            results.append(
                {
                    "clause_index": index,
                    "clause_text": clause_text,
                    "predicted_label": "REJECTED_FRAGMENT",
                    "retrieved_labels": [],
                    "retrieved_scores": [],
                }
            )

    if not validated:
        raise ValueError(
            "All extracted fragments were rejected by the clause validator. "
            "The document may not contain recognizable contract clauses."
        )

    validated_texts = [text for _, text in validated]

    # Batch-embed all clauses in one call instead of one request per clause.
    query_vectors = embed_queries(cohere_client, validated_texts)

    for (original_index, clause_text), query_vector in zip(validated, query_vectors):
        # Defined-term entries (e.g. inside a DEFINITIONS section) have no
        # matching CUAD label -- classifying them always produces a wrong
        # guess. Skip the retrieval + LLM call entirely and tag them
        # directly; cheaper and more accurate than forcing a guess.
        if is_definition(clause_text):
            results.append(
                {
                    "clause_index": original_index,
                    "clause_text": clause_text,
                    "predicted_label": "DEFINITION",
                    "retrieved_labels": [],
                    "retrieved_scores": [],
                }
            )
            continue

        result = classify_clause(
            clause_text=clause_text,
            query_vector=query_vector,
            qdrant_client=qdrant_client,
            labels=labels,
            label_definitions=label_definitions,
            top_k=top_k,
        )
        results.append(
            {
                "clause_index": original_index,
                "clause_text": clause_text,
                "predicted_label": result["predicted_label"],
                "retrieved_labels": result["retrieved_labels"],
                "retrieved_scores": result["retrieved_scores"],
            }
        )

    results.sort(key=lambda r: r["clause_index"])
    return results