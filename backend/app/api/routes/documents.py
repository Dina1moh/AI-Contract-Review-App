"""
POST /documents/classify

Accepts an uploaded contract (PDF or .txt), runs it through the RAG
classification pipeline (services.rag_service.classify_contract), and
returns a labeled breakdown of every clause found.

Location: app/api/routes/documents.py
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from ...models.schemas import ClauseResult, ContractClassificationResponse
from ...services.rag_service import classify_contract

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_SUFFIXES = {".pdf", ".txt"}


@router.post("/classify", response_model=ContractClassificationResponse)
async def classify_document(
    file: UploadFile = File(...),
) -> ContractClassificationResponse:
    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Upload a .pdf or .txt file.",
        )

    # Stream the upload to a temp file on disk -- classify_contract()
    # expects a Path, and contracts can be too large to hold comfortably
    # in memory.
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        results = classify_contract(tmp_path)
    except ValueError as exc:
        # No clauses detected, empty file, etc. -- a client-fixable error.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        # Ollama down, Cohere/Qdrant unreachable, etc. -- a server-side
        # dependency failure, not the client's fault.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return ContractClassificationResponse(
        filename=file.filename or tmp_path.name,
        total_clauses=len(results),
        clauses=[ClauseResult(**r) for r in results],
    )