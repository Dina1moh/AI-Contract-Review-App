from __future__ import annotations

from pathlib import Path

import pytest

from app.core.rag.clause_segmenter import _build_nodes, segment_document


ZTO_PDF = (
    Path(__file__).resolve().parents[2]
    / "ZtoExpressCaymanInc_20160930_F-1_EX-10.10_9752871_EX-10.10_Transportation Agreement.pdf"
)


def test_hierarchy_preserves_compound_parentheses():
    nodes = _build_nodes(
        [
            "5. Obligations are described in this section and remain binding.",
            "5(a) Party A shall provide the required information and notices.",
            "5(b) Party B shall perform the services in accordance with the agreement.",
        ],
        [(1, 1), (1, 1), (1, 1)],
        "docling",
    )
    assert len(nodes) == 1
    assert nodes[0].clause_id == "5."
    assert [child.clause_id for child in nodes[0].children] == ["5(a)", "5(b)"]


def test_text_document_uses_text_parser_and_keeps_same_line_boundaries(tmp_path: Path):
    source = tmp_path / "contract.txt"
    source.write_text(
        "1. Party A shall provide services under this agreement. "
        "2. Party B shall pay the agreed fees and expenses.",
        encoding="utf-8",
    )
    nodes = segment_document(source)
    assert [node.clause_number for node in nodes] == ["1.", "2."]
    assert all(node.parser == "text" for node in nodes)
    assert all(node.page_start == 1 and node.page_end == 1 for node in nodes)


def test_signature_metadata_is_marked_without_being_classified():
    nodes = _build_nodes(
        ["Party A: Name: Signature: Date: Party B: Name: Signature: Date:"],
        [(3, 3)],
        "docling",
    )
    assert nodes[0].metadata["is_signature_metadata"] is True


@pytest.mark.skipif(not ZTO_PDF.exists(), reason="ZTO regression PDF is not present")
def test_zto_top_level_clauses_do_not_collapse_into_three_records():
    nodes = segment_document(ZTO_PDF)
    numbers = {
        node.clause_number.rstrip(".")
        for node in nodes
        if node.clause_number and node.clause_number.rstrip(".").isdigit()
    }
    assert {str(number) for number in range(1, 16)} <= numbers
    assert len(nodes) > 3
