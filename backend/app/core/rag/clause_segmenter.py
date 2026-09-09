"""Contract-aware segmentation over structured parser blocks."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .hybrid_parser import parse_document
from .segmenter import (
    _clean_pages,
    _looks_like_all_caps_title,
    _marker_for_text,
    is_definition,
    split_into_clauses,
)


SIGNATURE_RE = re.compile(
    r"\b(?:party\s+[ab]|company\s+seal|signature|signed|name|title|date)\s*:",
    re.IGNORECASE,
)


@dataclass
class ClauseNode:
    clause_id: str
    clause_number: str | None
    heading: str | None
    text: str
    page_start: int | None
    page_end: int | None
    parser: str
    children: list["ClauseNode"] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


def _is_signature(text: str) -> bool:
    matches = len(SIGNATURE_RE.findall(text))
    return matches >= 2 and len(text) < 800


def _parent_id(marker: str | None) -> str | None:
    if not marker:
        return None
    match = re.match(r"^(\d+)\.\d+", marker)
    if match:
        return match.group(1)
    match = re.match(r"^(\d+)\s*\(", marker)
    if match:
        return match.group(1)
    if marker.startswith("("):
        return None
    return None


def _build_nodes(
    clauses: list[str],
    pages: list[tuple[int | None, int | None]],
    parser: str,
) -> list[ClauseNode]:
    nodes: list[ClauseNode] = []
    top_level: dict[str, ClauseNode] = {}
    for index, text in enumerate(clauses, start=1):
        marker = _marker_for_text(text)
        parent_id = _parent_id(marker)
        heading = text if _looks_like_all_caps_title(text) else None
        page_start, page_end = pages[index - 1] if index - 1 < len(pages) else (None, None)
        node = ClauseNode(
            clause_id=marker or f"clause-{index}",
            clause_number=marker,
            heading=heading,
            text=text,
            page_start=page_start,
            page_end=page_end,
            parser=parser,
            metadata={
                "is_definition": is_definition(text),
                "is_signature_metadata": _is_signature(text),
                "has_structural_marker": marker is not None,
            },
        )
        if parent_id and parent_id in top_level:
            top_level[parent_id].children.append(node)
        else:
            nodes.append(node)
            if marker and re.match(r"^\d+\.?$", marker):
                top_level[marker.rstrip(".")] = node
    return nodes


def segment_document(file_path: Path) -> list[ClauseNode]:
    """Parse once, normalize page structure, then segment deterministically."""
    parsed = parse_document(file_path)
    by_page: dict[int, list[str]] = {}
    for block in parsed.blocks:
        by_page.setdefault(block.page_start, []).append(block.text)
    pages = _clean_pages(["\n".join(by_page[number]) for number in sorted(by_page)])
    flattened_parts: list[str] = []
    page_ranges: list[tuple[int, int, int]] = []
    cursor = 0
    for page_number, page in enumerate(pages, start=1):
        if not page:
            continue
        start = cursor
        flattened_parts.append(page)
        cursor += len(page)
        page_ranges.append((start, cursor, page_number))
        flattened_parts.append("\n\n")
        cursor += 2
    flattened = "".join(flattened_parts)
    clauses = split_into_clauses(flattened)
    page_spans: list[tuple[int | None, int | None]] = []
    search_from = 0
    for clause in clauses:
        position = flattened.find(clause[: min(120, len(clause))], search_from)
        if position < 0:
            position = search_from
        end = position + len(clause)
        covered = [
            page
            for start, stop, page in page_ranges
            if start < end and stop > position
        ]
        page_spans.append(
            (min(covered) if covered else None, max(covered) if covered else None)
        )
        search_from = max(search_from + 1, end)
    return _build_nodes(clauses, page_spans, parsed.parser)
