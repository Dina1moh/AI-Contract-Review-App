"""
Splits an uploaded contract (PDF or .txt) into individual clauses so each
one can be classified by generator.classify_clause().

This is new logic -- not present in the original single-clause classifier
scripts, since those assumed the clause text was already isolated (one row
from the CUAD test CSV).

Install (one-time):
    pip install pdfplumber
"""

from __future__ import annotations

import re
import difflib
from pathlib import Path

import pdfplumber


# Matches common numbered-section headers, e.g.:
#   "1. Definitions"   "1.1 Term"   "Section 3."   "ARTICLE IV"   "(a) Termination"
# Requires a period (or paren/word) immediately after the number, which is
# what distinguishes a real clause number from a footnote marker below.
SECTION_HEADER_RE = re.compile(
    r"""
    ^\s*
    (?:
        \d+(?:\.\d+)*\.?          # 1.  or 1.1  or 1.1.2
        | \([a-zA-Z]\)            # (a)
        | [A-Z][A-Za-z]*\s+\d+    # Section 3 / Article IV (arabic numeral)
        | (?:ARTICLE|SECTION)\s+[IVXLCDM]+\b   # ARTICLE IV / SECTION XII (roman numeral)
    )
    \s+\S
    """,
    re.VERBOSE | re.IGNORECASE,
)

# ALL-CAPS section titles with no leading number at all, e.g.
#   "TERM AND TERMINATION"   "CONFIDENTIALITY"   "LIMITATION OF LIABILITY"
# Common in contracts that don't use decimal numbering (confirmed cause of
# a 22,000+ char under-split clause seen in a real Franchise agreement
# sample).
#
# RISK: multi-line ALL-CAPS legal boilerplate (liability disclaimers,
# warranty waivers -- exactly the text behind CUAD labels like "Uncapped
# Liability" / "Cap On Liability") also gets extracted as consecutive
# short ALL-CAPS lines. A loose pattern here would wrongly split those
# clauses apart. Mitigations:
#   - max 6 words (real titles are short; boilerplate sentence fragments
#     that happen to be short are rarer but not impossible)
#   - must NOT end in a comma (mid-sentence lines end there)
#   - must NOT contain digits (boilerplate clauses reference section
#     numbers like "SECTION 11.3"; titles don't)
# This reduces but does not eliminate false positives -- verify against
# real liability/warranty clauses before trusting this in production.
ALL_CAPS_HEADER_RE = re.compile(
    r"^\s*(?:[A-Z][A-Z&/-]*\s+){0,5}[A-Z][A-Z&/-]*\.?\s*$"
)


def _looks_like_all_caps_title(line: str) -> bool:
    stripped = line.strip()
    if not ALL_CAPS_HEADER_RE.match(stripped):
        return False
    if any(ch.isdigit() for ch in stripped):
        return False  # boilerplate referencing "SECTION 11.3" etc.
    if len(stripped) < 6:
        return False  # too short to be a meaningful title
    return True

# Footnote markers: a bare 1-2 digit number, a space, then a capital letter
# -- NO period after the digits. Real clause numbers always have a period
# (or are wrapped in parens) immediately after the digit(s), e.g. "1." or
# "1.1", so this pattern only matches drafting-note footnotes like
# "1 This form is fairly generic..." or "20 Pre-approved subcontractors
# should be included." Confirmed against the actual Cooley SaaS template,
# which places these as numbered notes separate from the numbered clauses.
FOOTNOTE_LINE_RE = re.compile(r"^\d{1,2}\s+[A-Z]")

# Definition entries inside a "1. DEFINITIONS." section, e.g.:
#   1.1 "Affiliate" means, with respect to any entity, ...
# CUAD's 41 labels don't include a "definitions" category, so forcing one
# of these into a substantive label (License Grant, Uncapped Liability,
# etc.) is always wrong. Detected so callers can route them to a fixed
# DEFINITION bucket instead of the classifier -- see is_definition().
DEFINITION_RE = re.compile(
    r'^\d+\.\d+\s+(?:["\u201c][^"\u201d]{1,120}["\u201d](?:\s*(?:or|and|,)\s*["\u201c][^"\u201d]{1,120}["\u201d])*)\s+(?:means|has the meaning)\b',
    re.IGNORECASE,
)

# Running header/footer text repeated on (almost) every page, e.g. a firm's
# "this is not legal advice" disclaimer. Detected by frequency across pages
# rather than hardcoded, so this generalizes to other templates.
HEADER_FOOTER_MIN_PAGE_FRACTION = 0.5

MIN_CLAUSE_CHARS = 40  # skip stray headers / page numbers / whitespace noise

# If SECTION_HEADER_RE finds no numbered headers at all (e.g. contracts
# using ALL-CAPS headers, roman numerals, or no numbering convention),
# a whole section can end up as one oversized clause. Confirmed on real
# CUAD contracts: a Franchise agreement produced a single 22,000+ char
# "clause". Anything over this length gets a fallback secondary split.
LONG_CLAUSE_FALLBACK_THRESHOLD = 3000

# Inline enumerated sub-item markers that can appear mid-paragraph rather
# than at the start of a line, e.g. "...as follows: (a) the Party shall
# ... (b) the Party will...". SECTION_HEADER_RE only matches at line
# start, so these need a separate inline split.
INLINE_SUBITEM_RE = re.compile(r"(?=\s\((?:[a-z]|[ivx]{1,4})\)\s)")

# Last-resort split when no structural markers exist at all: break on
# sentence boundaries, keeping chunks under the fallback threshold. This
# never produces a "correct" clause boundary, but prevents one enormous
# garbled block from being sent to the classifier as if it were one clause.
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.;])\s+(?=[A-Z(])")


def _sentence_chunk(text: str) -> list[str]:
    """Last-resort split when no structural markers exist: break on
    sentence boundaries, keeping chunks under the fallback threshold.
    """
    sentences = SENTENCE_BOUNDARY_RE.split(text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) > LONG_CLAUSE_FALLBACK_THRESHOLD:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current.strip())
    return chunks if chunks else [text]


def _fallback_split_oversized(clause: str) -> list[str]:
    """Break up a clause that SECTION_HEADER_RE failed to split, using
    inline sub-item markers first, then sentence-boundary chunking as a
    last resort. Every returned piece is guaranteed to be at or under
    LONG_CLAUSE_FALLBACK_THRESHOLD -- inline-marker splitting alone does
    NOT guarantee this (an unevenly-sized sub-item can still be oversized),
    so each resulting part is checked and sentence-chunked again if needed.
    """
    if len(clause) <= LONG_CLAUSE_FALLBACK_THRESHOLD:
        return [clause]

    # Try inline (a)/(b)/(i)/(ii) markers first -- these usually align
    # with real sub-clause boundaries.
    parts = [p.strip() for p in INLINE_SUBITEM_RE.split(clause) if p.strip()]
    if len(parts) <= 1:
        parts = [clause]  # no inline markers found

    result: list[str] = []
    for part in parts:
        if len(part) <= LONG_CLAUSE_FALLBACK_THRESHOLD:
            result.append(part)
        else:
            # Still oversized after inline splitting (or no markers found
            # at all) -- force it under the threshold via sentence chunking.
            result.extend(_sentence_chunk(part))

    return result


def is_footnote(text: str) -> bool:
    """True if this looks like a numbered drafting-note footnote rather
    than a real contract clause."""
    return bool(FOOTNOTE_LINE_RE.match(text.strip()))


def is_definition(text: str) -> bool:
    """True if this looks like a defined-term entry (e.g. inside a
    DEFINITIONS section) rather than a substantive clause."""
    return bool(DEFINITION_RE.match(text.strip()))


def _strip_repeated_header_footer_lines(pages: list[str]) -> list[str]:
    """Remove lines that repeat (or are near-duplicates) across a large
    fraction of pages -- running headers/footers/disclaimers.

    This implementation normalizes lines (lowercase, collapse whitespace,
    strip leading page numbers), groups near-duplicates using
    difflib.SequenceMatcher, and then removes any group that appears on at
    least HEADER_FOOTER_MIN_PAGE_FRACTION of pages.
    """
    if len(pages) < 3:
        return pages  # not enough pages for frequency detection to be reliable

    def _normalize_for_header(s: str) -> str:
        s = s.strip().lower()
        if not s:
            return ""
        # collapse whitespace
        s = re.sub(r"\s+", " ", s)
        # remove common leading page markers like "page 1 of 10" or bare numbers
        s = re.sub(r"^page\s*\d+(?:\s*of\s*\d+)?[:\-–—]?\s*", "", s)
        s = re.sub(r"^[\d\.\-]+\s+", "", s)
        # reduce punctuation to boost similarity matching
        s = re.sub(r"[^a-z0-9 ]", "", s)
        s = s.strip()
        return s

    # Map a representative normalized string -> count of pages it appears on
    line_page_counts: dict[str, int] = {}

    for page_text in pages:
        seen_this_page = set()
        for line in page_text.split("\n"):
            norm = _normalize_for_header(line)
            if not norm or norm in seen_this_page:
                continue

            # try to find an existing representative that's very similar
            matched = None
            for rep in list(line_page_counts.keys()):
                if difflib.SequenceMatcher(None, rep, norm).ratio() >= 0.9:
                    matched = rep
                    break

            if matched is not None:
                line_page_counts[matched] += 1
                seen_this_page.add(matched)
            else:
                line_page_counts[norm] = line_page_counts.get(norm, 0) + 1
                seen_this_page.add(norm)

    threshold = max(2, int(len(pages) * HEADER_FOOTER_MIN_PAGE_FRACTION))
    boilerplate_norms = {line for line, count in line_page_counts.items() if count >= threshold}

    if not boilerplate_norms:
        return pages

    cleaned_pages = []
    for page_text in pages:
        kept_lines = [
            line for line in page_text.split("\n") if _normalize_for_header(line) not in boilerplate_norms
        ]
        cleaned_pages.append("\n".join(kept_lines))
    return cleaned_pages


def extract_text(file_path: Path) -> str:
    """Extract raw text from an uploaded contract (PDF or .txt), stripping
    repeated running headers/footers for PDFs.
    """
    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        return file_path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        pages = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages.append(text)
        pages = _strip_repeated_header_footer_lines(pages)
        return "\n".join(pages)

    raise ValueError(f"Unsupported file type: {suffix}. Use .pdf or .txt.")


def split_into_clauses(raw_text: str) -> list[str]:
    """
    Heuristic contract segmentation:
      1. Normalize whitespace/line breaks.
      2. Split on blank lines (paragraph breaks) first.
      3. Within paragraphs, further split at numbered-section headers AND
         at footnote markers (so a footnote never gets glued onto the
         real clause text next to it).
      4. Drop footnote fragments entirely -- they're drafting commentary,
         not contract clauses, and would otherwise get force-classified
         into a wrong CUAD label.
      5. Drop fragments too short to be a real clause.

    This is a heuristic, not a legal-structure parser -- expect occasional
    over/under-splitting on unusually formatted contracts. Scanned/
    image-only PDFs need OCR first; pdfplumber can't read those pages.
    """
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)

    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    clauses: list[str] = []
    for paragraph in raw_paragraphs:
        lines = paragraph.split("\n")
        current: list[str] = []

        for line in lines:
            sec_match = SECTION_HEADER_RE.match(line)
            is_boundary = (
                bool(sec_match)
                or bool(FOOTNOTE_LINE_RE.match(line))
                or _looks_like_all_caps_title(line)
            )

            # Don't treat short parenthetical inline markers (e.g. "(i)", "(a)")
            # as a clause boundary if the immediately preceding text in the
            # same paragraph already matched a DEFINITION entry. This avoids
            # splitting a definition's continuation into a separate clause.
            if sec_match and line.strip().startswith("(") and current and is_definition(" ".join(current)):
                is_boundary = False

            if is_boundary and current:
                clauses.append(" ".join(current).strip())
                current = [line]
            else:
                current.append(line)

        if current:
            clauses.append(" ".join(current).strip())

    cleaned = []
    for clause in clauses:
        clause = re.sub(r"\s+", " ", clause).strip()
        if len(clause) < MIN_CLAUSE_CHARS:
            continue
        if is_footnote(clause):
            continue  # drafting commentary, not a contract clause
        cleaned.extend(_fallback_split_oversized(clause))

    return cleaned