# Production Hybrid Segmentation Architecture

## Data flow

`PDF/TXT upload -> structured parser -> normalization -> contract-aware
boundary detection -> clause tree -> validation -> existing CUAD/RAG -> risk
analysis -> API/frontend`

PDF files are parsed by Docling. TXT files use the text parser directly. The
parser returns ordered `DocumentBlock` objects with page ranges and element
kind (`text` or `table`). The clause segmenter consumes these blocks and never
parses PDF bytes or PDF coordinates itself.

## Responsibilities

Docling owns PDF reading order, paragraphs, headings, page provenance, layout
elements, and table extraction. Deterministic logic owns normalization,
repeated header/footer cleanup, numbering recognition, same-line boundaries,
multi-line continuation, nested clause relationships, signature metadata
detection, and validation.

Regex remains only as a small numbering/structure recognizer. It is not the PDF
parser and is not run directly against a PDF file.

## Clause hierarchy

`ClauseNode` is the internal representation:

```json
{
  "clause_id": "5",
  "clause_number": "5",
  "heading": "OBLIGATIONS",
  "text": "5. Obligations ...",
  "page_start": 2,
  "page_end": 3,
  "parser": "docling",
  "children": [
    {"clause_id": "5(a)", "clause_number": "5(a)", "text": "..."}
  ]
}
```

The service converts nodes to their full normalized `text` values before
calling the existing validator, embedder, Qdrant retriever, CUAD classifier,
and risk detector. Therefore downstream APIs continue to receive
`clause_text`, `clause_index`, and the existing classification fields.

## Fallback and errors

`CONTRACT_PARSER=hybrid` is the default. It attempts Docling first and uses a
clearly labeled `pdfplumber-fallback` parser result if Docling is unavailable
or cannot parse the file. The fallback is local, deterministic, and preserves
the application instead of returning an empty document. `CONTRACT_PARSER=docling`
turns Docling failure into a clear application error. `CONTRACT_PARSER=regex`
is retained only as an explicit compatibility/debug mode.

The progress stream reports structure extraction, clause-boundary detection,
validation, embedding, classification, and risk stages without changing the
frontend protocol.

## Validation

Before CUAD/RAG processing, the existing local validator checks whether each
node is a substantive clause. Signature and document metadata nodes are marked
and skipped from model classification. Definitions remain coherent and are
handled by the existing definition shortcut. Page furniture is removed using
repetition evidence, while page breaks are represented by the parser's page
provenance rather than treated as clause boundaries.

## Testing strategy

Unit tests cover same-line numbering, compact numbering, nested markers,
definitions, headings, table blocks, page spans, footnotes, and repeated
headers/footers. The ZTO Transportation Agreement is the required regression
fixture: clauses 1–15 must not collapse into three records, and nested items
must remain associated with their parent where the parsed structure exposes
them.

The environment used for this implementation did not have Python/Pytest or
Docling installed, and the target PDF was not available after the workspace
cleanup. Therefore no claim of an executed ZTO end-to-end result is made here;
the first deployment validation must install the production requirements and
run the fixture through `segment_document`.

## Known limitations

- OCR is disabled by default; image-only PDFs require an explicit OCR policy.
- Docling versions may change item labels or provenance object shapes.
- Page ranges for a clause are currently conservative when a parser block
  spans multiple pages.
- No labeled corpus exists in this repository for numerical boundary accuracy.
- Tables remain associated blocks; row-level legal semantics are not inferred.
