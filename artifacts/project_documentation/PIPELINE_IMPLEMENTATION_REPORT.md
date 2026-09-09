# Pipeline Implementation Report

Audit of the current repository at D:/gradation project/gradation project/application/cuad-rag-app/AI-contract-reviwe.

Audit date: 2026-09-08

This report documents the code that exists in the repository. It does not describe a target architecture. Production code was not modified during this audit.

## 1. Executive Summary

The application is a FastAPI + React application for clause-level contract classification. The runtime path is:

    React upload
      -> POST /documents/classify
      -> temporary local file
      -> pdfplumber or UTF-8 text extraction
      -> regex/paragraph clause segmentation
      -> per-segment local Ollama validation
      -> Cohere query embeddings
      -> Qdrant cuad_train retrieval
      -> local Ollama CUAD classification
      -> category-mapped Cohere legal-guidance retrieval from Qdrant legal_knowledge
      -> documented indicator detection + local Ollama risk reasoning
      -> Pydantic response model
      -> React rendering

Two vector collections are present in the local Qdrant store:

| Collection | Points | Dimension | Distance |
|---|---:|---:|---|
| cuad_train | 7,004 | 1,024 | Cosine |
| legal_knowledge | 30 | 1,024 | Cosine |

The current train CSV contains 10,545 rows from 410 documents, including 3,541 metadata rows and 7,004 substantive rows. load_labels() exposes 36 substantive CUAD labels.

The current legal source directory contains 17 ingestable markdown documents producing 19 chunks. The Qdrant legal collection contains 30 points: all 19 current chunk IDs are present, plus 11 Qdrant point IDs that do not match the current local chunk set. This is a verified storage discrepancy; the report does not assume those 11 points are duplicates or invalid without further history.

The existing test suite was run with Python 3.11 and reported 14 passed, 2 warnings. A bounded local probe found 22 non-empty PDF pages and 80,806 characters extracted directly with pdfplumber from the supplied sample PDF. A segmentation probe over the page-joined text produced 120 segments, but it did not apply the production repeated-header/footer cleanup because the full production extractor exceeded the bounded probe time. Those segmentation numbers are therefore explicitly labelled probe metrics, not exact production-output metrics.

## 2. Full End-to-End Pipeline

| Stage | File / symbol | Input | Output and transformation | Dependencies | Local / external | Error handling |
|---|---|---|---|---|---|---|
| Upload UI | frontend/src/App.jsx / handleFileChange | Browser-selected file | Stores File in React state; clears prior result/error | React | Browser local | No client-side size validation |
| HTTP request | frontend/src/App.jsx / handleSubmit | Selected file | FormData field named file; fetch POST /documents/classify | Fetch API | Local API through Vite proxy | Non-2xx detail shown |
| Route | backend/app/api/routes/documents.py / classify_document | FastAPI UploadFile | Checks suffix, writes named temp file, invokes service | FastAPI, tempfile, shutil | Local server | 400 suffix; 422 ValueError; 503 RuntimeError; finally deletes temp |
| Service | backend/app/services/rag_service.py / classify_contract | Temporary Path | Extracts, segments, validates, embeds, classifies, risk-analyzes | Segmenter, Cohere, Qdrant, Ollama | Mixed | Empty/no validated clauses ValueError; risk exceptions caught per clause |
| Extraction | backend/app/core/rag/segmenter.py / extract_text | PDF/TXT path | PDF pages with pdfplumber; TXT UTF-8 errors ignored | pdfplumber | Local | Unsupported suffix ValueError |
| Segmentation | segmenter.py / split_into_clauses | Extracted text | Paragraph and regex section splitting | Python regex | Local | Heuristic over/under-splitting possible |
| Validation | backend/app/core/rag/validator.py / is_real_clause | One segment | Ollama JSON yes/no; default fail-open | requests, Ollama | Local service | Ollama failure/unparseable output returns true |
| Query embedding | backend/app/core/rag/embedder.py / embed_queries | Validated clause strings | Cohere search_query vectors | Cohere API | External API | Retry for 429; other errors raise |
| CUAD retrieval | backend/app/core/rag/retriever.py / retrieve_similar | Query vector | Top-K cuad_train points excluding metadata | Qdrant | Local in observed runtime | Qdrant errors propagate |
| CUAD classification | backend/app/core/rag/generator.py / classify_clause | Clause, vector, examples, labels | Features + prompt + Ollama JSON; parses one label | Ollama | Local service | Unrecognized result becomes UNKNOWN |
| Legal retrieval | backend/app/core/legal_knowledge/retriever.py / retrieve_legal_guidance | Clause + predicted label | Category mapping, query embedding, category-filtered search | Cohere, Qdrant | Mixed | Empty category gives no guidance; no filtered hits uses unfiltered fallback |
| Risk detection | backend/app/core/risk/risk_detector.py / detect_clause_risk | Clause, CUAD type, guidance | Indicators + guidance + Ollama JSON | Ollama, legal retriever | Mixed | No guidance false result; service catches exceptions |
| Schema | backend/app/models/schemas.py | Clause dictionaries | Pydantic ClauseResult and ContractClassificationResponse | Pydantic | Local | Schema validation errors |
| Response | documents.py | Validated model | JSON response | FastAPI | Local | FastAPI serialization |
| Rendering | frontend/src/App.jsx | JSON response | Summary, labels, statuses, evidence, sources, snippets | React | Browser local | Network/API error shown |

## 3. Frontend Flow

frontend/src/main.jsx mounts App inside React.StrictMode and imports index.css.

frontend/src/App.jsx provides:

- File input advertising .pdf and .txt.
- Submit disabled without a file or while loading.
- Loading and error states.
- Results in API order.
- Clause label from clause_type with predicted_label fallback.
- Potential Risk, No Risk Indicator Detected, or Risk Analysis Not Configured for This Clause.
- Reason, exact evidence, source link, and expandable retrieved legal guidance.
- Expandable CUAD retrieved labels and scores.
- Summary counts for total clauses, boolean risks, configured no-risk clauses, and distinct displayed clause types.

frontend/vite.config.js uses port 5173 and proxies /documents to http://127.0.0.1:8000. frontend/package.json defines npm run build as vite build.

No client-side file-size limit, MIME verification, PDF/TXT content validation, or upload progress exists.

## 4. API Flow

POST /documents/classify accepts multipart/form-data with a required file field. Suffixes .pdf and .txt are accepted after lowercasing the filename suffix.

The route uses tempfile.NamedTemporaryFile(suffix=suffix, delete=False), streams with shutil.copyfileobj, invokes classify_contract, and deletes the temporary path in finally. Uploaded files are not retained by the route.

| Status | Condition |
|---:|---|
| 200 | Classification completes and response validates |
| 400 | Unsupported suffix |
| 422 | No clauses or all fragments rejected |
| 503 | RuntimeError from Cohere, Qdrant, Ollama, or another dependency |

backend/app/main.py mounts the documents router, defines /health, enables CORS for listed localhost Vite ports, and mounts backend/static only if that directory exists. backend/static was not verified as present. The /query route intentionally returns 501 for free-text contract Q&A.

## 5. Document Processing

Supported inputs are PDF and TXT. No application maximum or minimum upload byte size is configured. The segmenter minimum clause size is 40 characters after normalization; it is not a file-size limit.

PDF pages are extracted with pdfplumber page by page and joined with newline. TXT is read as UTF-8 with errors ignored. The upload exists only in the OS temporary directory selected by NamedTemporaryFile and is deleted after the request. Suffix rejection occurs before temporary-file creation.

The sample PDF Cooley SaaS Agreement ACC Form.pdf has 22 pages; all 22 produced non-empty direct pdfplumber text, totaling 80,806 characters before the segmenter cleanup routine.

## 6. Text Extraction

extract_text uses pdfplumber for PDF and Path.read_text for TXT.

For PDFs, _strip_repeated_header_footer_lines runs only for at least three pages. It normalizes lines, strips common page-number prefixes, reduces punctuation, groups near duplicates with difflib.SequenceMatcher at ratio 0.9 or greater, and removes groups present on at least half the pages, with a minimum threshold of two pages.

There is no OCR. Image-only PDFs may result in empty text and a later ValueError.

split_into_clauses normalizes line endings, reduces three-or-more newlines to two, splits blank-line paragraphs, joins lines, and collapses whitespace. Footnote-looking fragments are dropped. Definitions are recognized and later bypass CUAD classification by the service.

## 7. Segmentation

Responsible file: backend/app/core/rag/segmenter.py.

- SECTION_HEADER_RE recognizes numbered headers such as 1., 1.1, (a), Section 3, and Roman-numeral ARTICLE/SECTION forms.
- ALL_CAPS_HEADER_RE and _looks_like_all_caps_title recognize short all-capital headings, maximum six words, no digits, and minimum length six.
- FOOTNOTE_LINE_RE recognizes one or two digits followed by whitespace and a capital letter, without a period after the digits.
- DEFINITION_RE recognizes decimal subsections with a quoted term and means or has the meaning.
- INLINE_SUBITEM_RE recognizes inline letter/roman sub-items.
- SENTENCE_BOUNDARY_RE is used by sentence fallback after period or semicolon boundaries before a capital or opening parenthesis.

The algorithm splits blank-line paragraphs, treats numbered headers, footnote lines, and all-caps titles as boundaries, preserves definition continuations in a special parenthetical case, drops footnote fragments, skips output under MIN_CLAUSE_CHARS = 40, and calls _fallback_split_oversized above LONG_CLAUSE_FALLBACK_THRESHOLD = 3000. Oversized clauses are split on inline sub-items first and sentence boundaries second.

Definitions remain in the segment list. The service emits them as DEFINITION and skips CUAD/risk calls. The segmenter is heuristic, not a legal structure parser.

### Segmentation probe

This safe probe extracted each page with pdfplumber, joined pages, and called split_into_clauses directly. It did not apply production repeated-header/footer cleanup:

| Metric | Observed |
|---|---:|
| Page-joined characters | 80,848 |
| Direct-probe paragraphs | 1 |
| Segments | 120 |
| Average segment characters | 595.7 |
| Minimum segment characters | 51 |
| Maximum segment characters | 2,444 |
| Definition matches | 18 |
| Footnote-like source lines | 20 |
| Numbered header lines | 148 |
| All-caps title lines | 20 |
| Inline subitem markers | 100 |
| Output segments over 3,000 characters | 0 |

Exact production-path segmentation metrics are Not verified because the full extractor probe, including near-duplicate header/footer grouping, exceeded the bounded inspection window and was stopped safely.

## 8. Chunking

No separate chunking layer was found for uploaded contracts; clauses/segments are the CUAD retrieval units.

Legal knowledge uses a separate fixed-character chunker: chunk_text(text, chunk_size=1200, overlap=150). It collapses whitespace, slices by character offset, and overlaps adjacent chunks by 150 characters. It is not token-based or semantic.

Legal payloads contain retrieved_text, source_name, source_url, title, clause_category, document_filename, and deterministic UUID5 chunk_id. risk_indicators.md, source_catalog.md, and clause_coverage.md are excluded from ingestion.

## 9. Validation

backend/app/core/rag/validator.py asks Ollama whether each already-split fragment is a real complete clause. It does not segment.

The validator parses JSON first and then literal true/false. On Ollama failure or ambiguous output it uses fail_open=True and retains the fragment. Rejected fragments are returned as REJECTED_FRAGMENT with risk skipped. If all fragments are rejected, the service raises ValueError.

Definitions are returned as DEFINITION with risk skipped. Empty documents and no-clause conditions become HTTP 422.

## 10. CUAD Classification

CUAD is used for clause classification and retrieval. It is not used as a risk dataset.

data/splits/train/master_clauses_train.csv has 10,545 rows, 410 documents, 41 clause types including metadata, 3,541 metadata rows, and 7,004 substantive rows. config.load_labels filters is_metadata and exposes 36 labels.

data/scripts/dataset/split_cuad.py documents contract-level splitting, wide-to-long reshaping, stringified-list parsing, span explosion, metadata marking, and train/test overlap checks. Runtime classification uses the existing CSV.

prompt.load_label_definitions reads data/metadata/cuad_label_definitions.json, fills missing definitions from DEFAULT_LABEL_DEFINITIONS and writes the repaired file if needed, and rejects malformed/stale/incomplete mappings.

build_prompt includes the clause, all allowed labels and definitions, extracted legal features, and top-K retrieved TRAIN examples. It instructs Ollama to return one JSON object with one allowed clause_type.

classify_clause retrieves examples, extracts deterministic legal features, calls Ollama, and parses a label. parse_prediction accepts exact JSON, a JSON object regex, or a label substring; otherwise it returns UNKNOWN.

## 11. CUAD Retrieval

backend/app/core/rag/retriever.py uses Cohere query vectors against Qdrant collection cuad_train. The observed runtime model is embed-english-v3.0. Query input type is search_query. TOP_K is 5. The Qdrant filter excludes is_metadata == true. Scores are returned as retrieved_scores rounded to four decimals. No reranking was found.

Observed collection:

    name: cuad_train
    points: 7004
    vector dimension: 1024
    distance: Cosine
    payload: answer, clause_text, clause_type, document_id, is_metadata

Offline embed_train.py uses Cohere search_document, batch size 96, deterministic UUID5 IDs, and up to five attempts for HTTP 429. Runtime embedder.py uses configured batch size 96 and configured maximum retry count 8 for 429 responses.

## 12. Legal Knowledge RAG

### Ingestion

backend/app/core/legal_knowledge/ingest.py recursively reads markdown under LEGAL_KNOWLEDGE_PATH, requires source_name, source_url, title, clause_category, and document_filename front matter, chunks at 1,200 characters with 150 overlap, embeds with Cohere search_document, and upserts to legal_knowledge using Cosine vectors.

IDs are UUID5 from path and chunk index. Re-running unchanged files upserts the same IDs. The implementation does not delete points that no longer correspond to local files.

Current local source inventory:

| Source | Documents | Current chunks | Qdrant points | Categories represented |
|---|---:|---:|---:|---|
| World Commerce & Contracting | 9 | 10 | 15 | Assignment, Confidentiality, Indemnification, Insurance, Intellectual Property, Limitation of Liability, Payment, Termination, Warranties |
| World Intellectual Property Organization | 2 | 3 | 5 | Confidentiality, Intellectual Property |
| International Chamber of Commerce | 1 | 1 | 1 | Force Majeure |
| United Nations Commission on International Trade Law | 1 | 1 | 1 | Dispute Resolution |
| American Bar Association | 3 | 3 | 6 | Indemnification, Limitation of Liability, Termination |
| International Bar Association | 1 | 1 | 2 | Indemnification |

Current local IDs are all present in Qdrant. Qdrant also has 11 IDs absent from the current local chunk set. This verified discrepancy means current point count and current source-loader count are not identical.

### Retrieval

map_clause_category maps CUAD labels and text heuristics to legal categories. retrieve_legal_guidance embeds a combined category/terms/clause query using Cohere search_query, applies an exact clause_category filter, and returns up to LEGAL_KNOWLEDGE_TOP_K = 3. If no filtered points are returned, it queries without the filter.

Real local retrieval was successfully probed for Limitation of Liability, Indemnification, Termination, Intellectual Property, Confidentiality, Force Majeure, and Payment Terms. Returned metadata included source name, title, category, URL, score, and retrieved text.

## 13. Risk Detection

backend/app/core/risk/risk_detector.py performs Potential Risk Indicator Detection, not legal judgment.

The exact keys in RISK_INDICATOR_PATTERNS are:

1. Limitation of Liability
2. Indemnification
3. Termination
4. Intellectual Property
5. Confidentiality
6. Payment
7. Force Majeure
8. Dispute Resolution
9. Insurance
10. Warranties
11. Assignment

Flow:

1. Retrieve legal guidance using clause text and predicted CUAD type.
2. If guidance is empty, return false with the configured-category message.
3. Run category-specific regex indicators on the uploaded clause.
4. Prompt Ollama with clause, clause type, indicator labels, and guidance.
5. Parse JSON or a JSON object embedded in the response.
6. Sanitize evidence to contract text when possible.
7. Replace model source metadata unless it exactly matches a retrieved source.
8. Force false if model risk is true but no indicator matched.
9. Normalize generic risk types to documented specific types.
10. Apply clear-cap and clear-IP guards.

The structured result contains risk, risk_type, reason, evidence, source, and legal_knowledge. Risk types such as Potential Uncapped Liability, Unilateral Termination, Broad IP License Scope, Unclear Confidential Information Definition, and Unclear ADR Mechanics are system-derived potential review labels. Sources do not define the application boolean risk label.

The detector uses hard-coded Python indicator patterns corresponding to risk_indicators.md; it does not parse that markdown dynamically at runtime.

## 14. Ollama / LLM

Observed runtime configuration:

- URL: http://127.0.0.1:11434
- Model: llama3.2:3b
- Timeout: 60 seconds
- Retries: 3

generator.call_ollama posts to /api/generate with model, prompt, stream false, format json, temperature zero, num_gpu 0, use_mmap true, and num_ctx 4096. The validator uses the same endpoint and model with num_ctx 512.

The classification parser reads response from payload["response"]. Risk and classification parsers support direct JSON and regex-extracted JSON fallbacks.

A safe reachability probe did not establish a successful connection to port 11434. A real POST /documents/classify probe returned HTTP 503 due to Ollama unavailability. No mock LLM was used.

## 15. Embedding Pipelines

### CUAD

Offline embed_train.py reads substantive train clause text, calls Cohere search_document in batches of 96, and writes train_embeddings.json plus train_metadata.jsonl. build_qdrant.py loads those parallel files and upserts cuad_train.

Runtime embed_queries receives validated upload clauses and calls the configured Cohere model with search_query. Vectors are used for retrieval and are not persisted by the upload route.

### Legal knowledge

Offline ingest_legal_knowledge reads local legal markdown, fixed-character chunks it, calls Cohere search_document, and upserts legal_knowledge.

Runtime retrieve_legal_guidance builds one search_query per clause and uses its vector only for Qdrant retrieval.

Both observed collections use 1,024-dimensional vectors. Cohere is external; Qdrant and Ollama are local in the observed configuration.

## 16. Qdrant Architecture

make_qdrant_client uses QDRANT_URL if configured, otherwise QdrantClient(path=QDRANT_PATH). The observed store is data/qdrant_local. The collections are separate and have different payload schemas:

- cuad_train: CUAD training examples and classification payloads.
- legal_knowledge: source-backed legal guidance chunks and traceability metadata.

Both collections were read successfully and both use 1,024 dimensions and Cosine distance.

## 17. Data & Storage Map

    CUAD train CSV
      -> data/scripts/embedding/embed_train.py
      -> data/embeddings/train_embeddings.json
      -> data/embeddings/train_metadata.jsonl
      -> data/scripts/embedding/build_qdrant.py
      -> data/qdrant_local/collection/cuad_train/storage.sqlite

    Legal markdown
      -> backend/app/core/legal_knowledge/ingest.py
      -> Cohere document vectors
      -> data/qdrant_local/collection/legal_knowledge/storage.sqlite

    Uploaded contract
      -> OS temporary file
      -> extraction and segmentation in memory
      -> Cohere/Qdrant/Ollama processing
      -> FastAPI JSON response
      -> temporary file deleted

The upload service does not write runtime result files. Existing data/metadata results are repository artifacts, not proven outputs of the current request path.

## 18. API Schema

ContractClassificationResponse contains filename, total_clauses, and clauses. Each ClauseResult contains clause_index, clause_text, predicted_label, clause_type, retrieved_labels, retrieved_scores, risk, risk_type, reason, evidence, source, and legal_knowledge.

The source object has name, url, and title. Legal knowledge entries have source_name, source_url, title, clause_category, relevance_score, and retrieved_text. Legacy CUAD retrieval fields remain.

## 19. Frontend Rendering

The frontend renders API fields directly. Source URLs open in a new tab with noreferrer. Legal guidance is inside HTML details and displays snippets, not full documents. CUAD evidence is a separate details element displaying labels and scores.

The summary Supported Categories count is based on distinct displayed CUAD clause_type/predicted_label values whose reason does not contain “not configured”; it is not a direct legal-category count.

## 20. Tests

Inspected test file: backend/tests/test_legal_knowledge.py.

It covers ingestion metadata/chunks/upsert, documentation-only exclusion, category-filtered legal retrieval, guidance mappings, risk JSON parsing, unsupported categories, IP guards and normalization, Pydantic compatibility, and API compatibility with legacy/risk fields.

Observed result:

    14 passed, 2 warnings in 2.23s

Warnings came from the installed FastAPI/Starlette/httpx combination. No dedicated frontend test suite was found. npm run build was Not verified because Node/npm was unavailable on PATH; the existing Vite command could not find node.

## 21. Measured Metrics

| Metric | Value | Verification |
|---|---:|---|
| CUAD train rows | 10,545 | pandas |
| CUAD train documents | 410 | pandas |
| CUAD unique types including metadata | 41 | pandas |
| CUAD metadata rows | 3,541 | pandas |
| CUAD substantive rows/points | 7,004 | pandas/Qdrant |
| Loaded CUAD labels | 36 | load_labels |
| cuad_train dimension | 1,024 | Qdrant |
| legal_knowledge points | 30 | Qdrant |
| Current local legal documents | 17 | legal loader |
| Current local legal chunks | 19 | legal loader |
| Qdrant IDs absent from current local chunks | 11 | ID comparison |
| CUAD top-K | 5 | config |
| Legal top-K | 3 | config |
| Cohere batch size | 96 | config |
| Cohere retry count | 8 | config |
| Ollama timeout | 60 seconds | config |
| Ollama retry count | 3 | config |
| Legal chunk size/overlap | 1,200/150 chars | ingestion defaults |
| Sample PDF pages | 22 | pdfplumber |
| Sample direct characters | 80,806 | pdfplumber |
| Segmentation probe segments | 120 | bounded direct probe |
| Test result | 14 passed | pytest |

## 22. Supported Clause Categories

Runtime CUAD labels: Affiliate License-Licensee, Affiliate License-Licensor, Anti-Assignment, Audit Rights, Cap On Liability, Change Of Control, Competitive Restriction Exception, Covenant Not To Sue, Exclusivity, Governing Law, Insurance, Ip Ownership Assignment, Irrevocable Or Perpetual License, Joint Ip Ownership, License Grant, Liquidated Damages, Minimum Commitment, Most Favored Nation, No-Solicit Of Customers, No-Solicit Of Employees, Non-Compete, Non-Disparagement, Non-Transferable License, Notice Period To Terminate Renewal, Post-Termination Services, Price Restrictions, Renewal Term, Revenue/Profit Sharing, Rofr/Rofo/Rofn, Source Code Escrow, Termination For Convenience, Third Party Beneficiary, Uncapped Liability, Unlimited/All-You-Can-Eat-License, Volume Restriction, Warranty Duration.

Exact risk-pattern categories: Limitation of Liability, Indemnification, Termination, Intellectual Property, Confidentiality, Payment, Force Majeure, Dispute Resolution, Insurance, Warranties, Assignment.

The legal retriever has extra mappings/query terms for Non-Solicitation, Sub-Contracting, Suspension Rights, Termination Assistance, and Order of Precedence. Current local legal points and risk-pattern keys do not establish those as fully supported risk categories.

## 23. Legal Sources

Local source metadata is in backend/data/legal_knowledge/source_catalog.md. The local documents point to:

- World Commerce & Contracting: https://www.worldcc.foundation/how-we-help/resources-tools/contracting-principles.html
- WIPO: https://www.wipo.int/en/web/technology-transfer/agreements
- ICC: https://iccwbo.org/news-publications/icc-rules-guidelines/icc-force-majeure-and-hardship-clauses/
- UNCITRAL: https://uncitral.un.org/en/texts/arbitration/modellaw/commercial_arbitration
- ABA URLs listed in source_catalog.md
- IBA: https://www.ibanet.org/Handbook-for-lawyers/Chapter-2

SOURCE means local guidance text plus source metadata and URL. SYSTEM-DERIVED POTENTIAL RISK INDICATOR means the output generated from the uploaded clause, hard-coded indicator patterns, retrieved guidance, and Ollama response. A source does not define the application risk boolean.

The catalog describes local files as concise source-derived guidance summaries. It does not claim subscription-only or restricted documents are ingested.

## 24. Known Limitations

- No OCR path for scanned PDFs.
- Header/footer cleanup may be expensive on large documents.
- Segmentation is heuristic.
- Validator is Ollama-dependent and fail-open.
- CUAD and risk reasoning depend on Ollama.
- Cohere is an external API dependency.
- Qdrant has 11 point IDs absent from the current source-loader output.
- Legal ingestion upserts but does not remove stale points.
- Risk indicators are hard-coded in Python, not dynamically loaded from risk_indicators.md.
- Unsupported-category UI detection relies on reason text.
- No severity score exists.
- /query returns 501.
- Frontend build was not verified because Node/npm was unavailable.
- README still describes four MVP risk categories while current risk pattern keys contain 11; this report follows code.

## 25. Architecture Diagram

The companion Mermaid source is PIPELINE_ARCHITECTURE.mmd. It shows the online upload path and offline CUAD/legal ingestion paths.

## 26. Implementation Status

| Component | Status | Evidence | Notes |
|---|---|---|---|
| React upload UI | Implemented | frontend/src/App.jsx | No client-side size validation |
| Vite proxy | Implemented | frontend/vite.config.js | Port 5173 to backend 8000 |
| FastAPI upload route | Implemented | documents.py | Suffix check/temp cleanup |
| PDF/TXT extraction | Implemented | segmenter.py | No OCR |
| Heuristic segmentation | Implemented | segmenter.py | Exact production sample metrics Not verified |
| Segment validation | Implemented | validator.py | Ollama fail-open |
| CUAD labels | Implemented | config.py | 36 observed |
| CUAD Qdrant retrieval | Implemented | rag/retriever.py | 7,004 points |
| CUAD Ollama classification | Implemented | generator.py/prompt.py | Ollama unavailable at audit |
| Legal ingestion | Implemented | legal_knowledge/ingest.py | 19 current chunks |
| Legal retrieval | Implemented | legal_knowledge/retriever.py | Real probe succeeded |
| Risk detector | Implemented | risk_detector.py | 11 pattern categories |
| Source traceability | Implemented | risk result/legal payload mapping | Source constrained to retrieved metadata |
| Free-text Q&A | Not Implemented | query.py | Returns 501 |
| Frontend build | Not Verified | Node/npm unavailable | Existing dist is not proof |
| Real Ollama end-to-end success | Not Verified | Port probe failed; API 503 | No mock |
| Legal stale-point cleanup | Not Implemented | ingestion only upserts | 11 extra IDs observed |

## 27. File-by-File Inventory

### Runtime backend

- backend/app/main.py: FastAPI app, CORS, documents router, health, optional static mount.
- backend/app/api/routes/documents.py: upload validation, temp lifecycle, status mapping.
- backend/app/api/routes/health.py: separate health router implementation; main.py defines its own health route.
- backend/app/api/routes/query.py: intentional 501 free-text endpoint.
- backend/app/models/schemas.py: Pydantic response models.
- backend/app/services/rag_service.py: online orchestration.
- backend/app/core/config.py: paths, environment, model, collection, top-K and labels.
- backend/app/core/rag/segmenter.py: extraction and segmentation.
- backend/app/core/rag/validator.py: per-fragment validation.
- backend/app/core/rag/embedder.py: Cohere wrappers.
- backend/app/core/rag/retriever.py: CUAD Qdrant retrieval.
- backend/app/core/rag/prompt.py: label definitions and prompt.
- backend/app/core/rag/generator.py: features, Ollama classification, parsing.
- backend/app/core/legal_knowledge/ingest.py: legal parsing, chunking and upsert.
- backend/app/core/legal_knowledge/retriever.py: category mapping and retrieval.
- backend/app/core/risk/risk_detector.py: indicators, prompt, parsing, guards.

### Offline data

- data/scripts/dataset/split_cuad.py: CUAD split and reshape.
- data/scripts/embedding/embed_train.py: CUAD Cohere document embeddings.
- data/scripts/embedding/build_qdrant.py: CUAD Qdrant build.
- data/splits/train/master_clauses_train.csv: runtime train rows.
- data/embeddings/train_embeddings.json and train_metadata.jsonl: offline embedding artifacts.
- data/qdrant_local/collection/cuad_train/storage.sqlite: CUAD collection.
- data/qdrant_local/collection/legal_knowledge/storage.sqlite: legal collection.
- data/metadata/cuad_label_definitions.json: label definitions.

### Legal data

- backend/data/legal_knowledge/worldcc/*.md: nine local WorldCC documents.
- backend/data/legal_knowledge/wipo/*.md: two local WIPO documents.
- backend/data/legal_knowledge/icc/*.md: one local ICC document.
- backend/data/legal_knowledge/uncitral/*.md: one local UNCITRAL document.
- backend/data/legal_knowledge/aba/*.md: three local ABA documents.
- backend/data/legal_knowledge/iba/*.md: one local IBA document.
- risk_indicators.md: documented potential review indicators.
- source_catalog.md: source metadata and URLs.
- clause_coverage.md: local coverage table.

### Frontend/deployment

- frontend/src/App.jsx: upload/results UI.
- frontend/src/main.jsx: React mount.
- frontend/src/index.css: styles.
- frontend/vite.config.js: Vite proxy.
- frontend/package.json: scripts/dependencies.
- Dockerfile: Node build stage, Python runtime, static copy, Uvicorn command.
- requirements.txt: backend dependencies.
- pyproject.toml: project metadata/dependencies.

## Final Audit Outputs

- Master report: artifacts/project_documentation/PIPELINE_IMPLEMENTATION_REPORT.md
- Architecture diagram: artifacts/project_documentation/PIPELINE_ARCHITECTURE.mmd
- Structured inventory: artifacts/project_documentation/IMPLEMENTATION_INVENTORY.json

Files were created only under artifacts/project_documentation; application code was not modified.
