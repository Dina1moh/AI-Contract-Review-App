# Evaluation Comparison Report

## 1. Run

- Run name: `run_2026-09-22_1654_sample5_topk5`
- Total processed rows: `5`
- Retrieval-successful rows: `5`
- Model-successful rows: `5`
- Failed model rows: `0`

## 2. Dataset

- Train documents: `383`
- Test documents: `92`
- Train/Test document overlap: `0`
- Leakage status: `PASS`

## 3. Retrieval-Only Results

- RAG Top-1 Accuracy: `0.8000`
- Recall@3: `1.0000`
- Recall@5: `1.0000`
- MRR: `0.9000`
- RAG Top-1 correct: `4`

## 4. Qwen + RAG Results

- Model Accuracy: `1.0000`
- Macro Precision: `1.0000`
- Macro Recall: `1.0000`
- Macro F1: `1.0000`
- Correct: `5`
- Incorrect: `0`

## 5. RAG vs. Model

| Outcome | Cases |
|---|---:|
| Both Correct | 4 |
| RAG Correct / Model Wrong | 0 |
| RAG Wrong / Model Correct | 1 |
| Both Wrong | 0 |

- Model improvement cases: `1`
- Net correct gain: `1`
- Accuracy gain: `0.2000`

## 6. Interpretation

The RAG Top-1 label is the label of the highest-ranked retrieved TRAIN clause.

The final model prediction is the label returned by Qwen after seeing the retrieved clauses as few-shot evidence.

A model improvement occurs when the RAG Top-1 prediction is wrong and the Qwen prediction is correct.

Technical failures are not counted as model predictions.

## 7. Error Examples

### RAG Correct / Model Wrong

- None

### RAG Wrong / Model Correct

- `AIRTECHINTERNATIONALGROUPINC_05_08_2000-EX-10.4-FRANCHISE AGREEMENT.PDF` | `Exclusivity` | RAG=`License Grant` | Model=`Exclusivity`

### Both Correct

- `KINGPHARMACEUTICALSINC_08_09_2006-EX-10.1-PROMOTION AGREEMENT.PDF` | `Audit Rights` | RAG=`Audit Rights` | Model=`Audit Rights`
- `MOSSIMOINC_04_14_2000-EX-10.14-ENDORSEMENT AGREEMENT.PDF` | `Ip Ownership Assignment` | RAG=`Ip Ownership Assignment` | Model=`Ip Ownership Assignment`
- `CHANGEPOINTCORP_03_08_2000-EX-10.6-LICENSE AND HOSTING AGREEMENT.PDF` | `License Grant` | RAG=`License Grant` | Model=`License Grant`

### Both Wrong

- None

## 8. Configuration

- Embedding model: `embed-english-v3.0`
- Qdrant collection: `cuad_train`
- Qdrant path: `D:\peper_AI-legal-contract-review-master\dataset\qdrant_local`
- Ollama model: `qwen2.5:1.5b`
- Top-K: recorded in `run_metadata.json`
- Checkpoint frequency: `10`
- Ollama timeout: `180` seconds
