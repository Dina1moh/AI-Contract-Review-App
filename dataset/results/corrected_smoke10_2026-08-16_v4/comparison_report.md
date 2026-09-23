# Evaluation Comparison Report

## 1. Run

- Run name: `corrected_smoke10_2026-08-16_v4`
- Total processed rows: `10`
- Retrieval-successful rows: `10`
- Model-successful rows: `10`
- Failed model rows: `0`

## 2. Dataset

- Train documents: `383`
- Test documents: `92`
- Train/Test document overlap: `0`
- Leakage status: `PASS`

## 3. Retrieval-Only Results

- RAG Top-1 Accuracy: `0.6000`
- Recall@3: `0.9000`
- Recall@5: `0.9000`
- MRR: `0.7333`
- RAG Top-1 correct: `6`

## 4. Qwen + RAG Results

- Model Accuracy: `0.7000`
- Macro Precision: `0.6000`
- Macro Recall: `0.5500`
- Macro F1: `0.5667`
- Correct: `7`
- Incorrect: `3`

## 5. RAG vs. Model

| Outcome | Cases |
|---|---:|
| Both Correct | 6 |
| RAG Correct / Model Wrong | 0 |
| RAG Wrong / Model Correct | 1 |
| Both Wrong | 3 |

- Model improvement cases: `1`
- Net correct gain: `1`
- Accuracy gain: `0.1000`

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
- `INTERNATIONALFASTFOODCORP_04_04_1997-EX-99-FRANCHISE AGREEMENT.PDF` | `Audit Rights` | RAG=`Audit Rights` | Model=`Audit Rights`

### Both Wrong

- `CHANGEPOINTCORP_03_08_2000-EX-10.6-LICENSE AND HOSTING AGREEMENT.PDF` | `License Grant` | RAG=`Irrevocable Or Perpetual License` | Model=`Non-Transferable License`
- `INTERSECTENT,INC_05_11_2020-EX-10.1-SUPPLY AGREEMENT.PDF` | `Liquidated Damages` | RAG=`Change Of Control` | Model=`Change Of Control`
- `RareElementResourcesLtd_20171019_SC 13D_EX-99.4_10897534_EX-99.4_Intellectual Property Agreement.pdf` | `Exclusivity` | RAG=`Anti-Assignment` | Model=`Non-Transferable License`

## 8. Configuration

- Embedding model: `embed-english-v3.0`
- Qdrant collection: `cuad_train`
- Qdrant path: `I:\gradation project\mvp3\dataset\qdrant_local`
- Ollama model: `qwen2.5:1.5b`
- Top-K: recorded in `run_metadata.json`
- Checkpoint frequency: `10`
- Ollama timeout: `180` seconds
