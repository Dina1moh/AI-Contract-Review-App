# Evaluation Comparison Report

## 1. Run

- Run name: `run_2026-09-22_0455_sample500_topk5`
- Total processed rows: `500`
- Retrieval-successful rows: `493`
- Model-successful rows: `491`
- Failed model rows: `9`

## 2. Dataset

- Train documents: `383`
- Test documents: `92`
- Train/Test document overlap: `0`
- Leakage status: `PASS`

## 3. Retrieval-Only Results

- RAG Top-1 Accuracy: `0.6491`
- Recall@3: `0.8377`
- Recall@5: `0.8824`
- MRR: `0.7487`
- RAG Top-1 correct: `320`

## 4. Qwen + RAG Results

- Model Accuracy: `0.6415`
- Macro Precision: `0.5280`
- Macro Recall: `0.5093`
- Macro F1: `0.4889`
- Correct: `315`
- Incorrect: `176`

## 5. RAG vs. Model

| Outcome | Cases |
|---|---:|
| Both Correct | 243 |
| RAG Correct / Model Wrong | 76 |
| RAG Wrong / Model Correct | 72 |
| Both Wrong | 100 |

- Model improvement cases: `72`
- Net correct gain: `-4`
- Accuracy gain: `-0.0081`

## 6. Interpretation

The RAG Top-1 label is the label of the highest-ranked retrieved TRAIN clause.

The final model prediction is the label returned by Qwen after seeing the retrieved clauses as few-shot evidence.

A model improvement occurs when the RAG Top-1 prediction is wrong and the Qwen prediction is correct.

Technical failures are not counted as model predictions.

## 7. Error Examples

### RAG Correct / Model Wrong

- `CHANGEPOINTCORP_03_08_2000-EX-10.6-LICENSE AND HOSTING AGREEMENT.PDF` | `License Grant` | RAG=`License Grant` | Model=`Non-Transferable License`
- `AIRTECHINTERNATIONALGROUPINC_05_08_2000-EX-10.4-FRANCHISE AGREEMENT.PDF` | `Rofr/Rofo/Rofn` | RAG=`Rofr/Rofo/Rofn` | Model=`Change Of Control`
- `TomOnlineInc_20060501_20-F_EX-4.46_749700_EX-4.46_Co-Branding Agreement.pdf` | `Post-Termination Services` | RAG=`Post-Termination Services` | Model=`Non-Transferable License`

### RAG Wrong / Model Correct

- `AIRTECHINTERNATIONALGROUPINC_05_08_2000-EX-10.4-FRANCHISE AGREEMENT.PDF` | `Exclusivity` | RAG=`License Grant` | Model=`Exclusivity`
- `TomOnlineInc_20060501_20-F_EX-4.46_749700_EX-4.46_Co-Branding Agreement.pdf` | `Post-Termination Services` | RAG=`Audit Rights` | Model=`Post-Termination Services`
- `ENTERPRISEPRODUCTSPARTNERSLP_07_08_1998-EX-10.3-TRANSPORTATION CONTRACT.PDF` | `Volume Restriction` | RAG=`Minimum Commitment` | Model=`Volume Restriction`

### Both Correct

- `KINGPHARMACEUTICALSINC_08_09_2006-EX-10.1-PROMOTION AGREEMENT.PDF` | `Audit Rights` | RAG=`Audit Rights` | Model=`Audit Rights`
- `MOSSIMOINC_04_14_2000-EX-10.14-ENDORSEMENT AGREEMENT.PDF` | `Ip Ownership Assignment` | RAG=`Ip Ownership Assignment` | Model=`Ip Ownership Assignment`
- `INTERNATIONALFASTFOODCORP_04_04_1997-EX-99-FRANCHISE AGREEMENT.PDF` | `Audit Rights` | RAG=`Audit Rights` | Model=`Audit Rights`

### Both Wrong

- `INTERSECTENT,INC_05_11_2020-EX-10.1-SUPPLY AGREEMENT.PDF` | `Liquidated Damages` | RAG=`Change Of Control` | Model=`Change Of Control`
- `RareElementResourcesLtd_20171019_SC 13D_EX-99.4_10897534_EX-99.4_Intellectual Property Agreement.pdf` | `Exclusivity` | RAG=`Anti-Assignment` | Model=`Non-Transferable License`
- `MOELIS&CO_03_24_2014-EX-10.19-STRATEGIC ALLIANCE AGREEMENT.PDF` | `Rofr/Rofo/Rofn` | RAG=`Non-Transferable License` | Model=`Anti-Assignment`

## 8. Configuration

- Embedding model: `embed-english-v3.0`
- Qdrant collection: `cuad_train`
- Qdrant path: `D:\peper_AI-legal-contract-review-master\dataset\qdrant_local`
- Ollama model: `qwen2.5:1.5b`
- Top-K: recorded in `run_metadata.json`
- Checkpoint frequency: `10`
- Ollama timeout: `180` seconds
