# Evaluation Comparison Report

## 1. Dataset
- Evaluated clauses: `200`
- Train documents: `383`
- Test documents: `92`
- Train/Test document overlap: `0`

## 2. Retrieval-Only Results
- RAG Top-1 Accuracy: `0.645000`
- RAG Recall@3: `0.820000`
- RAG Recall@5: `0.870000`
- MRR: `0.739000`
- RAG correct count: `129`

## 3. Qwen Model Results
- Model Accuracy: `0.675000`
- Macro Precision: `0.608104`
- Macro Recall: `0.566322`
- Macro F1: `0.551337`

## 4. RAG vs Model Comparison
- Model correct count: `135`
- Cases where model improved over RAG: `33`
- Cases where model changed a correct RAG result into a wrong prediction: `27`
- Cases where both were wrong: `38`
- Cases where both were correct: `102`

## 5. Interpretation
- `model_improved` is `True` only when RAG is wrong and the final model is correct.
- The final model prediction is stored in `model_prediction`.
- The retrieval result is stored separately in `rag_top1_label` and the retrieved clause list.

## 6. Examples

### RAG correct / model wrong
- `CHANGEPOINTCORP_03_08_2000-EX-10.6-LICENSE AND HOSTING AGREEMENT.PDF` | `License Grant` | RAG=`License Grant` | Model=`Non-Transferable License`
- `AURASYSTEMSINC_06_16_2010-EX-10.25-STRATEGIC ALLIANCE AGREEMENT.PDF` | `Minimum Commitment` | RAG=`Minimum Commitment` | Model=`Volume Restriction`

### RAG wrong / model correct
- `TomOnlineInc_20060501_20-F_EX-4.46_749700_EX-4.46_Co-Branding Agreement.pdf` | `Post-Termination Services` | RAG=`Audit Rights` | Model=`Post-Termination Services`
- `ENTERPRISEPRODUCTSPARTNERSLP_07_08_1998-EX-10.3-TRANSPORTATION CONTRACT.PDF` | `Volume Restriction` | RAG=`Minimum Commitment` | Model=`Volume Restriction`

### Both correct
- `KINGPHARMACEUTICALSINC_08_09_2006-EX-10.1-PROMOTION AGREEMENT.PDF` | `Audit Rights` | RAG=`Audit Rights` | Model=`Audit Rights`
- `MOSSIMOINC_04_14_2000-EX-10.14-ENDORSEMENT AGREEMENT.PDF` | `Ip Ownership Assignment` | RAG=`Ip Ownership Assignment` | Model=`Ip Ownership Assignment`

### Both wrong
- `AIRTECHINTERNATIONALGROUPINC_05_08_2000-EX-10.4-FRANCHISE AGREEMENT.PDF` | `Exclusivity` | RAG=`License Grant` | Model=`License Grant`
- `INTERSECTENT,INC_05_11_2020-EX-10.1-SUPPLY AGREEMENT.PDF` | `Liquidated Damages` | RAG=`Change Of Control` | Model=`Change Of Control`

## 7. Train/Test Leakage Check
- Train/test document overlap: `0`
- Result: `PASS`
