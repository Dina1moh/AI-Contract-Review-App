# Evaluation Comparison Report

## Run

- Run name: `run_2026-09-22_1658_sample200_topk5`
- Total rows processed: `200`
- Successful predictions: `200`
- Technical failures: `0`

## Dataset

- Evaluated clauses: `200`
- Train documents: `383`
- Test documents: `92`
- Train/Test document overlap: `0`
- Leakage status: `PASS`

## Retrieval-Only Results

- RAG Top-1 Accuracy: `0.6450`
- RAG Recall@3: `0.8200`
- RAG Recall@5: `0.8700`
- MRR: `0.7390`
- RAG Top-1 Correct: `129`

## Qwen + RAG Results

- Model Accuracy: `0.6800`
- Macro Precision: `0.6028`
- Macro Recall: `0.5610`
- Macro F1: `0.5404`
- Model Correct: `136`
- Model Incorrect: `64`

## RAG vs. Model Comparison

| Outcome | Cases |
|---|---:|
| Both Correct | 103 |
| RAG Correct / Model Wrong | 26 |
| RAG Wrong / Model Correct | 33 |
| Both Wrong | 38 |

### Net Effect

- Model improvement cases: `33`
- Net correct gain: `7`
- RAG accuracy: `0.6450`
- Model accuracy: `0.6800`
- Accuracy gain: `0.0350`

## Interpretation

The RAG Top-1 result represents the first retrieved training clause type.

The Qwen prediction represents the final classification after the model receives the retrieved clauses as few-shot evidence.

A model improvement occurs when RAG Top-1 is incorrect but the final Qwen prediction is correct.

## Technical Failures

Technical failures are excluded from model scoring so infrastructure errors are not counted as model mistakes.

## Files Generated

- `test_predictions.csv`
- `retrieval_metrics.json`
- `model_metrics.json`
- `comparison_metrics.json`
- `classification_report.txt`
- `per_class_metrics.csv`
- `error_analysis.csv`
- `confusion_matrix.png`
- `rag_vs_model_accuracy.png`
- `comparison_report.md`
- `evaluation_metrics.csv`
