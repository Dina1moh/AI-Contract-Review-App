# Evaluation Comparison Report

## Run

- Run name: `run_2026-09-22_1613_sample500_topk5`
- Total rows processed: `500`
- Successful predictions: `405`
- Technical failures: `95`

## Dataset

- Evaluated clauses: `405`
- Train documents: `383`
- Test documents: `92`
- Train/Test document overlap: `0`
- Leakage status: `PASS`

## Retrieval-Only Results

- RAG Top-1 Accuracy: `0.6519`
- RAG Recall@3: `0.8395`
- RAG Recall@5: `0.8914`
- MRR: `0.7525`
- RAG Top-1 Correct: `264`

## Qwen + RAG Results

- Model Accuracy: `0.6370`
- Macro Precision: `0.4344`
- Macro Recall: `0.4088`
- Macro F1: `0.3999`
- Model Correct: `258`
- Model Incorrect: `147`

## RAG vs. Model Comparison

| Outcome | Cases |
|---|---:|
| Both Correct | 215 |
| RAG Correct / Model Wrong | 49 |
| RAG Wrong / Model Correct | 43 |
| Both Wrong | 98 |

### Net Effect

- Model improvement cases: `43`
- Net correct gain: `-6`
- RAG accuracy: `0.6519`
- Model accuracy: `0.6370`
- Accuracy gain: `-0.0148`

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
