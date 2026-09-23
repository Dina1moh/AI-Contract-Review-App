"""
Experiment 2 evaluation for the held-out CUAD TEST clauses.

Pipeline:
    TEST clause
        -> Cohere query embedding
        -> Qdrant Top-K retrieval
        -> deterministic legal-information extraction
        -> CUAD label definitions
        -> retrieved TRAIN evidence
        -> enhanced prompt
        -> local Qwen 2.5 1.5B via Ollama
        -> final label prediction
        -> RAG vs model comparison

This runner is intentionally separate from the baseline evaluation script.
"""

from __future__ import annotations

import argparse
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
)
from datetime import datetime
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

try:
    import config
except ModuleNotFoundError:  # pragma: no cover - fallback for package-style imports
    from src import config

try:
    from classify_experiment_02 import (
        build_prompt,
        call_ollama,
        classify_clause,
        embed_queries,
        extract_legal_information,
        load_label_definitions,
        make_cohere_client,
        make_qdrant_client,
        parse_prediction,
        retrieve_similar,
    )
except ModuleNotFoundError:  # pragma: no cover - fallback for package-style imports
    from src.classify_experiment_02 import (
        build_prompt,
        call_ollama,
        classify_clause,
        embed_queries,
        extract_legal_information,
        load_label_definitions,
        make_cohere_client,
        make_qdrant_client,
        parse_prediction,
        retrieve_similar,
    )


SEED = 42
DEFAULT_SAMPLE_SIZE = 200
DEFAULT_TOP_K = 5
CHECKPOINT_EVERY = int(10)
OLLAMA_CALL_TIMEOUT_SECONDS = int(180)
EXPERIMENT_DIR_NAME = f"experiment_02_sample{DEFAULT_SAMPLE_SIZE}_topk{DEFAULT_TOP_K}"

PREDICTION_COLUMNS = [
    "document_id",
    "clause_text",
    "true_clause_type",
    "rag_top1_label",
    "retrieved_clause_types",
    "retrieval_scores",
    "retrieved_examples",
    "extracted_features",
    "true_label_in_top1",
    "true_label_in_top3",
    "true_label_in_top5",
    "retrieval_success",
    "model_prediction",
    "model_success",
    "rag_top1_correct",
    "model_correct",
    "rag_vs_model",
    "model_improved",
    "raw_model_response",
    "error",
]


def create_run_directory(output_dir: Path | None = None) -> Path:
    if output_dir is None:
        output_dir = config.RESULTS_DIR / EXPERIMENT_DIR_NAME

    run_dir = output_dir.resolve()

    if run_dir.exists() and any(run_dir.iterdir()):
        counter = 2
        while True:
            candidate = run_dir.parent / f"{run_dir.name}_v{counter}"
            if not candidate.exists():
                run_dir = candidate
                break
            counter += 1

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_output_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "predictions": run_dir / "test_predictions.csv",
        "retrieval_metrics": run_dir / "retrieval_metrics.json",
        "model_metrics": run_dir / "model_metrics.json",
        "comparison_metrics": run_dir / "comparison_metrics.json",
        "classification_report": run_dir / "classification_report.txt",
        "per_class_metrics": run_dir / "per_class_metrics.csv",
        "error_analysis": run_dir / "error_analysis.csv",
        "comparison_report": run_dir / "comparison_report.md",
        "confusion_matrix": run_dir / "confusion_matrix.png",
        "rag_model_plot": run_dir / "rag_vs_model_accuracy.png",
        "run_metadata": run_dir / "run_metadata.json",
    }


def load_test_rows(sample: int = DEFAULT_SAMPLE_SIZE) -> pd.DataFrame:
    df = pd.read_csv(config.TEST_DATA_PATH)

    if "is_metadata" not in df.columns:
        df["is_metadata"] = False

    df = df[~df["is_metadata"].astype(bool)].reset_index(drop=True)

    if sample <= 0:
        raise ValueError("sample must be greater than 0")

    sample_size = min(sample, len(df))
    df = df.sample(n=sample_size, random_state=SEED).reset_index(drop=True)
    return df


def make_row_key(row: dict) -> str:
    clause_type = row.get("true_clause_type", row.get("clause_type", ""))
    return "||".join(
        [
            str(row["document_id"]),
            str(row["clause_text"]),
            str(clause_type),
        ]
    )


def is_completed(row: dict) -> bool:
    error = str(row.get("error", "") or "").strip()
    model_prediction = str(row.get("model_prediction", "") or "").strip()
    return error == "" and model_prediction not in {"", "ERROR"}


def load_existing_predictions(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}

    existing = pd.read_csv(path)
    if not set(PREDICTION_COLUMNS).issubset(existing.columns):
        print("Existing prediction file has an incompatible schema. Starting without resume.")
        return {}

    completed: dict[str, dict] = {}
    for _, row in existing.iterrows():
        row_dict = row.to_dict()
        if is_completed(row_dict):
            completed[make_row_key(row_dict)] = row_dict
    return completed


def save_predictions(rows_by_key: dict[str, dict], test_df: pd.DataFrame, output_path: Path) -> None:
    ordered_rows = []
    for _, row in test_df.iterrows():
        key = make_row_key(row)
        if key in rows_by_key:
            ordered_rows.append(rows_by_key[key])

    results = pd.DataFrame(ordered_rows, columns=PREDICTION_COLUMNS)
    results.to_csv(output_path, index=False, encoding="utf-8-sig")


def call_ollama_with_timeout(prompt: str) -> tuple[str, str]:
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(call_ollama, prompt)

    try:
        response = future.result(timeout=OLLAMA_CALL_TIMEOUT_SECONDS)
        return response, ""
    except FuturesTimeoutError:
        future.cancel()
        return "", f"Ollama timed out after {OLLAMA_CALL_TIMEOUT_SECONDS} seconds"
    except Exception as exc:
        return "", str(exc)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def compare_outcome(rag_correct: bool, model_correct: bool) -> str:
    if rag_correct and model_correct:
        return "Both Correct"
    if rag_correct and not model_correct:
        return "RAG Correct / Model Wrong"
    if not rag_correct and model_correct:
        return "RAG Wrong / Model Correct"
    return "Both Wrong"


def rank_hit(labels: list[str], true_label: str, rank: int) -> bool:
    return true_label in labels[:rank]


def parse_retrieved_labels(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if not value:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    return []


def compute_retrieval_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "evaluated_clauses": 0,
            "rag_top1_correct": 0,
            "rag_top1_accuracy": 0.0,
            "rag_recall_at_3": 0.0,
            "rag_recall_at_5": 0.0,
            "mrr": 0.0,
        }

    evaluated = len(df)
    top1 = int(df["true_label_in_top1"].sum())
    top3 = int(df["true_label_in_top3"].sum())
    top5 = int(df["true_label_in_top5"].sum())

    reciprocal_ranks = []
    for _, row in df.iterrows():
        retrieved = parse_retrieved_labels(row["retrieved_clause_types"])
        true_label = str(row["true_clause_type"]).strip()
        found_rank = None
        for index, label in enumerate(retrieved, start=1):
            if str(label).strip() == true_label:
                found_rank = index
                break
        reciprocal_ranks.append(1.0 / found_rank if found_rank else 0.0)

    return {
        "evaluated_clauses": evaluated,
        "rag_top1_correct": top1,
        "rag_top1_accuracy": top1 / evaluated,
        "rag_recall_at_3": top3 / evaluated,
        "rag_recall_at_5": top5 / evaluated,
        "mrr": sum(reciprocal_ranks) / evaluated,
    }


def compute_model_metrics(df: pd.DataFrame) -> tuple[dict, list[str], str]:
    if df.empty:
        return (
            {
                "evaluated_clauses": 0,
                "correct": 0,
                "incorrect": 0,
                "accuracy": 0.0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "macro_f1": 0.0,
                "labels": [],
            },
            [],
            "No successful model predictions.",
        )

    y_true = df["true_clause_type"].astype(str)
    y_pred = df["model_prediction"].astype(str)
    labels = sorted(set(y_true.unique()).union(set(y_pred.unique())))

    correct = int((y_true == y_pred).sum())
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    return (
        {
            "evaluated_clauses": len(df),
            "correct": correct,
            "incorrect": len(df) - correct,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(f1),
            "labels": labels,
        },
        labels,
        report,
    )


def compute_comparison(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "evaluated_clauses": 0,
            "rag_correct": 0,
            "model_correct": 0,
            "both_correct": 0,
            "rag_correct_model_wrong": 0,
            "rag_wrong_model_correct": 0,
            "both_wrong": 0,
            "model_improvement_cases": 0,
            "net_correct_gain": 0,
            "rag_accuracy": 0.0,
            "model_accuracy": 0.0,
            "accuracy_gain": 0.0,
        }

    rag = df["rag_top1_correct"].astype(bool)
    model = df["model_correct"].astype(bool)
    total = len(df)

    both_correct = int((rag & model).sum())
    rag_correct_model_wrong = int((rag & ~model).sum())
    rag_wrong_model_correct = int((~rag & model).sum())
    both_wrong = int((~rag & ~model).sum())
    rag_correct = int(rag.sum())
    model_correct = int(model.sum())

    return {
        "evaluated_clauses": total,
        "rag_correct": rag_correct,
        "model_correct": model_correct,
        "both_correct": both_correct,
        "rag_correct_model_wrong": rag_correct_model_wrong,
        "rag_wrong_model_correct": rag_wrong_model_correct,
        "both_wrong": both_wrong,
        "model_improvement_cases": rag_wrong_model_correct,
        "net_correct_gain": model_correct - rag_correct,
        "rag_accuracy": rag_correct / total,
        "model_accuracy": model_correct / total,
        "accuracy_gain": (model_correct / total) - (rag_correct / total),
    }


def save_per_class_metrics(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return

    y_true = df["true_clause_type"].astype(str)
    y_pred = df["model_prediction"].astype(str)
    labels = sorted(set(y_true.unique()).union(set(y_pred.unique())))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    result = pd.DataFrame(
        {
            "clause_type": labels,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support,
        }
    )
    result.to_csv(output_path, index=False, encoding="utf-8-sig")


def save_error_analysis(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        pd.DataFrame().to_csv(output_path, index=False)
        return

    columns = [
        "document_id",
        "clause_text",
        "true_clause_type",
        "rag_top1_label",
        "model_prediction",
        "retrieved_clause_types",
        "retrieval_scores",
        "retrieved_examples",
        "extracted_features",
        "rag_vs_model",
        "model_improved",
        "raw_model_response",
    ]
    available = [col for col in columns if col in df.columns]
    df[available].to_csv(output_path, index=False, encoding="utf-8-sig")


def save_confusion_matrix(df: pd.DataFrame, labels: list[str], output_path: Path) -> None:
    if df.empty:
        return

    y_true = df["true_clause_type"].astype(str)
    y_pred = df["model_prediction"].astype(str)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    n_labels = len(labels)
    fig, ax = plt.subplots(figsize=(max(14, min(0.55 * n_labels + 6, 42)), max(11, min(0.45 * n_labels + 5, 42))))
    image = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    ax.set_xticks(range(n_labels))
    ax.set_yticks(range(n_labels))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted clause type")
    ax.set_ylabel("True clause type")
    ax.set_title("Experiment 2 Clause Classification")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_accuracy_comparison(comparison: dict, output_path: Path) -> None:
    categories = ["RAG Top-1", "Experiment 2"]
    values = [comparison["rag_accuracy"], comparison["model_accuracy"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(categories, values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("RAG vs. Experiment 2 Accuracy")

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2%}", ha="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_comparison_report(
    output_path: Path,
    run_name: str,
    results_df: pd.DataFrame,
    retrieval_df: pd.DataFrame,
    model_df: pd.DataFrame,
    retrieval_metrics: dict,
    model_metrics: dict,
    comparison: dict,
) -> None:
    def examples(df: pd.DataFrame, limit: int = 5) -> str:
        if df.empty:
            return "No examples."
        sample = df.head(limit)
        lines = []
        for _, row in sample.iterrows():
            lines.append(
                f"- {row['document_id']} | true={row['true_clause_type']} | "
                f"rag={row['rag_top1_label']} | pred={row['model_prediction']}"
            )
        return "\n".join(lines)

    report = f"""# Experiment 2 Comparison Report

## Run

- Run name: `{run_name}`
- Processed rows: `{len(results_df)}`
- Retrieval successes: `{len(retrieval_df)}`
- Model successes: `{len(model_df)}`

## Retrieval

- RAG Top-1 Accuracy: `{retrieval_metrics['rag_top1_accuracy']:.4f}`
- Recall@3: `{retrieval_metrics['rag_recall_at_3']:.4f}`
- Recall@5: `{retrieval_metrics['rag_recall_at_5']:.4f}`
- MRR: `{retrieval_metrics['mrr']:.4f}`

## Model

- Accuracy: `{model_metrics['accuracy']:.4f}`
- Macro Precision: `{model_metrics['macro_precision']:.4f}`
- Macro Recall: `{model_metrics['macro_recall']:.4f}`
- Macro F1: `{model_metrics['macro_f1']:.4f}`

## Comparison

- Both Correct: `{comparison['both_correct']}`
- RAG Correct / Model Wrong: `{comparison['rag_correct_model_wrong']}`
- RAG Wrong / Model Correct: `{comparison['rag_wrong_model_correct']}`
- Both Wrong: `{comparison['both_wrong']}`
- Model improvement cases: `{comparison['model_improvement_cases']}`
- Net correct gain: `{comparison['net_correct_gain']}`
- Accuracy gain: `{comparison['accuracy_gain']:.4f}`

## Examples

### RAG Correct / Model Wrong

{examples(model_df[model_df["rag_vs_model"] == "RAG Correct / Model Wrong"])}

### RAG Wrong / Model Correct

{examples(model_df[model_df["rag_vs_model"] == "RAG Wrong / Model Correct"])}

### Both Correct

{examples(model_df[model_df["rag_vs_model"] == "Both Correct"])}

### Both Wrong

{examples(model_df[model_df["rag_vs_model"] == "Both Wrong"])}
"""
    output_path.write_text(report, encoding="utf-8")


def save_run_metadata(path: Path, output_dir: Path, sample: int, top_k: int, labels: list[str]) -> None:
    metadata = {
        "run_name": output_dir.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "sample": sample,
        "evaluation_type": "sample",
        "top_k": top_k,
        "labels_count": len(labels),
        "train_data_path": str(config.TRAIN_DATA_PATH),
        "test_data_path": str(config.TEST_DATA_PATH),
        "label_definitions_path": str(config.LABEL_DEFINITIONS_PATH),
        "embedding_model": config.COHERE_MODEL,
        "qdrant_collection": config.QDRANT_COLLECTION,
        "qdrant_path": config.QDRANT_PATH,
        "ollama_model": config.OLLAMA_MODEL,
        "ollama_url": config.OLLAMA_URL,
        "checkpoint_every": CHECKPOINT_EVERY,
        "ollama_timeout_seconds": OLLAMA_CALL_TIMEOUT_SECONDS,
        "experiment": "experiment_02",
    }
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def run_evaluation(sample: int, top_k: int, output_dir: Path) -> None:
    if sample != DEFAULT_SAMPLE_SIZE:
        raise ValueError(f"Experiment 2 must run with sample={DEFAULT_SAMPLE_SIZE}.")
    if top_k != DEFAULT_TOP_K:
        raise ValueError(f"Experiment 2 must run with top_k={DEFAULT_TOP_K}.")

    paths = get_output_paths(output_dir)

    print("=" * 72)
    print("EXPERIMENT 2: LEGAL INFORMATION EXTRACTION + RAG + QWEN")
    print("=" * 72)
    print(f"Output directory: {output_dir}")
    print(f"Evaluation size: {sample}")
    print(f"Top-K: {top_k}")
    print()

    labels = config.load_labels()
    label_definitions = load_label_definitions(labels)
    save_run_metadata(paths["run_metadata"], output_dir, sample, top_k, labels)

    cohere_client = make_cohere_client()
    qdrant_client = make_qdrant_client()
    test_df = load_test_rows(sample)

    print(f"Loaded {len(labels)} clause-type labels.")
    print(f"TEST clauses selected: {len(test_df)}")

    completed_rows = load_existing_predictions(paths["predictions"])
    if completed_rows:
        print(f"Resuming {len(completed_rows)} successful clauses from this run.")

    pending_texts: list[str] = []
    pending_keys: list[str] = []
    for _, row in test_df.iterrows():
        key = make_row_key(row)
        if key in completed_rows:
            continue
        pending_texts.append(str(row["clause_text"]))
        pending_keys.append(key)

    print(f"Pending clauses: {len(pending_texts)}")

    if pending_texts:
        print("Embedding pending TEST clauses...")
        query_vectors = embed_queries(cohere_client, pending_texts)
        if len(query_vectors) != len(pending_keys):
            raise RuntimeError("Number of Cohere embeddings does not match number of pending clauses.")
        vector_map = dict(zip(pending_keys, query_vectors))
    else:
        vector_map = {}

    newly_processed = 0
    started = time.time()

    for index, row in enumerate(test_df.itertuples(), start=1):
        key = make_row_key(
            {
                "document_id": row.document_id,
                "clause_text": row.clause_text,
                "true_clause_type": row.clause_type,
            }
        )

        if key in completed_rows:
            print(f"[{index}/{len(test_df)}] skip existing")
            continue

        print(f"[{index}/{len(test_df)}] evaluating")

        true_label = str(row.clause_type).strip()
        query_vector = vector_map[key]

        retrieved_examples: list[dict] = []
        retrieved_clause_types: list[str] = []
        retrieval_scores: list[float] = []
        extracted_features: dict[str, list[str]] = extract_legal_information(str(row.clause_text))
        rag_top1_label = ""
        raw_model_response = ""
        retrieval_success = False
        model_success = False
        model_prediction = "ERROR"
        error = ""

        try:
            retrieved_examples = retrieve_similar(
                qdrant_client=qdrant_client,
                query_vector=query_vector,
                top_k=top_k,
            )
            retrieved_clause_types = [item["clause_type"] for item in retrieved_examples]
            retrieval_scores = [round(float(item["score"]), 4) for item in retrieved_examples]
            rag_top1_label = retrieved_clause_types[0] if retrieved_clause_types else ""
            retrieval_success = bool(retrieved_examples)
        except Exception as exc:
            error = str(exc)

        if retrieval_success:
            try:
                prompt = build_prompt(
                    clause_text=str(row.clause_text),
                    examples=retrieved_examples,
                    labels=labels,
                    label_definitions=label_definitions,
                    extracted_features=extracted_features,
                )
                raw_model_response, model_error = call_ollama_with_timeout(prompt)
                if model_error:
                    raise RuntimeError(model_error)
                model_prediction = parse_prediction(raw_model_response, labels)
                model_success = model_prediction not in {"", "UNKNOWN", "ERROR"}
            except Exception as exc:
                error = str(exc)

        rag_top1_correct = retrieval_success and rag_top1_label == true_label
        model_correct = model_success and model_prediction == true_label

        result = {
            "document_id": row.document_id,
            "clause_text": row.clause_text,
            "true_clause_type": true_label,
            "rag_top1_label": rag_top1_label,
            "retrieved_clause_types": json.dumps(retrieved_clause_types, ensure_ascii=False),
            "retrieval_scores": json.dumps(retrieval_scores, ensure_ascii=False),
            "retrieved_examples": json.dumps(retrieved_examples, ensure_ascii=False),
            "extracted_features": json.dumps(extracted_features, ensure_ascii=False),
            "true_label_in_top1": rank_hit(retrieved_clause_types, true_label, 1),
            "true_label_in_top3": rank_hit(retrieved_clause_types, true_label, 3),
            "true_label_in_top5": rank_hit(retrieved_clause_types, true_label, 5),
            "retrieval_success": retrieval_success,
            "model_prediction": model_prediction,
            "model_success": model_success,
            "rag_top1_correct": rag_top1_correct,
            "model_correct": model_correct,
            "rag_vs_model": compare_outcome(rag_top1_correct, model_correct),
            "model_improved": (not rag_top1_correct and model_correct),
            "raw_model_response": raw_model_response,
            "error": error,
        }

        completed_rows[key] = result
        newly_processed += 1

        if newly_processed % CHECKPOINT_EVERY == 0 or index == len(test_df):
            save_predictions(completed_rows, test_df, paths["predictions"])
            elapsed = time.time() - started
            print(f"  checkpoint saved: {newly_processed} new clauses ({elapsed:.0f}s)")

    save_predictions(completed_rows, test_df, paths["predictions"])

    results_df = pd.DataFrame(
        [
            completed_rows[make_row_key(row)]
            for _, row in test_df.iterrows()
            if make_row_key(row) in completed_rows
        ],
        columns=PREDICTION_COLUMNS,
    )

    retrieval_df = results_df[results_df["retrieval_success"].astype(bool)].copy()
    model_df = results_df[
        results_df["model_success"].astype(bool)
        & (results_df["error"].fillna("").astype(str).str.strip() == "")
    ].copy()
    failed_df = results_df[~results_df["model_success"].astype(bool)].copy()

    retrieval_metrics = compute_retrieval_metrics(retrieval_df)
    model_metrics, model_labels, classification_text = compute_model_metrics(model_df)
    comparison = compute_comparison(model_df)

    paths["retrieval_metrics"].write_text(json.dumps(retrieval_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["model_metrics"].write_text(json.dumps(model_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["comparison_metrics"].write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")

    report_text = f"""Experiment 2 Classification Report
=================================

Total processed rows: {len(results_df)}
Successful model predictions: {len(model_df)}
Failed model predictions: {len(failed_df)}

Accuracy: {model_metrics['accuracy']:.6f}
Macro Precision: {model_metrics['macro_precision']:.6f}
Macro Recall: {model_metrics['macro_recall']:.6f}
Macro F1: {model_metrics['macro_f1']:.6f}

Per-Class Report
================
{classification_text}
"""
    paths["classification_report"].write_text(report_text, encoding="utf-8")

    save_per_class_metrics(model_df, paths["per_class_metrics"])
    save_error_analysis(model_df, paths["error_analysis"])
    save_confusion_matrix(model_df, model_labels, paths["confusion_matrix"])
    save_accuracy_comparison(comparison, paths["rag_model_plot"])
    save_comparison_report(
        output_path=paths["comparison_report"],
        run_name=output_dir.name,
        results_df=results_df,
        retrieval_df=retrieval_df,
        model_df=model_df,
        retrieval_metrics=retrieval_metrics,
        model_metrics=model_metrics,
        comparison=comparison,
    )

    print()
    print("=" * 72)
    print("EVALUATION RESULTS")
    print("=" * 72)
    print(f"Total rows processed:     {len(results_df)}")
    print(f"Retrieval successful:     {len(retrieval_df)}")
    print(f"Model successful:         {len(model_df)}")
    print(f"Failed rows:              {len(failed_df)}")
    print(f"RAG Top-1 Accuracy:       {retrieval_metrics['rag_top1_accuracy']:.4f}")
    print(f"RAG Recall@3:             {retrieval_metrics['rag_recall_at_3']:.4f}")
    print(f"RAG Recall@5:             {retrieval_metrics['rag_recall_at_5']:.4f}")
    print(f"MRR:                      {retrieval_metrics['mrr']:.4f}")
    print(f"Model Accuracy:           {model_metrics['accuracy']:.4f}")
    print(f"Model Macro Precision:    {model_metrics['macro_precision']:.4f}")
    print(f"Model Macro Recall:       {model_metrics['macro_recall']:.4f}")
    print(f"Model Macro F1:           {model_metrics['macro_f1']:.4f}")
    print(f"Model improvements:       {comparison['model_improvement_cases']}")
    print(f"Net correct gain:         {comparison['net_correct_gain']}")
    print(f"\nRUN DIRECTORY:\n{output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Experiment 2 sample size. Must remain 200 for the controlled run.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Experiment 2 retrieval depth. Must remain 5 for the controlled run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory root. Defaults to the Experiment 2 directory.",
    )

    args = parser.parse_args()

    if args.sample != DEFAULT_SAMPLE_SIZE:
        raise ValueError(f"--sample must be {DEFAULT_SAMPLE_SIZE} for Experiment 2.")
    if args.top_k != DEFAULT_TOP_K:
        raise ValueError(f"--top-k must be {DEFAULT_TOP_K} for Experiment 2.")

    output_dir = create_run_directory(args.output_dir)
    run_evaluation(sample=args.sample, top_k=args.top_k, output_dir=output_dir)


if __name__ == "__main__":
    main()
