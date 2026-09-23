"""
RAG + Qwen evaluation for the held-out CUAD TEST clauses.

Pipeline:
    TEST clause
        -> Cohere query embedding
        -> Qdrant Top-K retrieval
        -> RAG Top-1 baseline
        -> Few-shot prompt
        -> Local Qwen 2.5 1.5B
        -> Final model prediction
        -> RAG vs Model comparison

Each new run gets its own timestamped directory.

Resume behavior:
- A NEW execution creates a NEW run directory automatically.
- To resume an existing run, provide the same --output-dir.
- Failed rows are retried when the same run is resumed.
- Successful rows are skipped.

Example:
    python -B src\\Run_evaluation.py --sample 200

Resume:
    python -B src\\Run_evaluation.py --sample 200 ^
        --output-dir "dataset\\results\\run_2026-08-16_2300_sample200_topk5"
"""

from __future__ import annotations

import argparse
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
)
from datetime import datetime
import json
import os
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

import config
from classify import (
    build_prompt,
    call_ollama,
    embed_queries,
    make_cohere_client,
    make_qdrant_client,
    parse_prediction,
    retrieve_similar,
)


# ============================================================
# Configuration
# ============================================================

CHECKPOINT_EVERY = int(
    os.getenv("EVAL_CHECKPOINT_EVERY", "10")
)

OLLAMA_CALL_TIMEOUT_SECONDS = int(
    os.getenv("EVAL_OLLAMA_CALL_TIMEOUT_SECONDS", "180")
)


# ============================================================
# Output columns
# ============================================================

PREDICTION_COLUMNS = [
    "document_id",
    "clause_text",
    "true_clause_type",
    "rag_top1_label",
    "retrieved_clause_types",
    "retrieval_scores",
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


# ============================================================
# Run directory
# ============================================================

def create_run_directory(
    sample: int | None,
    top_k: int,
) -> Path:
    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H%M"
    )

    size_tag = (
        f"sample{sample}"
        if sample
        else "full"
    )

    base_name = (
        f"run_{timestamp}_{size_tag}_topk{top_k}"
    )

    run_dir = (
        config.RESULTS_DIR / base_name
    )

    counter = 2

    while run_dir.exists():
        run_dir = (
            config.RESULTS_DIR
            / f"{base_name}_v{counter}"
        )
        counter += 1

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    return run_dir


def get_output_paths(
    run_dir: Path,
) -> dict[str, Path]:
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
        "evaluation_metrics": run_dir / "evaluation_metrics.csv",
    }


# ============================================================
# Dataset
# ============================================================

def load_test_rows(
    sample: int | None,
    stratify: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(
        config.TEST_DATA_PATH
    )

    if "is_metadata" not in df.columns:
        df["is_metadata"] = False

    df = df[
        ~df["is_metadata"].astype(bool)
    ].reset_index(drop=True)

    if sample is not None:
        if sample <= 0:
            raise ValueError(
                "--sample must be greater than 0."
            )

        sample_size = min(
            sample,
            len(df),
        )

        if stratify:
            df = df.groupby("clause_type", group_keys=False).sample(
                frac=sample_size / len(df),
                random_state=42,
            )
            if len(df) < sample_size:
                extra = df.sample(n=sample_size - len(df), random_state=42, replace=True)
                df = pd.concat([df, extra], ignore_index=True)
            elif len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
        else:
            df = df.sample(n=sample_size, random_state=42)
            
        df = df.reset_index(drop=True)

    return df


# ============================================================
# Resume
# ============================================================

def make_row_key(row) -> str:
    clause_type = row.get(
        "true_clause_type",
        row.get("clause_type", ""),
    )

    return "||".join(
        [
            str(row["document_id"]),
            str(row["clause_text"]),
            str(clause_type),
        ]
    )


def is_completed(row: dict) -> bool:
    """
    Successful model predictions are resumable.
    ERROR/failed rows are intentionally retried.
    """
    error = str(
        row.get("error", "") or ""
    ).strip()

    model_prediction = str(
        row.get(
            "model_prediction",
            "",
        )
        or ""
    ).strip()

    return (
        error == ""
        and model_prediction
        not in {"", "ERROR"}
    )


def load_existing_predictions(
    path: Path,
) -> dict[str, dict]:
    if not path.exists():
        return {}

    existing = pd.read_csv(path)

    if not set(PREDICTION_COLUMNS).issubset(
        existing.columns
    ):
        print(
            "Existing prediction file has an incompatible "
            "schema. Starting without resume."
        )
        return {}

    completed = {}

    for _, row in existing.iterrows():
        row_dict = row.to_dict()
        key = make_row_key(row_dict)

        if is_completed(row_dict):
            completed[key] = row_dict

    return completed


def save_predictions(
    rows_by_key: dict[str, dict],
    test_df: pd.DataFrame,
    output_path: Path,
) -> None:
    ordered_rows = []

    for _, row in test_df.iterrows():
        key = make_row_key(row)

        if key in rows_by_key:
            ordered_rows.append(
                rows_by_key[key]
            )

    results = pd.DataFrame(
        ordered_rows,
        columns=PREDICTION_COLUMNS,
    )

    results.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# Ollama timeout
# ============================================================

def call_ollama_with_timeout(
    prompt: str,
) -> tuple[str, str]:
    executor = ThreadPoolExecutor(
        max_workers=1
    )

    future = executor.submit(
        call_ollama,
        prompt,
    )

    try:
        response = future.result(
            timeout=OLLAMA_CALL_TIMEOUT_SECONDS
        )

        return response, ""

    except FuturesTimeoutError:
        future.cancel()

        return (
            "",
            (
                "Ollama timed out after "
                f"{OLLAMA_CALL_TIMEOUT_SECONDS} seconds"
            ),
        )

    except Exception as exc:
        return "", str(exc)

    finally:
        executor.shutdown(
            wait=False,
            cancel_futures=True,
        )


# ============================================================
# Comparison
# ============================================================

def compare_outcome(
    rag_correct: bool,
    model_correct: bool,
) -> str:
    if rag_correct and model_correct:
        return "Both Correct"

    if rag_correct and not model_correct:
        return "RAG Correct / Model Wrong"

    if not rag_correct and model_correct:
        return "RAG Wrong / Model Correct"

    return "Both Wrong"


def rank_hit(
    labels: list[str],
    true_label: str,
    rank: int,
) -> bool:
    return true_label in labels[:rank]


def parse_retrieved_labels(
    value: object,
) -> list[str]:
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


# ============================================================
# Metrics
# ============================================================


def compute_classification_metrics(y_true: pd.Series, y_pred: pd.Series, method_name: str) -> dict:
    y_true = y_true.fillna("").astype(str).str.strip()
    y_pred = y_pred.fillna("").astype(str).str.strip()
    
    evaluated = len(y_true)
    if evaluated == 0:
        return {
            "Method": method_name,
            "Accuracy": 0.0,
            "Macro Precision": 0.0,
            "Macro Recall": 0.0,
            "Macro F1": 0.0,
            "Weighted Precision": 0.0,
            "Weighted Recall": 0.0,
            "Weighted F1": 0.0,
            "Evaluated Samples": 0,
            "Correct Predictions": 0,
        }

    labels = sorted(set(y_true.unique()).union(set(y_pred.unique())))
    correct = int((y_true == y_pred).sum())
    acc = float(accuracy_score(y_true, y_pred))

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    
    return {
        "Method": method_name,
        "Accuracy": acc,
        "Macro Precision": float(macro_p),
        "Macro Recall": float(macro_r),
        "Macro F1": float(macro_f1),
        "Weighted Precision": float(weighted_p),
        "Weighted Recall": float(weighted_r),
        "Weighted F1": float(weighted_f1),
        "Evaluated Samples": evaluated,
        "Correct Predictions": correct,
    }

def generate_evaluation_metrics_df(df: pd.DataFrame) -> pd.DataFrame:
    if "true_clause_type" not in df.columns:
        return pd.DataFrame()
    y_true = df["true_clause_type"]
    metrics_list = []
    
    methods = [
        ("RAG Top-1", "rag_top1_label"),
        ("Qwen Model", "model_prediction")
    ]
    
    for col in df.columns:
        if col.endswith("_prediction") and col not in ["model_prediction"]:
            methods.append((col.replace("_prediction", "").title() + " Prediction", col))
        elif col.endswith("_label") and col not in ["rag_top1_label", "true_clause_type"]:
            methods.append((col.replace("_label", "").title() + " Label", col))
            
    for method_name, col_name in methods:
        if col_name in df.columns:
            metrics_list.append(compute_classification_metrics(y_true, df[col_name], method_name))
            
    return pd.DataFrame(metrics_list)


def compute_retrieval_metrics(
    df: pd.DataFrame,
) -> dict:
    """
    Retrieval metrics use rows where retrieval succeeded.
    A later Ollama failure does not invalidate successful retrieval.
    """
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

    top1 = int(
        df["true_label_in_top1"].sum()
    )

    top3 = int(
        df["true_label_in_top3"].sum()
    )

    top5 = int(
        df["true_label_in_top5"].sum()
    )

    reciprocal_ranks = []

    for _, row in df.iterrows():
        retrieved = parse_retrieved_labels(
            row[
                "retrieved_clause_types"
            ]
        )

        true_label = str(
            row["true_clause_type"]
        ).strip()

        found_rank = None

        for index, label in enumerate(
            retrieved,
            start=1,
        ):
            if str(label).strip() == true_label:
                found_rank = index
                break

        reciprocal_ranks.append(
            1.0 / found_rank
            if found_rank
            else 0.0
        )

    mrr = (
        sum(reciprocal_ranks)
        / evaluated
    )

    return {
        "evaluated_clauses": evaluated,
        "rag_top1_correct": top1,
        "rag_top1_accuracy": top1 / evaluated,
        "rag_recall_at_3": top3 / evaluated,
        "rag_recall_at_5": top5 / evaluated,
        "mrr": mrr,
    }


def compute_model_metrics(
    df: pd.DataFrame,
) -> tuple[dict, list[str], str]:
    """
    ONLY successful model predictions are scored.
    """
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

    y_true = df[
        "true_clause_type"
    ].astype(str)

    y_pred = df[
        "model_prediction"
    ].astype(str)

    labels = sorted(
        set(y_true.unique()).union(
            set(y_pred.unique())
        )
    )

    correct = int(
        (y_true == y_pred).sum()
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
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
            "accuracy": float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),
            "macro_precision": float(
                precision
            ),
            "macro_recall": float(
                recall
            ),
            "macro_f1": float(
                f1
            ),
            "labels": labels,
        },
        labels,
        report,
    )


def compute_comparison(
    df: pd.DataFrame,
) -> dict:
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

    rag = df[
        "rag_top1_correct"
    ].astype(bool)

    model = df[
        "model_correct"
    ].astype(bool)

    both_correct = int(
        (rag & model).sum()
    )

    rag_correct_model_wrong = int(
        (rag & ~model).sum()
    )

    rag_wrong_model_correct = int(
        (~rag & model).sum()
    )

    both_wrong = int(
        (~rag & ~model).sum()
    )

    rag_correct = int(
        rag.sum()
    )

    model_correct = int(
        model.sum()
    )

    total = len(df)

    return {
        "evaluated_clauses": total,
        "rag_correct": rag_correct,
        "model_correct": model_correct,
        "both_correct": both_correct,
        "rag_correct_model_wrong": rag_correct_model_wrong,
        "rag_wrong_model_correct": rag_wrong_model_correct,
        "both_wrong": both_wrong,
        "model_improvement_cases":
            rag_wrong_model_correct,
        "net_correct_gain":
            model_correct - rag_correct,
        "rag_accuracy":
            rag_correct / total,
        "model_accuracy":
            model_correct / total,
        "accuracy_gain":
            (model_correct / total)
            - (rag_correct / total),
    }


# ============================================================
# Per-class / error analysis
# ============================================================

def save_per_class_metrics(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    if df.empty:
        return

    y_true = df[
        "true_clause_type"
    ].astype(str)

    y_pred = df[
        "model_prediction"
    ].astype(str)

    labels = sorted(
        set(y_true.unique()).union(
            set(y_pred.unique())
        )
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )
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

    result.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )


def save_error_analysis(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    if df.empty:
        pd.DataFrame().to_csv(
            output_path,
            index=False,
        )
        return

    columns = [
        "document_id",
        "clause_text",
        "true_clause_type",
        "rag_top1_label",
        "model_prediction",
        "retrieved_clause_types",
        "retrieval_scores",
        "rag_vs_model",
        "model_improved",
        "raw_model_response",
    ]

    available = [
        col
        for col in columns
        if col in df.columns
    ]

    df[available].to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# Figures
# ============================================================

def save_confusion_matrix(
    df: pd.DataFrame,
    labels: list[str],
    output_path: Path,
) -> None:
    if df.empty:
        return

    y_true = df[
        "true_clause_type"
    ].astype(str)

    y_pred = df[
        "model_prediction"
    ].astype(str)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    n_labels = len(labels)

    fig, ax = plt.subplots(
        figsize=(
            max(14, min(0.55 * n_labels + 6, 42)),
            max(11, min(0.45 * n_labels + 5, 42)),
        )
    )

    image = ax.imshow(
        matrix,
        cmap="Blues",
        interpolation="nearest",
    )

    ax.set_xticks(
        range(n_labels)
    )

    ax.set_yticks(
        range(n_labels)
    )

    ax.set_xticklabels(
        labels,
        rotation=90,
        fontsize=7,
    )

    ax.set_yticklabels(
        labels,
        fontsize=7,
    )

    ax.set_xlabel(
        "Predicted clause type"
    )

    ax.set_ylabel(
        "True clause type"
    )

    ax.set_title(
        "Qwen + RAG Clause Classification"
    )

    fig.colorbar(
        image,
        ax=ax,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_accuracy_comparison(
    comparison: dict,
    output_path: Path,
) -> None:
    categories = [
        "RAG Top-1",
        "Qwen + RAG",
    ]

    values = [
        comparison["rag_accuracy"],
        comparison["model_accuracy"],
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    bars = ax.bar(
        categories,
        values,
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.set_ylabel(
        "Accuracy"
    )

    ax.set_title(
        "RAG vs. Final Model Accuracy"
    )

    for bar, value in zip(
        bars,
        values,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.02,
            f"{value:.2%}",
            ha="center",
        )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Leakage
# ============================================================

def check_leakage() -> dict[str, int]:
    def load_document_ids(path: Path) -> set[str]:
        # Only the columns needed for overlap detection are loaded here to
        # avoid pulling the entire CSV into memory.
        df = pd.read_csv(
            path,
            usecols=lambda col: col in {
                "document_id",
                "is_metadata",
            },
            low_memory=False,
        )

        if "is_metadata" not in df.columns:
            df["is_metadata"] = False

        return set(
            df.loc[
                ~df["is_metadata"].astype(bool),
                "document_id",
            ].astype(str)
        )

    train_docs = load_document_ids(
        config.TRAIN_DATA_PATH
    )

    test_docs = load_document_ids(
        config.TEST_DATA_PATH
    )

    overlap = train_docs & test_docs

    return {
        "train_documents":
            len(train_docs),
        "test_documents":
            len(test_docs),
        "overlap_documents":
            len(overlap),
    }


# ============================================================
# Markdown report
# ============================================================

def save_comparison_report(
    output_path: Path,
    run_name: str,
    results_df: pd.DataFrame,
    retrieval_df: pd.DataFrame,
    model_df: pd.DataFrame,
    retrieval_metrics: dict,
    model_metrics: dict,
    comparison: dict,
    leakage: dict,
) -> None:
    def examples(
        subset: pd.DataFrame,
        limit: int = 3,
    ) -> str:
        if subset.empty:
            return "- None"

        lines = []

        for _, row in subset.head(
            limit
        ).iterrows():
            lines.append(
                f"- `{row['document_id']}` | "
                f"`{row['true_clause_type']}` | "
                f"RAG=`{row['rag_top1_label']}` | "
                f"Model=`{row['model_prediction']}`"
            )

        return "\n".join(lines)

    report = f"""# Evaluation Comparison Report

## 1. Run

- Run name: `{run_name}`
- Total processed rows: `{len(results_df)}`
- Retrieval-successful rows: `{len(retrieval_df)}`
- Model-successful rows: `{len(model_df)}`
- Failed model rows: `{len(results_df) - len(model_df)}`

## 2. Dataset

- Train documents: `{leakage['train_documents']}`
- Test documents: `{leakage['test_documents']}`
- Train/Test document overlap: `{leakage['overlap_documents']}`
- Leakage status: `{'PASS' if leakage['overlap_documents'] == 0 else 'FAIL'}`

## 3. Retrieval-Only Results

- RAG Top-1 Accuracy: `{retrieval_metrics['rag_top1_accuracy']:.4f}`
- Recall@3: `{retrieval_metrics['rag_recall_at_3']:.4f}`
- Recall@5: `{retrieval_metrics['rag_recall_at_5']:.4f}`
- MRR: `{retrieval_metrics['mrr']:.4f}`
- RAG Top-1 correct: `{retrieval_metrics['rag_top1_correct']}`

## 4. Qwen + RAG Results

- Model Accuracy: `{model_metrics['accuracy']:.4f}`
- Macro Precision: `{model_metrics['macro_precision']:.4f}`
- Macro Recall: `{model_metrics['macro_recall']:.4f}`
- Macro F1: `{model_metrics['macro_f1']:.4f}`
- Correct: `{model_metrics['correct']}`
- Incorrect: `{model_metrics['incorrect']}`

## 5. RAG vs. Model

| Outcome | Cases |
|---|---:|
| Both Correct | {comparison['both_correct']} |
| RAG Correct / Model Wrong | {comparison['rag_correct_model_wrong']} |
| RAG Wrong / Model Correct | {comparison['rag_wrong_model_correct']} |
| Both Wrong | {comparison['both_wrong']} |

- Model improvement cases: `{comparison['model_improvement_cases']}`
- Net correct gain: `{comparison['net_correct_gain']}`
- Accuracy gain: `{comparison['accuracy_gain']:.4f}`

## 6. Interpretation

The RAG Top-1 label is the label of the highest-ranked retrieved TRAIN clause.

The final model prediction is the label returned by Qwen after seeing the retrieved clauses as few-shot evidence.

A model improvement occurs when the RAG Top-1 prediction is wrong and the Qwen prediction is correct.

Technical failures are not counted as model predictions.

## 7. Error Examples

### RAG Correct / Model Wrong

{examples(model_df[
    model_df["rag_vs_model"]
    == "RAG Correct / Model Wrong"
])}

### RAG Wrong / Model Correct

{examples(model_df[
    model_df["rag_vs_model"]
    == "RAG Wrong / Model Correct"
])}

### Both Correct

{examples(model_df[
    model_df["rag_vs_model"]
    == "Both Correct"
])}

### Both Wrong

{examples(model_df[
    model_df["rag_vs_model"]
    == "Both Wrong"
])}

## 8. Configuration

- Embedding model: `{config.COHERE_MODEL}`
- Qdrant collection: `{config.QDRANT_COLLECTION}`
- Qdrant path: `{config.QDRANT_PATH}`
- Ollama model: `{config.OLLAMA_MODEL}`
- Top-K: recorded in `run_metadata.json`
- Checkpoint frequency: `{CHECKPOINT_EVERY}`
- Ollama timeout: `{OLLAMA_CALL_TIMEOUT_SECONDS}` seconds
"""

    output_path.write_text(
        report,
        encoding="utf-8",
    )


# ============================================================
# Run metadata
# ============================================================

def save_run_metadata(
    path: Path,
    output_dir: Path,
    sample: int | None,
    top_k: int,
) -> None:
    metadata = {
        "run_name": output_dir.name,
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "sample": sample,
        "evaluation_type":
            "sample" if sample else "full",
        "top_k": top_k,
        "train_data_path":
            str(config.TRAIN_DATA_PATH),
        "test_data_path":
            str(config.TEST_DATA_PATH),
        "embedding_model":
            config.COHERE_MODEL,
        "qdrant_collection":
            config.QDRANT_COLLECTION,
        "qdrant_path":
            config.QDRANT_PATH,
        "ollama_model":
            config.OLLAMA_MODEL,
        "ollama_url":
            config.OLLAMA_URL,
        "checkpoint_every":
            CHECKPOINT_EVERY,
        "ollama_timeout_seconds":
            OLLAMA_CALL_TIMEOUT_SECONDS,
    }

    path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ============================================================
# Evaluation
# ============================================================

def run_evaluation(
    sample: int | None,
    top_k: int,
    output_dir: Path,
    stratify: bool = False,
) -> None:
    paths = get_output_paths(
        output_dir
    )

    print("=" * 72)
    print("RAG + QWEN EVALUATION")
    print("=" * 72)
    print(
        f"Output directory: {output_dir}"
    )
    print(
        f"Evaluation size: "
        f"{sample if sample else 'FULL'}"
    )
    print(
        f"Top-K: {top_k}"
    )
    print(
        f"Stratified: {stratify}"
    )
    print()

    save_run_metadata(
        paths["run_metadata"],
        output_dir,
        sample,
        top_k,
    )

    # Create the Qdrant client before loading the CSVs so the local
    # embedded collection has the best chance to load without a transient
    # MemoryError on constrained machines.
    cohere_client = make_cohere_client()
    qdrant_client = make_qdrant_client()

    labels = config.load_labels()

    print(
        f"Loaded {len(labels)} clause-type labels."
    )

    test_df = load_test_rows(
        sample,
        stratify,
    )

    print(
        f"TEST clauses selected: "
        f"{len(test_df)}"
    )

    # Successful rows only are resumed.
    completed_rows = (
        load_existing_predictions(
            paths["predictions"]
        )
    )

    if completed_rows:
        print(
            f"Resuming {len(completed_rows)} "
            f"successful clauses from this run."
        )

    pending_texts = []
    pending_keys = []

    for _, row in test_df.iterrows():
        key = make_row_key(row)

        if key in completed_rows:
            continue

        pending_texts.append(
            row["clause_text"]
        )
        pending_keys.append(key)

    print(
        f"Pending clauses: "
        f"{len(pending_texts)}"
    )

    # --------------------------------------------------------
    # Embed only pending clauses
    # --------------------------------------------------------

    if pending_texts:
        print(
            "Embedding pending TEST clauses..."
        )

        query_vectors = embed_queries(
            cohere_client,
            pending_texts,
        )

        if len(query_vectors) != len(
            pending_keys
        ):
            raise RuntimeError(
                "Number of Cohere embeddings does "
                "not match number of pending clauses."
            )

        vector_map = dict(
            zip(
                pending_keys,
                query_vectors,
            )
        )
    else:
        vector_map = {}

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    newly_processed = 0
    started = time.time()

    for index, row in enumerate(
        test_df.itertuples(),
        start=1,
    ):
        key = make_row_key(
            {
                "document_id":
                    row.document_id,
                "clause_text":
                    row.clause_text,
                "true_clause_type":
                    row.clause_type,
            }
        )

        if key in completed_rows:
            print(
                f"[{index}/{len(test_df)}] "
                "skip existing"
            )
            continue

        print(
            f"[{index}/{len(test_df)}] "
            "evaluating"
        )

        retrieved_clause_types = []
        retrieval_scores = []

        rag_top1_label = ""
        raw_model_response = ""

        retrieval_success = False
        model_success = False

        model_prediction = "ERROR"
        error = ""

        query_vector = vector_map[key]

        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------

        try:
            retrieved = retrieve_similar(
                qdrant_client=qdrant_client,
                query_vector=query_vector,
                top_k=top_k,
            )

            retrieved_clause_types = [
                item["clause_type"]
                for item in retrieved
            ]

            retrieval_scores = [
                round(
                    float(item["score"]),
                    4,
                )
                for item in retrieved
            ]

            rag_top1_label = (
                retrieved_clause_types[0]
                if retrieved_clause_types
                else ""
            )

            retrieval_success = bool(
                retrieved
            )

        except Exception as exc:
            error = str(exc)
            retrieved = []

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        if retrieval_success:
            try:
                prompt = build_prompt(
                    row.clause_text,
                    retrieved,
                    labels,
                )

                (
                    raw_model_response,
                    model_error,
                ) = call_ollama_with_timeout(
                    prompt
                )

                if model_error:
                    raise RuntimeError(
                        model_error
                    )

                model_prediction = (
                    parse_prediction(
                        raw_model_response,
                        labels,
                    )
                )

                model_success = (
                    model_prediction
                    not in {
                        "",
                        "UNKNOWN",
                        "ERROR",
                    }
                )

            except Exception as exc:
                error = str(exc)

        # ----------------------------------------------------
        # Comparison fields
        # ----------------------------------------------------

        true_label = str(
            row.clause_type
        ).strip()

        rag_top1_correct = (
            retrieval_success
            and rag_top1_label
            == true_label
        )

        model_correct = (
            model_success
            and model_prediction
            == true_label
        )

        true_label_in_top1 = rank_hit(
            retrieved_clause_types,
            true_label,
            1,
        )

        true_label_in_top3 = rank_hit(
            retrieved_clause_types,
            true_label,
            3,
        )

        true_label_in_top5 = rank_hit(
            retrieved_clause_types,
            true_label,
            5,
        )

        rag_vs_model = compare_outcome(
            rag_top1_correct,
            model_correct,
        )

        model_improved = (
            not rag_top1_correct
            and model_correct
        )

        result = {
            "document_id":
                row.document_id,
            "clause_text":
                row.clause_text,
            "true_clause_type":
                true_label,
            "rag_top1_label":
                rag_top1_label,
            "retrieved_clause_types":
                json.dumps(
                    retrieved_clause_types,
                    ensure_ascii=False,
                ),
            "retrieval_scores":
                json.dumps(
                    retrieval_scores,
                    ensure_ascii=False,
                ),
            "true_label_in_top1":
                true_label_in_top1,
            "true_label_in_top3":
                true_label_in_top3,
            "true_label_in_top5":
                true_label_in_top5,
            "retrieval_success":
                retrieval_success,
            "model_prediction":
                model_prediction,
            "model_success":
                model_success,
            "rag_top1_correct":
                rag_top1_correct,
            "model_correct":
                model_correct,
            "rag_vs_model":
                rag_vs_model,
            "model_improved":
                model_improved,
            "raw_model_response":
                raw_model_response,
            "error":
                error,
        }

        # Store both successes and failures.
        completed_rows[key] = result

        newly_processed += 1

        # Checkpoint
        if (
            newly_processed
            % CHECKPOINT_EVERY
            == 0
            or index == len(test_df)
        ):
            save_predictions(
                completed_rows,
                test_df,
                paths["predictions"],
            )

            elapsed = (
                time.time() - started
            )

            print(
                f"  checkpoint saved: "
                f"{newly_processed} new clauses "
                f"({elapsed:.0f}s)"
            )

    # Final checkpoint
    save_predictions(
        completed_rows,
        test_df,
        paths["predictions"],
    )

    # --------------------------------------------------------
    # Final dataframe
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        [
            completed_rows[
                make_row_key(row)
            ]
            for _, row in test_df.iterrows()
            if make_row_key(row)
            in completed_rows
        ],
        columns=PREDICTION_COLUMNS,
    )

    # Retrieval evaluation:
    # only rows where Qdrant retrieval succeeded.
    retrieval_df = results_df[
        results_df[
            "retrieval_success"
        ].astype(bool)
    ].copy()

    # Model evaluation:
    # ONLY successful model predictions.
    model_df = results_df[
        results_df[
            "model_success"
        ].astype(bool)
        & (
            results_df[
                "error"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
            == ""
        )
    ].copy()

    failed_df = results_df[
        ~results_df[
            "model_success"
        ].astype(bool)
    ].copy()

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    retrieval_metrics = (
        compute_retrieval_metrics(
            retrieval_df
        )
    )

    (
        model_metrics,
        model_labels,
        classification_text,
    ) = compute_model_metrics(
        model_df
    )

    comparison = (
        compute_comparison(
            model_df
        )
    )

    leakage = check_leakage()

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    paths[
        "retrieval_metrics"
    ].write_text(
        json.dumps(
            retrieval_metrics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    paths[
        "model_metrics"
    ].write_text(
        json.dumps(
            model_metrics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    paths[
        "comparison_metrics"
    ].write_text(
        json.dumps(
            comparison,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report_text = f"""Qwen + RAG Classification Report
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

    paths[
        "classification_report"
    ].write_text(
        report_text,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Per-class metrics
    # --------------------------------------------------------

    eval_metrics_df = generate_evaluation_metrics_df(model_df)
    eval_metrics_df.to_csv(
        paths["evaluation_metrics"],
        index=False,
        encoding="utf-8-sig",
    )
    
    save_per_class_metrics(
        model_df,
        paths["per_class_metrics"],
    )

    save_error_analysis(
        model_df,
        paths["error_analysis"],
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    save_confusion_matrix(
        model_df,
        model_labels,
        paths["confusion_matrix"],
    )

    save_accuracy_comparison(
        comparison,
        paths["rag_model_plot"],
    )

    # --------------------------------------------------------
    # Markdown report
    # --------------------------------------------------------

    save_comparison_report(
        output_path=paths[
            "comparison_report"
        ],
        run_name=output_dir.name,
        results_df=results_df,
        retrieval_df=retrieval_df,
        model_df=model_df,
        retrieval_metrics=retrieval_metrics,
        model_metrics=model_metrics,
        comparison=comparison,
        leakage=leakage,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("EVALUATION RESULTS")
    print("=" * 72)

    print(
        f"Total rows processed:     "
        f"{len(results_df)}"
    )

    print(
        f"Retrieval successful:     "
        f"{len(retrieval_df)}"
    )

    print(
        f"Model successful:         "
        f"{len(model_df)}"
    )

    print(
        f"Failed rows:              "
        f"{len(failed_df)}"
    )

    print(
        f"RAG Top-1 Accuracy:       "
        f"{retrieval_metrics['rag_top1_accuracy']:.4f}"
    )

    print(
        f"RAG Recall@3:             "
        f"{retrieval_metrics['rag_recall_at_3']:.4f}"
    )

    print(
        f"RAG Recall@5:             "
        f"{retrieval_metrics['rag_recall_at_5']:.4f}"
    )

    print(
        f"MRR:                      "
        f"{retrieval_metrics['mrr']:.4f}"
    )

    print(
        f"Model Accuracy:           "
        f"{model_metrics['accuracy']:.4f}"
    )

    print(
        f"Model Macro F1:           "
        f"{model_metrics['macro_f1']:.4f}"
    )

    print(
        f"Model improvements:      "
        f"{comparison['model_improvement_cases']}"
    )

    print(
        f"Net correct gain:         "
        f"{comparison['net_correct_gain']}"
    )

    print(
        f"\nRUN DIRECTORY:\n"
        f"{output_dir}"
    )


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help=(
            "Evaluate a random sample of N TEST clauses. "
            "Without this argument, the full TEST set is used."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=config.TOP_K,
        help=(
            "Number of retrieved TRAIN examples "
            "used as few-shot evidence."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Use this exact directory for the run. "
            "If it already contains successful predictions, "
            "the run resumes there. Without this option, "
            "a NEW timestamped run directory is created."
        ),
    )

    parser.add_argument(
        "--stratify",
        action="store_true",
        help="Use stratified sampling based on clause_type.",
    )

    args = parser.parse_args()

    if args.sample is not None and args.sample <= 0:
        raise ValueError(
            "--sample must be greater than zero."
        )

    if args.top_k <= 0:
        raise ValueError(
            "--top-k must be greater than zero."
        )

    if args.output_dir is None:
        output_dir = create_run_directory(
            sample=args.sample,
            top_k=args.top_k,
        )
    else:
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    run_evaluation(
        sample=args.sample,
        top_k=args.top_k,
        output_dir=output_dir,
        stratify=args.stratify,
    )


if __name__ == "__main__":
    main()
