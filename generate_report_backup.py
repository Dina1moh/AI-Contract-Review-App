from __future__ import annotations

import argparse
import json
from pathlib import Path

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


# ---------------------------------------------------------------------
# Required columns from the new RAG-vs-Model evaluation
# ---------------------------------------------------------------------

REQUIRED_COLUMNS = {
    "document_id",
    "clause_text",
    "true_clause_type",
    "rag_top1_label",
    "retrieved_clause_types",
    "retrieval_scores",
    "true_label_in_top1",
    "true_label_in_top3",
    "true_label_in_top5",
    "model_prediction",
    "rag_top1_correct",
    "model_correct",
    "rag_vs_model",
    "model_improved",
    "raw_model_response",
    "error",
}


# ---------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------

def resolve_input_dir(input_dir: Path | None) -> Path:
    """
    Use an explicitly provided evaluation directory.

    If no directory is provided, use the latest run_* directory.
    """
    project_root = Path(__file__).resolve().parents[1]
    results_root = project_root / "dataset" / "results"

    if input_dir:
        return input_dir.resolve()

    run_dirs = sorted(
        results_root.glob("run_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not run_dirs:
        raise FileNotFoundError(
            f"No run directories found under: {results_root}"
        )

    return run_dirs[0]


def load_predictions(input_dir: Path) -> pd.DataFrame:
    predictions_path = input_dir / "test_predictions.csv"

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Missing predictions file:\n{predictions_path}"
        )

    df = pd.read_csv(predictions_path)

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            "Predictions file is missing required columns:\n"
            + "\n".join(f"- {column}" for column in sorted(missing))
        )

    return df


# ---------------------------------------------------------------------
# Cleaning and validation
# ---------------------------------------------------------------------

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize boolean columns because CSV files may store them as strings.
    boolean_columns = [
        "true_label_in_top1",
        "true_label_in_top3",
        "true_label_in_top5",
        "rag_top1_correct",
        "model_correct",
        "model_improved",
    ]

    for column in boolean_columns:
        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(
                {
                    "true": True,
                    "false": False,
                    "1": True,
                    "0": False,
                }
            )
            .fillna(False)
        )

    # Errors are technical failures, not model predictions.
    df["error"] = df["error"].fillna("").astype(str).str.strip()

    # Empty model predictions are treated as failures for reporting.
    df["model_prediction"] = (
        df["model_prediction"].fillna("").astype(str).str.strip()
    )

    df["true_clause_type"] = (
        df["true_clause_type"].fillna("").astype(str).str.strip()
    )

    df["rag_top1_label"] = (
        df["rag_top1_label"].fillna("").astype(str).str.strip()
    )

    return df


def get_successful_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows where the pipeline produced no technical error.

    This prevents runtime failures from being counted as model mistakes.
    """
    return df[df["error"] == ""].copy()


# ---------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------

def calculate_retrieval_metrics(df: pd.DataFrame) -> dict:
    evaluated = len(df)

    if evaluated == 0:
        return {
            "evaluated_clauses": 0,
            "rag_top1_correct": 0,
            "rag_top1_accuracy": 0.0,
            "rag_recall_at_3": 0.0,
            "rag_recall_at_5": 0.0,
            "mrr": 0.0,
        }

    top1_correct = int(df["true_label_in_top1"].sum())
    top3_correct = int(df["true_label_in_top3"].sum())
    top5_correct = int(df["true_label_in_top5"].sum())

    reciprocal_ranks = []

    for _, row in df.iterrows():
        true_label = row["true_clause_type"]

        try:
            retrieved_labels = json.loads(
                row["retrieved_clause_types"]
            )
        except (json.JSONDecodeError, TypeError):
            retrieved_labels = []

        rank = None

        for index, label in enumerate(retrieved_labels, start=1):
            if str(label).strip() == true_label:
                rank = index
                break

        reciprocal_ranks.append(
            1.0 / rank if rank else 0.0
        )

    mrr = sum(reciprocal_ranks) / evaluated

    return {
        "evaluated_clauses": evaluated,
        "rag_top1_correct": top1_correct,
        "rag_top1_accuracy": top1_correct / evaluated,
        "rag_recall_at_3": top3_correct / evaluated,
        "rag_recall_at_5": top5_correct / evaluated,
        "mrr": mrr,
    }


# ---------------------------------------------------------------------
# Model metrics
# ---------------------------------------------------------------------

def calculate_model_metrics(df: pd.DataFrame) -> tuple[dict, str]:
    y_true = df["true_clause_type"].astype(str)
    y_pred = df["model_prediction"].astype(str)

    evaluated = len(df)

    if evaluated == 0:
        empty_report = "No successful predictions available."
        return (
            {
                "evaluated_clauses": 0,
                "correct": 0,
                "incorrect": 0,
                "accuracy": 0.0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "macro_f1": 0.0,
            },
            empty_report,
        )

    accuracy = accuracy_score(y_true, y_pred)

    labels = sorted(
        set(y_true.unique()).union(set(y_pred.unique()))
    )

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

    correct = int((y_true == y_pred).sum())

    metrics = {
        "evaluated_clauses": evaluated,
        "correct": correct,
        "incorrect": evaluated - correct,
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }

    return metrics, report


# ---------------------------------------------------------------------
# RAG vs Model comparison
# ---------------------------------------------------------------------

def calculate_comparison(df: pd.DataFrame) -> dict:
    rag_correct = df["rag_top1_correct"]
    model_correct = df["model_correct"]

    both_correct = int((rag_correct & model_correct).sum())

    rag_correct_model_wrong = int(
        (rag_correct & ~model_correct).sum()
    )

    rag_wrong_model_correct = int(
        (~rag_correct & model_correct).sum()
    )

    both_wrong = int(
        (~rag_correct & ~model_correct).sum()
    )

    rag_accuracy = (
        rag_correct.mean() if len(df) else 0.0
    )

    model_accuracy = (
        model_correct.mean() if len(df) else 0.0
    )

    return {
        "evaluated_clauses": len(df),
        "rag_correct": int(rag_correct.sum()),
        "model_correct": int(model_correct.sum()),
        "both_correct": both_correct,
        "rag_correct_model_wrong": rag_correct_model_wrong,
        "rag_wrong_model_correct": rag_wrong_model_correct,
        "both_wrong": both_wrong,
        "model_improvement_cases": rag_wrong_model_correct,
        "net_correct_gain": int(
            model_correct.sum() - rag_correct.sum()
        ),
        "rag_accuracy": float(rag_accuracy),
        "model_accuracy": float(model_accuracy),
        "accuracy_gain": float(
            model_accuracy - rag_accuracy
        ),
    }


# ---------------------------------------------------------------------
# Per-class metrics
# ---------------------------------------------------------------------

def create_per_class_metrics(df: pd.DataFrame) -> pd.DataFrame:
    y_true = df["true_clause_type"].astype(str)
    y_pred = df["model_prediction"].astype(str)

    labels = sorted(
        set(y_true.unique()).union(set(y_pred.unique()))
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    return pd.DataFrame(
        {
            "clause_type": labels,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support,
        }
    )


# ---------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------

def create_error_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only cases where the retrieval and final model disagree
    or both are wrong.
    """
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

    return df[
        [
            column
            for column in columns
            if column in df.columns
        ]
    ].copy()


# ---------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------

def save_confusion_matrix(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    y_true = df["true_clause_type"].astype(str)
    y_pred = df["model_prediction"].astype(str)

    labels = sorted(
        set(y_true.unique()).union(set(y_pred.unique()))
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    n_labels = len(labels)

    figure_width = max(
        14,
        min(0.55 * n_labels + 6, 42),
    )

    figure_height = max(
        11,
        min(0.45 * n_labels + 5, 42),
    )

    fig, ax = plt.subplots(
        figsize=(figure_width, figure_height)
    )

    image = ax.imshow(
        cm,
        cmap="Blues",
        interpolation="nearest",
    )

    ax.set_xticks(range(n_labels))
    ax.set_yticks(range(n_labels))

    ax.set_xticklabels(
        labels,
        rotation=90,
        fontsize=7,
    )

    ax.set_yticklabels(
        labels,
        fontsize=7,
    )

    ax.set_xlabel("Predicted clause type")
    ax.set_ylabel("True clause type")
    ax.set_title("Qwen + RAG Clause Classification Confusion Matrix")

    fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# RAG vs Model visualization
# ---------------------------------------------------------------------

def save_rag_model_comparison_plot(
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

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        categories,
        values,
    )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Retrieval-Only vs. Final Model Accuracy")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.2%}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------

def save_comparison_report(
    output_path: Path,
    run_name: str,
    all_df: pd.DataFrame,
    successful_df: pd.DataFrame,
    retrieval_metrics: dict,
    model_metrics: dict,
    comparison: dict,
    leakage_info: dict,
) -> None:
    failed_count = len(all_df) - len(successful_df)

    report = f"""# Evaluation Comparison Report

## Run

- Run name: `{run_name}`
- Total rows processed: `{len(all_df)}`
- Successful predictions: `{len(successful_df)}`
- Technical failures: `{failed_count}`

## Dataset

- Evaluated clauses: `{len(successful_df)}`
- Train documents: `{leakage_info["train_documents"]}`
- Test documents: `{leakage_info["test_documents"]}`
- Train/Test document overlap: `{leakage_info["overlap_documents"]}`
- Leakage status: `{"PASS" if leakage_info["overlap_documents"] == 0 else "FAIL"}`

## Retrieval-Only Results

- RAG Top-1 Accuracy: `{retrieval_metrics["rag_top1_accuracy"]:.4f}`
- RAG Recall@3: `{retrieval_metrics["rag_recall_at_3"]:.4f}`
- RAG Recall@5: `{retrieval_metrics["rag_recall_at_5"]:.4f}`
- MRR: `{retrieval_metrics["mrr"]:.4f}`
- RAG Top-1 Correct: `{retrieval_metrics["rag_top1_correct"]}`

## Qwen + RAG Results

- Model Accuracy: `{model_metrics["accuracy"]:.4f}`
- Macro Precision: `{model_metrics["macro_precision"]:.4f}`
- Macro Recall: `{model_metrics["macro_recall"]:.4f}`
- Macro F1: `{model_metrics["macro_f1"]:.4f}`
- Model Correct: `{model_metrics["correct"]}`
- Model Incorrect: `{model_metrics["incorrect"]}`

## RAG vs. Model Comparison

| Outcome | Cases |
|---|---:|
| Both Correct | {comparison["both_correct"]} |
| RAG Correct / Model Wrong | {comparison["rag_correct_model_wrong"]} |
| RAG Wrong / Model Correct | {comparison["rag_wrong_model_correct"]} |
| Both Wrong | {comparison["both_wrong"]} |

### Net Effect

- Model improvement cases: `{comparison["model_improvement_cases"]}`
- Net correct gain: `{comparison["net_correct_gain"]}`
- RAG accuracy: `{comparison["rag_accuracy"]:.4f}`
- Model accuracy: `{comparison["model_accuracy"]:.4f}`
- Accuracy gain: `{comparison["accuracy_gain"]:.4f}`

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
"""

    output_path.write_text(
        report,
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# Leakage check
# ---------------------------------------------------------------------

def check_leakage() -> dict:
    project_root = Path(__file__).resolve().parents[1]

    train_path = (
        project_root
        / "dataset"
        / "splits"
        / "train"
        / "master_clauses_train.csv"
    )

    test_path = (
        project_root
        / "dataset"
        / "splits"
        / "test"
        / "master_clauses_test.csv"
    )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if "is_metadata" not in train_df.columns:
        train_df["is_metadata"] = False

    if "is_metadata" not in test_df.columns:
        test_df["is_metadata"] = False

    train_docs = set(
        train_df.loc[
            ~train_df["is_metadata"].astype(bool),
            "document_id",
        ].astype(str)
    )

    test_docs = set(
        test_df.loc[
            ~test_df["is_metadata"].astype(bool),
            "document_id",
        ].astype(str)
    )

    overlap = train_docs & test_docs

    return {
        "train_documents": len(train_docs),
        "test_documents": len(test_docs),
        "overlap_documents": len(overlap),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a complete RAG-vs-model evaluation report "
            "from one saved evaluation run."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=(
            "Evaluation run directory containing "
            "test_predictions.csv. If omitted, the latest run_* "
            "directory is used."
        ),
    )

    args = parser.parse_args()

    input_dir = resolve_input_dir(args.input_dir)

    print("=" * 70)
    print("GENERATE EVALUATION REPORT")
    print("=" * 70)
    print(f"Input run: {input_dir}")

    predictions = load_predictions(input_dir)
    predictions = prepare_dataframe(predictions)

    successful = get_successful_predictions(predictions)

    if successful.empty:
        raise RuntimeError(
            "No successful predictions are available for reporting."
        )

    retrieval_metrics = calculate_retrieval_metrics(successful)

    model_metrics, classification_text = calculate_model_metrics(
        successful
    )

    comparison = calculate_comparison(successful)

    per_class = create_per_class_metrics(successful)
    error_analysis = create_error_analysis(successful)

    leakage = check_leakage()

    # -------------------------------------------------------------
    # Save JSON metrics
    # -------------------------------------------------------------

    (input_dir / "retrieval_metrics.json").write_text(
        json.dumps(
            retrieval_metrics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    (input_dir / "model_metrics.json").write_text(
        json.dumps(
            model_metrics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    (input_dir / "comparison_metrics.json").write_text(
        json.dumps(
            comparison,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------
    # Save classification report
    # -------------------------------------------------------------

    classification_report_text = (
        "Qwen + RAG Classification Report\n"
        "=================================\n\n"
        f"Evaluated clauses: {len(successful)}\n"
        f"Accuracy: {model_metrics['accuracy']:.6f}\n"
        f"Macro Precision: {model_metrics['macro_precision']:.6f}\n"
        f"Macro Recall: {model_metrics['macro_recall']:.6f}\n"
        f"Macro F1: {model_metrics['macro_f1']:.6f}\n\n"
        "Per-Class Classification Report\n"
        "================================\n"
        f"{classification_text}\n"
    )

    (input_dir / "classification_report.txt").write_text(
        classification_report_text,
        encoding="utf-8",
    )

    # -------------------------------------------------------------
    # Save per-class and error analysis CSVs
    # -------------------------------------------------------------

    per_class.to_csv(
        input_dir / "per_class_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    error_analysis.to_csv(
        input_dir / "error_analysis.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------
    # Save figures
    # -------------------------------------------------------------

    save_confusion_matrix(
        successful,
        input_dir / "confusion_matrix.png",
    )

    save_rag_model_comparison_plot(
        comparison,
        input_dir / "rag_vs_model_accuracy.png",
    )

    # -------------------------------------------------------------
    # Save Markdown report
    # -------------------------------------------------------------

    save_comparison_report(
        input_dir / "comparison_report.md",
        run_name=input_dir.name,
        all_df=predictions,
        successful_df=successful,
        retrieval_metrics=retrieval_metrics,
        model_metrics=model_metrics,
        comparison=comparison,
        leakage_info=leakage,
    )

    # -------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------

    print("\n" + "=" * 70)
    print("REPORT GENERATED")
    print("=" * 70)

    print(f"Total rows:              {len(predictions)}")
    print(f"Successful predictions:  {len(successful)}")
    print(f"Technical failures:      {len(predictions) - len(successful)}")

    print(
        f"RAG Top-1 Accuracy:      "
        f"{retrieval_metrics['rag_top1_accuracy']:.4f}"
    )

    print(
        f"RAG Recall@3:            "
        f"{retrieval_metrics['rag_recall_at_3']:.4f}"
    )

    print(
        f"RAG Recall@5:            "
        f"{retrieval_metrics['rag_recall_at_5']:.4f}"
    )

    print(
        f"MRR:                     "
        f"{retrieval_metrics['mrr']:.4f}"
    )

    print(
        f"Model Accuracy:          "
        f"{model_metrics['accuracy']:.4f}"
    )

    print(
        f"Model Macro F1:          "
        f"{model_metrics['macro_f1']:.4f}"
    )

    print(
        f"Model improvements:      "
        f"{comparison['model_improvement_cases']}"
    )

    print(
        f"Net correct gain:        "
        f"{comparison['net_correct_gain']}"
    )

    print(f"\nReport directory:\n{input_dir}")


if __name__ == "__main__":
    main()
