import re

def update_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    unified_metrics_code = """
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
"""

    if "def compute_classification_metrics(" not in content:
        if "def calculate_retrieval_metrics" in content:
            content = content.replace("def calculate_retrieval_metrics", unified_metrics_code + "\n\ndef calculate_retrieval_metrics")
        elif "def compute_retrieval_metrics" in content:
            content = content.replace("def compute_retrieval_metrics", unified_metrics_code + "\n\ndef compute_retrieval_metrics")
            
    if "Run_evaluation.py" in filepath:
        save_code = """
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
"""
        if "evaluation_metrics.csv" not in content:
            content = content.replace('"run_metadata": run_dir / "run_metadata.json",', '"run_metadata": run_dir / "run_metadata.json",\n        "evaluation_metrics": run_dir / "evaluation_metrics.csv",')
            # Replace the exact function call
            content = content.replace('save_per_class_metrics(\n        model_df,\n        paths["per_class_metrics"],\n    )', save_code.strip())
            
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

update_file("src/Run_evaluation.py")
print("Done.")

