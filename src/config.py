import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

for dotenv_path in (
    PROJECT_ROOT / ".env",
    PROJECT_ROOT / "scripts" / ".env",
):
    if dotenv_path.exists():
        load_dotenv(dotenv_path)
        break


# The workspace shell is configured with a dead localhost proxy. Clear it
# here so Cohere/httpx can reach the real API while keeping localhost
# requests for Ollama and Qdrant unaffected.
for proxy_var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    proxy_value = os.getenv(proxy_var, "")
    if proxy_value.startswith("http://127.0.0.1:9") or proxy_value.startswith(
        "https://127.0.0.1:9"
    ):
        os.environ.pop(proxy_var, None)
        os.environ.pop(proxy_var.lower(), None)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
TRAIN_DATA_PATH = (
    PROJECT_ROOT / "dataset" / "splits" / "train" / "master_clauses_train.csv"
)
TEST_DATA_PATH = (
    PROJECT_ROOT / "dataset" / "splits" / "test" / "master_clauses_test.csv"
)

RESULTS_DIR = PROJECT_ROOT / "dataset" / "results"
LABEL_DEFINITIONS_PATH = (
    PROJECT_ROOT / "dataset" / "metadata" / "cuad_label_definitions.json"
)
PREDICTIONS_PATH = RESULTS_DIR / "test_predictions.csv"
METRICS_PATH = RESULTS_DIR / "metrics.json"
CLASSIFICATION_REPORT_PATH = RESULTS_DIR / "classification_report.csv"
CONFUSION_MATRIX_PNG_PATH = RESULTS_DIR / "confusion_matrix.png"
CONFUSION_MATRIX_CSV_PATH = RESULTS_DIR / "confusion_matrix.csv"


# --------------------------------------------------------------------------
# Cohere (must match the model used to build the Qdrant index in
# embed_train.py, or retrieval similarity will be meaningless)
# --------------------------------------------------------------------------
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
COHERE_MODEL = os.getenv(
    "COHERE_MODEL",
    os.getenv("COHERE_EMBED_MODEL", "embed-english-v3.0"),
)
COHERE_EMBED_BATCH_SIZE = int(os.getenv("COHERE_BATCH_SIZE", "96"))
COHERE_MAX_RETRY_ATTEMPTS = int(os.getenv("COHERE_MAX_RETRY_ATTEMPTS", "8"))


# --------------------------------------------------------------------------
# Qdrant (same collection built by build_qdrant_collection.py)
# --------------------------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_PATH = os.getenv(
    "QDRANT_PATH", str(PROJECT_ROOT / "dataset" / "qdrant_local")
)
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "cuad_train")


# --------------------------------------------------------------------------
# Retrieval + local Qwen (Ollama)
# --------------------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "5"))

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60"))
OLLAMA_MAX_RETRY_ATTEMPTS = int(os.getenv("OLLAMA_MAX_RETRY_ATTEMPTS", "3"))


def load_labels() -> list[str]:
    """The fixed set of substantive clause-type labels the model is allowed
    to choose from -- derived from the TRAIN split so it always matches
    exactly what's indexed in Qdrant (metadata fields like Document Name /
    Parties / dates are excluded, same as in embed_train.py).
    """
    import pandas as pd

    df = pd.read_csv(TRAIN_DATA_PATH)
    if "is_metadata" not in df.columns:
        df["is_metadata"] = False
    labels = sorted(df.loc[~df["is_metadata"].astype(bool), "clause_type"].unique())
    return labels
