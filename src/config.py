"""
config.py — Central configuration loaded from .env
All other modules import from here; never import os.environ directly.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve project root (two levels up from this file: src/ → project/)
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_RAW_DIR       = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR         = ROOT_DIR / "models"
NOTEBOOKS_DIR      = ROOT_DIR / "notebooks"

RAW_CSV            = DATA_RAW_DIR / "osteoporosis.csv"
PROCESSED_CSV      = DATA_PROCESSED_DIR / "osteoporosis_processed.csv"
MODEL_PATH         = MODELS_DIR / "best_model.joblib"
PREPROCESSOR_PATH  = MODELS_DIR / "preprocessor.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"

# ── Supabase ──────────────────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# ── LLM ───────────────────────────────────────────────────────────────────────
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")

# Decide which LLM backend to use
def llm_backend() -> str:
    if OPENAI_API_KEY:
        return "openai"
    if GEMINI_API_KEY:
        return "gemini"
    return "template"

# ── ML ────────────────────────────────────────────────────────────────────────
RANDOM_STATE   = 42
CV_FOLDS       = 5
TARGET_COLUMN  = "Osteoporosis"   # raw binary label in the CSV
TARGET_3CLASS  = "risk_label"     # Normal / Osteopenia / Osteoporosis
SMOTE_THRESHOLD = 0.20            # apply SMOTE if minority class < 20 %

CLASS_NAMES    = ["Normal", "Osteopenia", "Osteoporosis"]
CLASS_COLORS   = {"Normal": "#22c55e", "Osteopenia": "#f59e0b", "Osteoporosis": "#ef4444"}

# ── API ───────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
