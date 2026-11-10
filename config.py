import os
import re
from typing import Any
from dotenv import load_dotenv

# Muat variabel environment dari file .env
load_dotenv(override=True)

# Supress MLflow hint message
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

# API Keys support fleksibel (bisa GOOGLE_API_KEY atau GOOGLE_API)
GOOGLE_API_KEY = (
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GOOGLE_API")
    or os.getenv("GEMINI_API_KEY")
    or ""
).strip()

TAVILY_API_KEY = (
    os.getenv("TAVILY_API_KEY")
    or os.getenv("TAVILY_API")
    or ""
).strip()

# Sinkronkan kembali ke os.environ agar library resmi Google & Tavily mendeteksi otomatis
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GEMINI_API_KEY"] = GOOGLE_API_KEY

if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# Normalisasi nama model Gemini (menangani format seperti 'Gemini 3.5 Flash Lite' -> 'gemini-3.5-flash-lite')
def normalize_model_name(raw_name: str) -> str:
    if not raw_name:
        return "gemini-3.5-flash-lite"
    cleaned = raw_name.strip().lower()
    cleaned = re.sub(r"\s+", "-", cleaned)
    if cleaned.startswith("models/"):
        cleaned = cleaned[len("models/"):]
    return cleaned

raw_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
DEFAULT_MODEL = normalize_model_name(raw_model)

# MLflow Experiment Name
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Market_News_Agentic_Evaluation")

# Daftar portal berita keuangan terpercaya untuk Google Dorking
TRUSTED_FINANCE_DOMAINS = [
    "reuters.com",
    "bloomberg.com",
    "cnbc.com",
    "marketwatch.com",
    "wsj.com",
    "ft.com",
    "finance.yahoo.com",
    "investing.com",
    "coindesk.com"
]

def extract_text_content(response: Any) -> str:
    """
    Mengekstrak teks murni dari objek response LLM, baik bertipe string maupun list of blocks.
    """
    content = getattr(response, "content", response)
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            elif hasattr(block, "text"):
                parts.append(getattr(block, "text", ""))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)
