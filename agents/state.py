from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict


class MarketState(TypedDict):
    """
    State yang dibagi antar agen dalam LangGraph workflow.
    """
    user_query: str                          # Pertanyaan / fokus pasar dari user
    plan: Dict[str, Any]                     # Rencana riset dari Agent 1 (Planning Agent)
    dork_queries: List[Dict[str, str]]       # Daftar query Google Dorking dari Agent 2
    search_results: List[Dict[str, Any]]     # Berita 24 jam terakhir yang berhasil dikumpulkan
    market_report: str                       # Laporan akhir dari Agent 3 (Report Agent)
    evaluation_result: Optional[Dict[str, Any]] # Hasil evaluasi MLflow LLM-as-a-Judge

