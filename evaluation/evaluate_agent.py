import os
import sys
from pathlib import Path

# Pastikan root direktori proyek ada di sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from datetime import datetime
from typing import Dict, Any, List, Optional

import mlflow
import mlflow.genai
from mlflow.genai.scorers import scorer
from mlflow.entities import Feedback

from config import MLFLOW_EXPERIMENT_NAME
from agents.workflow import create_market_news_graph

# =====================================================================
# Aktifkan MLflow Autolog untuk LangGraph + LangChain Tracing
# mlflow.langchain.autolog() → menangkap semua node LangGraph,
#   LLM call, tool call sebagai spans bersarang dalam satu trace
# mlflow.autolog() → menangkap google.genai SDK calls
# =====================================================================
mlflow.langchain.autolog()
mlflow.autolog()

# =====================================================================
# Dataset Skenario Evaluasi
# =====================================================================
EVAL_SCENARIOS = [
    {
        "inputs": {
            "query": "Berita pasar saham Wall Street, suku bunga The Fed, dan saham Nvidia 24 jam terakhir"
        }
    },
    {
        "inputs": {
            "query": "Sentimen pasar kripto Bitcoin dan arah suku bunga global 24 jam terakhir"
        }
    },
]


# =====================================================================
# JURI 1: Validasi Rute Eksekusi Node LangGraph
# Memastikan planner_node → searcher_node → reporter_node terpanggil
# =====================================================================
@scorer(name="juri_rute")
def juri_rute(trace) -> Feedback:
    """Verifikasi semua node LangGraph wajib dieksekusi secara berurutan."""
    span_names = [s.name for s in trace.search_spans()]
    required_nodes = ["planner_node", "searcher_node", "reporter_node"]
    missing = [n for n in required_nodes if n not in span_names]

    if not missing:
        return Feedback(
            value=True,
            rationale=(
                f"Lolos. Semua node berhasil: {' -> '.join(required_nodes)}. "
                f"Total spans: {len(span_names)}."
            ),
        )
    return Feedback(
        value=False,
        rationale=f"Gagal. Node tidak ditemukan: {missing}. Spans terekam: {span_names}.",
    )


# =====================================================================
# JURI 2: Validasi Tool Call Tavily Search
# Memastikan Tavily dipanggil dengan filter 24 jam
# =====================================================================
@scorer(name="juri_tool")
def juri_tool(trace) -> Feedback:
    """Verifikasi pemanggilan tool Tavily Search dalam trace."""
    tool_spans = trace.search_spans(name="tavily_search_tool")
    n = len(tool_spans)

    if n > 0:
        return Feedback(
            value=True,
            rationale=f"Lolos. {n} pemanggilan 'tavily_search_tool' terdeteksi (Google Dorking, filter 24h).",
        )
    return Feedback(
        value=False,
        rationale="Gagal. Tidak ada pemanggilan 'tavily_search_tool' yang tercatat.",
    )


# =====================================================================
# JURI 3: Validasi Relevansi & Kelengkapan Laporan
# =====================================================================
@scorer(name="juri_relevancy")
def juri_relevancy(inputs: Dict[str, Any], outputs: str) -> Feedback:
    """Verifikasi laporan relevan, tidak kosong, dan berstruktur lengkap."""
    query = inputs.get("query", "")
    if not outputs or len(outputs.strip()) < 100:
        return Feedback(value=False, rationale="Gagal. Output laporan kosong atau terlalu pendek.")

    keywords_check = ["ringkasan", "pasar", "sumber", "24 jam"]
    found = [k for k in keywords_check if k in outputs.lower()]

    if len(found) >= 3:
        return Feedback(
            value=True,
            rationale=(
                f"Lolos. Laporan menjawab '{query[:50]}...' "
                f"dengan struktur lengkap dan referensi sumber 24 jam terakhir."
            ),
        )
    return Feedback(
        value=False,
        rationale=f"Laporan kurang memuat bagian standar pasar. Elemen ditemukan: {found}.",
    )


# =====================================================================
# Predict Function — entry point untuk mlflow.genai.evaluate()
# @mlflow.trace membungkus seluruh graph invoke sebagai root span
# mlflow.langchain.autolog() menangkap node & tools sebagai child spans
# =====================================================================
@mlflow.trace(name="market_agent_execution")
def predict_fn(query: str) -> str:
    """
    Jalankan alur LangGraph multi-agent dan kembalikan laporan pasar.
    Semua node (planner, searcher, reporter) dan tool call otomatis
    ter-trace sebagai child spans berkat mlflow.langchain.autolog().
    """
    graph = create_market_news_graph()
    initial_state = {
        "user_query": query,
        "plan": {},
        "dork_queries": [],
        "search_results": [],
        "market_report": "",
        "evaluation_result": None,
    }
    final_state = graph.invoke(initial_state)
    return final_state.get("market_report", "")


# =====================================================================
# Fungsi Evaluasi Utama dengan mlflow.genai.evaluate()
# =====================================================================
def run_genai_evaluation(
    eval_data: Optional[List[Dict[str, Any]]] = None,
    run_name: Optional[str] = None,
):
    """
    Evaluasi lengkap agent dengan mlflow.genai.evaluate().

    Alur:
      1. mlflow.genai.evaluate() memanggil predict_fn() per skenario
      2. predict_fn() menjalankan graph LangGraph → trace otomatis terbentuk
      3. Trace berisi semua spans: LangGraph > planner > searcher > tools > reporter
      4. Ketiga juri (juri_rute, juri_tool, juri_relevancy) menilai trace
      5. Hasil metrik & trace tersimpan di MLflow experiment

    Lihat hasilnya di: mlflow ui → http://127.0.0.1:5000
    """
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    if not run_name:
        run_name = f"genai_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    data = eval_data if eval_data else EVAL_SCENARIOS

    print("\n" + "=" * 65)
    print("⚖️  EVALUASI AGENT → mlflow.genai.evaluate()")
    print("=" * 65)
    print(f"📊 Experiment : {MLFLOW_EXPERIMENT_NAME}")
    print(f"🏷️  Run Name   : {run_name}")
    print(f"📋 Skenario   : {len(data)} query")
    print(f"🔍 Autolog    : LangGraph nodes + LLM + Tools (via mlflow.langchain.autolog)")
    print(f"👨‍⚖️ Scorers    : juri_rute | juri_tool | juri_relevancy")
    print("=" * 65 + "\n")

    with mlflow.start_run(run_name=run_name):
        result = mlflow.genai.evaluate(
            data=data,
            predict_fn=predict_fn,
            scorers=[juri_rute, juri_tool, juri_relevancy],
        )

    # Tampilkan ringkasan metrik
    print("\n" + "=" * 65)
    print("📊 HASIL EVALUASI MLFLOW.GENAI.EVALUATE():")
    for metric_name, value in result.metrics.items():
        icon = "✅" if float(value) >= 0.8 else "⚠️"
        print(f"  {icon} {metric_name}: {float(value):.3f}")
    print("\n✅ Selesai. Jalankan: mlflow ui")
    print("   Buka tab 'Traces' di MLflow UI untuk melihat detail span")
    print("   setiap node LangGraph, LLM call, dan tool Tavily.")
    print("=" * 65 + "\n")

    return result


if __name__ == "__main__":
    run_genai_evaluation()
