import sys
import argparse
from datetime import datetime

import mlflow
import mlflow.genai

from config import GOOGLE_API_KEY, TAVILY_API_KEY, DEFAULT_MODEL, MLFLOW_EXPERIMENT_NAME
from agents.workflow import create_market_news_graph
from evaluation.evaluate_agent import run_genai_evaluation, predict_fn

# =====================================================================
# Aktifkan MLflow Tracing untuk LangGraph + LangChain
# Harus dipanggil sebelum graph di-invoke agar semua node & tools ter-trace
# =====================================================================
mlflow.langchain.autolog()
mlflow.autolog()


def run_market_news_assistant(user_query: str, evaluate: bool = True):
    print("=" * 70)
    print("🚀 AI ASSISTANT: 24H MARKET NEWS SUMMARY (AGENTIC WORKFLOW)")
    print("=" * 70)
    print(f"🤖 LLM Model      : Google AI Studio ({DEFAULT_MODEL})")
    print(f"🔎 Search Tool     : Tavily Search (Google Dorking + 24h Filter)")
    print(f"🎯 Fokus Pasar     : {user_query}")
    print(f"📡 MLflow Tracing  : LangGraph nodes + LLM + Tools (autolog aktif)")
    print("=" * 70)

    if not GOOGLE_API_KEY:
        print("\n⚠️  PERINGATAN: GOOGLE_API_KEY belum terpasang!")
        print("Silakan tambahkan GOOGLE_API_KEY di file .env Anda.")
        return

    if not TAVILY_API_KEY:
        print("\n⚠️  PERHATIAN: TAVILY_API_KEY belum terpasang!")
        print("Agent berjalan dengan data peringatan sistem.")

    # Set experiment MLflow
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Jalankan graph dan tampilkan laporan
    print("\n[Orchestrator] Memulai eksekusi agent (trace MLflow aktif)...\n")
    report = predict_fn(query=user_query)

    print("\n" + "=" * 70)
    print("📄 HASIL LAPORAN RINGKASAN PASAR (24 JAM TERAKHIR)")
    print("=" * 70)
    print(report)
    print("=" * 70)

    # Simpan laporan ke file
    report_filename = "latest_market_report.md"
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n💾 Laporan disimpan ke: {report_filename}")

    # Evaluasi dengan mlflow.genai.evaluate()
    if evaluate:
        scenario = [{"inputs": {"query": user_query}}]
        run_name = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        eval_result = run_genai_evaluation(eval_data=scenario, run_name=run_name)
    else:
        eval_result = None

    print("\n✅ Selesai. Jalankan: mlflow ui")
    print("   Lihat Traces di http://127.0.0.1:5000\n")

    return {"market_report": report, "evaluation_result": eval_result}


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Market News Summary AI Assistant")
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Topik pasar yang ingin dianalisis (24 jam terakhir)",
    )
    parser.add_argument(
        "--eval-all",
        action="store_true",
        help="Jalankan seluruh skenario evaluasi (mlflow.genai.evaluate)",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Lewati evaluasi MLflow setelah laporan dibuat",
    )

    args = parser.parse_args()

    if args.eval_all:
        run_genai_evaluation()
        return

    if args.query:
        query = args.query
    else:
        default_query = (
            "Berita pasar saham AS, kebijakan suku bunga The Fed, "
            "dan sentimen saham teknologi 24 jam terakhir"
        )
        print(f"\nMasukkan fokus pasar (Enter = default):")
        print(f"Default: '{default_query}'")
        try:
            user_input = input(">> ").strip()
            query = user_input if user_input else default_query
        except (EOFError, KeyboardInterrupt):
            query = default_query

    run_market_news_assistant(user_query=query, evaluate=not args.no_eval)


if __name__ == "__main__":
    main()
