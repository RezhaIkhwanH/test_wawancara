import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import mlflow
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from config import GOOGLE_API_KEY, DEFAULT_MODEL, extract_text_content
from agents.state import MarketState


class ResearchTopic(BaseModel):
    category: str = Field(description="Kategori pasar, misal: Makro/Bank Sentral, Indeks Saham, Komoditas, Korporasi/Earnings, atau Geopolitik")
    focus_question: str = Field(description="Pertanyaan inti yang harus dijawab dari berita 24 jam terakhir")
    target_entities: List[str] = Field(description="Nama entitas, indeks, atau ticker relevan (contoh: ['Federal Reserve', 'S&P 500', 'Nvidia', 'Crude Oil'])")


class MarketPlan(BaseModel):
    objective: str = Field(description="Ringkasan objektif riset pasar yang dibutuhkan user")
    topics: List[ResearchTopic] = Field(description="Daftar 3 sampai 4 sub-topik riset spesifik")


@mlflow.trace(name="planner_node")
def plan_agent(state: MarketState) -> Dict[str, Any]:
    """
    Agent 1: Planning Agent
    Menganalisis permintaan pengguna dan merancang rencana riset pasar yang terarah
    untuk dikumpulkan dari berita internet 24 jam terakhir.
    """
    user_query = state.get("user_query") or "Ringkasan berita pasar saham dan ekonomi global 24 jam terakhir"

    llm = ChatGoogleGenerativeAI(
        model=DEFAULT_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2
    )

    system_prompt = """Anda adalah Senior Chief Market Strategist di bank investasi global.
Tugas Anda adalah merancang Rencana Riset Pasar (Market Research Plan) untuk memandu agen pencari internet.
Rencana riset harus secara spesifik menargetkan peristiwa-peristiwa yang terjadi atau dirilis dalam 24 JAM TERAKHIR.

Fokuskan pada:
1. Keputusan makroekonomi / kebijakan moneter suku bunga bank sentral (The Fed, ECB, BI, dll.)
2. Pergerakan indeks saham utama (S&P 500, Nasdaq, Dow Jones, IHSG, Nikkei)
3. Katalis berita emiten besar (tech giants, laporan laba/earnings, M&A)
4. Pasar obligasi/yield treasury, valuta asing (DXY/USD), komoditas (Minyak, Emas), atau kripto jika relevan."""

    user_prompt = f"Permintaan Analisis Pasar dari Pengguna:\n\"{user_query}\"\n\nSusun rencana riset terperinci untuk mencari informasi 24 jam terakhir."

    try:
        structured_llm = llm.with_structured_output(MarketPlan)
        plan_result = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        plan_dict = plan_result.model_dump()
    except Exception as e:
        # Fallback parsing jika structured output mengalami kendala
        raw_response = llm.invoke([
            SystemMessage(content=system_prompt + "\nKembalikan HANYA format JSON valid sesuai schema: {'objective': '...', 'topics': [{'category': '...', 'focus_question': '...', 'target_entities': [...]}]}"),
            HumanMessage(content=user_prompt)
        ])
        try:
            cleaned_text = extract_text_content(raw_response).strip().replace("```json", "").replace("```", "")
            plan_dict = json.loads(cleaned_text)
        except Exception:
            plan_dict = {
                "objective": f"Riset berita pasar 24 jam terakhir untuk: {user_query}",
                "topics": [
                    {
                        "category": "Makroekonomi & The Fed",
                        "focus_question": "Bagaimana arah suku bunga dan pernyataan pejabat The Fed 24 jam terakhir?",
                        "target_entities": ["Federal Reserve", "Jerome Powell", "Interest Rates", "Inflation"]
                    },
                    {
                        "category": "Indeks Saham Utama",
                        "focus_question": "Bagaimana penutupan dan pergerakan indeks S&P 500, Nasdaq, dan Dow Jones?",
                        "target_entities": ["S&P 500", "Nasdaq", "Wall Street", "Dow Jones"]
                    },
                    {
                        "category": "Sektor Saham & Komoditas",
                        "focus_question": "Apa berita penting terkait saham teknologi besar dan harga minyak/emas hari ini?",
                        "target_entities": ["Big Tech", "Nvidia", "Crude Oil", "Gold"]
                    }
                ]
            }

    print("\n[Agent 1: Planning Agent] Berhasil menyusun rencana riset:")
    print(f"- Objektif: {plan_dict.get('objective')}")
    for idx, t in enumerate(plan_dict.get("topics", []), 1):
        print(f"  {idx}. [{t.get('category')}] {t.get('focus_question')} (Entitas: {', '.join(t.get('target_entities', []))})")

    return {"plan": plan_dict}

