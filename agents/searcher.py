import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import mlflow
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from config import GOOGLE_API_KEY, DEFAULT_MODEL, TRUSTED_FINANCE_DOMAINS
from tools.search_tool import search_market_news_24h, build_google_dork_query
from agents.state import MarketState


class DorkQueryItem(BaseModel):
    category: str = Field(description="Kategori yang ditargetkan")
    dork_query: str = Field(description="String query Google Dorking lengkap (menggunakan site:, intitle:, OR, tanda petik)")
    search_intent: str = Field(description="Tujuan spesifik dari query dorking ini")


class DorkStrategy(BaseModel):
    queries: List[DorkQueryItem] = Field(description="Daftar query Google Dorking untuk dieksekusi")


@mlflow.trace(name="searcher_node")
def search_agent(state: MarketState) -> Dict[str, Any]:
    """
    Agent 2: Search Agent (Pencari Berita dengan Google Dorking & Tavily)
    Mengubah rencana riset dari Agent 1 menjadi strategi Google Dorking,
    kemudian mengeksekusi pencarian Tavily dengan filter ketat 24 jam terakhir.
    """
    plan = state.get("plan", {})
    topics = plan.get("topics", [])

    llm = ChatGoogleGenerativeAI(
        model=DEFAULT_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1
    )

    trusted_sites_str = " OR ".join([f"site:{d}" for d in TRUSTED_FINANCE_DOMAINS[:5]])

    system_prompt = f"""Anda adalah Spesialis Intelijen Informasi Pasar dan Ahli Google Dorking.
Tugas Anda adalah merumuskan query pencarian Google Dorking yang presisi untuk mencari berita pasar finansial 24 JAM TERAKHIR.

ATURAN GOOGLE DORKING:
1. Gunakan operator 'site:' untuk membatasi domain berita keuangan tepercaya seperti:
   ({trusted_sites_str})
2. Gunakan operator 'intitle:' untuk memastikan kata kunci utama ada di judul artikel (contoh: intitle:"Fed" OR intitle:"Wall Street").
3. Gunakan tanda kutip ganda '""' untuk exact phrase match (contoh: "interest rate", "S&P 500").
4. Buat 1 query Google Dorking yang kuat dan tajam untuk setiap topik riset yang diberikan.
5. Query akan dieksekusi di Tavily dengan filter 24 jam (time_range='day')."""

    plan_str = json.dumps(topics, indent=2, ensure_ascii=False)
    user_prompt = f"Berikut topik riset dari Planning Agent:\n{plan_str}\n\nBuat query Google Dorking optimal untuk setiap topik."

    try:
        structured_llm = llm.with_structured_output(DorkStrategy)
        dork_strategy = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        dork_list = [q.model_dump() for q in dork_strategy.queries]
    except Exception:
        # Fallback manual dork generation jika structured output terkendala
        dork_list = []
        for t in topics:
            category = t.get("category", "Market")
            entities = t.get("target_entities", ["market"])
            primary_entity = entities[0] if entities else "stocks"
            query_str = build_google_dork_query(
                keywords="market news update",
                target_domains=TRUSTED_FINANCE_DOMAINS[:4],
                intitle_terms=[primary_entity],
                exact_phrases=[primary_entity]
            )
            dork_list.append({
                "category": category,
                "dork_query": query_str,
                "search_intent": t.get("focus_question", "Cari update 24 jam")
            })

    print(f"\n[Agent 2: Search Agent] Menghasilkan {len(dork_list)} Query Google Dorking:")
    all_results = []
    seen_urls = set()

    for item in dork_list:
        dork_q = item.get("dork_query")
        cat = item.get("category")
        print(f"  🔍 Dork [{cat}]: {dork_q}")
        
        # Eksekusi pencarian Tavily dengan filter 24 jam terakhir
        results = search_market_news_24h(dork_q, max_results=4)
        print(f"     -> Ditemukan {len(results)} artikel terverifikasi.")

        for res in results:
            url = res.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                res["topic_category"] = cat
                all_results.append(res)

    print(f"[Agent 2: Search Agent] Total artikel terkumpul (unik): {len(all_results)} berita.\n")

    return {
        "dork_queries": dork_list,
        "search_results": all_results
    }

