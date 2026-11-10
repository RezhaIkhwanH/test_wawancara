import logging
from typing import List, Dict, Any, Optional
import mlflow
from tavily import TavilyClient
from config import TAVILY_API_KEY, TRUSTED_FINANCE_DOMAINS

logger = logging.getLogger(__name__)


def build_google_dork_query(
    keywords: str,
    target_domains: Optional[List[str]] = None,
    intitle_terms: Optional[List[str]] = None,
    exact_phrases: Optional[List[str]] = None
) -> str:
    """
    Menyusun query pencarian menggunakan teknik Google Dorking.
    Contoh:
      (site:reuters.com OR site:cnbc.com) intitle:"Fed" OR intitle:"interest rate" "inflation"
    """
    parts = []

    # Filter Domain (site:...)
    domains = target_domains if target_domains else TRUSTED_FINANCE_DOMAINS[:4]
    if domains:
        site_filters = [f"site:{d.strip()}" for d in domains]
        parts.append(f"({' OR '.join(site_filters)})")

    # Filter Judul (intitle:...)
    if intitle_terms:
        title_filters = [f'intitle:"{t.strip()}"' for t in intitle_terms]
        parts.append(f"({' OR '.join(title_filters)})")

    # Kata Kunci Tepat ("...")
    if exact_phrases:
        phrases = [f'"{p.strip()}"' for p in exact_phrases]
        parts.extend(phrases)

    # Keywords Bebas
    if keywords:
        parts.append(keywords.strip())

    return " ".join(parts)


@mlflow.trace(name="tavily_search_tool")
def search_market_news_24h(dork_query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Mencari berita pasar menggunakan Tavily Search API (Tool Call).
    Menggunakan filter ketat:
      - topic='news'
      - time_range='day' (membatasi hanya berita dalam 24 jam terakhir)
    Dicatat dalam trace MLflow sebagai span 'tavily_search_tool'.
    """
    if not TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY belum dikonfigurasi. Mengembalikan pesan petunjuk.")
        return [
            {
                "title": "Perhatian: TAVILY_API_KEY Belum Terpasang",
                "url": "https://tavily.com",
                "content": (
                    "Harap isi TAVILY_API_KEY di file .env Anda untuk melakukan scraping "
                    "berita pasar riil 24 jam terakhir dari Tavily Search."
                ),
                "published_date": "24h recent",
                "source": "System Warning"
            }
        ]

    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        
        # Panggil Tavily dengan time_range='day' khusus 24 jam terakhir
        response = client.search(
            query=dork_query,
            topic="news",
            time_range="day",  # Kunci pemfilteran 24 jam
            search_depth="advanced",
            max_results=max_results,
            include_answer=True,
            include_raw_content=False
        )

        results = []
        raw_items = response.get("results", [])
        
        for item in raw_items:
            results.append({
                "title": item.get("title", "No Title"),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "published_date": item.get("published_date", "Last 24 Hours"),
                "score": item.get("score", 0.0),
                "source": item.get("url", "").split("/")[2] if "//" in item.get("url", "") else "News Source"
            })

        # Jika ada jawaban instan AI Tavily, sertakan juga
        if response.get("answer"):
            results.insert(0, {
                "title": "Ringkasan Instan Tavily (24h Market Pulse)",
                "url": "https://tavily.com",
                "content": response.get("answer"),
                "published_date": "Last 24 Hours",
                "score": 1.0,
                "source": "Tavily AI Synthesis"
            })

        return results

    except Exception as e:
        logger.error(f"Error saat mencari berita via Tavily: {str(e)}")
        return [
            {
                "title": f"Gagal mengambil berita: {dork_query[:40]}...",
                "url": "N/A",
                "content": f"Terjadi kesalahan saat memanggil API Tavily: {str(e)}",
                "published_date": "N/A",
                "source": "Error"
            }
        ]
