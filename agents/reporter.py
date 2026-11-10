import json
from typing import Dict, Any
import mlflow
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from config import GOOGLE_API_KEY, DEFAULT_MODEL, extract_text_content
from agents.state import MarketState


@mlflow.trace(name="reporter_node")
def report_agent(state: MarketState) -> Dict[str, Any]:
    """
    Agent 3: Report Agent (Pembuat Laporan Ringkasan Berita Pasar)
    Menganalisis hasil pencarian berita 24 jam terakhir dan merangkumnya menjadi
    laporan komprehensif, terstruktur, berbasis data riil, dan profesional.
    """
    user_query = state.get("user_query", "Market News Summary")
    plan = state.get("plan", {})
    search_results = state.get("search_results", [])

    llm = ChatGoogleGenerativeAI(
        model=DEFAULT_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )

    system_prompt = """Anda adalah Kepala Riset Pasar dan Redaktur Utama di Institusi Finansial Global.
Tugas Anda adalah menulis laporan ringkasan berita pasar (Market Intelligence Summary) berdasarkan kumpulan berita yang terjadi dalam 24 JAM TERAKHIR.

PRINSIP PENULISAN:
1. Berbasis Fakta: Rujuk hanya pada fakta, angka, dan pernyataan yang ditemukan pada data berita.
2. Tajam & Terstruktur: Buat analisis yang actionable bagi investor dan pelaku pasar.
3. Transparansi Sumber: Sertakan referensi link ke media terpercaya di bagian akhir laporan.
4. Gaya Bahasa: Bahasa Indonesia profesional, jelas, bernada finansial / pasar modal yang tepat.

STRUKTUR LAPORAN WAJIB (MARKDOWN):
# 📊 Ringkasan Pasar Finansial Terkini (24 Jam Terakhir)
**Fokus Analisis:** [Ringkas pertanyaan/topik utama]
**Waktu Laporan:** [Tanggal & Jam Terkini]

---

### 1. 📌 Ringkasan Eksekutif (Executive Summary)
- Poin-poin sentimen utama pasar hari ini (Risk-on / Risk-off).
- Pergerakan pasar dominan.

### 2. 🏛️ Kebijakan Makroekonomi & Bank Sentral
- Pernyataan pejabat The Fed / ECB / Bank Sentral lainnya, data inflasi, atau arah suku bunga 24 jam terakhir.

### 3. 📈 Pergerakan Indeks Saham & Sektor Utama
- Kinerja Wall Street (S&P 500, Nasdaq, Dow Jones) dan bursa regional.
- Sektor yang memimpin penguatan (gainers) vs pelemahan (laggards).

### 4. 💡 Katalis & Berita Penggerak Pasar Terhangat
- Berita spesifik korporasi (Big Tech, earnings, kebijakan, M&A) atau rilis data ekonomi terkini.

### 5. 🛢️ Komoditas, Valas & Aset Alternatif
- Update harga Minyak Mentah (WTI/Brent), Emas, Indeks Dolar (DXY), atau Kripto (jika ada dalam data).

### 6. ⚖️ Sentimen Pasar & Faktor Risiko ke Depan
- Apa yang perlu diwaspadai pelaku pasar dalam 24-48 jam ke depan.

### 7. 🔗 Sumber Berita Terverifikasi (24 Jam Terakhir)
- Daftar bullet point judul berita dengan tautan link URL sumbernya ([Judul Berita](URL) - Media Sumber)."""

    # Format kumpulan berita agar rapi untuk LLM
    formatted_articles = []
    for idx, r in enumerate(search_results, 1):
        formatted_articles.append(
            f"[{idx}] Judul: {r.get('title')}\n"
            f"    Kategori: {r.get('topic_category', 'General')}\n"
            f"    Sumber: {r.get('source')} | Waktu: {r.get('published_date')}\n"
            f"    URL: {r.get('url')}\n"
            f"    Cuplikan: {r.get('content')}\n"
        )
    articles_str = "\n".join(formatted_articles) if formatted_articles else "Tidak ada artikel berita yang ditemukan."

    user_prompt = f"""Permintaan Pengguna: {user_query}

Tujuan Riset: {plan.get('objective', 'Market Overview')}

Kumpulan Berita 24 Jam Terakhir:
{articles_str}

Silakan susun Laporan Ringkasan Pasar lengkap dan profesional dalam format Markdown."""

    print("\n[Agent 3: Report Agent] Menyusun laporan analisis berita pasar...")
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    report_content = extract_text_content(response)

    print("[Agent 3: Report Agent] Laporan berhasil dibuat!\n")
    return {"market_report": report_content}

