from langgraph.graph import StateGraph, START, END
from agents.state import MarketState
from agents.planner import plan_agent
from agents.searcher import search_agent
from agents.reporter import report_agent


def create_market_news_graph():
    """
    Membangun arsitektur multi-agent LangGraph untuk riset berita pasar 24 jam.
    Alur:
      START -> Planner (Agent 1) -> Searcher (Agent 2) -> Reporter (Agent 3) -> END
    """
    workflow = StateGraph(MarketState)

    # Daftarkan ketiga agent sebagai node
    workflow.add_node("planner", plan_agent)
    workflow.add_node("searcher", search_agent)
    workflow.add_node("reporter", report_agent)

    # Tentukan jalur eksekusi berurutan (Sequential Agentic Pipeline)
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "searcher")
    workflow.add_edge("searcher", "reporter")
    workflow.add_edge("reporter", END)

    # Compile graph
    return workflow.compile()

