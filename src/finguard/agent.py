"""Optional LangGraph wrapper for the deterministic investigation workflow."""

from typing import TypedDict

from finguard.investigation import investigate
from finguard.rag import ComplianceRetriever


class InvestigationState(TypedDict, total=False):
    transaction: dict
    question: str
    result: dict


def build_investigation_graph(retriever: ComplianceRetriever):
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as error:
        raise RuntimeError("Install the LLM extra with `pip install -e '.[llm]'`.") from error

    def collect_evidence(state: InvestigationState) -> InvestigationState:
        return {
            "result": investigate(
                state["transaction"],
                retriever,
                state.get("question", ""),
            )
        }

    graph = StateGraph(InvestigationState)
    graph.add_node("collect_evidence", collect_evidence)
    graph.add_edge(START, "collect_evidence")
    graph.add_edge("collect_evidence", END)
    return graph.compile()
