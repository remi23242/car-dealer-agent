from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.nodes.calendar_node import calendar_node
from agent.nodes.rag_node import rag_node
from agent.nodes.response_node import response_node
from agent.router import router_node
from agent.state import AgentState

_ROUTE_MAP = {
    "car_query": "rag_node",
    "book_appointment": "calendar_node",
    "follow_up": "response_node",
    "greeting": "response_node",
    "farewell": "response_node",
    "social": "response_node",
}


def _route(state: AgentState) -> str:
    intent = state.get("intent") or "social"
    return intent if intent in _ROUTE_MAP else "social"


def build_graph(checkpointer: MemorySaver | None = None) -> CompiledStateGraph:
    g = StateGraph(AgentState)

    g.add_node("router_node", router_node)
    g.add_node("rag_node", rag_node)
    g.add_node("calendar_node", calendar_node)
    g.add_node("response_node", response_node)

    g.add_edge(START, "router_node")
    g.add_conditional_edges("router_node", _route, _ROUTE_MAP)
    g.add_edge("rag_node", "response_node")
    g.add_edge("calendar_node", "response_node")
    g.add_edge("response_node", END)

    return g.compile(checkpointer=checkpointer or MemorySaver())


# module-level singleton — import this in main.py and tests
graph: CompiledStateGraph = build_graph()
