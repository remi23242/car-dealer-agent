from typing import Annotated, Literal, Required, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Required[Annotated[list, add_messages]]
    intent: Literal["car_query", "book_appointment", "follow_up", "greeting", "farewell", "social"] | None
    user_email: str | None
    customer_name: str | None
    selected_slot: str | None
    available_slots: list[dict]
    meeting_link: str | None
    rag_context: str | None
    final_response: str | None
    last_results: list[str]
