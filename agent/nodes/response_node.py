import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic_settings import BaseSettings, SettingsConfigDict

from agent.state import AgentState

log = structlog.get_logger()

_FOLLOWUP_SYSTEM = """\
You are a car dealership voice assistant.
The user wants more detail about a car you previously described.
Respond in 1-2 natural conversational sentences. No markdown, no lists."""

_CONVERSATIONAL_SYSTEM = """\
You are Alex, a friendly voice assistant at Premier Auto Dealership on a live call.
You have memory of the entire conversation — use it to remember names, preferences, and context.
Reply in 1-2 short sentences. No markdown.
Rules:
- If the user mentions their name, acknowledge it warmly and use it going forward.
- If the user asks what you can do or who you are: you help customers find their perfect car and book test drives.
- If the user asks about something off-topic (emotions, weather, personal questions): respond briefly and warmly, then offer to help with cars or booking.
- Never repeat the exact same phrase twice. Be natural, warm, and varied.
CRITICAL: NEVER ask for appointment dates, times, names for booking, or attempt to create or confirm a booking yourself.
If the user agrees to schedule ("yeah", "sure", "ok") just say "Great! Let me check our available times." — the booking system handles everything else automatically."""

_SYSTEM_BASE = """\
You are Alex, a helpful car dealership voice assistant on a live call.
Use conversation history to remember context. 2 sentences max. No markdown."""


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    groq_api_key: str


_llm: ChatGroq | None = None


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        s = LLMSettings()
        _llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            streaming=True,
            api_key=s.groq_api_key,
        )
    return _llm


async def _llm_stream(messages: list, log_key: str) -> str:
    answer = ""
    async for event in _get_llm().astream_events(
        messages, config={"tags": ["tts-stream"]}, version="v2"
    ):
        if event["event"] == "on_chat_model_stream":
            token = event["data"]["chunk"].content
            if token:
                answer += token
    answer = answer.strip()
    log.info(log_key, chars=len(answer))
    return answer


async def response_node(state: AgentState) -> dict:
    intent = state.get("intent")
    history = list(state["messages"])

    # farewell → fast fixed response
    if intent == "farewell":
        answer = "Thanks for calling Premier Auto, have a great day!"
        log.info("response_node.farewell")
        return {"messages": [AIMessage(content=answer)], "final_response": answer}

    # fresh greeting (very first message) → fast fixed response
    if intent == "greeting" and len(history) <= 1:
        answer = "Thanks for calling! How can I help you today?"
        log.info("response_node.greeting.fresh")
        return {"messages": [AIMessage(content=answer)], "final_response": answer}

    # social or mid-call greeting → LLM with full conversation history
    if intent in ("social", "greeting"):
        messages_for_llm = [SystemMessage(content=_CONVERSATIONAL_SYSTEM)] + history
        answer = await _llm_stream(messages_for_llm, "response_node.social_llm")
        return {"messages": [AIMessage(content=answer)], "final_response": answer}

    # follow_up → describe last retrieved cars with stored inventory context
    if intent == "follow_up":
        last_results = state.get("last_results") or []
        if not last_results:
            # No inventory context — fall back to conversational LLM
            messages_for_llm = [SystemMessage(content=_CONVERSATIONAL_SYSTEM)] + history
            answer = await _llm_stream(messages_for_llm, "response_node.follow_up.no_context")
            return {"messages": [AIMessage(content=answer)], "final_response": answer}

        context = "\n\n---\n\n".join(last_results)
        messages = [
            SystemMessage(content=f"{_FOLLOWUP_SYSTEM}\n\nINVENTORY:\n{context}"),
            HumanMessage(content=history[-1].content),
        ]
        answer = await _llm_stream(messages, "response_node.follow_up.done")
        return {"messages": [AIMessage(content=answer)], "final_response": answer}

    # car_query / book_appointment: upstream node already set final_response — pass through
    if state.get("final_response"):
        log.info("response_node.passthrough")
        return {}

    # generic fallback: unexpected path — LLM with history
    system_parts = [_SYSTEM_BASE]
    meeting_link = state.get("meeting_link")
    if meeting_link:
        slot = state.get("selected_slot", "your selected time")
        system_parts.append(
            f"\n\nAPPOINTMENT CONFIRMED:\nSlot: {slot}\nGoogle Meet: {meeting_link}"
        )

    messages = [SystemMessage(content="\n".join(system_parts))] + history
    answer = await _llm_stream(messages, "response_node.fallback")
    return {"messages": [AIMessage(content=answer)], "final_response": answer}
