import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic_settings import BaseSettings, SettingsConfigDict

from agent.state import AgentState

log = structlog.get_logger()

_INTENT_SYSTEM = """\
You are an intent classifier for a car dealership voice assistant.
Classify the user's message into exactly one intent:
- car_query        : anything about cars, prices, specs, inventory, models, features — questions about car stock/inventory only
- book_appointment : wants to schedule, book, meet, or test drive; OR asks about available dates/times/slots for an appointment ("what times are available", "which dates work", "when can I come in")
- follow_up        : short references to a prior answer ("tell me more", "that one", "the first one", "elaborate", "go on", "what else")
- greeting         : hi, hello, hey, good morning, good evening, how are you, what's up
- farewell         : bye, goodbye, done, thank you bye, see you, have a good day, take care
- social           : everything else — emotions, random questions, weather, jokes, insults, gibberish, apologies, personal questions, anything not car-related

Reply with ONLY the intent label — no explanation, no punctuation."""


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    groq_api_key: str


_llm: ChatGroq | None = None


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        settings = LLMSettings()
        _llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=settings.groq_api_key,
        )
    return _llm


async def router_node(state: AgentState) -> dict:
    # Booking continuation — skip LLM when mid-flow
    available_slots = state.get("available_slots") or []
    selected_slot = state.get("selected_slot")
    user_email = state.get("user_email")
    if available_slots and not selected_slot:
        log.info("router.booking_continuation", step="select_slot")
        return {"intent": "book_appointment"}
    if selected_slot and not user_email:
        log.info("router.booking_continuation", step="collect_email")
        return {"intent": "book_appointment"}

    messages = state.get("messages") or []
    if not messages:
        log.warning("router.empty_messages")
        return {"intent": "greeting"}

    try:
        last = messages[-1].content

        # Include the immediately preceding AI message so the router can classify
        # short affirmatives ("yeah", "sure", "ok") correctly in context.
        router_msgs: list = [SystemMessage(content=_INTENT_SYSTEM)]
        if len(messages) >= 2:
            prev = messages[-2]
            if isinstance(prev, AIMessage):
                router_msgs.append(AIMessage(content=prev.content[:300]))
        router_msgs.append(HumanMessage(content=last))

        response = await _get_llm().ainvoke(router_msgs)
        raw = response.content.strip().lower()

        if "car_query" in raw:
            intent = "car_query"
        elif "book_appointment" in raw:
            intent = "book_appointment"
        elif "follow_up" in raw:
            intent = "follow_up"
        elif "greeting" in raw:
            intent = "greeting"
        elif "farewell" in raw:
            intent = "farewell"
        else:
            intent = "social"

        log.info("router.intent", raw=raw, classified=intent)
        return {"intent": intent}

    except IndexError:
        log.error("router.index_error")
        return {"intent": "greeting"}
    except Exception as exc:
        log.error("router.crash", error=str(exc))
        return {"intent": "greeting"}
