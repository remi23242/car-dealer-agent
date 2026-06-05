import asyncio
import re
import time
from collections import OrderedDict
from dataclasses import dataclass

import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from pydantic_settings import BaseSettings, SettingsConfigDict

from agent.state import AgentState
from rag.retriever import search_cars

log = structlog.get_logger()


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    groq_api_key: str


_llm_synth: ChatGroq | None = None


def _get_synth_llm() -> ChatGroq:
    global _llm_synth
    if _llm_synth is None:
        s = LLMSettings()
        _llm_synth = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            streaming=True,
            api_key=s.groq_api_key,
        )
    return _llm_synth


# ── regex param extraction (zero LLM calls, ~0ms) ────────────────────────────

_MAKES = {
    "toyota", "honda", "ford", "chevrolet", "chevy", "bmw", "mercedes",
    "audi", "volkswagen", "vw", "nissan", "hyundai", "kia", "subaru",
    "mazda", "lexus", "acura", "infiniti", "genesis", "volvo", "jeep",
    "ram", "gmc", "cadillac", "lincoln", "chrysler", "dodge", "tesla",
    "rivian", "lucid", "porsche", "ferrari", "lamborghini", "maserati",
    "alfa romeo", "fiat", "mitsubishi", "suzuki", "isuzu", "buick",
    "land rover", "jaguar", "mini", "bentley", "rolls royce",
}

# normalise aliases to canonical name used in metadata
_MAKE_ALIASES = {"chevy": "chevrolet", "vw": "volkswagen"}

_RE_YEAR = re.compile(r"\b(19[5-9]\d|20[0-3]\d)\b")
_RE_STYLE = re.compile(
    r"\b(suv|sedan|coupe|truck|hatchback|convertible|van|pickup|wagon|crossover|minivan)\b",
    re.I,
)
_RE_PRICE_UNDER = re.compile(r"under\s+\$?([\d,]+)k?", re.I)
_RE_PRICE_OVER = re.compile(r"(?:over|above|more than)\s+\$?([\d,]+)k?", re.I)
_RE_PRICE_RANGE = re.compile(r"(?:between\s+)?\$?([\d,]+)k?\s*(?:to|and|-)\s*\$?([\d,]+)k?", re.I)
_RE_PRICE_BUDGET = re.compile(r"budget\s+(?:of\s+)?\$?([\d,]+)k?", re.I)


def _parse_price(raw: str) -> int:
    val = int(raw.replace(",", ""))
    return val * 1000 if val < 1000 else val


@dataclass
class SearchParams:
    query: str
    make: str | None = None
    min_price: int | None = None
    max_price: int | None = None
    year: int | None = None
    vehicle_style: str | None = None


def _extract_params(user_text: str) -> SearchParams:
    text = user_text.lower()

    # make
    make: str | None = None
    for m in sorted(_MAKES, key=len, reverse=True):  # longest match first
        if m in text:
            make = _MAKE_ALIASES.get(m, m)
            break

    # year
    year: str | None = None
    if m := _RE_YEAR.search(text):
        year = int(m.group(1))

    # price
    min_price: int | None = None
    max_price: int | None = None
    if m := _RE_PRICE_RANGE.search(text):
        min_price = _parse_price(m.group(1))
        max_price = _parse_price(m.group(2))
    elif m := _RE_PRICE_UNDER.search(text):
        max_price = _parse_price(m.group(1))
    elif m := _RE_PRICE_OVER.search(text):
        min_price = _parse_price(m.group(1))
    elif m := _RE_PRICE_BUDGET.search(text):
        max_price = _parse_price(m.group(1))

    # vehicle style
    vehicle_style: str | None = None
    if m := _RE_STYLE.search(text):
        vehicle_style = m.group(1).lower()

    return SearchParams(
        query=user_text,
        make=make,
        min_price=min_price,
        max_price=max_price,
        year=year,
        vehicle_style=vehicle_style,
    )


# ── query answer cache (LRU, max 3 entries) ───────────────────────────────────

_ANSWER_CACHE: OrderedDict[str, str] = OrderedDict()
_CACHE_MAX = 3


def _cache_key(p: SearchParams) -> str:
    return f"{p.query}|{p.make}|{p.min_price}|{p.max_price}|{p.year}|{p.vehicle_style}"


def _cache_get(key: str) -> tuple[str, str] | None:
    if key not in _ANSWER_CACHE:
        return None
    _ANSWER_CACHE.move_to_end(key)
    return _ANSWER_CACHE[key]


def _cache_put(key: str, answer: str, context: str) -> None:
    if key in _ANSWER_CACHE:
        _ANSWER_CACHE.move_to_end(key)
    else:
        if len(_ANSWER_CACHE) >= _CACHE_MAX:
            _ANSWER_CACHE.popitem(last=False)
    _ANSWER_CACHE[key] = (answer, context)


# ── synthesis ─────────────────────────────────────────────────────────────────

_ANSWER_SYSTEM = """\
You are a concise car dealership assistant answering on a voice call.
Use ONLY the inventory results below to answer the user's specific question.
If the results do not contain information relevant to the question, say so in one sentence and stop — do NOT list cars.
If the results ARE relevant, mention the car name, price, and one key spec. 2 sentences max. No markdown. No lists.
Never use abbreviations — always write horsepower not HP, miles per gallon not MPG, all wheel drive not AWD, front wheel drive not FWD, rear wheel drive not RWD, 4 door not 4dr, V8 engine not V8.
Write ALL numbers as spoken English words: "one thousand and one horsepower" not "1001 horsepower", "two hundred horsepower" not "200 horsepower". For prices write digits followed by the word dollars: "1,705,769 dollars"."""


async def _synthesize(user_text: str, context: str) -> str:
    messages = [
        SystemMessage(content=f"{_ANSWER_SYSTEM}\n\nINVENTORY:\n{context}"),
        HumanMessage(content=user_text),
    ]
    t_synth = time.monotonic()
    log.info("timing.synth_start")
    full = ""
    first_token_logged = False
    async for event in _get_synth_llm().astream_events(
        messages,
        config={"tags": ["tts-stream"]},
        version="v2",
    ):
        if event["event"] == "on_chat_model_stream":
            token = event["data"]["chunk"].content
            if token:
                if not first_token_logged:
                    log.info("timing.first_token", ms=round((time.monotonic() - t_synth) * 1000))
                    first_token_logged = True
                full += token
    return full.strip()


# ── LangGraph tool (for future tool-calling integration) ──────────────────────

@tool
async def car_search_tool(
    query: str,
    make: str | None = None,
    max_price: int | None = None,
    min_price: int | None = None,
) -> str:
    """Search car inventory. Use when user asks about specs, pricing, or availability."""
    loop = asyncio.get_running_loop()
    docs = await loop.run_in_executor(
        None, lambda: search_cars(query, make=make, max_price=max_price, min_price=min_price)
    )
    if not docs:
        return "No cars found matching that criteria."
    return "\n\n---\n\n".join(d.page_content for d in docs)


# ── node ──────────────────────────────────────────────────────────────────────

async def rag_node(state: AgentState) -> dict:
    user_text = state["messages"][-1].content
    t0 = time.monotonic()

    # 1. extract structured search params (regex, ~0ms)
    params = _extract_params(user_text)
    log.info("timing.params_done", ms=round((time.monotonic() - t0) * 1000),
             make=params.make, style=params.vehicle_style,
             min_price=params.min_price, max_price=params.max_price)

    # 2. cache check — skip vector search + synthesis on repeat questions
    key = _cache_key(params)
    cached = _cache_get(key)
    if cached:
        answer, context = cached
        log.info("rag_node.cache_hit", query=params.query)
        return {
            "rag_context": context,
            "messages": [AIMessage(content=answer)],
            "final_response": answer,
        }

    # Prepend vehicle_style so vector search is anchored to the body type.
    search_query = f"{params.vehicle_style} {params.query}" if params.vehicle_style else params.query

    # 3. vector search with metadata filters (top_k=3 for lower latency)
    # Run in executor — HuggingFace embed + Chroma query are sync/blocking.
    loop = asyncio.get_running_loop()
    docs = await loop.run_in_executor(
        None,
        lambda: search_cars(
            query=search_query,
            make=params.make,
            min_price=params.min_price,
            max_price=params.max_price,
            year=params.year,
            top_k=3,
        ),
    )
    log.info("timing.rag_done", ms=round((time.monotonic() - t0) * 1000), n_docs=len(docs))

    if not docs:
        answer = "I couldn't find any cars matching that description in our inventory."
        log.info("rag_node.no_results")
        return {
            "rag_context": "",
            "messages": [AIMessage(content=answer)],
            "final_response": answer,
        }

    last_results = [d.page_content for d in docs]
    context = "\n\n---\n\n".join(last_results)

    # 4. synthesize — astream_events emits on_chat_model_stream tokens visible
    #    to call_handler's graph.astream_events for sentence-level TTS streaming
    answer = await _synthesize(user_text, context)
    log.info("timing.synth_done", ms=round((time.monotonic() - t0) * 1000), chars=len(answer))

    _cache_put(key, answer, context)

    return {
        "rag_context": context,
        "last_results": last_results,
        "messages": [AIMessage(content=answer)],
        "final_response": answer,
    }
