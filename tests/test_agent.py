"""
Three end-to-end agent conversations tested against the real graph + Chroma.
No mocks — runs actual Groq LLM + local vector store.
"""
import pytest
from langchain_core.messages import HumanMessage

from agent.graph import graph


def _invoke(message: str, thread_id: str) -> dict:
    """Synchronous wrapper — pytest-asyncio handles the event loop."""
    import asyncio
    config = {"configurable": {"thread_id": thread_id}}
    return asyncio.run(
        graph.ainvoke({"messages": [HumanMessage(content=message)]}, config=config)
    )


# ── conversation 1: Toyota Camry query ───────────────────────────────────────

@pytest.mark.asyncio
async def test_toyota_camry_intent() -> None:
    """Router classifies Toyota Camry question as car_query."""
    config = {"configurable": {"thread_id": "conv1-camry"}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="Tell me about the Toyota Camry")]},
        config=config,
    )
    assert result["intent"] == "car_query"


@pytest.mark.asyncio
async def test_toyota_camry_rag_context() -> None:
    """RAG retrieves Camry docs — context contains 'camry' (case-insensitive)."""
    config = {"configurable": {"thread_id": "conv1-camry-ctx"}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="Tell me about the Toyota Camry")]},
        config=config,
    )
    ctx = (result.get("rag_context") or "").lower()
    assert "camry" in ctx, f"Expected 'camry' in rag_context, got: {ctx[:200]}"


@pytest.mark.asyncio
async def test_toyota_camry_response_mentions_toyota() -> None:
    """Final response mentions Toyota and a price."""
    config = {"configurable": {"thread_id": "conv1-camry-resp"}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="Tell me about the Toyota Camry")]},
        config=config,
    )
    resp = (result.get("final_response") or "").lower()
    assert "toyota" in resp or "camry" in resp, f"Response missing car name: {resp[:200]}"
    assert "$" in resp or "price" in resp or "msrp" in resp, \
        f"Response missing price info: {resp[:200]}"


# ── conversation 2: cars under $25 000 ───────────────────────────────────────

@pytest.mark.asyncio
async def test_under_25k_intent() -> None:
    """Router classifies budget query as car_query."""
    config = {"configurable": {"thread_id": "conv2-price"}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="Show me cars under 25000 dollars")]},
        config=config,
    )
    assert result["intent"] == "car_query"


@pytest.mark.asyncio
async def test_under_25k_results_within_budget() -> None:
    """All retrieved docs have MSRP <= 25 000."""
    config = {"configurable": {"thread_id": "conv2-price-ctx"}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="Show me cars under 25000 dollars")]},
        config=config,
    )
    ctx = result.get("rag_context") or ""
    assert ctx, "rag_context should not be empty for a price-filtered query"

    # extract all MSRP values from context and verify none exceed the cap
    import re
    prices = [int(p.replace(",", "")) for p in re.findall(r"\$([\d,]+)", ctx)]
    over_budget = [p for p in prices if p > 25_000]
    assert not over_budget, f"Results contain prices over $25k: {over_budget}"


@pytest.mark.asyncio
async def test_under_25k_response_has_content() -> None:
    """Response is non-empty and references cars or price."""
    config = {"configurable": {"thread_id": "conv2-price-resp"}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="Show me cars under 25000 dollars")]},
        config=config,
    )
    resp = result.get("final_response") or ""
    assert len(resp) > 20, "Response too short"
    resp_lower = resp.lower()
    assert any(w in resp_lower for w in ("$", "price", "msrp", "car", "vehicle", "model")), \
        f"Response doesn't mention cars or prices: {resp[:200]}"


# ── conversation 3: book appointment ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_book_appointment_intent() -> None:
    """Router classifies booking request as book_appointment."""
    config = {"configurable": {"thread_id": "conv3-book"}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="I want to book a test drive")]},
        config=config,
    )
    assert result["intent"] == "book_appointment"


@pytest.mark.asyncio
async def test_book_appointment_offers_slots() -> None:
    """Turn 1 booking request — graph runs calendar_node, offers available slots."""
    config = {"configurable": {"thread_id": "conv3-book-slots"}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="I want to schedule a test drive for Saturday")]},
        config=config,
    )
    # step 1 ran — slots offered, no booking yet
    assert result.get("meeting_link") is None
    assert result.get("available_slots"), "Expected available_slots to be populated"
    assert result.get("final_response") is not None


@pytest.mark.asyncio
async def test_book_appointment_no_rag_context() -> None:
    """Booking intent should not trigger RAG retrieval."""
    config = {"configurable": {"thread_id": "conv3-book-norag"}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="Can I book a test drive?")]},
        config=config,
    )
    # rag_node never ran — context stays empty/None
    assert not result.get("rag_context"), \
        f"rag_context should be empty for booking intent, got: {result.get('rag_context')}"
