"""
Smoke-tests for router_node guard paths (no real LLM calls).

Covers:
  - empty messages list   -> "greeting"
  - missing messages key  -> "greeting"
  - LLM raises            -> "greeting" (not crash)
  - booking continuation  -> "book_appointment" (no LLM needed)

Run: uv run python scripts/test_router.py
"""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

# ── mock heavy deps before importing router ────────────────────────────────────
# Prevents langchain_groq from making network calls or slow init at import time
_mock_groq_module = MagicMock()
sys.modules.setdefault("langchain_groq", _mock_groq_module)

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from agent.router import router_node                         # noqa: E402


def _base_state(**overrides) -> dict:
    base = {
        "messages": [],
        "intent": None,
        "user_email": None,
        "customer_name": None,
        "selected_slot": None,
        "available_slots": [],
        "meeting_link": None,
        "rag_context": None,
        "final_response": None,
        "last_results": [],
    }
    base.update(overrides)
    return base


async def main() -> None:
    results: list[bool] = []
    sep = "=" * 60

    # ── test 1: empty messages list ───────────────────────────────────────────
    r1 = await router_node(_base_state(messages=[]))
    ok1 = r1.get("intent") == "greeting"
    results.append(ok1)
    print(f"\n{sep}")
    print(f"  TEST 1 — empty messages list  [{'PASS' if ok1 else 'FAIL'}]")
    print(f"{sep}")
    print(f"  intent: {r1.get('intent')!r}  (expected 'greeting')")

    # ── test 2: missing messages key entirely ─────────────────────────────────
    r2 = await router_node({})
    ok2 = r2.get("intent") == "greeting"
    results.append(ok2)
    print(f"\n{sep}")
    print(f"  TEST 2 — missing messages key  [{'PASS' if ok2 else 'FAIL'}]")
    print(f"{sep}")
    print(f"  intent: {r2.get('intent')!r}  (expected 'greeting')")

    # ── test 3: LLM raises RuntimeError ──────────────────────────────────────
    def _boom():
        raise RuntimeError("simulated LLM crash")

    import agent.router as _mod
    orig_get_llm = _mod._get_llm
    _mod._get_llm = _boom
    try:
        r3 = await router_node(_base_state(messages=[HumanMessage(content="hello")]))
    finally:
        _mod._get_llm = orig_get_llm
    ok3 = r3.get("intent") == "greeting"
    results.append(ok3)
    print(f"\n{sep}")
    print(f"  TEST 3 — LLM crash -> greeting  [{'PASS' if ok3 else 'FAIL'}]")
    print(f"{sep}")
    print(f"  intent: {r3.get('intent')!r}  (expected 'greeting')")

    # ── test 4: booking continuation (slots offered, no slot selected) ────────
    r4 = await router_node(_base_state(
        messages=[HumanMessage(content="the first one")],
        available_slots=[{"slot": "Monday 10 AM", "iso": "2026-06-08T10:00:00+05:00"}],
        selected_slot=None,
    ))
    ok4 = r4.get("intent") == "book_appointment"
    results.append(ok4)
    print(f"\n{sep}")
    print(f"  TEST 4 — booking continuation (slot select)  [{'PASS' if ok4 else 'FAIL'}]")
    print(f"{sep}")
    print(f"  intent: {r4.get('intent')!r}  (expected 'book_appointment')")

    # ── test 5: booking continuation (slot chosen, awaiting name) ─────────────
    r5 = await router_node(_base_state(
        messages=[HumanMessage(content="my name is Alex")],
        selected_slot="2026-06-08T10:00:00+05:00",
        user_email=None,
    ))
    ok5 = r5.get("intent") == "book_appointment"
    results.append(ok5)
    print(f"\n{sep}")
    print(f"  TEST 5 — booking continuation (collect name)  [{'PASS' if ok5 else 'FAIL'}]")
    print(f"{sep}")
    print(f"  intent: {r5.get('intent')!r}  (expected 'book_appointment')")

    # ── test 6: LLM returns valid intent ─────────────────────────────────────
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "car_query"
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    import agent.router as _mod2
    orig2 = _mod2._get_llm
    _mod2._get_llm = lambda: mock_llm
    try:
        r6 = await router_node(_base_state(messages=[HumanMessage(content="tell me about the Tesla Model S")]))
    finally:
        _mod2._get_llm = orig2
    ok6 = r6.get("intent") == "car_query"
    results.append(ok6)
    print(f"\n{sep}")
    print(f"  TEST 6 — LLM returns car_query  [{'PASS' if ok6 else 'FAIL'}]")
    print(f"{sep}")
    print(f"  intent: {r6.get('intent')!r}  (expected 'car_query')")

    # ── summary ───────────────────────────────────────────────────────────────
    passed = sum(results)
    total = len(results)
    print(f"\n{sep}")
    print(f"  {passed}/{total} tests passed {'OK' if passed == total else 'FAIL'}")
    print(f"{sep}\n")


if __name__ == "__main__":
    asyncio.run(main())
