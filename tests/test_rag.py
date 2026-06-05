import pytest


@pytest.mark.asyncio
async def test_search_cars_returns_results() -> None:
    from rag.retriever import search_cars
    results = search_cars("Toyota sedan", top_k=2)
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_search_cars_price_filter() -> None:
    from rag.retriever import search_cars
    results = search_cars("cheap car", max_price=20000, top_k=4)
    for doc in results:
        assert doc.metadata.get("msrp", 0) <= 20000
