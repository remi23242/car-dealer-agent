from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

CHROMA_PATH = "rag/chroma_db"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

_db: Chroma | None = None


def _get_db() -> Chroma:
    global _db
    if _db is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        _db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
            collection_name="car_inventory",
        )
    return _db


def _build_filter(
    make: str | None,
    max_price: int | None,
    min_price: int | None,
    year: int | None,
) -> dict | None:
    """Build a Chroma where-filter. Multiple conditions → $and clause."""
    clauses: list[dict] = []
    if make:
        clauses.append({"make": {"$eq": make.lower()}})
    if year:
        clauses.append({"year": {"$eq": year}})
    if max_price is not None:
        clauses.append({"msrp": {"$lte": max_price}})
    if min_price is not None:
        clauses.append({"msrp": {"$gte": min_price}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def search_cars(
    query: str,
    make: str | None = None,
    max_price: int | None = None,
    min_price: int | None = None,
    year: int | None = None,
    top_k: int = 4,
) -> list[Document]:
    db = _get_db()
    where = _build_filter(make, max_price, min_price, year)
    return db.similarity_search(query, k=top_k, filter=where)
