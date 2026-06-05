# RAG Data Pipeline — Kaggle Car Dataset

## Dataset to download

**Car Features and MSRP** by CooperUnion
- URL: https://www.kaggle.com/datasets/CooperUnion/cardataset
- File: `data.csv`
- Size: ~11,900 rows, 16 columns
- Originally scraped from Edmunds and Twitter

---

## How to download

### Option A — Kaggle CLI (recommended)
```bash
# Install kaggle CLI
uv add kaggle

# Put your kaggle.json credentials in:
# Windows: C:\Users\<you>\.kaggle\kaggle.json
# Get it from: kaggle.com → Account → API → Create New Token

kaggle datasets download -d CooperUnion/cardataset -p rag/data/ --unzip
# saves as: rag/data/data.csv
```

### Option B — Manual download
1. Go to https://www.kaggle.com/datasets/CooperUnion/cardataset
2. Click Download
3. Unzip and place `data.csv` in `rag/data/`

---

## Dataset columns

| Column | Type | Example | Used for |
|---|---|---|---|
| Make | string | Toyota | filter + embed |
| Model | string | Camry | filter + embed |
| Year | int | 2015 | filter + metadata |
| Engine Fuel Type | string | regular unleaded | embed |
| Engine HP | float | 203.0 | embed |
| Engine Cylinders | float | 4.0 | embed |
| Transmission Type | string | AUTOMATIC | embed |
| Driven_Wheels | string | front wheel drive | embed |
| Number of Doors | float | 4.0 | embed |
| Market Category | string | Crossover,Hatchback | embed |
| Vehicle Size | string | Compact | embed |
| Vehicle Style | string | 4dr SUV | embed |
| highway MPG | int | 28 | embed |
| city mpg | int | 22 | embed |
| Popularity | int | 1385 | ignore |
| MSRP | int | 21000 | filter + embed |

---

## Ingest script

```python
# rag/ingest.py
import pandas as pd
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import os

DATA_PATH = "rag/data/data.csv"
CHROMA_PATH = "rag/chroma_db"

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # drop rows missing critical fields
    df = df.dropna(subset=["Make", "Model", "Year", "MSRP", "Engine HP"])

    # drop duplicate make/model/year combos (keep first)
    df = df.drop_duplicates(subset=["Make", "Model", "Year"])

    # drop cars priced under $2000 (data quality issue)
    df = df[df["MSRP"] > 2000]

    # clean column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    return df


def row_to_document(row: pd.Series) -> Document:
    """Convert one CSV row into a LangChain Document for embedding."""
    content = f"""
{int(row['year'])} {row['make']} {row['model']}
Price (MSRP): ${int(row['msrp']):,}
Engine: {row.get('engine_hp', 'N/A')} HP, {row.get('engine_cylinders', 'N/A')} cylinders
Fuel type: {row.get('engine_fuel_type', 'N/A')}
Transmission: {row.get('transmission_type', 'N/A')}
Drive: {row.get('driven_wheels', 'N/A')}
Style: {row.get('vehicle_style', 'N/A')}, {row.get('vehicle_size', 'N/A')} size
Doors: {row.get('number_of_doors', 'N/A')}
Fuel economy: {row.get('city_mpg', 'N/A')} city / {row.get('highway_mpg', 'N/A')} highway MPG
Category: {row.get('market_category', 'N/A')}
    """.strip()

    metadata = {
        "make": str(row["make"]).lower(),
        "model": str(row["model"]).lower(),
        "year": int(row["year"]),
        "msrp": int(row["msrp"]),
        "vehicle_style": str(row.get("vehicle_style", "")),
        "vehicle_size": str(row.get("vehicle_size", "")),
        "driven_wheels": str(row.get("driven_wheels", "")),
        "engine_fuel_type": str(row.get("engine_fuel_type", "")),
        "transmission_type": str(row.get("transmission_type", "")),
    }

    return Document(page_content=content, metadata=metadata)


def ingest():
    print("Loading CSV...")
    df = load_and_clean(DATA_PATH)
    print(f"  {len(df)} rows after cleaning")

    print("Converting to documents...")
    docs = [row_to_document(row) for _, row in df.iterrows()]

    print("Embedding and storing in Chroma...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name="car_inventory"
    )
    print(f"  Done — {len(docs)} documents stored in {CHROMA_PATH}")
    return db


if __name__ == "__main__":
    ingest()
```

Run it once before starting the agent:
```bash
uv run python -m rag.ingest
```

---

## Retriever with metadata filtering

```python
# rag/retriever.py
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

CHROMA_PATH = "rag/chroma_db"

def get_retriever():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="car_inventory"
    )
    return db

def search_cars(
    query: str,
    make: str | None = None,
    max_price: int | None = None,
    min_price: int | None = None,
    year: int | None = None,
    top_k: int = 4
) -> list[Document]:
    """
    Semantic search with optional metadata filters.
    
    Examples:
        search_cars("fuel efficient SUV under 30000", max_price=30000)
        search_cars("Toyota sedan automatic", make="toyota")
        search_cars("sports car V8", min_price=40000)
    """
    db = get_retriever()

    # build Chroma where filter
    filters = {}
    if make:
        filters["make"] = make.lower()
    if year:
        filters["year"] = year
    if max_price and min_price:
        filters["$and"] = [
            {"msrp": {"$lte": max_price}},
            {"msrp": {"$gte": min_price}}
        ]
    elif max_price:
        filters["msrp"] = {"$lte": max_price}
    elif min_price:
        filters["msrp"] = {"$gte": min_price}

    results = db.similarity_search(
        query,
        k=top_k,
        filter=filters if filters else None
    )
    return results
```

---

## RAG node in LangGraph

```python
# agent/nodes/rag_node.py
from langchain_core.tools import tool
from rag.retriever import search_cars
from agent.state import AgentState

@tool
def car_search_tool(
    query: str,
    make: str | None = None,
    max_price: int | None = None,
    min_price: int | None = None
) -> str:
    """Search the car inventory. Use when user asks about car specs, pricing, or availability."""
    docs = search_cars(query, make=make, max_price=max_price, min_price=min_price)
    if not docs:
        return "No cars found matching that criteria."
    return "\n\n---\n\n".join(d.page_content for d in docs)


async def rag_node(state: AgentState) -> AgentState:
    """Retrieve relevant car info and add to state context."""
    last_message = state["messages"][-1].content
    results = search_cars(query=last_message, top_k=4)
    context = "\n\n---\n\n".join(d.page_content for d in results)
    return {**state, "rag_context": context}
```

---

## Example queries the RAG handles

| User says | How it filters |
|---|---|
| "Tell me about Toyota Camry" | semantic match on make + model |
| "Show me cars under $25,000" | metadata filter `msrp <= 25000` |
| "I want an SUV with good mileage" | semantic match on vehicle_style + mpg |
| "What Fords do you have?" | metadata filter `make = ford` |
| "Any manual transmission sports cars?" | semantic on transmission + market_category |
| "Cheap reliable sedans" | semantic match, no metadata filter needed |

---

## Chroma collection stats

After ingestion you should have approximately:
- ~11,000 documents (one per unique make/model/year)
- Collection name: `car_inventory`
- Stored at: `rag/chroma_db/` (gitignored)

Re-run `rag/ingest.py` any time you update the dataset.

---

## .gitignore additions

```
rag/data/
rag/chroma_db/
*.csv
```

Do not commit raw data or the vector DB — they're large and can be regenerated.
