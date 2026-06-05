import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

DATA_PATH = "rag/data/data.csv"
CHROMA_PATH = "rag/chroma_db"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["Make", "Model", "Year", "MSRP", "Engine HP"])
    df = df.drop_duplicates(subset=["Make", "Model", "Year"])
    df = df[df["MSRP"] > 2000]
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def row_to_document(row: pd.Series) -> Document:
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


def ingest() -> Chroma:
    print("Loading CSV...")
    df = load_and_clean(DATA_PATH)
    print(f"  {len(df)} rows after cleaning")

    print("Converting to documents...")
    docs = [row_to_document(row) for _, row in df.iterrows()]

    print("Embedding and storing in Chroma...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name="car_inventory",
    )
    print(f"  Done — {len(docs)} documents stored in {CHROMA_PATH}")
    return db


if __name__ == "__main__":
    ingest()
