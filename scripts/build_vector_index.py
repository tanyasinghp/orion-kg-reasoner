#!/usr/bin/env python3
"""Build FAISS vector index from Amazon Product Reviews entity descriptions."""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "amazon.csv")
INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "indices")
INDEX_PATH = os.path.join(INDEX_DIR, "entity_embeddings.index")
MAPPING_PATH = os.path.join(INDEX_DIR, "uri_mapping.json")


def read_entities_from_csv() -> list[dict]:
    import csv
    entities = []
    seen: set[str] = set()

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            product_id = row.get("product_id", row.get("name", ""))
            product_name = row.get("product_name", row.get("name", ""))
            about_product = row.get("about_product", row.get("description", ""))
            category = row.get("category", "")
            rating = row.get("rating", "")
            discounted_price = row.get("discounted_price", "")

            text_parts = [product_name]
            if category:
                text_parts.append(f"Category: {category}")
            if about_product:
                text_parts.append(about_product)
            if rating:
                text_parts.append(f"Rating: {rating}")
            if discounted_price:
                text_parts.append(f"Price: {discounted_price}")

            text = ". ".join(text_parts)

            import hashlib
            uri = f"http://amazon/kg/Product/{hashlib.md5(product_id.encode()).hexdigest()[:12]}"

            if uri not in seen:
                seen.add(uri)
                entities.append({
                    "uri": uri,
                    "name": product_name,
                    "types": ["Product"],
                    "text": text,
                })

    print(f"Read {len(entities)} entities from CSV.")
    return entities


def build_and_save_index(entities: list[dict]) -> None:
    from vector_db.faiss_search import FaissVectorSearch
    from config import VectorSearchConfig

    config = VectorSearchConfig(
        faiss_index_path=INDEX_PATH,
        faiss_mapping_path=MAPPING_PATH,
        embedding_model_name="all-MiniLM-L6-v2"
    )
    search = FaissVectorSearch(config)
    search.build_index(entities)
    print(f"FAISS index saved to: {INDEX_PATH}")
    print(f"URI mapping saved to: {MAPPING_PATH}")


if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        print(f"CSV not found at: {CSV_PATH}")
        print("Run scripts/download_data.py first.")
        sys.exit(1)

    os.makedirs(INDEX_DIR, exist_ok=True)
    entities = read_entities_from_csv()
    build_and_save_index(entities)
    print("Done!")
