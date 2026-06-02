#!/usr/bin/env python3
"""Load Amazon Product Reviews CSV into Apache Jena Fuseki as RDF triples."""

import os
import sys
import csv
import hashlib
import urllib.parse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

FUSEKI_URL = os.environ.get("FUSEKI_URL", "http://localhost:3030")
FUSEKI_DATASET = os.environ.get("FUSEKI_DATASET", "amazon")
CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "amazon.csv")


AMAZON_NS = "http://amazon/kg/"
AMAZON_P = f"{AMAZON_NS}p_"
AMAZON_R = f"{AMAZON_NS}r_"
RDF_TYPE = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
RDFS_LABEL = "<http://www.w3.org/2000/01/rdf-schema#label>"


def slugify(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def to_sparql_literal(value: str, datatype: str | None = None) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    if datatype:
        return f'"{escaped}"^^<{datatype}>'
    return f'"{escaped}"'


def generate_triples() -> list[str]:
    triples: list[str] = []

    BASE_UPDATE = f"PREFIX : <{AMAZON_NS}>"
    triples.append(f"DROP ALL")
    triples.append(BASE_UPDATE)

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        seen_categories: set[str] = set()
        seen_users: set[str] = set()

        for row in reader:
            product_id = slugify(row.get("product_id", row.get("name", "unknown")))
            product_uri = f"<{AMAZON_NS}Product/{product_id}>"
            product_name = row.get("product_name", row.get("name", ""))
            category = row.get("category", "")
            about_product = row.get("about_product", row.get("description", ""))
            discounted_price = row.get("discounted_price", "").replace("₹", "").replace(",", "").strip()
            actual_price = row.get("actual_price", "").replace("₹", "").replace(",", "").strip()
            discount_pct = row.get("discount_percentage", "").replace("%", "").strip()
            rating = row.get("rating", "").strip()
            rating_count = row.get("rating_count", "").replace(",", "").strip()
            user_id = row.get("user_id", "")
            user_name = row.get("user_name", "")
            review_id = row.get("review_id", slugify(f"review_{product_id}"))
            review_title = row.get("review_title", "")
            review_content = row.get("review_content", "")

            # Product triple
            prod_triples = f"""
DELETE {{ ?s ?p ?o }}
WHERE {{ ?s ?p ?o . FILTER(?s = {product_uri}) }};

INSERT DATA {{
  {product_uri} {RDF_TYPE} <{AMAZON_NS}Product> .
  {product_uri} <{AMAZON_P}product_id> {to_sparql_literal(product_id)} .
  {product_uri} <{AMAZON_P}name> {to_sparql_literal(product_name)} .
  {product_uri} {RDFS_LABEL} {to_sparql_literal(product_name)} .
"""
            if about_product:
                prod_triples += f"  {product_uri} <{AMAZON_P}about_product> {to_sparql_literal(about_product)} .\n"
            if discounted_price and discounted_price != "nan":
                prod_triples += f"  {product_uri} <{AMAZON_P}discounted_price> {to_sparql_literal(discounted_price, 'http://www.w3.org/2001/XMLSchema#decimal')} .\n"
            if actual_price and actual_price != "nan":
                prod_triples += f"  {product_uri} <{AMAZON_P}actual_price> {to_sparql_literal(actual_price, 'http://www.w3.org/2001/XMLSchema#decimal')} .\n"
            if discount_pct and discount_pct != "nan":
                prod_triples += f"  {product_uri} <{AMAZON_P}discount_percentage> {to_sparql_literal(discount_pct, 'http://www.w3.org/2001/XMLSchema#decimal')} .\n"
            if rating and rating != "nan":
                prod_triples += f"  {product_uri} <{AMAZON_P}rating> {to_sparql_literal(rating, 'http://www.w3.org/2001/XMLSchema#decimal')} .\n"
            if rating_count and rating_count != "nan":
                prod_triples += f"  {product_uri} <{AMAZON_P}rating_count> {to_sparql_literal(rating_count, 'http://www.w3.org/2001/XMLSchema#integer')} .\n"
            prod_triples += "}"
            triples.append(prod_triples)

            # Category triple
            if category and category not in seen_categories:
                seen_categories.add(category)
                cat_id = slugify(category)
                cat_uri = f"<{AMAZON_NS}Category/{cat_id}>"
                cat_triples = f"""
INSERT DATA {{
  {cat_uri} {RDF_TYPE} <{AMAZON_NS}Category> .
  {cat_uri} <{AMAZON_P}category_name> {to_sparql_literal(category)} .
  {cat_uri} {RDFS_LABEL} {to_sparql_literal(category)} .
}}
"""
                triples.append(cat_triples)
                # belongs_to relation
                cat_id_slug = slugify(category)
                cat_uri_final = f"<{AMAZON_NS}Category/{cat_id_slug}>"
                rel_triple = f"""
INSERT DATA {{
  {product_uri} <{AMAZON_R}belongs_to> {cat_uri_final} .
}}
"""
                triples.append(rel_triple)

            # User triple
            if user_id and user_id not in seen_users:
                seen_users.add(user_id)
                user_uri = f"<{AMAZON_NS}User/{slugify(user_id)}>"
                user_triples = f"""
INSERT DATA {{
  {user_uri} {RDF_TYPE} <{AMAZON_NS}User> .
  {user_uri} <{AMAZON_P}user_id> {to_sparql_literal(user_id)} .
  {user_uri} <{AMAZON_P}user_name> {to_sparql_literal(user_name)} .
  {user_uri} {RDFS_LABEL} {to_sparql_literal(user_name)} .
}}
"""
                triples.append(user_triples)

            # Review triple
            if review_content:
                review_uri = f"<{AMAZON_NS}Review/{slugify(f'{product_id}_{review_id}')}>"
                review_triples = f"""
INSERT DATA {{
  {review_uri} {RDF_TYPE} <{AMAZON_NS}Review> .
  {review_uri} <{AMAZON_P}review_id> {to_sparql_literal(review_id)} .
  {review_uri} <{AMAZON_P}review_title> {to_sparql_literal(review_title)} .
  {review_uri} <{AMAZON_P}review_content> {to_sparql_literal(review_content)} .
  {product_uri} <{AMAZON_R}has_review> {review_uri} .
"""
                if user_id:
                    user_uri_rev = f"<{AMAZON_NS}User/{slugify(user_id)}>"
                    review_triples += f"  {review_uri} <{AMAZON_R}written_by> {user_uri_rev} .\n"
                review_triples += "}"
                triples.append(review_triples)

            # If no review but product exists, still add belongs_to
            if category:
                cat_id_slug = slugify(f'cat_{category}')
                pass  # handled above

    return triples


def load_to_fuseki(update_queries: list[str]) -> None:
    import httpx

    for i, query in enumerate(update_queries):
        if not query.strip():
            continue
        url = f"{FUSEKI_URL}/{FUSEKI_DATASET}/update"
        try:
            with httpx.Client() as client:
                resp = client.post(
                    url,
                    data={"update": query},
                    headers={"Accept": "application/json"},
                    timeout=120.0
                )
                resp.raise_for_status()
            if i % 50 == 0:
                print(f"  Processed {i}/{len(update_queries)} updates...")
        except Exception as e:
            print(f"  Error at update {i}: {e}")


if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        print(f"CSV not found at: {CSV_PATH}")
        print("Run scripts/download_data.py first.")
        sys.exit(1)

    print("Generating RDF triples from CSV...")
    updates = generate_triples()
    print(f"Generated {len(updates)} SPARQL UPDATE statements.")

    print(f"Loading into Fuseki at {FUSEKI_URL}/{FUSEKI_DATASET}...")
    # Clear existing data and load
    initial_clear = f"DROP ALL"
    import httpx
    try:
        with httpx.Client() as client:
            resp = client.post(
                f"{FUSEKI_URL}/{FUSEKI_DATASET}/update",
                data={"update": initial_clear},
                timeout=120.0
            )
            resp.raise_for_status()
        print("Cleared existing data.")
    except Exception as e:
        print(f"Warning: could not clear: {e}")

    load_to_fuseki(updates)
    print("Done!")
