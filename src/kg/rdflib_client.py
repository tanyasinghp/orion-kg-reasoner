import csv
import hashlib
import json
from typing import Any

from rdflib import Graph, URIRef, Literal, BNode, RDF, RDFS
from rdflib.namespace import XSD

from config import KnowledgeGraphClientConfig
from kg.client import KnowledgeGraphClient
from kg.sparql import SparqlJsonResponse
from trace import trace


AMAZON_NS = "http://amazon/kg/"
AMAZON_P = f"{AMAZON_NS}p_"
AMAZON_R = f"{AMAZON_NS}r_"


def _slugify(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def _rdf_term_type(term: Any) -> str:
    if isinstance(term, URIRef):
        return "uri"
    if isinstance(term, Literal):
        return "literal"
    if isinstance(term, BNode):
        return "bnode"
    return "literal"


class RdfLibKnowledgeGraphClient(KnowledgeGraphClient):
    def __init__(self, config: KnowledgeGraphClientConfig) -> None:
        self.graph = Graph()
        if config.csv_path:
            try:
                self._load_csv(config.csv_path)
            except FileNotFoundError:
                print(f"Warning: CSV not found at {config.csv_path}. KG will be empty.")

    def _load_csv(self, csv_path: str) -> None:
        seen_products: set[str] = set()
        seen_categories: set[str] = set()
        seen_users: set[str] = set()
        product_ratings: dict[str, list[tuple[str, str]]] = {}

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                asins = row.get("asins", "")
                product_id = _slugify(asins or row.get("name", "unknown"))
                product_uri = URIRef(f"{AMAZON_NS}Product/{product_id}")
                product_name = row.get("name", "")

                categories_str = row.get("categories", "")
                first_category = categories_str.split(",")[0].strip() if categories_str else ""

                brand = row.get("brand", "")
                prices_raw = row.get("prices", "")
                price_max = ""
                price_min = ""
                if prices_raw:
                    try:
                        prices_data = json.loads(prices_raw)
                        if isinstance(prices_data, list) and prices_data:
                            price_max = str(prices_data[0].get("amountMax", ""))
                            price_min = str(prices_data[0].get("amountMin", ""))
                    except (json.JSONDecodeError, ValueError, IndexError):
                        pass

                review_rating = row.get("reviews.rating", "").strip()
                num_helpful = row.get("reviews.numHelpful", "").strip()
                user_name = row.get("reviews.username", "").strip()
                review_title = row.get("reviews.title", "").strip()
                review_content = row.get("reviews.text", "").strip()
                review_date = row.get("reviews.date", "").strip()
                do_recommend = row.get("reviews.doRecommend", "").strip()

                g = self.graph

                # --- Product (dedup by asins) ---
                if asins not in seen_products:
                    seen_products.add(asins)
                    g.add((product_uri, RDF.type, URIRef(f"{AMAZON_NS}Product")))
                    g.add((product_uri, URIRef(f"{AMAZON_P}product_id"), Literal(asins)))
                    g.add((product_uri, URIRef(f"{AMAZON_P}name"), Literal(product_name)))
                    g.add((product_uri, RDFS.label, Literal(product_name)))
                    if brand:
                        g.add((product_uri, URIRef(f"{AMAZON_P}about_product"), Literal(brand)))
                    if price_max and price_max != "nan":
                        g.add((product_uri, URIRef(f"{AMAZON_P}actual_price"), Literal(price_max, datatype=XSD.decimal)))
                    if price_min and price_min != "nan":
                        g.add((product_uri, URIRef(f"{AMAZON_P}discounted_price"), Literal(price_min, datatype=XSD.decimal)))
                    product_ratings[asins] = []

                # Track ratings to compute average
                if review_rating and review_rating != "nan":
                    product_ratings[asins].append((review_rating, num_helpful))

                # --- Category ---
                if first_category and first_category not in seen_categories:
                    seen_categories.add(first_category)
                    cat_id = _slugify(first_category)
                    cat_uri = URIRef(f"{AMAZON_NS}Category/{cat_id}")
                    g.add((cat_uri, RDF.type, URIRef(f"{AMAZON_NS}Category")))
                    g.add((cat_uri, URIRef(f"{AMAZON_P}category_name"), Literal(first_category)))
                    g.add((cat_uri, RDFS.label, Literal(first_category)))

                if first_category:
                    cat_id = _slugify(first_category)
                    cat_uri = URIRef(f"{AMAZON_NS}Category/{cat_id}")
                    g.add((product_uri, URIRef(f"{AMAZON_R}belongs_to"), cat_uri))

                # --- User ---
                user_id = _slugify(f"user_{user_name}") if user_name else _slugify(f"anon_{product_id}")
                if user_name and user_name not in seen_users:
                    seen_users.add(user_name)
                    user_uri = URIRef(f"{AMAZON_NS}User/{user_id}")
                    g.add((user_uri, RDF.type, URIRef(f"{AMAZON_NS}User")))
                    g.add((user_uri, URIRef(f"{AMAZON_P}user_id"), Literal(user_id)))
                    g.add((user_uri, URIRef(f"{AMAZON_P}user_name"), Literal(user_name)))
                    g.add((user_uri, RDFS.label, Literal(user_name)))

                # --- Review ---
                if review_content or review_title:
                    review_id_str = _slugify(f"{product_id}_{review_title}_{user_name}")
                    review_uri = URIRef(f"{AMAZON_NS}Review/{review_id_str}")
                    g.add((review_uri, RDF.type, URIRef(f"{AMAZON_NS}Review")))
                    g.add((review_uri, URIRef(f"{AMAZON_P}review_id"), Literal(review_id_str)))
                    if review_title:
                        g.add((review_uri, URIRef(f"{AMAZON_P}review_title"), Literal(review_title)))
                    if review_content:
                        g.add((review_uri, URIRef(f"{AMAZON_P}review_content"), Literal(review_content)))
                    if review_rating and review_rating != "nan":
                        g.add((review_uri, URIRef(f"{AMAZON_P}review_rating"), Literal(review_rating, datatype=XSD.decimal)))
                    if review_date:
                        g.add((review_uri, URIRef(f"{AMAZON_P}review_date"), Literal(review_date)))
                    if num_helpful and num_helpful != "nan":
                        g.add((review_uri, URIRef(f"{AMAZON_P}review_helpful"), Literal(num_helpful, datatype=XSD.integer)))
                    g.add((product_uri, URIRef(f"{AMAZON_R}has_review"), review_uri))
                    if user_name:
                        user_uri = URIRef(f"{AMAZON_NS}User/{user_id}")
                        g.add((review_uri, URIRef(f"{AMAZON_R}written_by"), user_uri))

        # Compute average product ratings
        for asins_key, ratings in product_ratings.items():
            if ratings:
                avg = sum(float(r[0]) for r in ratings) / len(ratings)
                total_helpful = sum(int(r[1]) for r in ratings if r[1] and r[1] != "nan")
                product_id_key = _slugify(asins_key or "unknown")
                product_uri = URIRef(f"{AMAZON_NS}Product/{product_id_key}")
                if asins_key:
                    g.add((product_uri, URIRef(f"{AMAZON_P}rating"), Literal(str(avg), datatype=XSD.decimal)))
                    if total_helpful > 0:
                        g.add((product_uri, URIRef(f"{AMAZON_P}rating_count"), Literal(total_helpful, datatype=XSD.integer)))

    async def execute_sparql_query(self, query: str, **kwargs: Any) -> SparqlJsonResponse:
        trace("rdflib_client.py", f"SPARQL SELECT ({len(query)} chars)", event_type="sparql", data={"query": query[:500]})
        try:
            raw = self.graph.query(query)
            var_names = [str(v) for v in raw.vars]
            bindings: list[dict[str, dict[str, str]]] = []
            for row in raw:
                binding: dict[str, dict[str, str]] = {}
                for var_name, term in zip(var_names, row):
                    if term is not None:
                        entry: dict[str, str] = {
                            "type": _rdf_term_type(term),
                            "value": str(term),
                        }
                        lang = getattr(term, "language", None)
                        if lang:
                            entry["xml:lang"] = lang
                        dt = getattr(term, "datatype", None)
                        if dt:
                            entry["datatype"] = str(dt)
                        binding[var_name] = entry
                bindings.append(binding)
            json_str = json.dumps({
                "head": {"vars": var_names},
                "results": {"bindings": bindings}
            })
            resp = SparqlJsonResponse.model_validate_json(json_str)
            n_bindings = len(resp.results.bindings) if resp.results else 0
            trace("rdflib_client.py", f"  Result: {n_bindings} bindings", event_type="sparql_result",
                  data={"bindings_count": n_bindings})
            return resp
        except Exception as exc:
            trace("rdflib_client.py", f"  SPARQL evaluation error: {exc}", event_type="sparql_error",
                  data={"error": str(exc)})
            return SparqlJsonResponse(head={"vars": []}, results={"bindings": []}, error=str(exc))

    async def execute_sparql_update(self, update: str) -> None:
        self.graph.update(update)
