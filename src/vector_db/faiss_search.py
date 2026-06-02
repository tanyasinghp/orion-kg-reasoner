import json
import os
from typing import Any

import numpy as np

from config import VectorSearchConfig
from vector_db.vector_search import VectorSearch, EntityEmbedding

import faiss


class FaissVectorSearch(VectorSearch):
    def __init__(self, config: VectorSearchConfig) -> None:
        self.config = config
        self.encoder = None
        self.index: faiss.Index | None = None
        self.uri_mapping: list[dict[str, Any]] = []
        self._load_index()

    def _get_encoder(self):
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(self.config.embedding_model_name)
        return self.encoder

    def _load_index(self) -> None:
        index_path = self.config.faiss_index_path
        mapping_path = self.config.faiss_mapping_path
        if os.path.exists(index_path) and os.path.exists(mapping_path):
            self.index = faiss.read_index(index_path)
            with open(mapping_path) as f:
                self.uri_mapping = json.load(f)

    def build_index(self, entities: list[dict[str, Any]]) -> None:
        texts = []
        self.uri_mapping = []
        for entity in entities:
            text = entity.get("text", "")
            if not text:
                continue
            texts.append(text)
            self.uri_mapping.append({
                "uri": entity["uri"],
                "name": entity.get("name", ""),
                "types": entity.get("types", []),
            })

        if not texts:
            return

        encoder = self._get_encoder()
        embeddings = encoder.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(np.array(embeddings).astype(np.float32))

        os.makedirs(os.path.dirname(self.config.faiss_index_path), exist_ok=True)
        faiss.write_index(self.index, self.config.faiss_index_path)
        with open(self.config.faiss_mapping_path, "w") as f:
            json.dump(self.uri_mapping, f, indent=2)

    async def search_entities(
        self,
        question: str,
        include_types: bool = False,
        limit: int = 10,
        **kwargs: Any
    ) -> list[EntityEmbedding]:
        if self.index is None:
            return []

        encoder = self._get_encoder()
        query_embedding = encoder.encode([question], normalize_embeddings=True)
        scores, indices = self.index.search(np.array(query_embedding).astype(np.float32), limit)

        results: list[EntityEmbedding] = []
        for i, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(self.uri_mapping):
                continue
            mapping = self.uri_mapping[idx]
            results.append(EntityEmbedding(
                uri=mapping["uri"],
                name=mapping.get("name", ""),
                types=mapping.get("types", []) if include_types else [],
                similarity=float(scores[0][i]),
            ))

        return results
