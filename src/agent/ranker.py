from collections.abc import Iterable

import yaml

from agent.agent_context import Context, Entity, Relation
from trace import trace


class Ranker:
    """A class providing functionality to rank entities and relations from the output of tool calls.

    Attributes:
        weights: Scoring weights for different categories of entities and relations.
    """

    def __init__(self, weights: dict[str, float]) -> None:
        """Create a ranker for entities and relations from the output of tool calls.

        Args:
            weights: Scoring weights for different categories of entities and relations.
        """
        self.weights = weights

    @staticmethod
    def from_yaml(path: str) -> "Ranker":
        """Load ranking weights from a YAML file.

        Args:
            path: Path to the YAML file containing the ranking weights.

        Returns:
            A fully initialized Ranker using the ranking weights given in the YAML file.
        """
        with open(path) as f:
            # YAML file format:
            #
            # weights:
            #   type_match: 1
            #   type_match_inverse_rank: 0.5
            data = yaml.safe_load(f)
            weights: dict[str, float] = data["weights"]

            return Ranker(weights)

    def _calculate_entity_score(self, entity: Entity) -> float:
        return sum(
            self.weights.get(feature_name, 0) * feature_value for feature_name, feature_value in entity.ranking_features.items()
        )

    def _calculate_relation_score(self, context: Context, relation: Relation) -> float:
        source_entity: Entity | None = context.get_entity(relation.source_id)
        source_entity_score = self._calculate_entity_score(source_entity) if source_entity is not None else 0

        target_entity: Entity | None = context.get_entity(relation.target_id)
        target_entity_score = self._calculate_entity_score(target_entity) if target_entity is not None else 0

        relation_score = self.weights.get(relation.relation, 0)

        return source_entity_score + relation_score + target_entity_score

    def rank_and_select_entities(self, entities: list[Entity], limit: int | None = None) -> list[Entity]:
        ranked_entities = sorted(entities, key=self._calculate_entity_score, reverse=True)
        selected = ranked_entities[:limit] if limit else ranked_entities
        trace("ranker.py", f"rank_and_select_entities: {len(entities)} → {len(selected)} (limit={limit})")
        for e in selected[:5]:
            trace("ranker.py", f"  [{e.id}] '{e.name}' score={self._calculate_entity_score(e):.2f} features={e.ranking_features}")
        return selected

    def rank_and_select_relations(self,
                                  context: Context,
                                  relations: Iterable[Relation],
                                  limit: int | None = None
                                  ) -> list[Relation]:
        relations_list = list(relations)
        ranked_relations = sorted(relations_list, key=lambda r: self._calculate_relation_score(context, r), reverse=True)
        selected = ranked_relations[:limit] if limit else ranked_relations
        trace("ranker.py", f"rank_and_select_relations: {len(relations_list)} → {len(selected)} (limit={limit})")
        return selected

    @staticmethod
    def calculate_type_match_score(entity_types: list[str], target_type: str) -> float:
        # TODO: provide more fine-grained match score based on the hierarchy in the ontology
        if target_type in entity_types:
            return 1
        else:
            return 0
