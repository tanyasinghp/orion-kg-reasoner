import re
from typing import TYPE_CHECKING
from enum import Enum
from pprint import pformat
if TYPE_CHECKING:
    from collections.abc import Generator

import yaml
import networkx as nx
from pydantic import BaseModel, Field


class Type(BaseModel):
    name: str = Field(frozen=True)
    uri: str = Field(frozen=True)


class Relation(BaseModel):
    subject: str = Field(frozen=True)
    relation_path: str = Field(frozen=True)
    object: str = Field(frozen=True)
    explanation: str | None = Field(frozen=True, default=None)

    def format(self) -> str:
        formatted_explanation = f" # {self.explanation}" if self.explanation else ""
        return f"({self.subject})-[{self.relation_path}]->({self.object}){formatted_explanation}"


class TypeRelationship(Enum):
    SUB_CLASS_OF = "subClassOf"

    def format(self) -> str:
        if self == TypeRelationship.SUB_CLASS_OF:
            return "<"
        else:
            return self.value


class Hierarchy(BaseModel):
    first_type: str = Field(frozen=True)
    relationship: TypeRelationship = Field(frozen=True)
    second_type: str = Field(frozen=True)

    def format(self) -> str:
        return f"({self.first_type}) {self.relationship.format()} ({self.second_type})"


class Property(BaseModel):
    name: str = Field(frozen=True)
    uri: str = Field(frozen=True)


class ExtraProperty(BaseModel):
    type: str = Field(frozen=True)
    path: str = Field(frozen=True)
    alias: str = Field(frozen=True)


class OntologyProvider:
    """A provider for the knowledge graph ontology to be used by the agent.

    A note on the different property attributes: Datatype properties and object properties were introduced for
    the Business Data Cloud (BDC) knowledge graph. Datatype properties cover things like length or name where the
    values of the properties are literals, while object properties refer to other entities in the knowledge graph.
    Datatype properties are organized by the entity type, meaning different types can theoretically have the properties
    with the same name but different values (URIs).
    Flat properties are not organized by entity type but “flat”, this applies to the S/4HANA knowledge graph.
    Extra properties are defined for the automatic extraction of extra keys, so far this is only used in the S/4HANA
    knowledge graph ontology as well.

    Attributes:
        types: Types defined in the ontology.
        type_name_to_uri_lookup_table: Lookup table mapping type names to knowledge graph URIs.
        case_insensitive_type_name_to_uri_lookup_table: Lookup table mapping case-insensitive type names
            to knowledge graph URIs.
        uri_to_type_name_lookup_table: Lookup table mapping knowledge graph URIs to type names.
        relations: Relations defined in the knowledge graph.
        type_hierarchy: Type hierarchy defined through type relationships (e.g. 'subClassOf').
        sub_classes_lookup_table: Lookup table mapping parent classes to their subclasses.
        super_classes_lookup_table: Lookup table mapping subclasses to their super class.
        ancestor_types: Lookup table mapping types to their ancestor types for types in a subclass relation.
        descendant_types: Lookup table mapping types to their descendant types for types in a subclass relation.
        object_properties: Object properties.
        datatype_properties: Lookup table for datatype properties mapping datatype name to property name to property.
        datatype_properties_reverse_lookup_table: Reverse lookup table for datatype properties mapping property name
            to type name to property URI.
        properties: Lookup table for "flat" properties mapping property name to property.
        property_uri_lookup_table: Lookup table for "flat" properties mapping property URI to property name.
        extra_properties: Extra properties to be automatically looked up for every new entity.
        ontology_extension: Ontology extension.
    """
    # TODO: Define abstract base class for ontology provider defining the public interface of the ontology needed by
    #  the agent and have different ontology classes for the BDC knowledge graph adn the S/4HANA knowledge graph to make
    #  the class attributes (see properties) less confusing.

    def __init__(self, types: list[Type], # noqa: PLR0913
                 type_name_to_uri_lookup_table: dict[str, str],
                 case_insensitive_type_name_to_uri_lookup_table: dict[str, str],
                 uri_to_type_name_lookup_table: dict[str, str],
                 relations: list[Relation],
                 type_hierarchy: list[Hierarchy],
                 sub_classes_lookup_table: dict[str, list[str]],
                 super_classes_lookup_table: dict[str, str],
                 ancestor_types: dict[str, set[str]],
                 descendant_types: dict[str, set[str]],
                 object_properties: dict[str, Property],
                 datatype_properties: dict[str, dict[str, Property]],
                 datatype_properties_reverse_lookup_table: dict[str, dict[str, str]],
                 properties: dict[str, Property],
                 property_uri_lookup_table: dict[str, str],
                 extra_properties: list[ExtraProperty],
                 ontology_extension: list[str]
                 ) -> None:
        self.types = types
        self.type_name_to_uri_lookup_table = type_name_to_uri_lookup_table
        self.case_insensitive_type_name_to_uri_lookup_table = case_insensitive_type_name_to_uri_lookup_table
        self.uri_to_type_name_lookup_table = uri_to_type_name_lookup_table
        self.relations = relations
        self.type_hierarchy = type_hierarchy
        self.sub_classes_lookup_table = sub_classes_lookup_table
        self.super_classes_lookup_table = super_classes_lookup_table
        self.ancestor_types = ancestor_types
        self.descendant_types = descendant_types
        self.object_properties = object_properties
        self.datatype_properties = datatype_properties
        self.datatype_properties_reverse_lookup_table = datatype_properties_reverse_lookup_table
        self.properties = properties
        self.property_uri_lookup_table = property_uri_lookup_table
        self.extra_properties = extra_properties
        self.ontology_extension = ontology_extension


    @staticmethod
    def from_yaml(path: str) -> "OntologyProvider": # noqa: PLR0912, PLR0915
        with open(path) as f:
            data = yaml.safe_load(f)

        # Types
        types: list[Type] = []
        type_name_to_uri_lookup_table: dict[str, str] = {}
        case_insensitive_type_name_to_uri_lookup_table: dict[str, str] = {}
        uri_to_type_name_lookup_table: dict[str, str] = {}

        for type_name, uri in data.get("types", {}).items():
            normalized_type_name = OntologyProvider.normalize_name(type_name)
            types.append(Type(name=normalized_type_name, uri=uri))
            type_name_to_uri_lookup_table[normalized_type_name] = uri
            case_insensitive_type_name_to_uri_lookup_table[normalized_type_name.lower()] = uri
            uri_to_type_name_lookup_table[uri] = normalized_type_name

        # Relations
        relations: list[Relation] = []
        for relation_data in data.get("relations", []):
            # Relation with explanation (quadruple)
            if len(relation_data) == 4: # noqa: PLR2004
                relation = Relation(subject=relation_data[0], relation_path=relation_data[1],
                                    object=relation_data[2], explanation=relation_data[3])
                relations.append(relation)
            # Relation without explanation (triple)
            elif len(relation_data) == 3: # noqa: PLR2004
                relation = Relation(subject=relation_data[0], relation_path=relation_data[1], object=relation_data[2])
                relations.append(relation)
            else:
                raise Exception(f"Unexpected relation data '{relation_data}'."
                                f"Relations must have 3 or 4 elements: "
                                f"subject, relation path, object and optionally an explanation.")

        # Type hierarchy
        type_hierarchy: list[Hierarchy] = []
        sub_classes_lookup_table: dict[str, list[str]] = {} # Lookup table mapping parent classes to their subclasses
        super_classes_lookup_table: dict[str, str] = {} # Lookup table mapping subclasses to their super class

        for hierarchy_data in data.get("hierarchy", []):
            if len(hierarchy_data) == 3: # noqa: PLR2004
                first_type = hierarchy_data[0]
                type_relationship = TypeRelationship(hierarchy_data[1])
                second_type = hierarchy_data[2]
                hierarchy = Hierarchy(first_type=first_type, relationship=type_relationship, second_type=second_type)
                type_hierarchy.append(hierarchy)

                if type_relationship == TypeRelationship.SUB_CLASS_OF:
                    sub_classes_lookup_table.setdefault(second_type, []).append(first_type)
                    super_classes_lookup_table[first_type] = second_type

            else:
                raise Exception(f"Unexpected type hierarchy data '{hierarchy_data}'."
                                f"A type hierarchy must have exactly 3 elements: "
                                f"first type, type relationship, and second type.")

        # Calculate ancestor and descendant types for types in a subclass relation
        ancestor_types: dict[str, set[str]] = {
            type_name: OntologyProvider.get_ancestor_types(type_name, super_classes_lookup_table)
            for type_name in type_name_to_uri_lookup_table.keys()
        }
        descendant_types: dict[str, set[str]] = {
            type_name: OntologyProvider.get_descendant_types(type_name, sub_classes_lookup_table)
            for type_name in type_name_to_uri_lookup_table.keys()
        }

        # Object properties
        object_properties: dict[str, Property] = {}
        for name, uri in data.get("object_properties", {}).items():
            normalized_name = OntologyProvider.normalize_name(name)
            object_property = Property(name=normalized_name, uri=uri)
            object_properties[normalized_name] = object_property

        # Datatype properties
        datatype_properties: dict[str, dict[str, Property]] = {} # datatype name -> property name -> property
        for type_name, property_data in data.get("datatype_properties", {}).items():
            type_properties: dict[str, Property] = {}
            for property_name, property_uri in property_data.items():
                normalized_property_name = OntologyProvider.normalize_name(property_name)
                type_properties[normalized_property_name] = Property(name=normalized_property_name, uri=property_uri)

            normalized_type_name = OntologyProvider.normalize_name(type_name)
            datatype_properties[normalized_type_name] = type_properties

        # Reverse lookup table for datatype properties (property name -> type name -> property URI)
        datatype_properties_reverse_lookup_table: dict[str, dict[str, str]] = {}
        for type_name, properties in datatype_properties.items():
            for property_name, data_type_property in properties.items():
                property_uri = data_type_property.uri
                datatype_properties_reverse_lookup_table.setdefault(property_name, {})[type_name] = property_uri

        # Flat properties (optional)
        properties: dict[str, Property] = {} # property name -> property
        property_uri_lookup_table: dict[str, str] = {} # property URI -> property name
        for property_name, property_uri in data.get("properties", {}).items():
            normalized_property_name = OntologyProvider.normalize_name(property_name)
            properties[normalized_property_name] = Property(name=normalized_property_name, uri=property_uri)
            property_uri_lookup_table[property_uri] = normalized_property_name

        # Extra properties to be automatically looked up for every new entity (optional)
        extra_properties: list[ExtraProperty] = []
        for extra_property_data in data.get("extra_properties", []):
            property_type = extra_property_data[0]
            path = extra_property_data[1]
            alias = extra_property_data[2]
            extra_properties.append(ExtraProperty(type=property_type, path=path, alias=alias))

        # optional extension to ontology description in the prompt
        ontology_extension: list[str] = data.get("ontology_extension", [])

        ontology_provider = OntologyProvider(
            types, type_name_to_uri_lookup_table, case_insensitive_type_name_to_uri_lookup_table,
            uri_to_type_name_lookup_table, relations, type_hierarchy, sub_classes_lookup_table,
            super_classes_lookup_table, ancestor_types, descendant_types, object_properties, datatype_properties,
            datatype_properties_reverse_lookup_table, properties, property_uri_lookup_table, extra_properties,
            ontology_extension
        )

        # Minimize the ontology based on the type hierarchy
        ontology_provider.condense_ontology()

        return ontology_provider

    @staticmethod
    def normalize_name(name: str) -> str:
        return name.replace(" ", "_")

    @staticmethod
    def get_ancestor_types(type_name: str, super_classes_lookup_table: dict[str, str]) -> set[str]:
        # TODO: Why do we mark the given type itself as an ancestor?
        ancestor_types = {type_name}
        # If the given type has a super class, i.e., a 'subClassOf' relation between the given type and another
        # type exists in the type hierarchy of the ontology, mark the super class as an ancestor of the given type.
        # Then recursively determine the ancestor types of the super class.
        if (super_class := super_classes_lookup_table.get(type_name)) and super_class not in ancestor_types:
            super_class_ancestor_types = OntologyProvider.get_ancestor_types(super_class, super_classes_lookup_table)
            return ancestor_types.union(super_class_ancestor_types)
        else:
            return ancestor_types

    @staticmethod
    def get_descendant_types(type_name: str, sub_classes_lookup_table: dict[str, list[str]]) -> set[str]:
        # TODO: Why do we mark the given type itself as a descendant type?
        descendant_types = {type_name}
        for subclass in sub_classes_lookup_table.get(type_name, []):
            if subclass not in descendant_types:
                subclass_descendant_types = OntologyProvider.get_descendant_types(subclass, sub_classes_lookup_table)
                descendant_types.update(subclass_descendant_types)

        return descendant_types

    def condense_ontology(self) -> None:
        """Condense the ontology by tyring to remove redundant information.

        This method condenses the datatype properties definitions by removing all properties that are also defined
        in a parent class.
        """
        # Create type hierarchy graph
        type_hierarchy_graph = nx.DiGraph()
        edges = [
            (hierarchy.second_type, hierarchy.first_type) # (parent class, subclass)
            for hierarchy in self.type_hierarchy
            if hierarchy.relationship == TypeRelationship.SUB_CLASS_OF
        ]
        type_hierarchy_graph.add_edges_from(edges)

        # Validate that the given type hierarchy doesn't contain any cycles
        if not nx.is_directed_acyclic_graph(type_hierarchy_graph):
            raise ValueError("Cycle detected: the type hierarchy in the ontology do not form a DAG.")

        # Get types in topologically sorted order from parent classes to subclasses
        sorted_types: Generator[str] = nx.topological_sort(type_hierarchy_graph)

        # Condense the datatype properties following the type order
        for type_name in sorted_types:
            properties: dict[str, Property] = self.datatype_properties.get(type_name, {}) # property name -> property
            ancestor_properties: set[str] = set() # property names

            for ancestor_type in self.ancestor_types[type_name]:
                # The ancestor types of a type contain the type itself, thus we ignore it here
                if ancestor_type != type_name:
                    ancestor_properties.update(self.datatype_properties.get(ancestor_type, {}).keys())

            self.datatype_properties[type_name] = {
                property_name: prop for property_name, prop in properties.items()
                if property_name not in ancestor_properties
            }

    def get_type_uri(self, type_name: str, default: str | None = None, case_sensitive: bool = True) -> str | None:
        """Get the URI of the type with the given name.

        Args:
            type_name: Name of the type.
            default: Default value to return if no type with the given name is present in the ontology.
            case_sensitive: Perform a case-sensitive or case-insensitive search. When a case-sensitive search
                is requested, this method only returns the URI of the type if the type name corresponding to the
                type URI matches the given type name exactly.

        Returns:
            The URI of type if a type with the given name is in this ontology, else default.
        """
        if case_sensitive:
            return self.type_name_to_uri_lookup_table.get(type_name, default)
        else:
            return self.case_insensitive_type_name_to_uri_lookup_table.get(type_name.lower(), default)

    def get_type_by_uri(self, uri: str, default: str | None = None) -> str | None:
        """Get the type with the given URI.

        Args:
            uri: Type URI.
            default: Default value to return if no type with the given URI is present in the ontology.

        Returns:
            The name of the type if a type with the given URI is in this ontology, else default.
        """
        return self.uri_to_type_name_lookup_table.get(uri, default)

    def get_property_uri(self, property_name: str, default: str | None = None) -> str | None:
        """Get the URI of the property with the given name.

        Args:
            property_name: Name of the property.
            default: Default value to return if no property with the given name is present in the ontology.

        Returns:
            The URI of the property if a property with the given name is in this ontology, else default.
        """
        # object properties
        if property_name in self.object_properties:
            return self.object_properties[property_name].uri
        # flat property dictionary
        elif property_name in self.properties:
            return self.properties[property_name].uri
        # type-dependent properties
        else:
            # TODO: Use a more sophisticated logic here to select the URI based on the type,
            #  e.g., prioritizing the ones appearing in the selected subset of ontology
            for _, property_uri in self.datatype_properties_reverse_lookup_table.get(property_name, {}).items():
                # TODO: for now just return the first match
                return property_uri

        return default

    def resolve_path_properties(self, path: str) -> str:
        """Resolve all relations and property aliases in the given path to their corresponding URIs.

        A path is a chain of relations that goes from one entity to another following the relations specified in the
        ontology. In a path, inverse relations are denoted with the prefix "^". Two relations can be linked with the
        infix operator "/", and alternative paths can be specified with the infix operator "|".

        For example, given the two relation patterns [E1, R1, E2], [E3, R2, E2] in the ontology, the path = 'R1 / ^R2'
        can be defined to navigate from E1 to E3; or path = '( ^R1 | ^R2 )' to navigate from E2 to E1 or E3.

        Args:
            path: Path of the property.

        Returns:
            The given path where all property aliases have been replaced by their corresponding property URIs.
            For example, given the ontology defines a property 'Is_In_Table', the path '^Is_In_Table' would be
            resolved to the path '^<http://sap.com/datasphere/deepsea/repository#r_isInTable>'.
        """
        # Use regex to find all property aliases in the path string with their positions
        matches = re.finditer(r"[^\/\|\^\(\)\s]+", path)
        aliases_with_positions = [(match.group(), match.start(), match.end()) for match in matches]

        # Sort the aliases by their start position in reverse order to
        # ensure that the replacements don't interfere with each other
        aliases_with_positions.sort(key=lambda triple: triple[1], reverse=True)

        resolved_path = path
        for relation, start, end in aliases_with_positions:
            # TODO: Don't use a fallback value here. Thrown an exception instead which results in a max priority
            #  message in the agent loop indicating to the LLM that it used a relation which doesn't exist in the
            #  ontology.
            uri = self.get_property_uri(relation, default="xxx")
            resolved_path = resolved_path[:start] + f"<{uri}>" + resolved_path[end:]

        return resolved_path

    def resolve_property_path(self, path: str) -> str | None:
        """Resolve a property path to a single predicate URI.

        Tries the entire path as a direct property name first (with spaces normalized
        to underscores). If that fails, falls back to resolve_path_properties for complex
        paths containing /, |, ^ operators. Returns None if unresolvable.

        Args:
            path: Property path string (e.g. 'Discounted Price' or '^belongs_to/p_name').

        Returns:
            The resolved predicate URI wrapped in angle brackets, or None if unresolvable.
        """
        normalized = path.replace(" ", "_")
        uri = self.get_property_uri(normalized)
        if uri is not None:
            return f"<{uri}>"
        resolved = self.resolve_path_properties(path)
        if "<xxx>" in resolved:
            return None
        return resolved

    def list_property_paths(self) -> list[str]:
        """List all available property paths (both datatype and object properties).

        Returns:
            A sorted list of all property names known to the ontology, with underscores
            replaced by spaces for readability.
        """
        paths: set[str] = set()
        props: dict[str, Property] = {}
        props.update(self.object_properties)
        props.update(self.properties)
        paths.update(p.name.replace("_", " ") for p in props.values())
        for type_name, type_props in self.datatype_properties.items():
            paths.update(p.name.replace("_", " ") for p in type_props.values())
        return sorted(paths)

    def format(self) -> str:
        """Get a formatted string representation of the ontology suitable for an LLM prompt.

        Returns:
            A formatted string containing the entity types, the type hierarchy, and the relations of the ontology.
        """
        ontology = (f"ENTITY_TYPES = {list(self.type_name_to_uri_lookup_table.keys())}\n"
                    f"TYPE_HIERARCHY = {self.format_hierarchy(self.type_hierarchy)}\n"
                    f"RELATION_PATTERNS = {self.format_relations(self.relations)}")
        return ontology

    def format_subset(self, include_types: list[str]) -> str:
        """Get a formatted string representation of a subset of the ontology suitable for an LLM prompt.

        Args:
            include_types: Reduce the ontology to only contain information related to these types.

        Returns:
            A formatted string containing the entity types, the type hierarchy, the relations, and the
            data type properties of the ontology which has been reduced to only contain information related
            to the given types.
        """
        # Find all subclasses and super-classes of the included types
        selected_types: set[str] = {ancestor for t in include_types for ancestor in self.ancestor_types.get(t, set())}

        # Select the type hierarchy relations where a selected type is a subclass
        selected_type_hierarchy: list[Hierarchy] = [
            type_hierarchy for type_hierarchy in self.type_hierarchy
            if (type_hierarchy.first_type in selected_types) and
               (type_hierarchy.relationship == TypeRelationship.SUB_CLASS_OF)
        ]

        # Select all relations referencing at least one of the selected types
        selected_relations: list[Relation] = [
            relation for relation in self.relations
            if (relation.subject in selected_types) or (relation.object in selected_types)
        ]

        # Collect datetype properties of selected types into a lookup table mapping type names to property names
        selected_properties: dict[str, list[str]] = {
            t: list(self.datatype_properties.get(t, {}).keys()) for t in selected_types
        }

        ontology = (f"ENTITY_TYPES = {list(selected_types)}\n"
                    f"TYPE_HIERARCHY = {self.format_hierarchy(selected_type_hierarchy)}\n"
                    f"RELATION_PATTERNS = {self.format_relations(selected_relations)}\n"
                    f"DATATYPE_PROPERTIES = {self.format_properties(selected_properties)}")
        return ontology

    @staticmethod
    def format_hierarchy(type_hierarchy: list[Hierarchy]) -> str:
        return "[" + ", ".join(hierarchy.format() for hierarchy in type_hierarchy) + "]"

    @staticmethod
    def format_relations(relations: list[Relation]) -> str:
        formatted_relations = ", \n".join(f"\t{relation.format()}" for relation in relations)
        return f"[\n{formatted_relations}\n]"

    @staticmethod
    def format_properties(properties: dict[str, list[str]]) -> str:
        return pformat(properties, compact=True)
