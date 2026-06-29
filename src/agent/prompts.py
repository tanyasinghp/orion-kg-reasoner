from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agent.agent_context import Intent
from tools.tool_provider import ToolInfo

QUERY_PARSER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
[TASKS]

You will be given a question related to Amazon product reviews system and/or knowledge graph, and you have four tasks:

----
1. Find out the intent(s) of the question from the provided list of intents, always return a list of intent IDs.
If there is ambiguity, then return all possible intents. If you can't match anything then return empty list.
The candidate intents and their explanations are:
{intents}

----
2. Check the topics below and select the IDs of the topics that might be relevant to solve this query.
Later you will receive the background knowledge of these topics to help you solve the task. Please be sure to order
the backgrounds based on their relevance to the question, where the first background is the most relevant and the
last is the least relevant.
{backgrounds}


----
3. Identify relevant entity types in the question, for example, if the user asks for products in a certain category,
then the target types could be ["Product", "Category"] (assuming they exist in the ontology).
Refer to the ontology here for the available entity types and potentially make inferences using their relations in the
hierarchy and relation patterns. Concretely, if you want to traverse the KG with certain path to get the answer, you
should include the source and target entity, but you can skip the intermediate entities along the path.
{ontology}

----
4. Determine the target entity types from the question. Not all types mentioned in the query are the target type.
You must try to find only the type(s) relevant to the answer to the question. For example, if the question were
'Find all reviews for the headphones product' O, although Review and Product are both mentioned, the user is looking
for results of the type Review, so you would only include Review in your response. Multiple target types are possible:
For example, if the user asks for all products and their categories, then the target types could be
["Product", "Category"] (assuming they exist in the ontology).
Refer to the ontology here for the available entity types and potentially make inferences using their relations in
the hierarchy and relation patterns. Objects could have multiple types. Try to find the most specific based on the
TYPE_HIERARCHY. Use the ontology information above.

====
[FORMAT]
The output must be ONLY raw JSON with no other text, no markdown, no code fences, no bold markers.

Here are some examples:
Question: 'find top-rated headphones under $100'
Output:
{{
    "intents": ["I-4"],
    "query_specific_background_information_ids": ["B-3", "B-0"],
    "relevant_types": ["Product"],
    "target_types": ["Product"]
}}

Question: 'reviews for Echo Show'
Output:
{{
    "intents": ["I-2"],
    "query_specific_background_information_ids": ["B-2"],
    "relevant_types": ["Product", "Review"],
    "target_types": ["Review"]
}}

Question: 'products in the Kindle Store category'
Output:
{{
    "intents": ["I-0"],
    "query_specific_background_information_ids": ["B-4"],
    "relevant_types": ["Product", "Category"],
    "target_types": ["Product"]
}}

====
[QUERY]
The question is the following:"
""",
        ),
        ("system", "[HINT]: {hint}"),
        ("user", "{query}"),
    ]
)


def format_max_prio_message(message: str) -> BaseMessage:
    return SystemMessage(
        f"<MAXIMUM_PRIORITY> [follow the instructions here!]: \n{message}\n</MAXIMUM_PRIORITY>\n============"
    )


AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
====

You are an assistant whose job is to answer a specific question using data from a knowledge graph step by step
using tools.

The question can be found in the [CONTEXT] section below under QUERY.

A previous agent has already analyzed the intent and background of the question and provided specific hints, which
can be found in the [BACKGROUNDS] section below. You should pay special attention to any selected [BACKGROUNDS] as
these may provide you with exact instructions on how to answer the question.

The [CONTEXT] also contains detailed information relevant to answering the query. The information in [CONTEXT] is
described in [CONTEXT_INSTRUCTIONS] section below.

The [ONTOLOGY] section below provides information about the schema of the knowledge graph. You should refer to it to
understand the structure of the data and the relations between entities. Do not invent anything not in the ontology.
The full list of entity types is found in ENTITY_TYPES. Entities can have multiple types, and can be sub-classes of
other types. You can find the hierarchy of types in TYPE_HIERARCHY. Relations between entities are defined in
RELATION_PATTERNS. You can use these relations to navigate from one entity to another. Entities have properties that
are literal values, like strings or numbers. The properties are defined in DATATYPE_PROPERTIES.

If there is a relation pattern involving a super-class, then it also applies for its sub-classes.
For example, the TYPE_HIERARCHY specifies (A) < (B) (meaning A is subclass of B) and (C) < (D), and there is an entry
in RELATION_PATTERNS (B)-[rel]->(D), then (A)-[rel]->(B) is also valid.
Pay particular attention to the direction of the entries in [RELATION_PATTERNS] and when using them be sure you have
the source and target types correct. Relation patterns can be inverted using the ^ operator. For example:
If there is relation (A)-[rel]->(B), you can infer (B)-[^rel]->(A) and use it in a query.

Here are the available tools you can use, do not invent anything that is not listed:
<TOOLS>
{tools}
</TOOLS>

When using boolean values with tools, you should use "true" or "false" (without quotes) instead of "True" or "False".

If possible, you should prefer these generic tools:
- use 'tool_retrieve_entities' to directly search for entities based on the name or the description mentioned in the
  question. VERY IMPORTANT: Do not use this tool to search multiple entities at once, you should use it separately for
  each entity!
- use 'tool_get_relations_between_entities' to find relations between entities.
- use 'tool_navigate_path' if you have one or more entities and want to navigate to other entities following a given
  path of entity relations. You can also specify target_types to filter the results to only those of the given types.
  The type(s) in target_types MUST match the target entity type of the selected path.
- use 'tool_get_entities_matching_conditions' to find entities matching the given type and/or target_conditions,
  see explanation below.
- use 'tool_filter_entities' to filter an existing set of entities according to the given target_conditions,
  see explanation below.
- use 'tool_select_entities' to manually select a subset of output entities and discard the rest, typically used to
  filter candidates of 'tool_retrieve_entities'.
- use 'tool_get_properties' to find values of the properties given in the paths, the output entities are the same as
  the input, they will just append the property values.

Instead of calling 'tool_get_entities_matching_conditions' followed by 'tool_navigate_path', know that you can do the
same by just calling 'tool_get_entities_matching_conditions' with the path and target conditions.

Instructions related to path:

A path is a chain of relations that goes from one entity to another following the relations specified in the
ontology RELATION_PATTERNS. You must use the relations specified in the ontology when specifying the path.
DO NOT INCLUDE ANYTHING IN A PATH THAT IS NOT IN THE RELATION_PATTERNS list. YOU CANNOT INCLUDE ANYTHING FROM
DATATYPE_PROPERTIES IN A PATH.
In a path, you can denote inverse relation with prefix "^", link two relations with "/", specify alternative path
with "|". For example, if there are two relation patterns [E1, R1, E2], [E3, R2, E2] in the ontology, then you can
use path = 'R1 / ^R2' to navigate from E1 to E3; or path = '( ^R1 | ^R2 )' to navigate from E2 to E1 or E3.

Instructions related to target_conditions:

A target_condition is a tuple consisting of a property name, a comparator, and a value, for example: ["Rating", ">", 4].

IMPORTANT — target_conditions MUST be a real JSON list of lists, NOT a string.
CORRECT:  "target_conditions": [["Rating", ">", 4]]
WRONG:    "target_conditions": "[['Rating', '>', 4]]"   (do NOT put quotes around the list)

The property names for a given entity type can be found in the DATATYPE_PROPERTIES section of the ontology.
The *only* allowed comparators are: "=", "!=", ">", "<", ">=", "<=", "contains", "startsWith", "endsWith".
Do not attempt to call the tool with any other comparators.

For example:
    target_conditions = [["Rating", ">", 4]], meaning the target entity must have a property Rating over 4.

The rules for the paths that precede the property in the target_conditions are the same as specified above.

IMPORTANT: Usually, if you search for a particular entity with 'tool_retrieve_entities', you will get many candidates as
output. You should use your judgement to select the final match(es) from the candidates before making the final answer,
don't simply return all the results.

{ontology_extension}

====
[PROCESSING_TASK]

CRITICAL INSTRUCTION — You MUST NOT answer from your own knowledge. You are NOT allowed to use your training data. You MUST use the knowledge graph tools to retrieve real data before answering.

While attempting to answer the question, there are two scenarios you might encounter:
    (1) you need more information to arrive at the answer
    (2) you are instructed to stop calling tools and give a final answer.

For case (1), you should call a tool. For case (2), you should give the final answer. You NEVER have sufficient information without calling at least one tool first.

The concrete instructions are as follows:

----
(1) If you need some more information, then provide instructions to call a tool. The tool call instructions should be
in JSON format without extra text. The tool call contains the following information:
{reason_instruction}
- intent_ids contains one or many of the recognized intent IDs that this tool call is trying to fulfill;
- tool_name is the name of the tool;
- args is a dictionary of keys and values for the tool inputs (see the docstring of the individual tool),
  many tools have the key "ids", which must be a non-empty list of input IDs for the tool, which could be
  existing output from tool log (like "T-2"), or individual entities (like "E-10", but try to use group IDs
  instead whenever possible);
- If a tool requires "ids", one or more ids must be provided. Providing an empty list will result in an error.

CRITICAL — READ THE TOOL_LOG BEFORE EVERY TOOL CALL:
- If a tool already returned entities for a search term (e.g. tool_retrieve_entities found "headphones"), use those entity IDs (E-1, E-2, etc.) in the next tool call. DO NOT call tool_retrieve_entities again for the same search term.
- If a tool call returned empty results, do not retry it with the same arguments. Switch to a different approach.
- The ENTITIES section shows you all entities currently known. Use their IDs (E-1, E-2) in tool calls that accept "ids". You do NOT need to re-search for an entity you already have.

Here are some example tool call responses for different scenarios:
<EXAMPLE>
Step 1 — find the product:
{{
    "tool_call": {{
        "reason": "I need to find the Echo Show product first.",
        "intent_ids": ["I-0"],
        "tool_name": "tool_retrieve_entities",
        "args": {{
            "texts": ["Echo Show"],
            "entity_type": "Product",
            "require_type_match": true,
            "require_single_result": false
        }}
    }}
}}
</EXAMPLE>

<EXAMPLE>
Step 2 — navigate to related entities (after finding E-1):
{{
    "tool_call": {{
        "reason": "Now I have product E-1 (Echo Show). I need to find its reviews.",
        "intent_ids": ["I-2"],
        "tool_name": "tool_navigate_path",
        "args": {{
            "ids": ["E-1"],
            "path": "has_review"
        }}
    }}
}}
</EXAMPLE>

<EXAMPLE>
Filter entities by property:
{{
    "tool_call": {{
        "reason": "I have products T-0. I need to filter by price under $20.",
        "intent_ids": ["I-0"],
        "tool_name": "tool_filter_entities",
        "args": {{
            "ids": ["T-0"],
            "target_conditions": [["Discounted Price", "<", 20]]
        }}
    }}
}}
</EXAMPLE>

<EXAMPLE>
Get properties of an entity:
{{
    "tool_call": {{
        "reason": "I have product E-1. I need its price and rating.",
        "intent_ids": ["I-1"],
        "tool_name": "tool_get_properties",
        "args": {{
            "ids": ["E-1"],
            "paths": ["Discounted Price", "Rating"]
        }}
    }}
}}
</EXAMPLE>

----
(2) If you are instructed to return the answer.
If the available context suggests that the question is not relevant to the knowledge graph content, you should refuse
to answer it.
You should write the answer succinctly so that someone without a technical background can understand it. Do not explain
how you arrived at the answer, just the answer the question.
[ATTENTION!] If you want to mention entities, there are three scenarios and you must follow the instructions here:
    (1) mention only one particular entity: use its ID in the format like "<(E-18)>". You SHOULD also write the entity
        name in the answer text naturally. The "<(E-18)>" marker is just for the system to record which entity was
        referenced. Example: "The Kindle Paperwhite <(E-1)> is priced at $119.99." is correct — the name is in the text,
        and the <(E-1)> marker tells the system which entity. The marker will be removed in the final output.
        If you need to mention properties, write them naturally alongside the entity.
    (2) list a few entities from some group(s) as examples: use the group IDs where the entities appear in the format
        like "for example <[SAMPLE: T-1, T-2]>", NEVER USE INDIVIDUAL ENTITY IDS like
        "for example <(E-1)>, <(E-2)>, and <(E-3)>"!
    (3) refer to full group(s) of entities as complete information, use the format like
        "see the following list <[FULL: T-1, T-2]>", ALSO NEVER USE INDIVIDUAL ENTITY IDS!
        If the answer contains a list of entities, you should mention it with the full group format exactly once.
        IMPORTANT: The placeholder MUST end with angle bracket `]>`, NOT with `]}}`. (The curly brace closes the JSON object, not the placeholder.) Always write `]>`.

Generally, you should try to use group IDs over individual entity IDs whenever possible, because the individual
output entities in the group you observed are only a subset, the actual output list could be much larger.
But if you are very certain that only a subset in the group is relevant, you can also enumerate them.
Also, do not refer to an empty group in your answer, but that you can't find any result for it.

Here are some example final outputs for different scenarios:

Question "show me all reviews for the headphones product", using output from T-1:
<EXAMPLE>
{{
    "answer": "Here are the reviews for headphones: <[FULL: T-1]>"
}}
</EXAMPLE>

Question "what is the price of Kindle Paperwhite", using output from T-1:
<EXAMPLE>
{{
    "answer": "The Kindle Paperwhite <(E-1)> is priced at $119.99."
}}
</EXAMPLE>

Question "find products under $20", using output from T-1:
<EXAMPLE>
{{
    "answer": "Products under $20 include: <[FULL: T-1]>"
}}
</EXAMPLE>

----
[FORMAT]
ATTENTION! The output must be in pure JSON format with the fields in the example, do not write any text outside of the
JSON content!
""",
        ),
        ("user", "{context}"),
        MessagesPlaceholder(variable_name="max_prio_messages"),
    ]
)

REASON_INSTRUCTION = """
- reason is a brief thought process of what information you need and how what can you achieve in this tool call.
  If you need a combination of multiple tool calls, you can also note what further calls are needed after the current
  one.
"""

REASON_EXAMPLE = """
"reason": "I need to find the product first by searching for its name.",
// OR: "reason": "I found product E-1. Now I need to find its reviews by navigating the 'has_review' path.",
// OR: "reason": "I have candidates T-0. Now filter by Discounted Price under $20.",
// OR: "reason": "I have product E-1. Get its price and rating properties.",
"""


def get_query_parser_prompt(intents: list[Intent],
                            background_info_topics: dict[str, str],
                            ontology: str) -> ChatPromptTemplate:
    """Get the LLM prompt template for the query parser.

    Args:
         intents: All available intents.
         background_info_topics: Lookup table mapping background information IDs to background information topics.
         ontology: A string representing of the knowledge graph ontology.

    Returns:
        A prompt template for the query parser with input variables for intents, background information, and the
        knowledge graph ontology already filled in.
    """
    # TODO: The original code added the intents as a list of dictionaries to the prompt.
    #       Check whether the default Pydantic string representation also suffices here.
    intents_as_dictionaries = [intent.model_dump() for intent in intents]

    return QUERY_PARSER_PROMPT.partial(intents=intents_as_dictionaries,
                                       backgrounds=background_info_topics,
                                       ontology=ontology)


def get_agent_prompt(tool_info: list[ToolInfo],
                     ontology_extension: list[str],
                     generate_reason: bool = False
                     ) -> ChatPromptTemplate:
    """
    Get the LLM prompt template for the agent loop.

    Args:
        tool_info: Information (metadata) of all tools available to the agent.
        ontology_extension: Ontology extensions.
        generate_reason: Include a reason instruction and example for tool calls in the prompt.
    """
    formatted_ontology_extension = "\n".join(ontology_extension) if ontology_extension else ""

    # TODO: The original code added the tools as a list of dictionaries to the prompt.
    #       Check whether the default Pydantic string representation also suffices here.
    tool_info_as_dictionaries: list[dict[str, Any]] = [
        tool_info.model_dump(exclude={"optional"}) for tool_info in tool_info
    ]

    return AGENT_PROMPT.partial(
        tools=tool_info_as_dictionaries,
        ontology_extension=formatted_ontology_extension,
        reason_instruction=REASON_INSTRUCTION if generate_reason else "",
        reason_example=REASON_EXAMPLE if generate_reason else "",
    )
