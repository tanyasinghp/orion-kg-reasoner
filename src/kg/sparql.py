from typing import Self

from pydantic import BaseModel, Field, model_validator


class SparqlHeadField(BaseModel):
    """A model for the 'head' member in the SPARQL 1.1 query results JSON format.

    The 'head' member gives the variables mentioned in the results and may contain a 'link' member.

    Attributes:
        vars: The 'vars' member is an array giving the names of the variables used in the results. These are the
            projected variables from the query. A variable is not necessarily given a value in every query solution
            of the results.
        link: The optional 'link' member gives an array of URIs, as strings, to refer for further information. The
            specification does not define the format and content of these link references.
    """
    vars: list[str] = Field(frozen=True)
    link: list[str] | None = Field(frozen=True, default_factory=list)


class SparqlRdfTerm(BaseModel):
    """A model for an RDF term encoded as a JSON object as defined in the SPARQL 1.1 query results JSON format.

    This class provides a model to deserialize an RDF term (IRI, literal or blank node) encoded as a JSON object.
    All aspects of the RDF term are represented. The JSON object has a 'type' member and other members depending on
    the specific kind of RDF term.

    The blank node label is scoped to the result object. That is, two blank nodes with the same label in a single
    SPARQL Results JSON object are the same blank node. This is not an indication for any internal system identifier
    the SPARQL processor may use. Use of the same label in another SPARQL Results JSON object does not imply it is
    the same blank node.

    Attributes:
        type: The type of the RDF term.
        value: The value of the RDF term.
        datatype: The datatype of the RDF term.
    """
    type: str = Field(frozen=True)
    value: str = Field(frozen=True)
    datatype: str | None = Field(frozen=True, default=None)


class SparqlResultsField(BaseModel):
    """A model for the 'results' member in the SPARQL 1.1 query results JSON format.

    The value of the 'results' member is an object with a single key called 'bindings' containing the query solutions.

    Attributes:
        bindings: The value of the 'bindings' member is an array with zero or more elements, one element per query
            solution. Each query solution is a JSON object. Each key of this object is a variable name from the query
            solution. The value for a given variable name is a JSON object that encodes the variable's bound value,
            an RDF term. There are zero elements in the array if the query returned an empty solution sequence.
            Variable names do not include the initial '?' or '$' character. Each variable name that appears as a key
            within the 'bindings' array will have appeared in the 'vars' array in the result header.
    """
    bindings: list[dict[str, SparqlRdfTerm]]


class SparqlJsonResponse(BaseModel):
    """A model for the SPARQL 1.1 query results JSON format.

    This class provides a model to deserialize SPARQL results (SELECT and ASK query forms) given in a JSON format
    meeting the `SPARQL 1.1 Query Results JSON Format`_.

    The results of a SPARQL Query are serialized in JSON as a single top-level JSON object. This object has a
    'head' member and either a 'results' member or a 'boolean' member, depending on the query form (SELECT or ASK,
    respectively). Other keys, with different names, may be present in the JSON results object but are not defined by
    the specification.

    Attributes:
        head: The 'head' member gives the variables mentioned in the results and may contain a 'link' member.
        boolean: The result of an ASK query form is encoded by the 'boolean' member, which takes either the
            JSON value 'true' or the JSON value 'false'.
        results: The value of the 'results' member is an object with a single key called 'bindings' containing
            the query solutions.
        error: Error message if the SPARQL query failed during evaluation.

    .. _SPARQL 1.1 Query Results JSON Format:
        https://www.w3.org/TR/sparql11-results-json/
    """
    head: SparqlHeadField = Field(frozen=True)
    boolean: bool | None = Field(frozen=True, default=None)
    results: SparqlResultsField | None = Field(frozen=True, default=None)
    error: str | None = Field(default=None)

    @model_validator(mode="after")
    def exclusive_result_fields(self) -> Self:
        if self.boolean is None and self.results is None:
            raise ValueError("Expected only one of these fields : [`boolean`, `results`]")
        return self
