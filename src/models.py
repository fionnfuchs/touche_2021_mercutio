from dataclasses import dataclass, field
from typing import List, Dict
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Query:
    # type 0: normal query, type 1: expanded query
    query_type: int = 0
    query_text: str = field(default_factory=str)
    is_phrase_search: bool = False


@dataclass_json
@dataclass
class ChatNoirResult:
    UUID: str
    trec_id: str
    text: str = field(repr=False)
    page_rank: float
    spam_rank: float
    score: float
    snippet: str = field(default_factory=str)
    is_cleaned: bool = False


@dataclass_json
@dataclass
class Document:
    chat_noir_result: ChatNoirResult
    rank: int = field(default_factory=int)
    simple_term_score: int = field(default_factory=int)
    classifier_score: float = field(default_factory=float)
    scores: Dict[str, float] = field(default_factory=dict)
    combined_score: float = field(default_factory=float)


@dataclass_json
@dataclass
class ProcessingObject:
    """The object to be processed by the pipeline."""

    query: Query = field(default=Query())
    # create a dict with the document's UID as key and the document as value
    documents: Dict[str, Document] = field(default_factory=dict)

    def copy_with(self, query: Query):
        return ProcessingObject(query=query)


@dataclass_json
@dataclass
class Topic:
    """
    This object represents one topic.
    It can contain multiple ProcessingObjects that each describe a query and their retrieved results
    """

    # this field represents the complete original query
    topic_query: str
    topic_number: int

    # version of the topic object, in case of direct saving to a database
    version: str = "v0"

    # these are the documents that we want to return for this topic
    result_docs: List[Document] = field(default_factory=list)

    processing_objects: List[ProcessingObject] = field(default_factory=list)
