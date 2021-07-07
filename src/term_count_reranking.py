from typing import List

from config import Config
from pipeline import Pipe
from models import Topic
import pandas as pd
from log import get_child_logger
from query_expansion import SpacyWrapper, PosType

logger = get_child_logger(__file__)

POS_TYPE_NAMES = {
    "entity": PosType.ENTITY,
    "verb": PosType.VERB,
    "feature": PosType.FEATURE,
}


class TermCountRanking(Pipe):
    def __init__(
        self,
        config: Config,
    ):
        self.spacy = SpacyWrapper()
        self.pos_types = self._load_pos_types(config)
        self.factor_b = config.get("term_counts", "factor_b", expected_type=float)
        if self.factor_b == 0:
            raise ValueError("factor 'b' cannot be zero.")

    def _load_pos_types(self, config) -> List[PosType]:
        l = config.get("term_counts", "pos_types", expected_type=list)
        detected_types = []
        for pos_type in l:
            if pos_type not in POS_TYPE_NAMES:
                raise ValueError(
                    "Invalid config: %s is no known pos type. Choose one of: %s"
                    % (pos_type, POS_TYPE_NAMES.keys())
                )
            detected_types.append(POS_TYPE_NAMES[pos_type])
        return detected_types

    def process(self, topic: Topic) -> Topic:
        query_terms = set()
        for pos_type in self.pos_types:
            tags = self.spacy.get_tags_of_type(topic.topic_query.lower(), pos_type)
            for tag in tags:
                query_terms.add(tag)

        n_query_terms = len(query_terms)
        if n_query_terms == 0:
            return topic
        for po in topic.processing_objects:
            for uuid in po.documents:
                d = po.documents[uuid]
                score = self.calc_score(d.chat_noir_result.text, query_terms)
                po.documents[uuid].scores["term_counts"] = score
        return topic

    def calc_score(self, doc_text, query_terms):
        total_score = 0
        for term in query_terms:
            counts = doc_text.lower().count(term.lower())
            if counts == 0:
                return 0
            total_score += 1 - 1 / (self.factor_b * counts + 1)
        return total_score / len(query_terms)
