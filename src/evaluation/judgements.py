import os
from dataclasses import dataclass
from typing import List, Dict, Any
import enum


@dataclass
class ConfusionMatrix:
    tp: int
    tn: int
    fp: int
    fn: int


class UnknownRelevanceStrategy(enum.Enum):
    ASSUME_NOT_RELEVANT = "assume_not_relevant"
    ASSUME_RELEVANT = "assume_relevant"
    IGNORE = "ignore"


class RelevanceJudgements:
    def __init__(
        self,
        qrels_path: str = None,
        relevance_threshold: int = 1,
        unknown_relevance_strategy: UnknownRelevanceStrategy = UnknownRelevanceStrategy.ASSUME_NOT_RELEVANT,
    ):
        if qrels_path is None:
            source_dir = os.path.split(__file__)[0]
            qrels_path = os.path.join(
                os.path.split(source_dir)[0], "..", "material", "task2-relevance.qrels"
            )
        self.relevance_stats: List[Dict[str, Any]] = self.__load_relevance_stats(
            qrels_path
        )
        self.relevance_threshold = relevance_threshold
        self.unknown_relevance_strategy = unknown_relevance_strategy

    def __load_relevance_stats(self, qrels_path) -> list:
        """Loads the relevance stats into list of dicts.

        These dicts contain the keys:

        topic : int (for topic number), UUID : str (for the documents UUID) and relevance : int.

        :rtype list:
        """

        result = []
        with open(qrels_path, "r") as relevance_file:
            for line in relevance_file.readlines():
                if line.strip() == "":
                    continue
                line_split = line.strip().split(" ")
                result.append(
                    {
                        "topic": int(line_split[0]),
                        "UUID": line_split[2],
                        "relevance": int(line_split[3]),
                    }
                )

        return result

    def get_relevance(self, uuid: str, topic: int) -> int:
        """Checks the relevance for UUID and topic number

        :param str uuid: UUID of the document
        :param int topic: topic number of the queried topic from topics.xml
        """

        for rel_dict in self.relevance_stats:
            if topic == rel_dict["topic"] and uuid == rel_dict["UUID"]:
                return rel_dict["relevance"]
        # relevance unknown; return relevance according to strategy
        if (
            self.unknown_relevance_strategy
            == UnknownRelevanceStrategy.ASSUME_NOT_RELEVANT
        ):
            return 0
        elif (
            self.unknown_relevance_strategy == UnknownRelevanceStrategy.ASSUME_RELEVANT
        ):
            return 1
        elif self.unknown_relevance_strategy == UnknownRelevanceStrategy.IGNORE:
            return -1
        else:
            raise ValueError("Unknown strategy: %s" % self.unknown_relevance_strategy)

    def get_judgements(self, topic: int) -> List:
        filtered = [judg for judg in self.relevance_stats if judg["topic"] == topic]
        sorted_by_relevance = sorted(
            filtered, key=lambda judg: judg["relevance"], reverse=True
        )
        return sorted_by_relevance

    def make_confusion_matrix(
        self, topic_id: int, predicted_doc_uuids: List[str]
    ) -> ConfusionMatrix:
        judgements = self.get_judgements(topic_id)  # a.k.a "actual" positive labels
        relevant_uuids = {
            judg["UUID"]
            for judg in judgements
            if judg["relevance"] >= self.relevance_threshold
        }
        non_relevant_uuids = {
            judg["UUID"] for judg in judgements if judg["UUID"] not in relevant_uuids
        }
        tp = len(
            [
                uuid
                for uuid in predicted_doc_uuids
                if self.get_relevance(uuid, topic_id) > 0
            ]
        )
        fp = len(
            [
                uuid
                for uuid in predicted_doc_uuids
                if self.get_relevance(uuid, topic_id) == 0
            ]
        )
        fn = len([uuid for uuid in relevant_uuids if uuid not in predicted_doc_uuids])
        # only true negatives are counted here for which an explicit relevance judgement is provided
        # the number of remaining true negatives is hard to calculate because the entire corpus of chat noir
        # must be considered for that
        tn = len(
            [uuid for uuid in non_relevant_uuids if uuid not in predicted_doc_uuids]
        )
        return ConfusionMatrix(tp=tp, tn=tn, fp=fp, fn=fn)
