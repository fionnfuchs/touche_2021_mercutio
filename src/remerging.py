from pipeline import Pipe
from models import Topic
from log import get_child_logger

logger = get_child_logger(__file__)

from log import get_child_logger

logger = get_child_logger(__file__)


class Remerging(Pipe):
    """An implementation of the Pipe object"""

    def __init__(
        self,
        config,
    ):

        self.weights = config.get("remerging", expected_type=dict)

        logger.info("Remerging weights: " + str(self.weights))

    def process(self, topic: Topic) -> Topic:
        result_dict = {}
        for po in topic.processing_objects:
            for uuid in po.documents:
                result_dict[uuid] = po.documents[uuid]
        result_docs = list(result_dict.values())

        if len(result_docs) == 0:
            logger.error(f"Cannot remerge documents, received empty document list!")
            raise RuntimeError("Received empty list of documents to remerge!")

        spam_rank_max = max([x.chat_noir_result.spam_rank for x in result_docs])
        page_rank_max = max([x.chat_noir_result.page_rank for x in result_docs])
        score_max = max([x.chat_noir_result.score for x in result_docs])

        max_scores = {}

        for doc in result_docs:
            for scorename, score in doc.scores.items():
                if scorename in max_scores:
                    if max_scores[scorename] < score:
                        max_scores[scorename] = score
                    else:
                        continue
                else:
                    max_scores[scorename] = score

        for doc in result_docs:
            doc.combined_score = 0.0
            for scorename, score in doc.scores.items():
                if max_scores[scorename] != 0:
                    weight = 1
                    if scorename in self.weights.keys():
                        weight = self.weights[scorename]
                    doc.combined_score += (score / max_scores[scorename]) * weight

            doc.combined_score -= (
                doc.chat_noir_result.spam_rank / spam_rank_max
            ) * self.weights["spam_rank"]
            doc.combined_score += (
                doc.chat_noir_result.page_rank / page_rank_max
            ) * self.weights["page_rank"]
            doc.combined_score += (
                doc.chat_noir_result.score / score_max
            ) * self.weights["score"]

        result_docs.sort(
            key=lambda x: (x.combined_score),
            reverse=True,
        )

        for i in range(len(result_docs)):
            result_docs[i].rank = i

        topic.result_docs = result_docs

        return topic
