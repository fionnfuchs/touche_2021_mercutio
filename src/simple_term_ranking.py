from pipeline import Pipe
from models import Topic
import pandas as pd
from log import get_child_logger

logger = get_child_logger(__file__)


class SimpleTermRanking(Pipe):
    def __init__(
        self,
        config,
    ):
        self.simple_terms = open("material/simple_terms.txt", "r").readlines()

    def process(self, topic: Topic) -> Topic:
        for po in topic.processing_objects:
            for uuid in po.documents:
                d = po.documents[uuid]
                text = d.chat_noir_result.text
                if text == None:
                    logger.warn(
                        f"Cannot process document: {d}, because chat noir text is None!"
                    )
                    continue
                score = 0
                for term in self.simple_terms:
                    if term in text:
                        score += 1
                po.documents[uuid].scores["simple_terms"] = score
        return topic
