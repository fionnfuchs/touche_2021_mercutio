from typing import Optional

import pandas

from config import Config
from log import get_child_logger
from models import Topic
from pipeline import Pipe

DEFAULT_GROUP_NAME = "ul-t2-mercutio"
DEFAULT_Q0_VALUE = "Q0"

logger = get_child_logger(__file__)


class TrecGenerator(Pipe):
    def __init__(
        self,
        config: Config,
        output_file: str,
        group_name=DEFAULT_GROUP_NAME,
        q0_value=DEFAULT_Q0_VALUE,
    ):
        # note: not all variables are read from the config, as the TrecGenerator is not intended to be configured via it
        # (instead, the --trec CLI flag should be used for it)
        self.output_file = output_file
        self.q0_value = q0_value
        run_name = config.get("dir", "run_name")
        self.tag_name = f"{group_name}-{run_name}"
        # Create the dataframe which will be the trec in-memory representation
        # see https://webis.de/events/touche-21/shared-task-2.html for more details
        self.df = pandas.DataFrame(columns=["qid", "Q0", "doc", "rank", "score", "tag"])

    def process(self, topic: Topic) -> Optional[Topic]:
        for doc in topic.result_docs:
            # a row is like "1 Q0 clueweb12-en0010-85-29836 1 17.89 myGroupMyMethod"
            row = {
                "qid": topic.topic_number,
                "Q0": self.q0_value,
                "doc": doc.chat_noir_result.trec_id,
                "rank": doc.rank,
                "score": doc.combined_score,
                "tag": self.tag_name,
            }
            self.df = self.df.append(row, ignore_index=True)
        return topic

    def cleanup(self):
        logger.info("Write trec to %s", self.output_file)
        df["score"] = df["score"].apply(self._clean_score)
        self.df.to_csv(self.output_file, sep=" ", index=False)

    def _clean_score(self, value):
        if "[[" not in str(value):
            return value
        else:
            return float(str(value).replace("[[", "").replace("]]", ""))
