from pipeline import Pipe
from models import Topic
from typing import Optional
import os
import pickle
from config import Config

data_path = os.path.split(__file__)[0]
data_path = os.path.join(os.path.split(data_path)[0], "cache")

from log import get_child_logger

logger = get_child_logger(__file__)


class PickleTopicResults(Pipe):
    """
    Pipe for saving the current state of a topic, can be added in every part.
    Uses the topic version as identifier
    """

    def __init__(self, config: Config):
        """ """
        self.config = config
        # the name of the current pipeline run f.e. "no-synonym-queries"
        # this name is used as a filename for the caching and the evaluation results
        self.result_id = config.get(
            "dir",
            "run_name",
            expected_type=str,
            raise_exc=True,
            default="v0",
        )

        self.results_path = os.path.join(data_path, self.result_id)
        os.makedirs(self.results_path, exist_ok=True)
        logger.info("using %s to save results" % (self.results_path))
        with open(self.results_path + os.sep + "config", "wb+") as config_file:
            pickle.dump(self.config, config_file)

    def process(self, topic: Topic) -> Optional[Topic]:
        topic_file_path = os.path.join(self.results_path, str(topic.topic_number))
        with open(topic_file_path, "wb+") as topic_file:
            pickle.dump(topic, topic_file)


def load_topic_objects(results_id: str):
    """Loads the topics from a given version, returns them as list. Returns None if version not available"""

    result = []
    results_path = os.path.join(data_path, results_id)

    with open(results_path + os.sep + "config", "rb") as config_file:
        config = pickle.load(config_file)

    if not os.path.isdir(results_path):
        logger.error(f"ERROR: Can't find data path {results_path}")
        return None

    for i in range(1, 51):

        topic_file_path = os.path.join(results_path, str(i))
        if not os.path.isfile(topic_file_path):
            logger.error(
                f"Error: No saved results for named {topic_file_path} available."
            )
            continue

        with open(topic_file_path, "rb") as topic_file:
            topic = pickle.load(topic_file)

            result.append(topic)
    return config, result
