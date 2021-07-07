from typing import Optional

from log import get_child_logger
from models import Topic

logger = get_child_logger(__file__)


class Pipeline:
    """The Pipeline is responsible for retrieving and ranking documents for queries."""

    def __init__(self):
        self.pipes = []

    def add_pipe(self, pipe):
        """Adds a pipe to the pipeline.

        :param Pipe pipe: The pipe to be added to the pipeline
        """

        self.pipes.append(pipe)

    def process(self, topic: Topic) -> Topic:
        for pipe in self.pipes:
            logger.info(
                "Starting pipe %s for Topic %d"
                % (pipe.__class__.__name__, topic.topic_number)
            )
            modified_topic = pipe.process(topic)
            if modified_topic is not None:
                topic = modified_topic
        return topic

    def cleanup(self):
        for pipe in self.pipes:
            pipe.cleanup()


class Pipe:
    """Interface for pipes."""

    def process(self, topic: Topic) -> Optional[Topic]:
        """Processes an array of objects.

        :param topic: Topic object that is processed and modified by the pipe
        :return: Can return the modified or new Topic object or None if the topic object is modified in-place
        """
        raise NotImplementedError

    def cleanup(self):
        pass


class BasicPipe(Pipe):
    """A basic pipe implementation for testing purposes."""

    def process(self, topic: Topic):
        """Processes an array of objects. Prints the query text of each object.

        :param topic: Topic to be processed by the pipe
        :return: Returns the processed list of objects.
        """
        for obj in topic.processing_objects:
            print(obj.query.query_text)
        return topic


class WrongPipePieceExeption(Exception):
    """An Exception defined for the case a Pipeline step is in a place it shouldnt be."""

    def __init__(self, pipe: Pipe, reason: str):
        super().__init__(
            f"Pipe {pipe.__class__.__name__} not in place. Reason: {reason}"
        )
