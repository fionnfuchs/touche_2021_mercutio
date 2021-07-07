import argparse
import os

from classifier_ranking import ClassifierRanking
from config import Config
from log import get_child_logger

from chat_noir import ChatNoirPipeLine
from doc_cleaning import DocumentCleaningPipeline

from models import Topic, ProcessingObject, Query

from pipeline import Pipeline
from query_expansion import QueryExpansion
from remerging import Remerging

from pickling import PickleTopicResults, load_topic_objects

from simple_term_ranking import SimpleTermRanking

from reevaluation_pipeline import ReEvaluationPipeline

import xml.etree.ElementTree as ET

import evaluation

from targer_ranking import TARGER_Rerank
from term_count_reranking import TermCountRanking
from trec import TrecGenerator

logger = get_child_logger(__file__)

AVAILABLE_PIPES = {
    "chat_noir": ChatNoirPipeLine,
    "query_expansion": QueryExpansion,
    "cleaning": DocumentCleaningPipeline,
    "remerging": Remerging,
    "pickling": PickleTopicResults,
    "simple_term_ranking": SimpleTermRanking,
    "classifier_ranking": ClassifierRanking,
    "targer_ranking": TARGER_Rerank,
    "reevaluate": ReEvaluationPipeline,
    "term_counts": TermCountRanking,
}


def main():
    cli_args = parse_cli()
    config = Config(path=cli_args.config)

    pl = load_pipeline(config, cli_args)

    if cli_args.identifier is not None:
        # ignore old config
        _, topics = load_topic_objects(cli_args.identifier)
    else:
        topics = load_topics(
            config.get(
                "dir",
                "topic_input",
                expected_type=str,
                raise_exc=False,
                default=None,
            )
        )
    if cli_args.limit_topics is not None:
        # process the first n-topics
        topics = topics[0 : cli_args.limit_topics]
    elif cli_args.single_topic is not None:
        if cli_args.single_topic > 0 and cli_args.single_topic <= 50:
            # process only one specified topic
            topics = [topics[cli_args.single_topic - 1]]
        else:
            logger.error(
                "The specfied topic number does not exist! Valid topic numbers: 1-50"
            )
            return

    for topic in topics:
        pl.process(topic)
    pl.cleanup()


def load_pipeline(config, cli_args) -> Pipeline:
    steps = config.get("pipeline", raise_exc=True, expected_type=list)
    pipeline = Pipeline()
    for pipe_name in steps:
        if pipe_name not in AVAILABLE_PIPES:
            raise ValueError("Pipe name %s is not known" % pipe_name)
        pipe = AVAILABLE_PIPES[pipe_name](config)
        pipeline.add_pipe(pipe)

    # add reevalation to pipeline if cli param is set
    if cli_args.judge == True:
        logger.info("Adding reevaluation pipe")
        pipeline.add_pipe(AVAILABLE_PIPES["reevaluate"](config))
    # add trec pipe if cli param is set
    if cli_args.trec is not None:
        logger.info("Adding trec pipe")
        pipeline.add_pipe(TrecGenerator(config, output_file=cli_args.trec))
    return pipeline


def load_topics(input_dir, version_name="v0"):
    topics = []
    tree = ET.parse(os.path.join(input_dir, "topics.xml"))
    root = tree.getroot()
    for child in root:
        query = child[1].text.replace("\n", "")
        number = int(child[0].text)
        t = Topic(
            topic_query=query,
            topic_number=number,
            version=version_name,
            processing_objects=[
                ProcessingObject(query=Query(query_type=0, query_text=query))
            ],
        )
        topics.append(t)
    return topics


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        help="Path to the configuration file",
        default=os.path.join(os.path.dirname(__file__), os.pardir, "config.yaml"),
    )
    parser.add_argument(
        "--limit-topics",
        "-l",
        help="Only process first n topics",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--single-topic",
        "-t",
        help="Only process a single topic specified by this number. This parameter is only used if the '--limit-topics' parameter is not set! Valid topic numbers: 1-50",
        type=int,
    )
    parser.add_argument(
        "--judge",
        "-j",
        help="If this flag is set, than the judgement/ reevalation pipeline is started. Combine this flag with the '-l' parameter to only judge the given topic.",
        action="store_true",
    )
    parser.add_argument(
        "--identifier",
        "-i",
        help="If this flag is set the topics are loaded from a the given identifier (e.g test). NOTE: The pipeline steps are executed regardless, so make sure the correct steps are set in the config.",
        type=str,
    )
    parser.add_argument(
        "--trec", help="Writes a trec file with the whole ranking", type=str
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
