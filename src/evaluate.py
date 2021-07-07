import argparse
import os

import pandas

from evaluation.metrics import calc_unweighted_measurements, calc_ndcgs
from evaluation.judgements import RelevanceJudgements, UnknownRelevanceStrategy
from log import get_child_logger
import pickling

logger = get_child_logger(__file__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qrels",
        help="Path to the qrels file with relevance judgements.",
        default=None,  # see class RelevanceJudgements for the default value
    )

    parser.add_argument(
        "-o",
        "--output",
        default=os.path.join(os.path.dirname(__file__), "..", "evaluation"),
    )

    parser.add_argument(
        "-i",
        "--ranking-id",
        help="REQUIRED The name of the ranking which will be evaluated.",
    )
    parser.add_argument(
        "-s",
        "--strategy",
        help=f"Strategy for handling unknown relevance. Choose one of: {[strategy.value for strategy in UnknownRelevanceStrategy ]}",
        default=UnknownRelevanceStrategy.ASSUME_NOT_RELEVANT,
        type=lambda x: UnknownRelevanceStrategy(x),
    )
    return parser.parse_args()


def run_evaluate():
    args = parse_args()

    judgements = RelevanceJudgements(
        qrels_path=args.qrels, unknown_relevance_strategy=args.strategy
    )
    # also returns the config used, not sure what to do with it
    used_config, topics = pickling.load_topic_objects(args.ranking_id)

    # the name of the current pipeline run f.e. "no-synonym-queries"
    # this name is used as a filename for the caching and the evaluation results
    run_name = used_config.get(
        "dir",
        "run_name",
        expected_type=str,
        raise_exc=True,
        default="v0",
    )
    evaluation_folder = args.output

    metrics = []
    for topic in topics:
        trec_ids = [doc.chat_noir_result.trec_id for doc in topic.result_docs]
        cm = judgements.make_confusion_matrix(topic.topic_number, trec_ids)
        topic_metrics = calc_unweighted_measurements(cm)
        topic_metrics.update({"topic_number": topic.topic_number})
        ndcgs = calc_ndcgs(topic, judgements)
        topic_metrics.update(ndcgs)
        metrics.append(topic_metrics)
    df = pandas.DataFrame(metrics)
    metrics_path = os.path.join(evaluation_folder, run_name + "_metrics.csv")
    average_path = os.path.join(evaluation_folder, run_name + "_average_metrics.csv")
    df.to_csv(metrics_path)

    aggregated = pandas.DataFrame(
        {"mean": df.mean(), "min": df.min(), "max": df.max(), "std": df.std()}
    )
    aggregated.to_csv(average_path)
    logger.info("Wrote metrics to %s and %s", metrics_path, average_path)
    print("\nAggregated metrics:")
    print(aggregated)


if __name__ == "__main__":
    run_evaluate()
