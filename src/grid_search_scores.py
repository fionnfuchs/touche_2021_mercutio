import argparse
import itertools
import os

import pandas

from evaluation.metrics import calc_unweighted_measurements, calc_ndcgs
from evaluation.judgements import RelevanceJudgements, UnknownRelevanceStrategy
from log import get_child_logger
import pickling
from pipeline import Pipeline
from remerging import Remerging

logger = get_child_logger(__file__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src")
    parser.add_argument(
        "--qrels",
        help="Path to the qrels file with relevance judgements.",
        default=None,  # see class RelevanceJudgements for the default value
    )

    parser.add_argument(
        "-o",
        "--output",
        default=os.path.join(os.path.dirname(__file__), "..", "gridsearch"),
    )
    parser.add_argument(
        "-i",
        "--ranking-id",
        help="The id of the ranking run that is used for the grid search.",
    )
    parser.add_argument(
        "-s",
        "--strategy",
        help=f"Strategy for handling unknown relevance. Choose one of: {[strategy.value for strategy in UnknownRelevanceStrategy ]}",
        default=UnknownRelevanceStrategy.ASSUME_NOT_RELEVANT,
        type=lambda x: UnknownRelevanceStrategy(x),
    )
    parser.add_argument(
        "--start", help="Start value for the weights", type=float, default=0
    )
    parser.add_argument(
        "--end", help="End value for the weights", default=1.4, type=float
    )
    parser.add_argument(
        "--step", help="Step size for the grid search", default=0.2, type=float
    )
    parser.add_argument(
        "--ignore",
        help="Names of weights that will be ignored in the grid search. One of them should always be ignored.",
        default=["score"],
        nargs="+",
    )
    return parser.parse_args()


def run_gridsearch():
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

    if not (os.path.exists(args.output) and os.path.isdir(args.output)):
        raise RuntimeError(
            "%s must be a directory and must exist. Please create it." % args.output
        )
    ignore_str = ",i-".join(args.ignore)
    gridsearch_id = (
        f"{run_name}-ignore-{ignore_str}-{args.start}-{args.end}-{args.step}"
    )
    output_file = os.path.join(args.output, gridsearch_id + ".csv")

    evaluations = []
    for config, pl in iter_pipelines(
        used_config,
        start=args.start,
        end=args.end,
        step=args.step,
        ignore_scores=args.ignore,
    ):
        for topic in topics:
            pl.process(topic)
        metrics = evaluate_topics(judgements, topics)
        row = config
        row.update(
            {
                "ndcg_5": metrics.loc["ndcg_5", "mean"],
                "ndcg_10": metrics.loc["ndcg_10", "mean"],
                "ndcg_all": metrics.loc["ndcg_all", "mean"],
            }
        )
        evaluations.append(row)
    logger.info("Write to %s", output_file)
    df = pandas.DataFrame(evaluations)
    df = df.sort_values(by=["ndcg_5"], ascending=False)
    df.to_csv(output_file, index=False)
    print(df)


def iter_pipelines(config, start=0, end=1, step=0.2, ignore_scores=None):
    if ignore_scores is None:
        ignore_scores = ["score"]
    pl = Pipeline()
    remerging = Remerging(config)
    pl.add_pipe(remerging)
    weight_names = [
        weight_name
        for weight_name in remerging.weights.keys()
        if weight_name not in ignore_scores
    ]
    weights = [w for w in float_range(start, end, step)]
    for weight_combis in itertools.product(*[weights for _ in weight_names]):
        params = {}
        for i, weight in enumerate(weight_combis):
            name = weight_names[i]
            remerging.weights[name] = weight
            params[name] = weight
        yield params, pl


def float_range(start, end, step: float):
    current = start
    while current <= end:
        yield float(current)
        current += step


def evaluate_topics(judgements, topics):
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
    aggregated = pandas.DataFrame(
        {"mean": df.mean(), "min": df.min(), "max": df.max(), "std": df.std()}
    )
    return aggregated


if __name__ == "__main__":
    run_gridsearch()
