import csv
import math
import os
from typing import List, Tuple

from config import Config
from evaluation.judgements import RelevanceJudgements, ConfusionMatrix
from log import get_child_logger
from models import Topic

logger = get_child_logger(__file__)


def __get_IDCG(relevance_stats, topic: int, p: int, rest_relevance: int = 0):
    """Calculates the IDCG (ideal DCG) of a topic

    :param int topic: topic number of the queried topic from topics.xml
    :param int p: length of result array
    :param int rest_relevance: defines the relevance of the fictionary fill up documents
    """

    # get all UUIDS relevant for this topic
    relevant_documents = []
    for d in relevance_stats:
        if d["topic"] == topic:
            relevant_documents.append(d)
    dcg = 0
    rank = 0
    for i, d in enumerate(
        sorted(
            relevant_documents,
            key=lambda dictionary: dictionary["relevance"],
            reverse=True,
        )
    ):
        rank = i + 1
        # makes sure that
        if rank > p:
            break
        dcg = dcg + (d["relevance"] / math.log2(rank + 1))

    # no use in calculating with rel=0
    if p > rank and rest_relevance > 0:
        for r in range(rank, p + 1):
            dcg = dcg + (rest_relevance / math.log2(r + 1))

    return dcg


def __get_DCG(judgements, doc_list, topic_number) -> Tuple[float, int]:
    """Calculates Discounted Cumulative Gain"""
    dcg = 0
    n_docs = 0
    for i, doc in enumerate(doc_list):
        rel = judgements.get_relevance(
            uuid=doc.chat_noir_result.trec_id,
            topic=topic_number,
        )
        if rel < 0:
            # relevance cannot be determined; ignore
            continue
        rank = i + 1
        dcg = dcg + safe_divide(rel, math.log2(rank + 1))
        n_docs += 1
    return dcg, n_docs


def calc_ndcgs(topic, judgements):
    relevance_stats = judgements.relevance_stats
    if len(topic.result_docs) <= 0:
        return {"ndcg_5": 0, "ndcg_10": 0, "ndcg_all": 0}
    # calculates the NDCG@5
    dcg_5, _ = __get_DCG(
        judgements,
        doc_list=topic.result_docs[:5],
        topic_number=topic.topic_number,
    )
    ndcg_5 = dcg_5 / __get_IDCG(relevance_stats, topic=topic.topic_number, p=5)

    # calculates the NDCG@10
    dcg_10, _ = __get_DCG(
        judgements,
        doc_list=topic.result_docs[:10],
        topic_number=topic.topic_number,
    )
    ndcg_10 = dcg_10 / __get_IDCG(relevance_stats, topic=topic.topic_number, p=10)

    # calculates the all NDCG
    dcg_all, n_docs = __get_DCG(
        judgements,
        doc_list=topic.result_docs,
        topic_number=topic.topic_number,
    )
    ndcg_all = dcg_all / __get_IDCG(relevance_stats, topic=topic.topic_number, p=n_docs)

    return {"ndcg_5": ndcg_5, "ndcg_10": ndcg_10, "ndcg_all": ndcg_all}


def get_recall(
    relevance_stats, topic: Topic, relevance_threshold: int = 1
) -> Tuple[int, int]:
    """
    Counts how many relevant documents were found for ONE topic.
    A document is relevant, if it is present in the provided relevance file.
    This function therefore counts how many relevant documents are in the provided relevance file
    and also how many of these documents were found by us.

    :param Topic topic: the topic object, for which the number of found relevant documents should be returned
    :param int relevance_threshold: only documents with a relevance >= relevance_threshold are going to be counted

    :returns tuple(the number of ALL relevant documents (recall), the number of found relevant documents)
    """

    number_of_relevant_documents = 0
    recall_count = 0

    # iterate over all relevance stats to find the matching trec_id
    for stat in relevance_stats:

        # we want to find relevant documents with the same topic number and the same UUID
        if stat["topic"] == topic.topic_number:

            # check relevance threshold
            # if it is lower, than the threshold, than skip this document
            if stat["relevance"] < relevance_threshold:
                continue

            number_of_relevant_documents += 1

            # iterate over all found documents
            for result_doc in topic.result_docs:
                if stat["UUID"] == result_doc.chat_noir_result.trec_id:
                    recall_count += 1
                    continue

    return number_of_relevant_documents, recall_count


def calc_unweighted_measurements(cm: ConfusionMatrix) -> dict:
    """
    Calculates different metrics for the values of a confusion matrix.
    For terminology see https://en.wikipedia.org/wiki/Precision_and_recall
    """
    tp, fp, fn, tn = cm.tp, cm.fp, cm.fn, cm.tn
    sd = safe_divide
    metrics = dict()
    p = tp + fn
    n = tn + fp
    metrics["precision"] = sd(tp, (tp + fp))
    metrics["recall"] = sd(tp, p)
    metrics["f1_score"] = sd(2 * tp, (2 * tp + fp + fn))
    metrics["accuracy"] = sd((tp + tn), (p + n))
    metrics["positives"] = p
    metrics["negatives"] = n
    metrics["tnr"] = sd(tn, n)
    metrics["npv"] = sd(tn, (tn + fn))
    metrics["fpr"] = sd(fp, n)
    metrics["fdr"] = sd(fp, (fp + tp))
    metrics["for"] = sd(fn, (fn + tn))
    metrics["fnr"] = sd(fn, (fn + tp))
    metrics["balanced_accuracy"] = (metrics["recall"] + metrics["tnr"]) / 2
    metrics["true_negatives"] = tn
    metrics["true_positives"] = tp
    metrics["false_negatives"] = fn
    metrics["false_positives"] = fp
    metrics["kappa"] = 1 - sd(
        1 - metrics["accuracy"],
        1 - sd((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn), pow(p + n, 2)),
    )
    metrics["mcc"] = sd(
        tp * tn - fp * fn, pow((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0.5)
    )
    metrics["support"] = n + p
    return metrics


def safe_divide(q1, q2) -> float:
    try:
        value = q1 / q2
    except ZeroDivisionError:
        value = float("NaN")
    return value
