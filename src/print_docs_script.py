"""
    A script to print out all results for the given topic
"""

import argparse
import os

from log import get_child_logger
import pickling
from models import Topic
from config import Config
import hashlib
from typing import Dict, List

logger = get_child_logger(__file__)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--ranking-id",
        dest="ranking_id",
        default="v0",
        help="The id of the ranking which will be evaluated.",
    )

    parser.add_argument(
        "-t",
        "--topic_number",
        dest="topic_number",
        default=-1,
        type=int,
        help="If set to a value between 1 and 50, only this specific topic is going to be printed.",
    )

    parser.add_argument(
        "-l",
        "--length_of_output",
        default=None,
        type=int,
        help="If set to a value > 0, only the first n chars of the text are displayed.",
    )

    parser.add_argument(
        "-q",
        "--only_queries",
        action="store_true",
        help="If this flag is set, then only queries are printed.",
    )

    parser.add_argument(
        "-d",
        "--count_doubles",
        action="store_true",
        help="If this flag is set, then all doubles of documents are counted and printed.",
    )

    return parser.parse_args()


def __print_topic(t: Topic, length_of_output: int = None, only_queries: bool = False):

    print(f"\nTopic Query: {t.topic_query} Topic Number: {t.topic_number}")
    if only_queries == True:
        for po in t.processing_objects:
            print(f"{po.query.query_text}")
        return

    print(f"\nTopic Query: {t.topic_query} Topic Number: {t.topic_number}")
    print("Documents:")
    for d in t.result_docs:
        print(f"trec_id: {d.chat_noir_result.trec_id}")
        print(f"Our Stats\t -> Rank: {d.rank}, Score: {d.combined_score}")
        print(
            f"Chat Noir Stats\t -> Score: {d.chat_noir_result.score}, Spam Rank: {d.chat_noir_result.spam_rank}, Spam Rank: {d.chat_noir_result.page_rank}"
        )
        print(
            f"Snippet: {d.chat_noir_result.snippet.replace('<em>','').replace('</em>','')}"
        )

        try:
            print(f"Text: {d.chat_noir_result.text[:length_of_output]}")
        except UnicodeEncodeError as e:
            print(f"Could not encode characters in text, skipping it. Error: {e}")
        print(
            "---------------------------------------------------------------------------------------\n"
        )


def __count_doubles(t: Topic):
    hash_dict: Dict[str, List[str]] = {}
    pos_dict: Dict[str, List[str]] = {}
    document_counter = 0

    # the average position, in which these doubles occur
    doubles_pos = []

    for i, d in enumerate(t.result_docs):
        document_counter += 1
        try:
            hash_temp = hashlib.sha256(
                d.chat_noir_result.text.encode("utf8")
            ).hexdigest()
        except UnicodeEncodeError:
            # i have absolutely no idea how to handle this error
            continue

        if hash_temp in hash_dict:
            hash_dict[hash_temp].append(d.chat_noir_result.UUID)
            pos_dict[hash_temp].append(i + 1)
        else:
            hash_dict[hash_temp] = [
                d.chat_noir_result.UUID,
            ]
            pos_dict[hash_temp] = [
                i + 1,
            ]

    doubles_list = []
    most_duplicates = 0
    count_duplicate_docs = 0
    for key in hash_dict:
        if len(hash_dict[key]) > 1:
            doubles_list.append(hash_dict[key])
            count_duplicate_docs += len(hash_dict[key])
        if len(pos_dict[key]) > most_duplicates:
            most_duplicates = len(pos_dict[key])

    print(f"Topic {t.topic_number} - {t.topic_query}:")
    print(
        f"-> found: {len(doubles_list)} duplicate instances, with total {count_duplicate_docs} documents in {document_counter} documents with most duplicates: {most_duplicates}"
    )
    print(f"at positions: {[x for x in pos_dict.values() if len(x)>1]}")


def run_pretty_print():
    args = parse_args()

    requested_topic_number = args.topic_number

    output_char_length = args.length_of_output

    # load all topics from pickle
    _, topics = pickling.load_topic_objects(args.ranking_id)

    print_queries_only = args.only_queries

    if requested_topic_number >= 1 and requested_topic_number <= 50:
        if args.count_doubles:
            __count_doubles(topics[requested_topic_number - 1])
            return
    else:
        if args.count_doubles:
            for topic in topics:
                __count_doubles(topic)
            return

    if requested_topic_number >= 1 and requested_topic_number <= 50:
        __print_topic(
            topics[requested_topic_number - 1], output_char_length, print_queries_only
        )
    else:
        for topic in topics:
            __print_topic(topic, output_char_length, print_queries_only)


if __name__ == "__main__":
    run_pretty_print()
