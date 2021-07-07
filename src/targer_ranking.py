from models import Topic, Document, ProcessingObject
from pipeline import Pipe, WrongPipePieceExeption
from enum import Enum
from typing import List
from log import get_child_logger
import re
import string
import traceback

import requests


targer_base_uri = "https://demo.webis.de/targer-api/"


logger = get_child_logger(__file__)


# lazy targer cashe
targer_cashe = {}


class TARGER_MODELS(Enum):
    """
    Models for the TARGER API:
    Combo -> Classifies input text to argument structure (Combo model - big dataset)
    ES -> Classifies input text to argument structure (Essays model, fasttext embeddings)
    ES_dep -> Classifies input text to argument structure (Essays model, dependency based)
    IBM -> Classifies input text to argument structure (IBM model, fasttext - big dataset)
    NewPE -> Classifies input text to argument structure (Essays model, fasttext embeddings)
    NewWD -> Classifies input text to argument structure (Essays model, fasttext - big dataset)
    WD -> Classifies input text to argument structure (WebD model, fasttext - big dataset)
    WD_dep -> Classifies input text to argument structure (WebD model, dependency based)
    """

    Combo = "classifyCombo"
    ES = "classifyES"
    ES_dep = "classifyES_dep"
    IBM = "classifyIBM"
    NewPE = "classifyNewPE"
    NewWD = "classifyNewWD"
    WD = "classifyWD"
    WD_dep = "classifyWD_dep"


class TARGER_Rerank(Pipe):
    def __init__(self, config):
        self.model = TARGER_MODELS[
            config.get("targer", "model", expected_type=str, default="Combo")
        ]
        self.min_confidence = config.get(
            "targer", "min_confidence", expected_type=float, default=0.75
        )
        self.arg_label = config.get(
            "targer", "arg_label", expected_type=str, default="both"
        )
        self.error_val_default = config.get(
            "targer", "error_default", expected_type=float, default=-1
        )

        self.ignore_punctuation = config.get(
            "targer", "ignore_punctuation", expected_type=bool, default=True
        )

        if self.arg_label == "claim":
            self.allowed_regex = re.compile(r"C-[A-Z]")
        elif self.arg_label == "premise":
            self.allowed_regex = re.compile(r"P-[A-Z]")
        else:
            self.allowed_regex = re.compile(r"[CP]-[A-Z]")

    def __query_targer_for_text(self, text: str) -> List[dict]:

        headers = {"Accept": "application/json", "Content-Type": "text/plain"}
        URI = targer_base_uri + self.model.value
        try:
            response = requests.post(
                URI,
                data=text.encode("utf-8"),
            )
        except Exception:
            logger.exception("Error in posting to targer:", stack_info=True)
            return None

        if response.status_code >= 400:
            logger.error(
                f"Error reaching TARGER API: Status {str(response.status_code)}; text length: {len(text)}"
            )
            return None

        return [d for l in response.json() for d in l]

    def __get_value_for_response(self, tagged_list) -> float:
        rel_word_count = 0
        punctuation_count = 0

        for tagged_word in tagged_list:
            if (
                self.allowed_regex.match(tagged_word["label"])
                and float(tagged_word["prob"]) >= self.min_confidence
                and tagged_word["token"] not in string.punctuation
            ):
                rel_word_count += 1
            elif tagged_word["token"] in string.punctuation:
                punctuation_count += 1

        counted_words = (
            len(tagged_list) - punctuation_count
            if self.ignore_punctuation
            else len(tagged_list)
        )

        return rel_word_count / counted_words

    def _get_targer_eval_value(self, text: str) -> float:
        try:
            tagged_list = self.__query_targer_for_text(text)
            if tagged_list is not None:
                v = self.__get_value_for_response(tagged_list)
                return v
            else:
                logger.warning(f"Error fetching targer evaluation")
                return self.error_val_default
        except Exception:
            logger.exception("Error requesting from targer:", exc_info=True)
            return self.error_val_default

    def process(self, topic: Topic):

        for po in topic.processing_objects:
            result_values = []

            if len(po.documents) < 1:
                logger.info(f"No results for {po.query.query_text}")
                continue

            for uuid in po.documents:
                d = po.documents[uuid]
                if uuid in targer_cashe:
                    score = targer_cashe[uuid]
                else:
                    if d.chat_noir_result.text == None:
                        logger.warn(
                            f"Cannot targ-rank document: {d}, because chat noir text is None!"
                        )
                        continue
                    score = self._get_targer_eval_value(d.chat_noir_result.text)

                if not uuid in targer_cashe and score != self.error_val_default:
                    targer_cashe[uuid] = score
                result_values.append(score)
                # set score between 0 and 1 (and error_default) to key targer
                po.documents[uuid].scores["targer"] = score

            error_count = 0
            result_sum = 0
            for s in result_values:
                result_sum += s
                if s == self.error_val_default:
                    error_count += 1

            error_rate = error_count / len(result_values)
            avg_rating = result_sum / len(result_values)

            logger.info(
                f"TARGER result for query '{po.query.query_text}': default error val: {str(error_rate)} . Avg rating: {str(avg_rating)}; Min: {str(min(result_values))} Max: {str(max(result_values))}"
            )

        return topic


def query_targer_for_text(text: str, model=TARGER_MODELS.Combo):
    headers = {"Accept": "application/json", "Content-Type": "text/plain"}
    URI = targer_base_uri + model.value
    response = requests.post(URI, data=text).json()

    return response
