from log import get_child_logger
from pipeline import Pipe
from models import Topic
from typing import Dict, Optional, List
from config import Config

import glob
import os
import dataclasses
import datetime

logger = get_child_logger(__file__)


@dataclasses.dataclass
class EvaluationObject:
    """
    A class to save the state of an evaluation.
    It also saves if an evaluation was previously loaded from a .qrels file or is a 'fresh' evaluation.
    """

    eval_value: int
    was_previously_evaluated: bool = False


class ReEvaluationPipeline(Pipe):
    """
    A pipeline that helps re-evaluating the received, merged and ranked chat noir results.
    This pipe needs interaction with the user.
    It returns the first n (100) results after the merging pipeline.
    The text appears one-by-one, the user has to read the text and evaluate if the text is relevant regarding the topic query.

    The results are then saved in an extra .qrels file.

    """

    # the number of documents one should evaluate
    NUMBER_OF_DOCS_TO_EVALUATE = 100

    def __init__(self, config: Config) -> None:
        self.evaluated_doc_dict: Dict[str, EvaluationObject] = {}

        # get 'material' dirname
        self.material_dirname = config.get("dir", "qrels_input", raise_exc=True)

    def __load_qrels_files(self, topic_number: int, material_dirname: str) -> None:
        """
        Loads the .qrels files from the 'material/' folder and parses the content of these files.
        The parsed content is then put into the evaluation dict.
        """
        # load all .qrels files from /material folder
        for f_name in glob.glob(material_dirname + "/" + "*.qrels"):
            logger.debug(f"Opening {f_name} to load evaluation values")
            with open(f_name, "r") as f:
                for line in f.readlines():
                    # every line in the .qrels files should have the following structure:
                    #    TOPIC_NUMBER 0 TREC_ID RELEVANCE
                    (topic_nr, _, trec_id, relevance) = line.split(" ")

                    # this could throw an exception
                    try:
                        # only load evaluated topic, if the topic number match
                        # because the same trec-id could be used in different topics
                        if int(topic_nr) == int(topic_number):
                            self.evaluated_doc_dict[str(trec_id)] = EvaluationObject(
                                int(relevance), was_previously_evaluated=True
                            )
                    except Exception as e:
                        logger.warn(f"Could not parse entry in .qrels file! Error:{e}")

    def __save_evaluation_as_qrels_file(self, topic_number: int) -> None:
        """
        Saves the results of the evaluation to a qrels file.
        Only saves values, that were not previously taken from another qrels file.

        The files created by this method follow this naming scheme:
            t_TOPICNUMBER_evaluation_UTCTIMESTAMP.qrels
        This allows only unique entry in every .qrels file.
        (Glaubt mir, ich hab mir deswgen nen kopf gemacht und das ist der beste Weg - zumindest ohne viel Mehr-Aufwand (; )
        """
        logger.info("Saving evaluation results...")
        # get current utc millisecond timestamp -> is a unique number
        utc_now_str = str(int(datetime.datetime.utcnow().timestamp() * 1000))
        file_name = f"t_{topic_number}_evaluation_{utc_now_str}.qrels"

        dir = os.path.dirname("material/")

        # the string that is written to the file (not appending line per line because less IO operations)
        line_list: List[str] = []

        for trec_id, evaluationObj in self.evaluated_doc_dict.items():
            # only add items if not already evaluated (makes merging easier later)
            if evaluationObj.was_previously_evaluated == False:
                line_list.append(
                    f"{topic_number} 0 {trec_id} {evaluationObj.eval_value}\n"
                )

        with open(os.path.join(dir, file_name), "w") as f:
            f.writelines(line_list)

        logger.info(f"Wrote all evaluations to {os.path.join(dir, file_name)}!")

        return

    def process(self, topic: Topic) -> Optional[Topic]:
        # clear doc evaluation dict
        self.evaluated_doc_dict.clear()

        # initialize the evaluation dict with trec_ids and pre-evaluated values
        logger.info("Loading .qrels files")
        self.__load_qrels_files(topic.topic_number, self.material_dirname)
        logger.info(f"Loaded {len(self.evaluated_doc_dict)} evaluated docs")

        print(
            "\n\t\t----------------------\n\t\t\tATTENTION\n\t\t----------------------"
        )
        print(
            """
            I am now going to display result texts after the merging step.
            You are asked to evaluate the texts.
            To evaluate the texts, please enter a number between 0 and 2.
            Results that are already evaluated by another person (or by you) are not going to be displayed.
            After this pipeline ends, the evaluations are stored in an extra '.qrels' file.
            You can stop the pipeline at every time, for this you need to enter 'quit' or 'q'."""
        )
        print(
            f"\n\n==> Evaluate topic {topic.topic_number} - Query: {topic.topic_query}"
        )
        has_quitted = False
        for i, doc in enumerate(topic.result_docs):
            if has_quitted == True:
                break

            if i > self.NUMBER_OF_DOCS_TO_EVALUATE:
                logger.info(
                    f"Already have {self.NUMBER_OF_DOCS_TO_EVALUATE} evaluated documents for this topic!"
                )
                break
            if doc.chat_noir_result.trec_id in self.evaluated_doc_dict:
                logger.info(
                    f"Document with trec_id: {doc.chat_noir_result.trec_id} already evaluated! Skipping it..."
                )
                continue

            print(
                "-----------------------------------------------------------------------------------------------"
            )
            print(
                "-----------------------------------------------------------------------------------------------"
            )
            print(
                f"\nEvaluating document: {i}/{self.NUMBER_OF_DOCS_TO_EVALUATE} trec_id:{doc.chat_noir_result.trec_id} UUID:{doc.chat_noir_result.UUID}"
            )
            print(
                "-----------------------------CHAT NOIR TEXT-----------------------------------------------------"
            )
            print(doc.chat_noir_result.text)

            print(
                "-----------------------------CHAT NOIR SNIPPET-------------------------------------------------"
            )
            # remove all <em> and </em> tags
            print(doc.chat_noir_result.snippet.replace("<em>", "").replace("</em>", ""))
            print(
                "-----------------------------------------------------------------------------------------------"
            )
            print("Please evaluate the document now! (enter 'q'/ 'quit' to quit)")

            # an endless while loop, which only gets repeated if the user enters a wrong evaluation number
            while True:
                user_input = input("Evaluation: ")
                try:
                    if user_input == "quit" or user_input == "q":
                        print("Stopping evaluation...")
                        has_quitted = True
                        break

                    evaluation_number = int(user_input)

                    if evaluation_number not in range(0, 3):
                        raise ValueError(
                            f"{evaluation_number} is not a valid judge value. 0,1,2 are valid judge values!"
                        )

                    self.evaluated_doc_dict[
                        doc.chat_noir_result.trec_id
                    ] = EvaluationObject(
                        eval_value=evaluation_number, was_previously_evaluated=False
                    )

                    # break when the user input was OK
                    break

                except ValueError as ve:
                    print(f"Please enter a valid number! Error: {ve}")

        print("Done evaluating the documents!")

        self.__save_evaluation_as_qrels_file(topic.topic_number)

        # return nothing
        return None
