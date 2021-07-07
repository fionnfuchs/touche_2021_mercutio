from bs4 import BeautifulSoup
from log import get_child_logger
from pipeline import Pipe
from models import Document, ChatNoirResult

logger = get_child_logger(__file__)


class DocumentCleaningPipeline(Pipe):
    """Pipeline process to clean the documents by removing HTML tags"""

    def __init__(self, config):
        pass

    def __clean_html_text(self, text) -> str:
        """extracts text from an HTML string and returns it"""
        # create soup object
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    def process(self, topic):
        """cleans the chat noir result documents"""
        logger.info("Starting cleaning pipeline...")
        # iterate over all ProcessingObjects and therefore all queries
        for obj in topic.processing_objects:
            # iterate over the dict of documents and clean the text from every document
            for doc_name in obj.documents:
                # check if there really exists a chat_noir_result object for this document
                if obj.documents[doc_name] != None:
                    if obj.documents[doc_name].chat_noir_result != None:
                        # check if already cleaned
                        if obj.documents[doc_name].chat_noir_result.is_cleaned == True:
                            continue

                        if obj.documents[doc_name].chat_noir_result.text != None:
                            obj.documents[
                                doc_name
                            ].chat_noir_result.text = self.__clean_html_text(
                                obj.documents[doc_name].chat_noir_result.text
                            )
                            obj.documents[doc_name].chat_noir_result.is_cleaned = True

        logger.info("Cleaning pipeline finished")
        return topic
