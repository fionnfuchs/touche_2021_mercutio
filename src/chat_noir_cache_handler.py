import os
import logging
import sys
import pickle  # for serializing the web-cache

from log import get_child_logger
from models import Document, ChatNoirResult
from typing import Dict, List
import hashlib
import operator

logger = get_child_logger(__file__)


def __sort_chat_noir_results_by_score(c: ChatNoirResult):
    return c.score


class CacheObject:
    def __init__(self):
        # a dict that stores a list of string UUIDs for every query
        self.query_dict: Dict[str, List[str]] = {}

        # a dict that stores every chat noir result by its UUID value
        self.document_dict: Dict[str, ChatNoirResult] = {}

    def add_new_query(
        self, query_str: str, chat_noir_result_list: List[ChatNoirResult]
    ):
        """
        Adds a new query with a list of chat noir results to the cache.
        """
        # a list to contain all chat noir result document UUIDs for this query
        doc_uuid_list = []

        # create list of document UUIDs and
        # add chat noir results by their UUID to the document_dict
        for i, _ in enumerate(chat_noir_result_list):
            doc_uuid_list.append(chat_noir_result_list[i].UUID)
            self.document_dict[chat_noir_result_list[i].UUID] = chat_noir_result_list[i]

        # check if the query is already present in the cache
        # if this is the case, compare the cached UUID lists, if they are not the same,
        # then merge them and delete duplicates
        # this case should only occur if the number of retrieved documents per query is changed
        if query_str in self.query_dict:
            # merges two lists and removes duplicates
            self.query_dict[query_str] += list(
                set(doc_uuid_list) - set(self.query_dict[query_str])
            )
        else:
            # query is new to cache, just add UUDs
            # add new entry to cache with document list
            self.query_dict[query_str] = doc_uuid_list

    def get_chat_noir_results_by_query(
        self, query_str: str, top_n: int = 100
    ) -> List[ChatNoirResult]:
        """
        Returns the first n cached chat noir results for a given query.
        The returned results are sorted by their chat noir score.
        Returns 'None' if the query was not already cached.
        """
        # check existance of query in cache
        # if no entry exists, return 'None'
        if query_str not in self.query_dict:
            logger.debug(
                "Cannot get documents for query '%s'. It was not cached. Returning 'None'."
                % (query_str)
            )
            return None

        result_document_list: List[ChatNoirResult] = []

        # get list of cached document UUIDs
        doc_uuid_list = self.query_dict[query_str]

        # get the document for every uuid in the list
        for uuid in doc_uuid_list:
            result_document_list.append(self.document_dict[uuid])
        logger.debug(
            "cache returned %d documents for %s"
            % (len(result_document_list), query_str)
        )

        result_document_list.sort(key=operator.attrgetter("score"))

        return result_document_list[:top_n]


class CacheHandler:
    def __init__(self, cache_filename):
        self.cache_filename = cache_filename
        self.cacheObject = None

    def load_cache(self):
        """Initialize CacheObject from saved json file"""
        logger.info("Loading chat noir cache file")
        # check if the cache file exists
        if os.path.exists(self.cache_filename):
            with open(self.cache_filename, "r+b") as f:
                self.cacheObject = pickle.load(f)
                if self.cacheObject != None:
                    logger.debug("Successfully loaded chat noir cache file")
                else:
                    logger.warn(
                        "Something went wrong when trying to load cache object!"
                    )
                    self.cacheObject = CacheObject()
        else:
            self.cacheObject = CacheObject()
            logger.warn("Chat noir cache file does not yet exist!")

    def save_cache(self):
        logger.info("Saving cache results to %s" % (self.cache_filename))
        with open(self.cache_filename, "w+b") as f:
            pickle.dump(self.cacheObject, f)

    def add_new_query_cache(self, query: str, chat_noir_results: List[ChatNoirResult]):
        """
        Adds a new query and a list of chat noir results to the cache file.
        """
        self.cacheObject.add_new_query(query, chat_noir_results)

    def get_cache_entry(self, query: str, top_n: int = 100) -> List[ChatNoirResult]:
        """Returns list of ChatNoirResults for this query or None if the cache object does not hold values for this query"""
        return self.cacheObject.get_chat_noir_results_by_query(query, top_n)
