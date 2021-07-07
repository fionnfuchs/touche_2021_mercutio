import os
import logging
import asyncio
import aiohttp

from log import get_child_logger
from pipeline import Pipe
from models import Document, ChatNoirResult, ProcessingObject, Topic
import chat_noir_cache_handler
from typing import Dict, List

logger = get_child_logger(__file__)


class ChatNoirPipeLine(Pipe):
    """An implementation of the Pipe object"""

    def __init__(self, config):
        # get config params
        # specifies the number of documents retrieved per query
        self.docs_per_query = config.get(
            "chat_noir", "docs_per_query", expected_type=int, default=100
        )

        # documents with a score lower than this value are not included, if None it is not filtered
        self.score_threshold = config.get(
            "chat_noir",
            "score_threshold",
            expected_type=float,
            raise_exc=False,
            default=None,
        )

        # documents with a spam rank higher than this value are not included, if None it is not filtered
        self.spam_rank_threshold = config.get(
            "chat_noir",
            "spam_threshold",
            expected_type=float,
            raise_exc=False,
            default=None,
        )

        # documents with a page rank lower than this value are not included, if None it is not filtered
        self.page_rank_threshold = config.get(
            "chat_noir",
            "page_rank_treshold",
            expected_type=float,
            raise_exc=False,
            default=None,
        )

        # init chat noir handler
        self.chat_noir_handler = ChatNoirHandler(
            chat_noir_api_key=config.get("chat_noir", "api_key", expected_type=str),
            number_of_query_retries=config.get(
                "chat_noir",
                "number_of_query_retries",
                expected_type=int,
                raise_exc=False,
                default=4,
            ),
        )

        self.is_cache_enabled = config.get(
            "chat_noir", "is_caching_enabled", expected_type=bool, raise_exc=True
        )

        if self.is_cache_enabled == True:
            # the cache filename of the chat noir result cache
            # this value does not really need to be changed
            cache_filename = config.get(
                "chat_noir",
                "cache_filename",
                expected_type=str,
                raise_exc=True,
            )
            cacheHandler = chat_noir_cache_handler.CacheHandler(cache_filename)
            cacheHandler.load_cache()
            self.cacheHandler = cacheHandler
        else:
            self.cacheHandler = None

    async def __handleResult(self, obj, result, session):
        # only filter by this value if it is not None
        # this allows to disable this filter completely
        if self.score_threshold != None and self.score_threshold >= 0:
            # only include documents with a scoring higher than the threshold
            if float(result["score"]) < self.score_threshold:
                return

        if self.spam_rank_threshold != None and self.spam_rank_threshold >= 0:
            # only check spam_rank if not null entry
            if str(result["spam_rank"]) != "null":
                # skip results with a spam rank too high
                if float(result["spam_rank"]) > self.spam_rank_threshold:
                    return

        if self.page_rank_threshold != None and self.page_rank_threshold >= 0:
            # only check page_rank if not null entry
            if str(result["page_rank"]) != "null":
                # skip results with a spam rank too high
                if float(result["page_rank"]) > self.page_rank_threshold:
                    return

        # query the complete document based on UUID
        result_document_text = await self.chat_noir_handler.retrieve_full_document(
            session, result["uuid"]
        )

        # create a chat noir result object
        chat_noir_result = ChatNoirResult(
            UUID=result["uuid"],
            trec_id=result["trec_id"],
            text=result_document_text,
            page_rank=result["page_rank"],
            spam_rank=result["spam_rank"],
            score=result["score"],
            snippet=result["snippet"],
        )

        # append new Document object
        obj.documents[result["uuid"]] = Document(chat_noir_result)

    async def __process_single_obj(
        self,
        obj: ProcessingObject,
        session: aiohttp.ClientSession,
        number_of_documents=100,
    ):
        cache_result = None
        if self.is_cache_enabled:
            # check if there exists a cache value for the given query
            # if cache_result is None, it means that there exists no cache values for this query
            cache_result: List[ChatNoirResult] = self.cacheHandler.get_cache_entry(
                obj.query.query_text, number_of_documents
            )
            if cache_result != None:
                # the cache result is a list of ChatNoirResults and the fingerprint of the website's HTML
                logger.info(
                    'Using cache results for query "%s". Got %d chat-noir-results'
                    % (obj.query.query_text, len(cache_result))
                )
                for cresult in cache_result:
                    # put all chat noir results in the document dict
                    # we do not need to create a deep-copy of the chat noir result
                    # since they are not going to be changed (hopefully ;)
                    obj.documents[cresult.UUID] = Document(
                        cresult
                    )  # create a 'Document' from 'chat_noir_result'
                return

        # if caching is disabled, the cache_result is None too
        if cache_result == None:

            query_response = None

            # check if a phrase search is requested or a simple search query
            # the simple search query is the default query of the chat-noir search engine
            # it provides modern search engine features like conjunctions and disjunctions
            logger.info('Querying: "%s"' % (obj.query.query_text))
            if obj.query.is_phrase_search == False:
                query_response = await self.chat_noir_handler.query_simple_search(
                    session, obj.query.query_text, results_per_page=number_of_documents
                )
            else:
                query_response = await self.chat_noir_handler.query_phrase_search(
                    session, obj.query.query_text, results_per_page=number_of_documents
                )

            if query_response == None or "error" in query_response:
                logger.debug(
                    "Chat-Noir-handler returned 'None', not processing/saving query \"%s\""
                    % (obj.query.query_text)
                )
                # skipping errors
                # it is already logged by the ChatNoirHandler
                return

            result_list = query_response["results"]

            sub_tasks = []

            for result in result_list:
                sub_tasks.append(self.__handleResult(obj, result, session))

            await asyncio.gather(*sub_tasks, return_exceptions=True)

            # save results to cache if caching is enabled
            if self.is_cache_enabled:
                self.cacheHandler.add_new_query_cache(
                    query=obj.query.query_text,
                    chat_noir_results=[
                        doc.chat_noir_result for doc in list(obj.documents.values())
                    ],
                )

    def process(self, topic: Topic, loop=None):
        """Processes an array of objects. Prints the query text of each object.

        :param list(PipelineObject) objs: Objects to be processed by the pipe
        :return: Returns the processed list of objects.
        """

        # set loop if it wasn set before
        if loop == None:
            loop = asyncio.get_event_loop()

        async def fetch_results(topic, number_documents):
            """This function fetches all results from the chat-noir-api"""
            # set limits for TCP connections
            conn = aiohttp.TCPConnector(limit=30, limit_per_host=30)
            tasks = []
            async with aiohttp.ClientSession(connector=conn) as session:
                # iterate over all ProcessingObjects and therefore all queries
                for obj in topic.processing_objects:
                    tasks.append(
                        self.__process_single_obj(
                            obj,
                            session,
                            number_of_documents=number_documents,
                        )
                    )

                await asyncio.gather(*tasks, return_exceptions=True)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(fetch_results(topic, self.docs_per_query))

        # delete all document objects with chat noir text = None
        for po in topic.processing_objects:
            for key in po.documents:
                if po.documents[key].chat_noir_result.text == None:
                    po.documents[key] = None
                    logger.warn(
                        f"Deleting document with UUID: {key}, because chat noir text was not retrieved."
                    )

            # clear up all dict entries set to None from previous step
            for uuid in [uuid for uuid in po.documents if po.documents[uuid] == None]:
                del po.documents[uuid]

        # save cache files, so if the pipeline fails, we still have the cache until this point
        if self.is_cache_enabled:
            self.cacheHandler.save_cache()
        return topic

    def cleanup(self):
        # save cache results to file
        # the chat noir results are cached at the end, so results are only saved to the cache
        # if the pipeline run was successful
        if self.is_cache_enabled:
            self.cacheHandler.save_cache()


class ChatNoirHandler:
    """The ChatNoirHandler shall be responsible for all kind of request against the ChatNoir Api: https://www.chatnoir.eu/doc/api/."""

    # default index we should use for the chat noir api, this needs to be tuple, currently we only use cw12
    __index = "cw12"
    # default number of results we expect from a query
    __number_of_results = 100
    # default url for the chat noir api
    __base_api_url = "https://www.chatnoir.eu/"

    # number of query retries after receiving an error
    __number_of_query_retries = 4

    def __init__(self, chat_noir_api_key, number_of_query_retries=4):
        if chat_noir_api_key == None or len(chat_noir_api_key) == 0:
            logger.error("no valid chat noir api key was passed to the handler")
        else:
            self.__CHAT_NOIR_API_KEY = chat_noir_api_key
            __number_of_query_retries = number_of_query_retries

    async def query_simple_search(
        self,
        session,
        query_string,
        pagination_begin=0,
        results_per_page=__number_of_results,
        is_explained=False,
        is_pretty=True,
    ) -> dict:
        """Creates a POST request to query the chat noir api.
        This query is a 'simple search' request to query simple words and wordgroups.

        :param str query_string: The query that is issued to the API. It can contain chat noir specific operators like 'AND','OR', ... (see API doc)
        :param int pagination_begin: Result pagination begin
        :param int results_per_page : Number of results per page
        :param bool is_explained : Return additional scoring information (boolean flag)
        :param bool is_pretty : Format the response in a more human-readable way
        :return: If an error occured, a dict with 'error' is returned. If there was no error, a dict with 'meta' and 'results' is returned.
        :rtype: dict

        meta : dict
            a dictionary containing the following items
            query_time : int
                query time in milliseconds
            total_results : int
                number of total hits
            indices : list
                list of indices that were searched
        results : list
            a list containing multiple results with the following properties
                score : float
                    ranking score of this result
                uuid : str
                    Webis UUID of this document
                index : str
                    index the document was retrieved from
                trec_id : str
                    TREC ID of the result if available (null otherwise)
                target_hostname : str
                    web host this document was crawled from
                target_uri : str
                    full web URI
                page_rank : int
                    page rank of this document if available (null otherwise)
                spam_rank : int
                    spam rank of this document if available (null otherwise)
                title : str
                    document title with highlights
                snippet : str
                    document body snippet with highlights
                explanation : str
                    additional scoring information if explain was set to true
        """

        request_specifier = "/api/v1/_search"

        request_data = {
            "apikey": self.__CHAT_NOIR_API_KEY,
            "query": query_string,
            "index": self.__index,
            "from": pagination_begin,
            "size": results_per_page,
            "explain": is_explained,
            "pretty": is_pretty,
        }

        # number of tries a query when an error occurs
        query_request_counter = self.__number_of_query_retries

        try:
            while True:
                logger.debug("Requesting simple search query: %s" % (query_string))
                async with session.post(
                    url=self.__base_api_url + request_specifier, data=request_data
                ) as response:

                    # check status of response (200 response is desired)
                    status = response.status
                    if status != 200:
                        # in case of error, decrease request_counter and try query again
                        query_request_counter -= 1
                        if query_request_counter <= 0:
                            logger.warn(
                                'Could not retrieve documents for query "%s", HTTP Status %d'
                                % (query_string, status)
                            )
                            return None
                        else:
                            logger.debug(
                                'Retrying query "%s" (%d retries left)'
                                % (query_string, query_request_counter)
                            )
                    else:
                        return await response.json()

        except Exception as err:
            # not a good practice -> pokemon error handling "gotta catch them all"
            logger.error(f"query_simple_search error occurred: {err}")
            return None

    async def query_phrase_search(
        self,
        session,
        query_string,
        slop=0,
        pagination_begin=0,
        results_per_page=__number_of_results,
        is_minimal=False,
        is_explained=False,
        is_pretty=True,
    ) -> dict:
        """Creates a POST request to query the chat noir api.
        This query is a 'phrase search' request to retrieve snippets containing certain fixed phrases.

        :param str query_string : The query that is issued to the API. It can contain chat noir specific operators like 'AND','OR', ... (see API doc).
        :param int slop : How far terms in a phrase may be apart (valid values: 0, 1, 2; default: 0).
        :param int pagination_begin : Result pagination begin
        :param int results_per_page : Number of results per page
        :param bool is_minimal : Reduce result list to 'score', 'uuid', 'target_uri' and 'snippet' for each hit.
        :param bool is_explained : Return additional scoring information.
        :param bool is_pretty : Format the response in a more human-readable way.
        :return: If an error occured, a dict with 'error' is returned. If there was no error, a dict with 'meta' and 'results' is returned.
        :rtype: dict

        meta : dict
            a dictionary containing the following items
            query_time : int
                query time in milliseconds
            total_results : int
                number of total hits
            indices : list
                list of indices that were searched
        results : list
            a list containing multiple results with the following properties
                score : float
                    ranking score of this result
                uuid : str
                    Webis UUID of this document
                index : str
                    index the document was retrieved from
                trec_id : str
                    TREC ID of the result if available (null otherwise)
                target_hostname : str
                    web host this document was crawled from
                target_uri : str
                    full web URI
                page_rank : int
                    page rank of this document if available (null otherwise)
                spam_rank : int
                    spam rank of this document if available (null otherwise)
                title : str
                    document title with highlights
                snippet : str
                    document body snippet with highlights
                explanation : str
                    additional scoring information if explain was set to true
        """

        request_specifier = "/api/v1/_phrases"

        request_data = {
            "apikey": self.__CHAT_NOIR_API_KEY,
            "query": query_string,
            "slop": slop,
            "index": self.__index,
            "from": pagination_begin,
            "size": results_per_page,
            "minimal": is_minimal,
            "explain": is_explained,
            "pretty": is_pretty,
        }

        # number of tries a query when an error occurs
        query_request_counter = self.__number_of_query_retries

        try:
            while True:
                logger.debug("Requesting phrase search query: %s" % (query_string))
                async with session.post(
                    url=self.__base_api_url + request_specifier, data=request_data
                ) as response:
                    # check status of response (200 response is desired)
                    status = response.status
                    if status != 200:
                        # in case of error, decrease request_counter and try query again
                        query_request_counter -= 1
                        if query_request_counter <= 0:
                            logger.warn(
                                'Could not retrieve documents for query "%s", HTTP Status %d'
                                % (query_string, status)
                            )
                            return None
                        else:
                            logger.debug(
                                'Retrying the query "%s" (%d times left)'
                                % (query_string, query_request_counter)
                            )
                    else:
                        return await response.json()
        except Exception as err:
            logger.error(f"error occurred: {err}")
            return None

    async def retrieve_full_document(self, session, UUID, is_plain=True) -> str:
        """Retrieves full documents from chat noir api
        The used index is the same specified in the class attribute.

        :param str UUID : UUID of the requested document
        :param bool is_plain : returns documents in plain text rendering with basic HTML-subset
        :return: full HTML contents of a document
        :rtype: str
        """

        if len(UUID) == 0 or UUID == None:
            logger.error("canno retrieve chat noir document, no UUID specified!")
            return {}

        if is_plain:
            # this is as cool as the 'requests' way of handling query parameters, but its enough for this project ;)
            request_specifier = "/cache?uuid={uuid:s}&index={index:s}&raw&plain".format(
                uuid=UUID, index=self.__index
            )
        else:
            request_specifier = "/cache?uuid={uuid:s}&index={index:s}&raw".format(
                uuid=UUID, index=self.__index
            )

        # number of tries a query when an error occurs
        query_request_counter = self.__number_of_query_retries

        try:
            while True:
                logger.debug(
                    "Requesting full document: %s"
                    % (self.__base_api_url + request_specifier)
                )
                async with session.get(
                    self.__base_api_url + request_specifier
                ) as response:
                    # check status of response (200 is desired)
                    status = response.status
                    if status != 200:
                        query_request_counter -= 1
                        if query_request_counter <= 0:
                            logger.warn(
                                "Could not retrieve document %s, HTTP Code: %d"
                                % (UUID, status)
                            )
                            return None
                        else:
                            logger.debug(
                                "Retrying the document query (%d retries left)"
                                % (query_request_counter)
                            )
                    else:
                        logger.debug("Received document response for %s" % (UUID))
                        return await response.text()

        except Exception as err:
            logger.error(f"retrieve_full_document error: {err}")
            return None
