import itertools
from typing import List, Dict
from dataclasses import fields

import nltk
from nltk.corpus import wordnet

from config import Config
from log import get_child_logger
from pipeline import Pipe
from models import Topic, ProcessingObject, Query

import gensim.downloader
from gensim.models import Word2Vec
from enum import Enum

import Levenshtein

import spacy
import re

from transformers import pipeline

logger = get_child_logger(__file__)

CONFIG_BASE = "query_expansion"


class PosType(Enum):
    ENTITY = 1
    FEATURE = 2
    VERB = 3


models_mem = {}


class GensimWrapper:
    """
    A class to handle w2v (word-2-vec) operations based on gensim.
    This wrapper can handle self-trained models by passing the path to the model or download a pre-trained gensim model.
    """

    def __init__(
        self,
        model_name: str,
        n_synonyms: int = 10,
        min_levenshtein_distance: int = 2,
        is_using_self_trained_model: bool = False,
    ):
        self.model = None
        if is_using_self_trained_model == True:
            logger.debug(f"Loading self-trained w2v model: {model_name} ...")
            self.model = Word2Vec.load(model_name)
        else:
            logger.debug(f"Loading pre-trained w2v model: {model_name} ...")
            self.model = gensim.downloader.load(model_name)
        self.n_synonyms = n_synonyms
        self.min_levenshtein_distance = min_levenshtein_distance

    def get_synonyms(self, word: str) -> List[str]:
        word = word.lower()
        if word not in self.model.wv:
            logger.warn(
                f"Word {word} could not be found in the gensim corpus! Skipping it..."
            )
            return []

        similarities = self.model.wv.similar_by_word(word=word, topn=self.n_synonyms)

        # remove similarity value to only keep the word
        similarities = [word[0] for word in similarities]

        # remove all synonyms with a levenshtein distance <= 2
        similarities = [
            sym
            for sym in similarities
            if Levenshtein.distance(word, sym) > self.min_levenshtein_distance
        ]

        return similarities

    def get_semantic_similiar(self, word: str, pos_words: list, neg_words: list):
        word = word.lower()
        if word not in self.model.wv:
            return []

        # remove all words which are not in the vocab
        pos_words = [x for x in pos_words if x in self.model.wv]
        neg_words = [x for x in neg_words if x in self.model.wv]

        pos_words.insert(0, word)

        similarities = self.model.wv.most_similar(
            positive=pos_words, negative=neg_words, topn=self.n_synonyms
        )

        # remove similarity value to only keep the word
        similarities = [word[0] for word in similarities]

        # remove all synonyms with a levenshtein distance <= 2
        similarities = [
            sym
            for sym in similarities
            if Levenshtein.distance(word, sym) > self.min_levenshtein_distance
        ]

        return similarities


class SpacyWrapper:
    TAGS = {
        PosType.ENTITY: {"PROPN", "NOUN"},
        PosType.FEATURE: {"ADJ"},
        PosType.VERB: {"VERB"},
    }

    def __init__(self):
        self.model = spacy.load("en_core_web_lg")

    def get_tags_of_type(self, sentence: str, tag_type: PosType):
        """
        Returns all words from the sentence with the specified POS tag type.
        """
        spacy_tags = SpacyWrapper.TAGS[tag_type]
        doc = self.model(sentence)
        # Token and Tag
        tags = [str(token) for token in doc if token.pos_ in spacy_tags]
        return tags

    def get_tag_for_word(self, word: str):
        """ "
        Returns the POS tag type for a given word.
        """
        doc = self.model(word)
        for token in doc:
            return token.pos_


class QueryExpansion(Pipe):
    def __init__(self, config: Config):
        self.__config = config
        self.__use_synonyms = self.__config.get(
            CONFIG_BASE, "enabled", expected_type=bool
        )

        if self.__use_synonyms == False:
            return

        self.__modes = self.__config.get(
            CONFIG_BASE,
            "mode",
            expected_type=str,
            default="bilbo_baggins",
        ).split(",")

        self.number_of_additional_context_words = self.__config.get(
            CONFIG_BASE,
            "number_of_additional_context_words",
            expected_type=int,
            default=1,
        )

        self.__use_default_query = self.__config.get(
            CONFIG_BASE,
            "use_default_query_everytime",
            expected_type=bool,
            default=True,
        )

        gensim_model_name = self.__config.get(
            CONFIG_BASE,
            "gensim_model_name",
            expected_type=str,
        )

        # get synonym backend model type (pre vs self-trained)
        w2v_model_type = self.__config.get(
            CONFIG_BASE,
            "synonym_source",
            expected_type=str,
            choices=["pretrained_gensim", "selftrained_w2v"],
            default="pretrained_gensim",
        )

        self.__max_number_of_synonyms = self.__config.get(
            CONFIG_BASE, "max_number_of_synonyms", expected_type=int, default=2
        )

        # minimal levenshtein distance between a word and its synonym
        min_levenshtein_distance = self.__config.get(
            CONFIG_BASE, "min_levensthein_distance", expected_type=int, default=2
        )

        self.__use_text_generation_prefix = self.__config.get(
            CONFIG_BASE, "use_text_generation_prefix", expected_type=bool, default=False
        )

        self.__fill_mask__model = None
        self.__mask_token = None
        self.__text_generation_model = None

        # load masked model if specified by mode
        if "masked_model_expansion" in self.__modes:
            # this can either be a path to a self trained model, or a pre-trained model like gpt2
            masked_model = self.__config.get(
                CONFIG_BASE,
                "masked_langue_model",
                expected_type=str,
                raise_exc=True,
            )

            logger.debug(f"Loading masked BERT model from {masked_model}")

            self.__fill_mask__model = pipeline(
                "fill-mask",
                model=masked_model,
                tokenizer=masked_model,
            )

            self.__mask_token = self.__config.get(
                CONFIG_BASE,
                "masked_token",
                expected_type=str,
                raise_exc=True,
            )

        # load masked model if specified by mode
        if "text_generation" in self.__modes:
            # this can either be a path to a self trained model, or a pre-trained model like gpt2
            text_generation_model = self.__config.get(
                CONFIG_BASE,
                "text_generation_model",
                expected_type=str,
                raise_exc=True,
            )

            logger.debug(
                f"Loading model for text-generation from {text_generation_model}"
            )

            self.__text_generation_model = pipeline(
                "text-generation",
                model=text_generation_model,
                tokenizer=text_generation_model,
            )

        self.synonyms_backend = None

        # only load w2v models if needed
        if (
            "en_mass" in self.__modes
            or "only_nouns" in self.__modes
            or "bilbo_baggins" in self.__modes
        ):
            if w2v_model_type == "pretrained_gensim":
                # gensim (w2v) is synonym backend
                self.synonyms_backend = GensimWrapper(
                    gensim_model_name,
                    min_levenshtein_distance=min_levenshtein_distance,
                    is_using_self_trained_model=False,
                    n_synonyms=self.__max_number_of_synonyms,
                )
            elif w2v_model_type == "selftrained_w2v":
                # load self trained w2v model
                gensim_model_name = self.__config.get(
                    CONFIG_BASE,
                    "path_w2v_trained_model",
                    expected_type=str,
                    raise_exc=True,
                )
                # gensim (w2v) is synonym backend
                self.synonyms_backend = GensimWrapper(
                    gensim_model_name,
                    min_levenshtein_distance=min_levenshtein_distance,
                    is_using_self_trained_model=True,
                    n_synonyms=self.__max_number_of_synonyms,
                )
            else:
                logger.error(
                    f"Cannot load w2v model because the synonym source: {w2v_model_type} is not known!"
                )

            self.nltk_wrapper = NLTKWrapper(
                min_levenshtein_distance=min_levenshtein_distance,
                number_of_synonyms=self.__max_number_of_synonyms,
            )

        # POS tagging backend is always spacy
        self.pos_backend = SpacyWrapper()

    def process(self, topic: Topic) -> Topic:
        if self.__use_synonyms:
            topic.processing_objects = list(
                itertools.chain(
                    *[self.expand_queries(po) for po in topic.processing_objects]
                )
            )
        logger.info(
            "Expanded queries to: %s",
            [po.query.query_text for po in topic.processing_objects],
        )
        return topic

    def __remove_duplicates(self, word_list: List[str]) -> List[str]:
        """
        Removes duplicate words from a list of strings.
        Only removes words, if they are exactly the same string.
        Words are not normalized or lowered before removing them, so the procedure
        is case-sensitive.
        This approach keeps the ordering of the initial word list, this is nice, so that we have deterministic query

        """

        # here comes the extra complicated move to remove duplicate words from a query
        # this approach always keeps words which are at the beginning of a query and only removes duplicate words
        # that occur later in the query
        unique_word_list = []

        for word in word_list:
            if word not in unique_word_list:
                unique_word_list.append(word)

        return unique_word_list

    def expand_queries(self, po: ProcessingObject) -> List[ProcessingObject]:
        """
        expanding queries to retrieve more results
        currently there are the following approaches:
            - en_mass
            - nouns_only
            - bilbo_baggins
            - masked_model_expansion
            - text_generation
        Or any combination of these modes
        """
        logger.info("Expansion mode is '%s'", self.__modes)

        # retrieve all tagged words for a query
        named_nouns = self.pos_backend.get_tags_of_type(
            po.query.query_text, tag_type=PosType.ENTITY
        )

        features = self.pos_backend.get_tags_of_type(
            po.query.query_text, tag_type=PosType.FEATURE
        )

        verbs = self.pos_backend.get_tags_of_type(
            po.query.query_text, tag_type=PosType.VERB
        )

        # merge all tagged words
        tagged_words = named_nouns.copy()
        tagged_words.extend(features)

        tagged_words = self.__remove_duplicates(tagged_words)
        named_nouns = self.__remove_duplicates(named_nouns)

        # if we dont have any nouns, use features as noun replacements
        # mostly relevant for "nouns_only" mode
        if len(named_nouns) == 0:
            logger.debug(
                f"Could not find nouns in query {po.query.query_text}, using features instead of nouns."
            )
            named_nouns.extend(features)
            named_nouns.extend(verbs)
            tagged_words = named_nouns

        # a dict to hold the closest similar words from nltk, w2c and nltk+w2v
        synonym_dict = {}
        if (
            "en_mass" in self.__modes
            or "only_nouns" in self.__modes
            or "bilbo_baggins" in self.__modes
        ):
            for word in tagged_words:

                synonyms = []

                if self.number_of_additional_context_words > 0:
                    # nltk synonyms (positive words for w2v)
                    pos_words = self.nltk_wrapper.get_synonyms(word)
                    neg_words = self.nltk_wrapper.get_antonyms(word)

                    w2v_nltk_synonyms = self.synonyms_backend.get_semantic_similiar(
                        word,
                        pos_words[: self.number_of_additional_context_words],
                        neg_words[: self.number_of_additional_context_words],
                    )  # only use two other words to specify the w2v word

                    logger.info("NLTK synonyms: %s" % (pos_words))
                    logger.info("NLTK antonyms: %s" % (neg_words))
                    logger.info("w2v with nltk bias : %s\n" % (w2v_nltk_synonyms))

                    # remove all duplicate entries by converting to set and back to list
                    synonyms = w2v_nltk_synonyms

                else:
                    w2v_synonyms = self.synonyms_backend.get_synonyms(word)
                    synonyms = w2v_synonyms

                synonyms = self.__remove_duplicates(synonyms)
                synonym_dict[word] = synonyms

        logger.info("Synonyms: %s" % (synonym_dict))

        generated_pobjs: List[ProcessingObject] = []

        # add default query if config.yaml said so
        if self.__use_default_query == True:
            generated_pobjs.append(po)

        if "en_mass" in self.__modes:
            # replace every word from the tagged words, with its similar word (if there exists one)
            for word in tagged_words:
                # only replace words, if there exists a synonym for the original word
                for syn in synonym_dict[word]:
                    generated_pobjs.append(
                        copy_po(
                            po,
                            new_query_text=po.query.query_text.replace(
                                word.lower(),
                                syn,
                            ),
                            q_type=1,
                        )
                    )

        if "only_nouns" in self.__modes:
            noun_string = None

            # only uses all the nouns from the original query
            if len(named_nouns) > 1:
                noun_string = " AND ".join(named_nouns)
            elif len(named_nouns) == 1:
                noun_string = named_nouns[0]

            # add query with default nouns
            generated_pobjs.append(
                copy_po(
                    po,
                    new_query_text=noun_string,
                    q_type=1,
                ),
            )

            # replace every word from the tagged words, with its similar word (if there exists one)
            for word in named_nouns:
                # only replace words, if there exists a synonym for the original word
                for syn in synonym_dict[word]:
                    generated_pobjs.append(
                        copy_po(
                            po,
                            new_query_text=noun_string.replace(
                                word.lower(),
                                syn,
                            ),
                            q_type=1,
                        )
                    )

        if "bilbo_baggins" in self.__modes:
            # see http://ceur-ws.org/Vol-2696/paper_130.pdf

            # if we dont use the default query, then we need to add it here, since this mode requires the default query
            if self.__use_default_query == False:
                generated_pobjs.append(po)

            antonyms = []
            # tagged words consists of features (adjectives) and nouns
            for word in tagged_words:
                # add all antonyms of the nouns and features
                antonyms += self.nltk_wrapper.get_antonyms(word)

            synonyms = []
            for key in synonym_dict:
                synonyms += synonym_dict[key]

            expanded_query2 = " AND ".join(named_nouns)
            expanded_query3 = " ".join(self.__remove_duplicates(named_nouns + features))
            expanded_query4 = " OR ".join(
                self.__remove_duplicates(named_nouns + features + synonyms + antonyms)
            )

            if len(expanded_query2) > 0:
                generated_pobjs.append(
                    copy_po(
                        po,
                        new_query_text=expanded_query2,
                        q_type=1,
                    ),
                )
            if len(expanded_query3) > 0:
                generated_pobjs.append(
                    copy_po(
                        po,
                        new_query_text=expanded_query3,
                        q_type=1,
                    ),
                )
            if len(expanded_query4) > 0:
                generated_pobjs.append(
                    copy_po(
                        po,
                        new_query_text=expanded_query4,
                        q_type=1,
                    ),
                )

        if "masked_model_expansion" in self.__modes:
            # this mode utilizes a self trained small masked-BERT model to replace a '<mask>'
            # token with the most likely word at this position

            # iteratively replace all nouns with the '<mask>' tag
            original_query_text = po.query.query_text

            for word in named_nouns:
                masked_query = original_query_text.replace(word, self.__mask_token, 1)
                logger.debug(f"masked query: {masked_query}")
                # hard-try the exception of a word, that is not known to the query
                try:
                    # replace <mask> tag with predictions
                    masked_results = self.__fill_mask__model(masked_query)
                except IndexError as e:
                    logger.warn(
                        f"Word: {word} is not known to the model. Exception: {e}"
                    )
                    continue

                # only use first 'max_number_of_synonyms' mask replacements
                masked_results = masked_results[: self.__max_number_of_synonyms]

                for entry in masked_results:
                    new_query_tmp = entry["sequence"]
                    if len(new_query_tmp) > 0:
                        generated_pobjs.append(
                            copy_po(
                                po,
                                new_query_text=new_query_tmp.replace("<s>", "").replace(
                                    "</s>", ""
                                ),
                                q_type=1,
                            )
                        )

        if "text_generation" in self.__modes:
            original_query_text = po.query.query_text

            gen_text_query = original_query_text

            if self.__use_text_generation_prefix == True:
                if len(features) > 0:
                    gen_text_query += " " + features[0].capitalize()

            gen_text = self.__text_generation_model(
                gen_text_query,
                # num_beams=5,
                # no_repeat_ngram_size=2,
                # early_stopping=True,
                # seed=0, setting seed useful?
                max_length=35,  # Maximum length that will be used by default in the generate method of the model.
                do_samples=True,  # Flag that will be used by default in the generate method of the model. Whether or not to use sampling ; use greedy decoding otherwise.
                top_k=50,  # Number of highest probability vocabulary tokens to keep for top-k-filtering that will be used by default in the generate method of the model.
                top_p=0.8,  # Value that will be used by default in the generate method of the model for top_p. If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.
                temperature=0.7,  # The value used to module the next token probabilities that will be used by default in the generate method of the model. Must be strictly positive.
                repetition_penalty=1.5,  # Parameter for repetition penalty that will be used by default in the generate method of the model. 1.0 means no penalty.
                num_return_sequences=3,
            )

            for t in gen_text:
                t_str = t["generated_text"].replace('"', "").replace("\\", "")
                # clean all links starting with www./https./http
                re.sub("[^\s\d]+\.[^\s\d]+", "", t_str)
                re.sub("[^a-zA-Z0-9 ]+", "", t_str)

                logger.info(f"text generated query: {t_str}")
                query_nouns = self.pos_backend.get_tags_of_type(
                    t_str, tag_type=PosType.ENTITY
                )
                query_nouns = [x.lower() for x in query_nouns]
                query_nouns = self.__remove_duplicates(query_nouns)

                generated_pobjs.append(
                    copy_po(
                        po,
                        new_query_text=" ".join(query_nouns),
                        q_type=1,
                    )
                )

        # delete duplicate queries by creating dict over new queries
        gobj_dict: Dict[str, ProcessingObject] = {}
        for _, val1 in enumerate(generated_pobjs):
            # dont add the same query, if it is lower cased
            if val1.query.query_text.lower() in gobj_dict:
                continue
            elif val1.query.query_text in gobj_dict:
                continue
            else:
                gobj_dict[val1.query.query_text] = val1

        # the generated_pobjs
        return gobj_dict.values()


class NLTKWrapper:
    """
    A NLTK wrapper around the wordnet dataset to retrieve synonyms and antonyms of specific words.
    """

    def __init__(self, min_levenshtein_distance: int = 2, number_of_synonyms: int = 2):
        nltk.download("wordnet")
        self.__min_levenshtein_distance = min_levenshtein_distance
        self.__max_number_of_synonyms = number_of_synonyms

    def get_synonyms(self, word):
        word = word.lower()
        synonyms = []
        for syn in wordnet.synsets(word):
            for lm in syn.lemmas():
                # check levenshtein distance, only add if the distance is greater than the specified levenshtein dist
                if (
                    Levenshtein.distance(word, lm.name())
                    > self.__min_levenshtein_distance
                ):
                    synonyms.append(lm.name().replace("_", " "))
        return list(set(synonyms))[: self.__max_number_of_synonyms]

    def get_antonyms(self, word):
        word = word.lower()
        antonyms = []
        for syn in wordnet.synsets(word):
            for lm in syn.lemmas():
                if lm.antonyms():
                    # check levenshtein distance, only add if the distance is greater than the specified levenshtein distance
                    if (
                        Levenshtein.distance(word, lm.name())
                        > self.__min_levenshtein_distance
                    ):
                        antonyms.append(lm.name().replace("_", " "))
        return list(set(antonyms))[: self.__max_number_of_synonyms]


def copy_po(processing_obj, new_query_text: str, q_type: int = 0):
    # create an unitialized query
    new_query = Query()

    # make a copy of the old query object
    # all attributes are copied over to the new instance
    for field in fields(Query):
        setattr(new_query, field.name, getattr(processing_obj.query, field.name))

    new_query.query_type = q_type

    # assign new query text value
    new_query.query_text = new_query_text
    return processing_obj.copy_with(query=new_query)
