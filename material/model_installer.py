import nltk
from nltk.corpus import wordnet

from transformers import pipeline

import gensim.downloader
from gensim.models import Word2Vec

print("Downloading nltk")
nltk.download("wordnet")

print("Downloading gensim w2v model")
gensim.downloader.load("glove-wiki-gigaword-300")

print("Downloading gpt2 model")
pipeline("fill-mask", model="roberta-base", tokenizer="roberta-base")

print("Downloading roberta model")
pipeline("text-generation", model="gpt2", tokenizer="gpt2")
