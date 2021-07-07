#!/bin/sh

# choose which one we want to install
#SPACY_CORPUS='en_core_web_sm'
#SPACY_CORPUS='en_core_web_md'
SPACY_CORPUS='en_core_web_lg'

echo "Installing: " $SPACY_CORPUS

pipenv run python3 -m spacy download "$SPACY_CORPUS"

