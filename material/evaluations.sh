#!/bin/sh

for f in cache/*; do
    RUN_NAME=$(echo "$f" | sed 's/cache\///g')
    echo "\n-> evaluating: $RUN_NAME\n"
    pipenv run python3 src/evaluate.py --qrels material/task2-relevance_self.qrels -i $RUN_NAME
done
