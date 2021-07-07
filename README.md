# Touché Task 2: Comparative Argument Retrieval - Mercutio

## Mirrored Repository Info 

This repository is mirrored from this [GitLab repository](https://git.informatik.uni-leipzig.de/depressed-spiders/comparative-argument-retrieval). This repository will not be updated. Please work with the [GitLab repository](https://git.informatik.uni-leipzig.de/depressed-spiders/comparative-argument-retrieval) if possible.

## Usage

### Preparation

Install `docker` and get a api key from [Chat Noir](https://www.chatnoir.eu/doc/api/).

### Docker

Build the docker image with

```shell script
docker build . -t mercutio -f docker/Dockerfile
```

Afterwards, run it with:

```shell script
docker run -e "CHAT_NOIR_API_KEY=$CHAT_NOIR_API_KEY" mercutio:latest
```

The api key needs to be set as an environment variable or replaced in the line above.

To specify a custom config (e.g. `configs/baseline.yaml`), use:

```shell script
docker run -e "CHAT_NOIR_API_KEY=$CHAT_NOIR_API_KEY" -v $(pwd)/configs/baseline.yaml:/app/config.yaml mercutio:latest
```

Instead of building the docker image on your own, you can also use one from hub.docker.io that we uploaded there. Download it with:

```
docker pull procrastimax/mercutio
```

All available options can be listed with `--help`:

```
usage: main.py [-h] [--config CONFIG] [--limit-topics LIMIT_TOPICS] [--single-topic SINGLE_TOPIC] [--judge]
               [--identifier IDENTIFIER] [--trec TREC]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG, -c CONFIG
                        Path to the configuration file
  --limit-topics LIMIT_TOPICS, -l LIMIT_TOPICS
                        Only process first n topics
  --single-topic SINGLE_TOPIC, -t SINGLE_TOPIC
                        Only process a single topic specified by this number. This parameter is only used if the '--
                        limit-topics' parameter is not set! Valid topic numbers: 1-50
  --judge, -j           If this flag is set, than the judgement/ reevalation pipeline is started. Combine this flag
                        with the '-l' parameter to only judge the given topic.
  --identifier IDENTIFIER, -i IDENTIFIER
                        If this flag is set the topics are loaded from a the given identifier (e.g test). NOTE: The
                        pipeline steps are executed regardless, so make sure the correct steps are set in the
                        config.
  --trec TREC           Writes a trec file with the whole ranking
```



### Evaluation

Run `pipenv run src/evaluate.py -i v0` for evaluating the ranking with the name `v0`. It generates various metrics and writes files into `evaluation/`.

All options for the evaluation are:

```
usage: evaluate.py [-h] [--qrels QRELS] [-o OUTPUT] [-i RANKING_ID]
                   [-s STRATEGY]

optional arguments:
  -h, --help            show this help message and exit
  --qrels QRELS         Path to the qrels file with relevance judgements.
  -o OUTPUT, --output OUTPUT
  -i RANKING_ID, --ranking-id RANKING_ID
                        REQUIRED The name of the ranking which will be
                        evaluated.
  -s STRATEGY, --strategy STRATEGY
                        Strategy for handling unknown relevance. Choose one
                        of: ['assume_not_relevant', 'assume_relevant',
                        'ignore']

```

### Grid Search

After the results of a specific configuration are retrieved from ChatNoir, a grid search for the best weights of the Remerging Pipe can be run:

```shell script
pipenv run src/grid_search_scores.py -i [run name] --start 0.5 --end 1.2 --step 0.1
```

This would use the serialized documents from specified run for testing all possible weight combinations between `0.5` and `1.2` (with `0.1` interval steps). The results are then saved in a csv in the directory `gridsearch/`.

Specific scores can be excluded from the grid search with the `--ignore` parameter. One of them should always be excluded: 
by default the ChatNoir `score` will not be changed by the grid search. All ignored weight values are read from the specified run's configuration.

All options are:

```
usage: grid_search_scores.py [-h] [--src SRC] [--qrels QRELS] [-o OUTPUT]
                             [-i RANKING_ID] [-s STRATEGY] [--start START]
                             [--end END] [--step STEP]
                             [--ignore IGNORE [IGNORE ...]]

optional arguments:
  -h, --help            show this help message and exit
  --src SRC
  --qrels QRELS         Path to the qrels file with relevance judgements.
  -o OUTPUT, --output OUTPUT
  -i RANKING_ID, --ranking-id RANKING_ID
                        The id of the ranking run that is used for the grid
                        search.
  -s STRATEGY, --strategy STRATEGY
                        Strategy for handling unknown relevance. Choose one
                        of: ['assume_not_relevant', 'assume_relevant',
                        'ignore']
  --start START         Start value for the weights
  --end END             End value for the weights
  --step STEP           Step size for the grid search
  --ignore IGNORE [IGNORE ...]
                        Names of weights that will be ignored in the grid
                        search. One of them should always be ignored.

```

## Development

### Setup
For a system that already has python3 configured and pip3 installed, install pipenv with: `pip3 install --user pipenv`.
Then install all needed python packages to run this project: `pipenv install`.
