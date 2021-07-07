# Touché Task 2

Resources for the second project available as part of the "[Advanced Information Retrieval](https://temir.org/teaching/information-retrieval-ws-2020-21/information-retrieval-ws-2020-21.html)" course at Leipzig University.

# Source Dataset
## [Chatnoir API](https://www.chatnoir.eu/doc/api/)

[Bevendorff et al. Elastic ChatNoir: Search Engine for the ClueWeb and the Common Crawl. (ECIR 2018)](https://webis.de/publications.html?q=Chatnoir#stein_2018c)

Since the web crawl used in Task 2 of Touché is too large to index individually, we provide access to the data via the [Chatnoir API](https://www.chatnoir.eu/doc/api/). It can be queried to retrieve pre-selected subsets of the complete data, possibly with a multitude of different queries per topic.

We will hand out API keys to every group deciding for Task 2.

# Training & Evaluation Datasets

## [Touche Dataset](./Dataset\ Touche\ 2020)

[Bondarenko et al. Overview of Touché 2020: Argument Retrieval. (CLEF 2020 Evaluation Labs)](https://webis.de/publications.html#stein_2020v)

This directory contains data from the first shared subtask on Argument Retrieval, [Touché 2020](https://events.webis.de/touche-20/).
It consists of three files:

The `topics.xml` file contains exemplary search topics as denoted below:

```xml
<topics>
    <topic>
        <number>2</number>
        <title>
            Which is better, a laptop or a desktop?
        </title>
        <description>
            A user wants to buy a new PC but has no prior preferences. They want to find arguments that show in what personal situation what kind of machine is preferable. This can range from situations like frequent traveling where a mobile device is to be favored to situations of a rather "stationary" gaming desktop PC.  
        </description>
        <narrative>
            Highly relevant documents will describe the major similarities and dissimilarities of laptops and desktops along with the respective advantages and disadvantages of specific usage scenarios. A comparison of the technical and architectural characteristics without personal opinion, recommendation, or pros/cons is not relevant.
        </narrative>
    </topic>
    
    ...
    ...
    ...
    
</topics>
```

`<number>` denotes a unique topic identifer and `<title>` denotes the query as to be entered into the retrieval system.

The other file in this folder, `task2-relevance.qrels`, contains relevance judgements from last years Touché lab. It can be used to train of evaluate your approaches.

Qrel files are formatted according the the standard TREC layout with 4 whitespace-separated columns: `qid`, `Q0`, `docid`, `relevancy`, where `qid` refers to the topic number (as also found in `topics.xml`), `Q0` is deprecated and contains only zeroes, `docid` is an ID identfying a document in the `args.me` corpus, and `relevancy` denotes the relevance of a a document to a query on a scale from -2 to 4 (irrelevant to highly relevant).
