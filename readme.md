# Neuromorph training

There are 4 different models in this repository. To train and evaluate them, it is neccessary to assign
environment variables DATA\_DIR, OUT\_DIR and EMBEDDINGS for different file paths and specify config.py files.
You then need to execute train.py and evaluate.py files. For all the models, DATA_DIR = data and EMBEDDINGS = embeddings.

## Multiclass classifiers

For models 1 and 2:

* Training: softmax/scripts/train.py
* Evaluation: softmax/scripts/evaluate.py

### Model 1

* OUT\_DIR = softmax/emb\_tag\_sum/output
* configuration: softmax/emb\_tag\_sum/config.py

### Model 2

* OUT\_DIR = softmax/emb\_cat\_sum/output
* configuration: softmax/emb\_cat\_sum/config.py

## Sequential models

For models 3 and 4:

* Training: seq2seq/scripts/train.py
* Evaluation: seq2seq/scripts/evaluate.py

### Model 3

* OUT\_DIR = seq2seq/emb\_tag\_sum/output
* configuration: seq2seq/emb\_tag\_sum/config.py

### Model 4

* OUT\_DIR = seq2seq/emb\_cat\_sum/output
* configuration: seq2seq/emb\_cat\_sum/config.py

## Example

Here's an example of a bash script for training and evaluating model 1:

\#!/bin/bash

set -e

export DATA\_DIR=data

export OUT\_DIR=softmax/emb\_tag\_sum/output

export EMBEDDINGS_DIR=embeddings

python softmax/scripts/train.py --config softmax/emb_tag_sum/config.py

python softmax/scripts/evaluate.py --test --config softmax/emb\_tag\_sum/config.py
