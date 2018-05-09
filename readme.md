# INLP Sequence Tagging Project, TU Wien, 2018S

## Goal:
Predict sequence of test-data via
* standard CRF predictor
* More modern approach (some machine learning algorithm) using the same features as the CRF predictor
* The same machine learning algorithm using word vectors

## How to create a CRF++ model & evaluate it:
* install CRF++
* if you encounter a libcrfpp.so.0 error: https://stackoverflow.com/a/25885693
* create a model: `crf_learn crf/templates/template_1.txt train.txt crf/models/model_1`
* predict test data sequence tags with our model: `crf_test -m crf/model/model_1 test.txt > crf/predictions/model_1_test_pred.txt`
* evaluate the predictions using conlleval: `./crf/conlleval.pl -d "\t" < crf/predictions/model_1_test_pred.txt > crf/conlleval_results/conlleval_model_1.txt`

## Resources:
CoNLL 2000 Sequence Chunking dataset and toolsCoNLL 2000 Sequence Chunking dataset and tools: https://www.clips.uantwerpen.be/conll2000/chunking/

## Requirements:
* CRF++
* Python modules: scikit, pandas

## Group Members:
* Maximilian Moser (01326252)
* Wolfgang Weintritt (01327191)
