#!/bin/bash
#takes one parameter (the template name after the prefix "template_", and generates a model, predictions, and evaluates the predictions via conlleval.

crf_learn crf/templates/template_${1}.txt train.txt crf/models/model_${1}
echo "Trained model. crf/models/model_${1}"

crf_test -m crf/models/model_${1} test.txt > crf/predictions/model_${1}_test_pred.txt
echo "Created predictions. crf/predictions/model_${1}_test_pred.txt"

./crf/conlleval.pl -d "\t" < crf/predictions/model_${1}_test_pred.txt > crf/conlleval_results/conlleval_model_${1}.txt
echo "Evaluated predictions. crf/conlleval_results/conlleval_model_${1}.txt"
