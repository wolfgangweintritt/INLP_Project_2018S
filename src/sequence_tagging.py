#!/usr/bin/env python3
"""parse training and test data. predict CHUNK-tags via MLP classifier. create outputfile in matching format for conlleval."""
import time
from datetime import datetime
from gensim.models import word2vec
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from util.classes import ParsedInput
import argparse
import pandas as pd
import util.parse_input as input_parser
import util.parse_output as output_parser
import sys

label_encoder = LabelEncoder()


def main():
    startTime = datetime.now()
    descr  = "Applies machine-learning techniques (Neural Networks) to solve Chunking."
    epilog = "By Wolfgang Weintritt and Maximilian Moser, 2018"
    ap = argparse.ArgumentParser(description=descr, epilog=epilog)
    ap.add_argument("--word2vec", "-w", help="use word vectors (word2vec)", action="store_true")
    ap.add_argument("--just_word2vec", "-j", help="use just word vectors no CRF features (next POS, prev POS)", action="store_true")
    ap.add_argument("--classifier", "-c", help="choose classifier", choices=["MLP", "RF", "SVM"], default="MLP")
    ap.add_argument("--grid-search", "-g", help="perform grid search over a pre-selected parameter space", action="store_true")
    ap.add_argument("--percent-of-train-data", "-p", help="only use some percent of the training data, to speedup the process", type=int, choices=range(1, 101), default=100)
    args = ap.parse_args()
    use_word_vectors = args.word2vec
    just_word2vec = args.just_word2vec
    classifier = args.classifier
    use_grid_search = args.grid_search
    percent_of_train_data = args.percent_of_train_data

    parsed_input = parse_input_and_get_dataframe("train.txt", "test.txt", use_word_vectors, just_word2vec, percent_of_train_data)
    df                        = parsed_input.data
    df_target                 = parsed_input.data_target
    test_data                 = parsed_input.test_data
    test_data_sentence_ending = parsed_input.test_data_sentence_ending
    words                     = parsed_input.words
    split_line                = parsed_input.split_line

    if classifier == "MLP":
        clf = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=200)
    elif classifier == "RF":
        clf = RandomForestClassifier(n_estimators=200)
    else:  # classifier = "SVM"
        clf = SVC()
        #clf = NuSVC()
    if use_grid_search:
        print("tuning hyper-parameters...")
        if classifier == "MLP":
            params = {"hidden_layer_sizes": [(10,), (10, 20, 10), (100,)], "max_iter": [200, 500]}
        elif classifier == "RF":
            params = {'n_estimators': [10, 40, 70]}
        else:  # classifier = "SVM"
            params = {'kernel': ['poly', 'rbf', 'sigmoid', 'precomputed']}

        clf = GridSearchCV(clf, params, n_jobs=-1)

    print("training classifier...")
    clf.fit(df[:split_line], df_target[:split_line])
    if use_grid_search:
        print("best parameters: {}".format(clf.best_params_))
    print("classifying test data...")
    test_data_predictions = clf.predict(df[split_line:])

    # pprint(df_target[split_line:])
    # pprint(test_data_predictions)

    filename = get_output_filename(classifier, percent_of_train_data, use_word_vectors, just_word2vec)
    print("creating output file and running conlleval...")
    output_parser.parse_output(test_data, words[split_line:], label_encoder.inverse_transform(test_data_predictions), test_data_sentence_ending, filename)
    print("done, took: {}".format((datetime.now() - startTime).seconds))


def parse_input_and_get_dataframe(train_filename: str, test_filename: str, word_vectors: bool, just_word2vec: bool, percent_of_train_data: int) -> ParsedInput:
    vector_size                          = 100
    train_data, _                        = input_parser.parse_input(train_filename)
    test_data, test_data_sentence_ending = input_parser.parse_input(test_filename)
    # len train_data: 211727
    # len test_data: 47377
    train_data_len = int(len(train_data) * (percent_of_train_data / 100))
    train_data = train_data[0:train_data_len]
    print("train data size: {}".format(len(train_data)))
    print("test data size: {}".format(len(test_data)))

    split_line = len(train_data)
    # merged data size: 259104
    # we have to merge the data because of one-hot encoding
    merged_data = train_data + test_data
    words = [w for (w, p, pp, np, c) in merged_data]
    print("merged data size: {}".format(len(merged_data)))

    if word_vectors or just_word2vec:
        print("creating word vectors...")

        # for word2vec, we need to create list of sentences (which are lists of words):
        #  e.g. [["Her", "name", "is", "Puck", "and", "she", "likes", "to", "solo-mid"], ...]
        sentences = []
        sentence  = []
        for word, pos, prevpos, nextpos, chunk in merged_data:
            sentence.append(word)
            if nextpos == "AFTER_SENTENCE":
                # meaning we currently have the end of the sentence
                sentences.append(sentence)
                sentence = []
        
        # create the word vectors from our corpus
        # with min_count=1 s.t. no words get filtered out
        vectors = word2vec.Word2Vec(sentences, size=vector_size, min_count=1)

        # replace the words by their vectors
        for itm in merged_data:
            word = itm.pop(0)
            itm.insert(0, vectors[word])

    if just_word2vec:  # leave out POS tags
        merged_data = [[w, p, c] for (w, p, pp, np, c) in merged_data]

    # create the dataframe (split up target and data, label-encode target)
    print("creating dataframe...")
    df = pd.DataFrame(merged_data)
    df_target = label_encoder.fit_transform(df.iloc[:, -1])
    df = df.iloc[:, 0:-1]

    print("doing one-hot encoding...")
    if not word_vectors and not just_word2vec:
        # if we don't have word vectors, we need to one-hot encode everything
        df = pd.get_dummies(df, sparse=True)
    else:
        # else, we have to one-hot encode everything except for the vectors
        tmp_wv = df.iloc[:, 0]

        # unpack the vectors in the first column, such that we don't have
        # any vectors in the dataframe (which scikit doesn't like)
        tmp_wv = pd.DataFrame([v for v in tmp_wv.values])

        tmp_r = df.iloc[:, 1:]
        tmp_r = pd.get_dummies(tmp_r, sparse=True)
        df = pd.concat([tmp_wv, tmp_r], axis=1)

    return ParsedInput(df, df_target, test_data, test_data_sentence_ending, words, split_line)


def get_output_filename(classifier: str, percent_of_train_data: int, use_word_vectors: bool, just_word2vec: bool) -> str:
    input_features = "CRF"
    if use_word_vectors:
        input_features = "WV_CRF"
    if just_word2vec:
        input_features = "JustWV"
    return "_{}_{}_{}P_{}".format(classifier, input_features, percent_of_train_data, str(int(time.time())))


if __name__ == '__main__':
    exit = main()
    sys.exit(exit)
