#!/usr/bin/env python3
"""parse training and test data. predict CHUNK-tags via MLP classifier. create outputfile in matching format for conlleval."""
import sys
from pprint import pprint
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from typing import Tuple, List
import pandas as pd
import parsers.parse_input as input_parser
import parsers.parse_output as output_parser

NOT_IN_SENTENCE_POS = "NIS"
label_encoder = LabelEncoder()


def main():
    df, df_target, test_data, test_sentence_ending, split_line = parse_input_and_get_dataframe("train.txt", "test.txt")

    print("training MLP classifier...")
    mlp_clf = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam')
    mlp_clf.fit(df[:split_line], df_target[:split_line])
    print("classifying test data...")
    test_data_predictions = mlp_clf.predict(df[split_line:])

    # pprint(df_target[split_line:])
    # pprint(test_data_predictions)

    output_parser.parse_output(test_data, label_encoder.inverse_transform(test_data_predictions), test_sentence_ending)


def parse_input_and_get_dataframe(train_filename: str, test_filename: str) -> Tuple[pd.DataFrame, pd.DataFrame, List, List, int]:
    train_data, _                        = input_parser.parse_input(train_filename)
    test_data, test_data_sentence_ending = input_parser.parse_input(test_filename)
    # len train_data: 211727
    # len test_data: 47377
    pprint("train data size: {}".format(len(train_data)))
    pprint("test data size: {}".format(len(test_data)))

    # reduce data size for test purposes
    # train_data = train_data[0:100]
    # test_data  = test_data [0:100]
    split_line = len(train_data)
    # merged data size: 259104
    # we have to merge the data because of one-hot encoding
    merged_data = train_data + test_data
    print("merged data size: {}".format(len(merged_data)))

    print("creating dataframe...")
    df = pd.DataFrame(merged_data)
    df_target = label_encoder.fit_transform(df.iloc[:, -1])
    df = df.iloc[:, 0:-1]

    print("doing one-hot encoding...")
    df = pd.get_dummies(df, sparse=True)
    return df, df_target, test_data, test_data_sentence_ending, split_line


if __name__ == '__main__':
    exit = main()
    sys.exit(exit)
