#!/usr/bin/env python3

from pprint import pprint
from typing import List
import time


def parse_output(test_df: List, words: List, test_data_predictions: List, test_sentence_ending: List):
    """create output file with fitting structure for conlleval"""

    test_sentence_ending.reverse()
    with open('mlp/output/output' + str(int(time.time())) + '.txt', 'w') as output_file:
        line_number = 0
        for idx in range(0, len(test_df)):
            if line_number == test_sentence_ending[-1]: # insert empty lines after sentences
                output_file.write('\n')
                test_sentence_ending.pop()
                line_number += 1
            # structure: word POS-tag CHUNK-tag predicted-CHUNK-tag
            output_file.write('{} {} {} {}\n'.format(words[idx], test_df[idx][1], test_df[idx][4], test_data_predictions[idx]))
            line_number += 1
