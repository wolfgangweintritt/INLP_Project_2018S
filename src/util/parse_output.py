#!/usr/bin/env python3
import subprocess
from pprint import pprint
from typing import List


def parse_output(test_df: List, words: List, test_data_predictions: List, test_sentence_ending: List, filename: str):
    """create output file with fitting structure for conlleval"""

    test_sentence_ending.reverse()
    output_filename  = 'outputs/output' + filename + '.txt'
    results_filename = 'results/result' + filename + '.txt'
    with open(output_filename, 'w') as output_file:
        line_number = 0
        for idx in range(0, len(test_df)):
            if line_number == test_sentence_ending[-1]: # insert empty lines after sentences
                output_file.write('\n')
                test_sentence_ending.pop()
                line_number += 1
            # structure: word POS-tag CHUNK-tag predicted-CHUNK-tag
            output_file.write('{}\t{}\t{}\t{}\n'.format(words[idx], test_df[idx][1], test_df[idx][4], test_data_predictions[idx]))
            line_number += 1

    with open(output_filename, 'r') as output_file:
        with open(results_filename, 'w') as results_file:
            subprocess.call(["./crf/conlleval.pl", "-d", "\t"], stdin=output_file, stdout=results_file)
