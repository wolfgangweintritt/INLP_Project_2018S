#!/usr/bin/env python3

from pprint import pprint
from typing import Tuple, List


BEFORE_SENTENCE_POS = "BEFORE_SENTENCE"
AFTER_SENTENCE_POS  = "AFTER_SENTENCE"


def parse_input(filename: str) -> Tuple[List, List]:
    """parse input file; add POS-tag of previous word, next word, save sentence-end positions"""
    # output has format [(WORD, POS, PREV-POS, NEXT-POS, CHUNK)]

    prev_POS = BEFORE_SENTENCE_POS
    prev_prev_POS = BEFORE_SENTENCE_POS
    data = []
    sentence_ending = []
    with open(filename, 'r') as input_file:
        line_number = 0
        for line in input_file:
            line = line.strip()
            if not line: # empty line
                sentence_ending.append(line_number)
                prev_POS = BEFORE_SENTENCE_POS
                prev_prev_POS = BEFORE_SENTENCE_POS
            else:
                fields = line.split(' ')
                newLine = [fields[0], fields[1], prev_POS, prev_prev_POS, AFTER_SENTENCE_POS, AFTER_SENTENCE_POS, fields[2]]
                if (len(sentence_ending) == 0 or sentence_ending[-1] < line_number - 1) and len(data) > 0:
                    data[-1][4] = fields[1] # change next_POS of last line
                    prev_prev_POS = data[-1][1] # save POS of previous line for the next line
                if (len(sentence_ending) == 0 or sentence_ending[-1] < line_number - 2) and len(data) > 1:
                    data[-2][5] = fields[1]  # change next_next_POS of 2nd last line
                prev_POS = fields[1] # save POS for next line

                data.append(newLine)

            line_number += 1

    # pprint(sentence_ending)
    # pprint(data)
    return data, sentence_ending

# test
# parse_input('train.txt')