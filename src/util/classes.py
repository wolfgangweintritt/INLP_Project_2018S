#!/usr/bin/env python3

class ParsedInput:
    """Class for wrapping parts of the parsed input"""
    def __init__(self, data, data_target, test_data, test_data_sentence_ending, words, split_line):
        self.data                      = data
        self.data_target               = data_target
        self.test_data                 = test_data
        self.test_data_sentence_ending = test_data_sentence_ending
        self.words                     = words
        self.split_line                = split_line