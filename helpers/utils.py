# home directory
from pathlib import Path
import json
import os
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

import numpy as np


HOME = str(Path.home())


# def find_described_noun(tagged_words, adjective):
#     chunks = ne_chunk(tagged_words)
#
#     for chunk in chunks:
#         if hasattr(chunk, "label") and chunk.label() == "NP":
#             []


# def extract_nouns_adjectives_dict(answer) -> dict:
#     nouns = {}
#     sentences = sent_tokenize(answer)
#     for sentence in sentences:
#         tokens = word_tokenize(sentence)
#         tags = pos_tag(tokens)
#         chunks = ne_chunk(tags)
#         for chunk in chunks:
#                 for word, tag in chunk.leaves():
#                     if tag.startswith("JJ"):
#                         if
#
#     nouns = {}
#     for sentence in sentences:
#         words = word_tokenize(sentence)
#         tagged_words = pos_tag(words)
#         for i in range(len(tagged_words)):
#             word, tag = tagged_words[i]
#             if tag == "ADJ":


def get_adjective_frequencies(answer):
    adjectives = {}
    sentences = sent_tokenize(answer)
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        tags = pos_tag(tokens)
        for word, tag in tags:
            if tag.startwith("JJ"):
                adjectives[word] = adjectives.get(word, 0) + 1
    return adjectives


def create_adjective_frequency_dict(level):
    level_adjectives = {}
    for answer in level:
        adjectives = get_adjective_frequencies(answer)
        for adjective, count in adjectives.items():
            level_adjectives[adjective] = level_adjectives.get(adjective, 0) + count
    return level_adjectives


def read_json(file):
    with open(file) as f:
        data = json.load(f)
    return data


def save_json(data, file):
    with open(file, "w") as f:
        json.dump(data, f, indent=2)


def validate_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
