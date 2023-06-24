import math
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import spacy

from scripts.answer_preprocessing import answer_preprocessing
from helpers.utils import validate_path


def get_term_frequency(answer):
    term_frequency = {}
    for word in answer:
        count = term_frequency.get(word, 0)
        term_frequency[word] = count + 1
    return term_frequency


def create_frequency_dict(answers):
    frequency_dict = {}
    for answer in answers:
        term_frequency = get_term_frequency(answer)
        for word, frequency in term_frequency.items():
            count = frequency_dict.get(word, 0)
            frequency_dict[word] = count + frequency
    return frequency_dict


class AdjectivesFreq:
    def __init__(self, answers):
        self.nlp = spacy.load("en_core_web_sm")
        self.answers = answers
        self.nouns = self._get_frequent_nouns()
        self.noun_adj_counts = self._get_noun_adj_counts()
        self.id2word = None
        self.frequency_dict = create_frequency_dict(answers)
        self.corpus = None

    def _get_frequent_nouns(self):
        nouns = set()
        for answer in self.answers:
            doc = self.nlp(answer.lower())
            for token in doc:
                if token.pos_ == "NOUN":
                    nouns.add(token.text)
        return nouns

    def _get_noun_adj_counts(self):
        noun_adj_counts = {}
        for noun in self.nouns:
            adj_counts = {}
            for answer in self.answers:
                doc = self.nlp(answer.lower())
                for token in doc:
                    if token.text == noun:
                        for child in token.children:
                            if child.pos_ == "ADJ":
                                adj_counts[child.text] = adj_counts.get(child.text, 0) + 1
            noun_adj_counts[noun] = adj_counts
        for noun, adj_counts in noun_adj_counts.items():
            noun_adj_counts[noun] = {adj: count for adj, count in
                                     sorted(adj_counts.items(), key=lambda x: x[1], reverse=True)}
        totals = {noun: 0 for noun, adj_counts in noun_adj_counts.items()}
        for noun, adj_counts in noun_adj_counts.items():
            for adj, count in adj_counts.items():
                totals[noun] += count
        noun_adj_counts = {noun: adj_counts for noun, adj_counts in
                                     sorted(noun_adj_counts.items(), key=lambda x: totals[x[0]], reverse=True)}

        return noun_adj_counts

    def plot_top_nouns(self, path, filename, n_nouns=9, n_adjectives=5):
        validate_path(path)
        fig, axes = plt.subplots(math.ceil(n_nouns / 3), 3, sharex=True)
        for i, (noun, adj_counts) in enumerate(list(self.noun_adj_counts.items())[:n_nouns]):
            top_adj_counts = dict(list(adj_counts.items())[:n_adjectives])
            ax = axes.flatten()[i]
            ax.grid(zorder=0)
            yticks = [" "]*n_adjectives
            values = [0]*n_adjectives
            for i, adj in enumerate(list(top_adj_counts.keys())):
                yticks[i] = adj
            for i, count in enumerate(list(top_adj_counts.values())):
                values[i] = count
            ax.barh(range(n_adjectives), values, zorder=2)
            ax.invert_yaxis()
            ax.set_yticks(range(n_adjectives), yticks)
            ax.set_title(noun)
        fig.subplots_adjust(wspace=0.8, hspace=0.5)
        fig.supxlabel('Count')
        fig.supylabel('Term')
        plt.savefig(os.path.join(path, filename))
        plt.close()

    def plot_nouns(self, path, filename, nouns, n_adjectives=5):
        validate_path(path)
        n_nouns = len(nouns)
        fig, axes = plt.subplots(math.ceil(n_nouns / 3), 3)
        noun_adj_counts = {noun: self.noun_adj_counts[noun] for noun in nouns}
        for i, (noun, adj_counts) in enumerate(noun_adj_counts.items()):
            top_adj_counts = dict(list(adj_counts.items())[:n_adjectives])
            ax = axes.flatten()[i]
            ax.grid(zorder=0)
            ax.barh(top_adj_counts.keys(), top_adj_counts.values(), zorder=2)
            ax.set_yticks(top_adj_counts.keys())
            ax.set_title(noun)
        fig.subplots_adjust(wspace=0.8)
        fig.supxlabel('Beta')
        fig.supylabel('Term')
        plt.savefig(os.path.join(path, filename))


if __name__ == "__main__":
    pass
    # answers = pd.read_excel('data/FID=386 and FID=387.ods', sheet_name='FID=386')
    # _answers = answers.fs_answer
    # answers = answers.astype(str)
    # adjective_freq = AdjectivesFreq(_answers)
    # adjective_freq.plot_top_nouns(os.path.join('data', 'fid386'), f'Adjectives_{specific}.png')
    # levels = answers.groupby(by="level")
    # answers = answers.astype(str)
    # adjective_freq = AdjectivesFreq(answers)
    # adjective_freq.plot_top_nouns(os.path.join('data', 'test'), f'test.png', n_nouns=6)
