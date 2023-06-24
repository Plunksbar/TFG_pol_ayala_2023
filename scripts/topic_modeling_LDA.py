import os
import pandas as pd
from pprint import pprint
from tqdm import tqdm
import gensim
from gensim.models import CoherenceModel
import gensim.corpora as corpora
import re
# import pyLDAvis
# import pyLDAvis.gensim_models as gensimvi
import matplotlib.pyplot as plt
import numpy as np
import math

from services.answer_preprocessing import answer_preprocessing
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


class LDA:
    def __init__(self, answers, max_words=20):
        self.lda_model = None
        self.answers = answers
        self.frequency_dict = create_frequency_dict(answers)
        self.bigrams = None
        self.bigram_answers = None
        self.id2word = None
        self.corpus = None

    def _topics_to_graph(self, path, filename):
        validate_path(path)
        columns = 3
        rows = math.ceil(len(self.lda_model.print_topics()) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(10, 3 * rows))
        # fig.tight_layout(pad=3)

        for i, topic in enumerate(self.lda_model.print_topics()):
            term_scores = re.findall('\d{1}.\d+\*"\w+"', topic[1])
            scores = [float(re.findall("\d{1}.\d+", score)[0]) for score in term_scores]
            labels = [re.findall("[a-zA-Z]+", score)[0] for score in term_scores]
            y_pos = np.arange(len(labels))

            ax = axes.flatten()[i]
            ax.grid(zorder=0)
            ax.barh(y_pos, scores, zorder=2)
            ax.set_yticks(y_pos, labels=labels)
            ax.invert_yaxis()
            ax.set_title(i)
        fig.subplots_adjust(wspace=0.8)
        fig.supxlabel('Beta')
        fig.supylabel('Term')
        plt.savefig(os.path.join(path, filename))

    def generate_bigrams(self, path, min_count=5, threshold=100):
        validate_path(path)
        if self.bigrams:
            return self.bigrams
        self.bigrams = gensim.models.Phrases(self.answers)

        bigram_scores = list(zip(self.bigrams.export_phrases().keys(), self.bigrams.export_phrases().values()))
        bigram_scores.sort(key=lambda x: x[1])
        if len(bigram_scores) == 0:
            print("No bigrams Found")
            return self.bigrams

        terms = list(map(list, zip(*bigram_scores)))[0][-max:]
        scores = list(map(list, zip(*bigram_scores)))[1][-max:]
        y_pos = np.arange(len(terms))

        fig = plt.figure()
        plt.grid(zorder=0)
        plt.barh(y_pos, scores, zorder=2)
        plt.yticks(y_pos, labels=terms)
        plt.title('Top bigrams')
        plt.savefig(os.path.join(path, 'Bigrams.png'))

        self.bigram_answers = self.bigrams[self.answers]

        return self.bigrams

    def model_topics(self, path, filename, num_topics=None):
        validate_path(path)
        self.id2word = corpora.Dictionary(self.answers)
        self.corpus = [self.id2word.doc2bow(answer) for answer in self.answers]

        if num_topics == None:
            num_topics = 10
        self.lda_model = gensim.models.LdaModel(corpus=self.corpus,
                                                id2word=self.id2word,
                                                num_topics=num_topics,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)

        self._topics_to_graph(path, filename)

        return self.lda_model.print_topics()


if __name__ == "__main__":
    pass
    # answers = pd.read_excel('data/FID=390 and FID=391.ods', sheet_name='FID=390')\
    #                     .set_index('fs_id').query('level == 0').fs_answer
    # answers = answers.astype(str)
    # answers = answers.apply(answer_preprocessing)
    # lda_analysis = LDA(answers)
    # bigrams = lda_analysis.generate_bigrams(os.path.join('data', "fid390", 'bigrams'))
    # lda_analysis.model_topics(os.path.join('data', "fid390", 'model_topics'))

