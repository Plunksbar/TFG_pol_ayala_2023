from wordcloud import WordCloud
import os
import pandas as pd
import matplotlib.pyplot as plt

from helpers.utils import validate_path
from services.answer_preprocessing import answer_preprocessing

# set_of_words = ["scalability", "openness", "heterogeneity", "concurrency", "transparency", "resource sharing",
#                 "fault-tolerance"]
set_of_words = ["personalized learning", "increased access to information", "innovative teaching methods"]


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


def create_document_frequency(answers, set_of_words):
    frequency_dict = {word: 0 for word in set_of_words}
    for word in set_of_words:
        for answer in answers:
            if word in answer.lower():
                frequency_dict[word] = frequency_dict[word] + 1
    return frequency_dict


def create_word_cloud(max_words, frequency_dict):
    word_cloud = WordCloud(background_color='black', max_words=max_words)
    word_cloud.generate_from_frequencies(frequency_dict)
    return word_cloud


class WordClouder:
    def __init__(self, answers, max_words=20):
        self.answers = answers
        self.frequency_dict = create_frequency_dict(answers)
        # self.document_frequency_dict = create_document_frequency(answers, set_of_words=set_of_words)
        self.max_words = max_words
        self.word_cloud = create_word_cloud(self.max_words, self.frequency_dict)

    def save_word_cloud(self, path, filename):
        validate_path(path)
        self.word_cloud.to_file(os.path.join(path, filename))


if __name__ == "__main__":
    pass
    # # Load dataset
    # data = pd.read_excel('data/FID=009.ods', sheet_name='FID=009')
    # activity_id = 'fid009'
    # # Make general WordCloud
    # answers = data.fs_answer
    # answers = answers.astype(str)
    # answers = answers.apply(lambda x: answer_preprocessing(x, root_mode='none'))
    # word_cloud = WordClouder(answers, max_words=50)
    # word_cloud.save_word_cloud(os.path.join('data', activity_id), f'WordCloud_50.png')
    # # WordCLoud by levels
    # levels = data.groupby(by="level")
    # for name, group in levels:
    #     _answers = group.fs_answer
    #     _answers = _answers.astype(str)
    #     _answers = _answers.apply(lambda x: answer_preprocessing(x))
    #     word_cloud = WordClouder(_answers, max_words=50)
    #     word_cloud.save_word_cloud(os.path.join('data', activity_id), f'WordCloud_{"".join(["level", str(name), "_"])}50.png')