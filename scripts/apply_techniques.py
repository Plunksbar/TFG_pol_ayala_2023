import pandas as pd
import os

from services.answer_preprocessing import answer_preprocessing
from services.word_cloud import WordClouder
from services.topic_modeling_LDA import LDA
from services.adjectives_frequency import AdjectivesFreq


def LDA_modeling(activity_id, answers, specific=''):
    answers = answers.astype(str)
    answers = answers.apply(lambda x: answer_preprocessing(x, root_mode='none'))
    lda_analysis = LDA(answers)
    # lda_analysis.generate_bigrams(os.path.join('data', activity_id, 'bigrams'))
    lda_analysis.model_topics(os.path.join('data', activity_id), f'ModelTopics_{specific}{2}.png', num_topics=2)
    lda_analysis.model_topics(os.path.join('data', activity_id), f'ModelTopics_{specific}{3}.png', num_topics=3)
    lda_analysis.model_topics(os.path.join('data', activity_id), f'ModelTopics_{specific}{4}.png', num_topics=4)
    lda_analysis.model_topics(os.path.join('data', activity_id), f'ModelTopics_{specific}{5}.png', num_topics=5)


def wordcloud(activity_id, answers, specific=''):
    # answers = answers.astype(str)
    # answers = answers.apply(answer_preprocessing)
    word_cloud = WordClouder(answers, max_words=50)
    word_cloud.save_word_cloud(os.path.join('data', activity_id), f'WordCloud_{specific}{50}.png')


def adjective_count(activity_id, answers, specific=''):
    answers = answers.astype(str)
    adjective_freq = AdjectivesFreq(answers)
    adjective_freq.plot_top_nouns(os.path.join('data', activity_id), f'Adjectives_{specific}.png')


def main():
    activities = {
        # 'fid390': pd.read_excel('data/FID=390 and FID=391.ods', sheet_name='FID=390')\
        #                 .set_index('fs_id').query('level == 0').fs_answer,
        # 'fid391': pd.read_excel('data/FID=390 and FID=391.ods', sheet_name='FID=391')\
        #                 .set_index('fs_id').query('level == 0').fs_answer,
        # 'fid395': pd.read_excel('data/FID=395 and FID=396.ods', sheet_name='FID=395')\
        #                 .set_index('fs_id').query('level == 0').fs_answer,
        # 'fid396': pd.read_excel('data/FID=395 and FID=396.ods', sheet_name='FID=396')\
        #                 .set_index('fs_id').query('level == 0').fs_answer
    }
    multilevel = {
        # 'fid386': pd.read_excel('data/FID=386 and FID=387.ods', sheet_name='FID=386'),
        # 'fid387': pd.read_excel('data/FID=386 and FID=387.ods', sheet_name='FID=387'),
        'fid009': pd.read_excel('data/FID=009.ods', sheet_name='FID=009')
    }

    for activity_id, answers in activities.items():
        wordcloud(activity_id, answers)
        LDA_modeling(activity_id, answers)
        adjective_count(activity_id, answers)
    for activity_id, answers in multilevel.items():
        _answers = answers.fs_answer
        wordcloud(activity_id, _answers)
        LDA_modeling(activity_id, _answers)
        adjective_count(activity_id, _answers)
        # levels = answers.groupby(by="level")
        # for name, group in levels:
        #     _answers = group.fs_answer
        #     wordcloud(activity_id, _answers, ''.join(['level', str(name), '_']))
        #     LDA_modeling(activity_id, _answers, ''.join(['level', str(name), '_']))
        #     adjective_count(activity_id, _answers, ''.join(['level', str(name), '_']))


if __name__ == '__main__':
    main()
