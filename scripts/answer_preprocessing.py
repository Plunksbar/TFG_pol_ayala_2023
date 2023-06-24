import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import enchant
from langdetect import detect
from textblob import TextBlob


def stemmer(answer):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in answer]


def lemmatize(answer):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in answer]


def correct_misspelling(misspelling, dictionary):
    suggestions = dictionary.suggest(misspelling)
    if suggestions:
        # print(misspelling)
        # print(suggestions)
        return suggestions[0]
    else:
        print(misspelling, '; Unable to correct misspelling')
        return misspelling


def check_spelling(answer):
    dictionary = enchant.Dict('en')
    for i, word in enumerate(answer):
        if not dictionary.check(word):
            answer[i] = correct_misspelling(word, dictionary)
    return answer


def translate_answer(answer, in_lang, to='en'):
    blob = TextBlob(answer)
    return str(blob.translate(from_lang=in_lang, to=to))


def remove_stopwords(answer, language='en'):
    lang_codes = {
        'en': 'english',
        'fr': 'french',
        'es': 'spanish',
        'cat': 'catalan'
    }
    stop_words = set(stopwords.words(lang_codes[language]))
    return [token for token in answer if token not in stop_words]


def regularize(answer):
    return [token for token in answer if token.isalpha()]


def tokenize(answer):
    return word_tokenize(answer)


def answer_preprocessing(answer, root_mode='lemmatize'):
    assert root_mode in ['lemmatize', 'stem', 'none', 'both']
    if len(answer) < 3:
        return answer

    language = detect(str(answer))
    language = 'en' if language not in ['en', 'cat', 'es'] else language

    answer = answer.lower()
    answer = tokenize(answer)
    answer = regularize(answer)
    if language != 'en':
        print(answer)
        answer = translate_answer(' '.join(answer), in_lang=language)
        answer = answer.lower()
        answer = tokenize(answer)
        answer = regularize(answer)
        print(answer)
        language = 'en'
    answer = remove_stopwords(answer)
    answer = check_spelling(answer)

    if root_mode == 'lemmatize':
        answer = lemmatize(answer)
    if root_mode == 'stem':
        answer = stemmer(answer)
    if root_mode == 'both':
        answer = lemmatize(answer)
        answer = stemmer(answer)

    return answer


if __name__ == '__main__':
    pass
    # sample_answers = pd.read_excel('data/FID=390 and FID=391.ods')
    # sample_answers = sample_answers[['level', 'individual_level', 'fs_answer']]
    # sample_answers['fs_clean'] = sample_answers.fs_answer.apply(lambda x: answer_preprocessing(x, root_mode='lemmatize'))
    # print(sample_answers['fs_clean'])