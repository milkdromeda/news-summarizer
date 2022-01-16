"""Data preprocessing

"""

import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import re

stemmer = PorterStemmer()

raw_df = pd.read_csv('news_summary_more.csv', encoding='iso-8859-1')
print(raw_df.head())


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[<>()|&©ø\[\]\'\",;?~*!]", ' ', str(text))  # removes unnecessary punctuation

    text = re.sub(r"(\.\s+)", ' ', str(text))  # removes punctuation at end of words(not between)
    text = re.sub(r"(\:\s+)", ' ', str(text))

    text = re.sub(r"(\s+)", ' ', str(text))  # remove multiple spaces
    text = re.sub(r"(\s+.\s+)", ' ', str(text))  # removes single characters between spaces
    return text


def tokenise(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    return tokens


def lemmatizer(text):
    lemmatizer = WordNetLemmatizer()
    lemm_text = [lemmatizer.lemmatize(word) for word in text]
    return lemm_text


raw_df['clean'] = raw_df['text'].apply(lambda x: clean_text(x))
raw_df['token'] = raw_df['clean'].apply(lambda x: tokenise(x))
raw_df['lemma'] = raw_df['token'].apply(lambda x: lemmatizer(x))
