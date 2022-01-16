"""Data preprocessing

"""

import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import re
lemmatizer = WordNetLemmatizer()
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
    # filtered_words = [w for w in tokens if len(w) > 2 if w not in stopwords.words('english')]
    # stem_words = [stemmer.stem(w) for w in filtered_words]
    # lemma_words = [lemmatizer.lemmatize(w) for w in stem_words]
    return tokens


raw_df['clean'] = raw_df["text"].apply(lambda x: clean_text(x))
raw_df['token'] = raw_df['clean'].apply(lambda x: tokenise(x))
