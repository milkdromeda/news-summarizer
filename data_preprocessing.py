"""Data preprocessing

"""

import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re


# def main():
#     """Carries out these actions if this file is the main file"""



def clean_text(text: str) -> str:
    """Return text with punctuation, whitespace and redundant characters removed."""
    text = text.lower()
    text = re.sub(r"[<>()|&©ø\[\]\'\",;?~*!]", ' ', str(text))  # removes unnecessary punctuation

    text = re.sub(r"(\.\s+)", ' ', str(text))  # removes punctuation at end of words(not between)
    text = re.sub(r"(\:\s+)", ' ', str(text))

    text = re.sub(r"(\s+)", ' ', str(text))  # remove multiple spaces
    text = re.sub(r"(\s+.\s+)", ' ', str(text))  # removes single characters between spaces
    return text


def tokenise(text: str) -> list[str]:
    """
    Return text as a list form with each item in the list being a word.
    """
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    return tokens


def lemmatizer(text: list[str]) -> list[str]:
    """Return the list of text with words lemmatized."""
    lemmatizer = WordNetLemmatizer()
    lemm_text = [lemmatizer.lemmatize(word) for word in text]
    return lemm_text


# def remove_special(text):
#     """Return text with all non-alphanumeric characters removed."""
#     return [char for char in text if char.isalnum() or char == ' ']


if __name__ == '__main__':
    raw_df = pd.read_csv('news_summary_more.csv', encoding='iso-8859-1')
    print(raw_df.head())
    raw_df['processed'] = raw_df['text'].apply(lambda x: clean_text(x))
    # raw_df['processed'] = raw_df['processed'].apply(lambda x: remove_special(x))
    raw_df['processed'] = raw_df['processed'].apply(lambda x: tokenise(x))
    raw_df['processed'] = raw_df['processed'].apply(lambda x: lemmatizer(x))
