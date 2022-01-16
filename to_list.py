"""
Make processed data into lists
"""
import pandas as pd

raw_df = pd.read_csv('news_summary_more.csv', encoding='iso-8859-1')
# used raw but change code to processed csv
listed_df = pd.DataFrame


def to_list(text: pd.DataFrame) -> list:
    """Make text from a single string into a list of strings."""
    return [word for word in text]


listed_df['summary'] = raw_df['summary'].apply(lambda entry: to_list(entry))
listed_df['text'] = raw_df['text'].apply(lambda entry: to_list(entry))
