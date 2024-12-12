import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

retweet_pattern = r"^RT[\s]+"
hyperlink_pattern = r"https?://[^\s\n\r]+"


def clean_text(text: str) -> str:
    cleaned_text = re.sub(retweet_pattern, "", text)
    cleaned_text = re.sub(hyperlink_pattern, "", cleaned_text)
    cleaned_text = re.sub("#", "", cleaned_text)
    return cleaned_text


def tokenize_text(text: str) -> list[str]:
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tokens = tokenizer.tokenize(text)
    return tokens


def remove_stopwords_punctuations(tokens: list[str]) -> list[str]:
    stopwords_english = stopwords.words("english")
    cleaned_tokens = [
        token
        for token in tokens
        if token not in stopwords_english and token not in string.punctuation
    ]
    return cleaned_tokens


def perform_stemming(tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    stemmed_text = [stemmer.stem(token) for token in tokens]
    return stemmed_text


def process_tweet(text: str) -> list[str]:
    cleaned_text = clean_text(text)
    tokens = tokenize_text(cleaned_text)
    cleaned_tokens = remove_stopwords_punctuations(tokens)
    stemmed_text = perform_stemming(cleaned_tokens)
    return stemmed_text


def build_freqs(tweets, ys):
    assert len(tweets) == len(ys), "length of labels should match the length of tweets"
    freqs = {}
    for tweet, y in zip(tweets, ys):
        for token in process_tweet(tweet):
            token_label_pair = (token, y)
            freqs[token_label_pair] = freqs.get(token_label_pair, 0) + 1
    return freqs
