import re
import string

import numpy as np
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


def process_tweet(tweet: str) -> list[str]:
    """Process tweet function.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet

    """
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words("english")
    # remove stock market tickers like $GE
    tweet = re.sub(r"\$\w*", "", tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r"^RT[\s]+", "", tweet)
    # remove hyperlinks
    tweet = re.sub(r"https?://[^\s\n\r]+", "", tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r"#", "", tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (
            word not in stopwords_english  # remove stopwords
            and word not in string.punctuation
        ):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean


def build_freqs(tweets: list[str], ys: np.ndarray) -> dict:
    """Build frequencies.
    Input:
        tweets: a list of tweets
        ys: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    yslist = np.squeeze(ys).tolist()

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs
