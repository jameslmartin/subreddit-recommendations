import numpy as np
import pandas as pd
import sqlite3
import gensim
import re
from nltk.corpus import stopwords
import nltk

sql_conn = sqlite3.connect('../data/database.sqlite')
mathematics = pd.read_sql("SELECT subreddit, body FROM May2015 WHERE subreddit == 'mathematics'",sql_conn)
computerscience = pd.read_sql("SELECT subreddit, body FROM May2015 WHERE subreddit == 'computerscience'",sql_conn)
history = pd.read_sql("SELECT subreddit, body FROM May2015 WHERE subreddit == 'history'",sql_conn)
philosophy = pd.read_sql("SELECT subreddit, body FROM May2015 WHERE subreddit == 'philosophy'",sql_conn)
elifive = pd.read_sql("SELECT subreddit, body FROM May2015 WHERE subreddit == 'explainlikeimfive'",sql_conn)
askanthro = pd.read_sql("SELECT subreddit, body FROM May2015 WHERE subreddit == 'AskAnthropology'",sql_conn)
homebrewing = pd.read_sql("SELECT subreddit, body FROM May2015 WHERE subreddit == 'Homebrewing'",sql_conn)
bicycling = pd.read_sql("SELECT subreddit, body FROM May2015 WHERE subreddit == 'bicycling'", sql_conn)
food = pd.read_sql("SELECT subreddit, body FROM May2015 WHERE subreddit == 'food'", sql_conn)
science = pd.read_sql("SELECT subreddit, body FROM May2015 WHERE subreddit == 'science'", sql_conn)
movies = pd.read_sql("SELECT subreddit, body FROM May2015 WHERE subreddit == 'movies'", sql_conn)
books = pd.read_sql("SELECT subreddit, body FROM May2015 WHERE subreddit == 'books'", sql_conn)

all_frames = [bicycling, history, philosophy, elifive, homebrewing, askanthro, mathematics, computerscience, food, science, movies, books]
all_data = pd.concat(all_frames, ignore_index=True)
# Takes a sentence in a comment and converts it to a list of words.
def comment_to_wordlist(comment, remove_stopwords=False ):
    comment = re.sub("[^a-zA-Z]"," ", comment)
    words = comment.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    return(words)

# Takes a comment and converts it to an array of sentences
def comment_to_sentence(comment, tokenizer, remove_stopwords=False):
    raw_sentences = tokenizer.tokenize(comment.strip())
    
    sentences = []
    for s in raw_sentences:
        if len(s)>0:
            sentences.append(comment_to_wordlist(s, remove_stopwords))
    return sentences

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

comments = []
ground_truth_labels = []
for i in range(len(all_data)):
    comments.append(comment_to_wordlist(all_data.iloc[i]['body'], tokenizer))
    ground_truth_labels.append(all_data.iloc[i]['subreddit']) # Could also just slice this array out

def reduce_comments(words, model):
    # Pre-allocate average vector representation
    average = np.zeros(300, dtype=np.float64)
    in_vocab = 0
    for word in words:
        if word in model.vocab:
            in_vocab += 1
            average = average + np.array(model[word], dtype=float)
    return np.divide(average,in_vocab)

from gensim.models import word2vec

m = "300features_10minwords_10context"
current_model = word2vec.Word2Vec.load('new_models/' + m);
single_rep_comments = map(lambda comment: reduce_comments(comment, current_model), comments)
d = np.asarray(single_rep_comments, dtype=np.float64)

# Pull out indices of rows where the vector representation does not contain a NaN
# Not sure how NaNs crept in here, possibly due to overflow issues
ix_non_nans = np.where(~(np.isnan(d).any(axis=1)))[0]

# Filter out NaNs on comments and labels
d = d[ix_non_nans]
gtls = np.asarray(ground_truth_labels)[ix_non_nans]

# Cluster using KMeans, initializing k = 12 (one for each of the subreddits we drew from)
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 12)
kmeans.fit(d)
