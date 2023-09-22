from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from pandas import DataFrame
import time
import pandas as pd
import numpy as np

norm = "l2"
max_df = 0.75
min_df = 3
ngram_range = (1, 3)
n_components = 100
token_pattern = r"(?u)\b\w\w+\b"


stop_words = ['xbi', 'in.', 'ft.', 'oz.', 'gal.', 'mm.', 'cm.', 'deg.', 'volt.', 'watt.', 'amp.', 'lb.', 'deg.',
              'and', 'in', 'that', 'l', 'r', ".", "&", "h", "w", "a", "d"]


def create_tfidf_svd_features(df_all, columns):

    time_start = time.time()
    print "Creating tfidf svd features, it may take some time"

    df = DataFrame()
    df['id'] = df_all['id']

    for c in columns:

        tfidf_vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, max_features=None,
                                           strip_accents='unicode', analyzer='word', token_pattern=token_pattern,
                                           ngram_range=ngram_range, use_idf=1, smooth_idf=1, sublinear_tf=1,
                                           stop_words=stop_words, norm=norm, vocabulary=None)

        X = tfidf_vectorizer.fit_transform(df_all[c])

        svd = TruncatedSVD(n_components=n_components, n_iter=15)
        X = svd.fit_transform(X)

        svd_columns = ['tfidf_svd_' + c + str(i) for i in range(n_components)]

        df1 = DataFrame(X, columns=svd_columns)

        df[svd_columns] = df1[svd_columns]

    print "Tfidf svd features are created, time elapsed " + str(time.time()-time_start) + " sec."

    return df


def create_ctfidf_svd_features(df_all):
    """ Create tfidf svd features with common vocabulary
    :param df_all:
    :return:
    """

    time_start = time.time()
    print "Creating tfidf svd features, it may take some time"

    #ser_text = pd.concat([df_all['query'], df_all['title'], df_all['description']], axis=0)
    ser_text = pd.concat([df_all['query'], df_all['title']], axis=0)
    tfidf_vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, max_features=None,
                                       strip_accents='unicode', analyzer='word', token_pattern=token_pattern,
                                       ngram_range=ngram_range, use_idf=1, smooth_idf=1, sublinear_tf=1,
                                       stop_words=stop_words, norm=norm, vocabulary=None)

    X = tfidf_vectorizer.fit_transform(ser_text)

    svd = TruncatedSVD(n_components=n_components, n_iter=15)
    X = svd.fit_transform(X)

    df = DataFrame()
    df['id'] = df_all['id']

    X_query = X[0:len(df_all)]
    svd_columns = ['ctfidf_svd_' + 'query' + str(i) for i in range(n_components)]
    df1 = DataFrame(X_query, columns=svd_columns)
    df[svd_columns] = df1[svd_columns]

    X_title = X[len(df_all):2*len(df_all)]
    svd_columns = ['ctfidf_svd_' + 'title' + str(i) for i in range(n_components)]
    df2 = DataFrame(X_title, columns=svd_columns)
    df[svd_columns] = df2[svd_columns]

    print "Tfidf svd features are created, time elapsed, time elapsed " + str(time.time()-time_start) + " sec."

    return df


def cos_sim(x, y):
    return cosine_similarity(x, y)[0][0]


def my_cosine_similarity(X, Y):

    time_start = time.time()
    print "my cosine similarity"

    Z = map(cos_sim, X, Y)

    print "finished my cosine similarity, time elapsed " + str(time.time()-time_start) + " sec."
    return Z


def create_cosine_similarity_features(df_all):

    time_start = time.time()
    print "Creating cosine similarity features, it may take some time"

    #ser_text = pd.concat([df_all['query'], df_all['title'], df_all['description']], axis=0)
    ser_text = pd.concat([df_all['query'], df_all['title']], axis=0)
    tfidf_vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, max_features=None,
                                       strip_accents='unicode', analyzer='word', token_pattern=token_pattern,
                                       ngram_range=ngram_range, use_idf=1, smooth_idf=1, sublinear_tf=1,
                                       stop_words=stop_words, norm=norm, vocabulary=None)

    X = tfidf_vectorizer.fit_transform(ser_text)
    X_query = X[0:len(df_all)]
    X_title = X[len(df_all):2*len(df_all)]
    #X_description = X[2*len(df_all):3*len(df_all)]

    df = DataFrame()
    df['id'] = df_all['id']
    df['query_cosine_similarity_query'] = my_cosine_similarity(X_query, X_title)
    #df['query_cosine_similarity_description'] = my_cosine_similarity(X_query, X_description)
    #df['title_cosine_similarity_description'] = my_cosine_similarity(X_description, X_title)

    print "Cosine similarity features are created, time elapsed " + str(time.time()-time_start) + " sec."

    return df
