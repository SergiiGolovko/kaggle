import time
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from pandas import DataFrame
from _estimation import feature_classification


def cos_sim(x, y):
    return cosine_similarity([x], [y])[0][0]


def my_cosine_similarity(X, Y):

    time_start = time.time()
    print "my cosine similarity"

    Z = map(cos_sim, X, Y)

    print "finished my cosine similarity, time elapsed " + str(time.time()-time_start) + " sec."
    return Z

global_metric = ""


def pairwise_dist(x, y):
    global global_metric
    return pairwise_distances([x], [y], metric=global_metric)[0][0]


def my_pairwise_distance(X, Y, metric):

    global global_metric
    time_start = time.time()
    global_metric = metric
    print "my pairwise distance"

    Z = map(pairwise_dist, X, Y)

    print "finished my pairwise dist, time elapsed " + str(time.time()-time_start) + " sec."
    return Z


def create_word2vec_distances(word2vec):

    df = DataFrame()
    df['id'] = word2vec['id']

    query_columns = [c for c in word2vec.columns if 'query' in c]
    title_columns = [c for c in word2vec.columns if 'title' in c]

    query_m = word2vec[query_columns].values
    title_m = word2vec[title_columns].values

    metrics = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']

    df['cosine_dist'] = my_cosine_similarity(query_m, title_m)

    for metric in metrics:
        df['pairwise_dist' + metric] = my_pairwise_distance(query_m, title_m, metric)

    df['cosine_dist'] = my_cosine_similarity(query_m, title_m)

    return df


if __name__ == "__main__":

    df_train = pd.read_csv('input/train_clean.csv')
    df_test = pd.read_csv('input/test_clean.csv')

    df_all = pd.concat([df_train, df_test], ignore_index=True)

    feat_dict = feature_classification(df_all)

    df = create_word2vec_distances(df_all[feat_dict['set12']+["id"]])
    df.to_csv('input/tfidf_dist_features.csv', index=False)

    #word2vec = pd.read_csv('input/word2vec_features.csv')
    #df = create_word2vec_distances(word2vec)
    #df.to_csv('input/word2vec_dist_features.csv', index=False)

