from __future__ import division
import numpy as np
import pandas as pd
import time
import sys
from gensim.models import Word2Vec
from pandas import DataFrame
from _query_features import create_query_features

from nltk.stem.snowball import SnowballStemmer # 0.003 improvement but takes twice as long as PorterStemmer
stemmer = SnowballStemmer('english')

num_features = 300
TEST_MODE = False
STEMMING = False


def makeFeatureVec(str1, model, word_dictionary):

    featureVec = np.zeros((num_features,), dtype="float32")
    words = str1.split()
    nwords = len(words)

    for word in words:
        if word in word_dictionary:
            featureVec = np.add(featureVec, model[word_dictionary[word]])

    #
    # Divide the result by the number of words to get the average
    if nwords > 0:
        featureVec = featureVec / nwords

    return list(featureVec)


def load_pretrained_model():

    time_start = time.time()
    print "Start loading pretrained model: Google News"

    model = Word2Vec.load_word2vec_format("pretrain_models/GoogleNews.bin", binary=True)

    words = list(set(model.index2word))

    if STEMMING:
        words_stemmed = [stemmer.stem(word) for word in words]
        word_dictionary = dict(zip(words_stemmed, words))
    else:
        word_dictionary = dict(zip(words, words))

    print "Model is loaded, time elapsed is: " + str(time.time()-time_start) + " sec. "

    return model, word_dictionary


def create_word2vec_features(df_all, model, word_dictionary):

    time_start = time.time()

    print "Creating Word2 Vec Features, It may take some time"

    if STEMMING:
        columns = ['query', 'title']
    else:
        columns = ['not_lemma_query', 'not_lemma_title']

    df = DataFrame()
    df['id'] = df_all['id']

    for c in columns:
        print "doing word2vec for column " + c + " time elapsed " + str(time.time()-time_start) + " sec."
        df[c + "features"] = df_all[c].map(lambda x: makeFeatureVec(x, model, word_dictionary))
        for i in range(num_features):
            df[c + "word2vec" + str(i)] = df[c + "features"].map(lambda x: x[i])
        df.drop([c+'features'], axis=1, inplace=True)

    print "Word2Vec features are created, time elapsed " + str(time.time()-time_start) + " sec."

    return df


def makeSimilarityFeatures(str1, str2, model, word_dictionary):

    words1, words2 = set(str1.split()), set(str2.split())
    # list of list of similarities
    sims = []

    for word1 in words1:

        sims.append([0])

        # check if word1 in dictionary
        if word1 not in word_dictionary:
            continue

        # create similarities with each word from word2 set
        for word2 in words2:
            # check if word2 in dictionary
            if word2 not in word_dictionary:
                continue

            # calculate similarity between words
            sim = model.similarity(word_dictionary[word1], word_dictionary[word2])
            sims[-1].append(sim)

            # if sim close to 1, no need to find better match
            if sim > 1 - 1.e-7:
                break

    # calculate the max of similarity for each word in words1
    sims = [max(s) for s in sims]
    sims = np.array(sims)

    # return different statistics: max, min, median, std
    return sims.max(), sims.min(), sims.mean(), sims.std()


def create_word2vec_similarities_features(df_all, model, word_dictionary):

    time_start = time.time()

    print "Creating Word2 Vec Similarity Features, It may take some time"

    # number of statistics to produce
    n_statistics = 4

    df = DataFrame()
    df['id'] = df_all['id']

    if STEMMING:
        query_name, title_name = 'query', 'title'
    else:
        query_name, title_name = 'not_lemma_query', 'not_lemma_title'

    df["query_title_word2vec_sim"] = df_all.apply(lambda x: makeSimilarityFeatures(x[query_name], x[title_name],
                                                                                   model,
                                                                                   word_dictionary), axis=1)
    for i in range(n_statistics):
        df["query_title_word2vec_sim" + str(i)] = df["query_title_word2vec_sim"].map(lambda x: x[i])

    df.drop(["query_title_word2vec_sim"], axis=1, inplace=True)

    print "Word2Vec features are created, time elapsed " + str(time.time()-time_start) + " sec."

    return df


if __name__ == '__main__':

    time_start = time.time()

    print "################CREATING WORD 2 VEC FEATURES - IT MAY TAKE SOME TIME - ####################"

    reload(sys)
    sys.setdefaultencoding('utf8')

    # read all cleaned
    df_all = pd.read_csv('input/all_cleaned.csv', encoding="ISO-8859-1")

    # load pretrained model
    model, word_dictionary = load_pretrained_model()

    if TEST_MODE:
        df_all = df_all[0:100]

    df = create_word2vec_features(df_all, model, word_dictionary)
    df.to_csv('input/word2vec_features.csv', index=False)

    df = create_word2vec_similarities_features(df_all, model, word_dictionary)
    df.to_csv('input/word2vec_similarity_features.csv', index=False)

    df = create_query_features(df_all, columns=['not_lemma_title'], qcol='not_lemma_query')
    df.to_csv('input/query_features.csv')

    print "done! time elapsed " + str(time.time()-time_start) + " sec."
    print "################CREATING FEATURES####################"

