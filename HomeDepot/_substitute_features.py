from __future__ import division
import pandas as pd
import time
#from pandas import DataFrame
from jellyfish import damerau_levenshtein_distance
from _query_features import create_query_features

DIST_THRESHOLD = 0.2

def replace_with_closest(str1, str2):
    """ Replace
    :param str1: sting which needs to be modified
    :param str2: vocabulary string
    :return: modified string
    """
    str1, str2 = str(str1), str(str2)
    words1, words2 = str1.split(), list(set(str2.split()))
    swords1 = []

    for word1 in words1:
        if word1 in words2:
            swords1.append(word1)
        else:
            #find the closest word
            min_d = 1000
            sword1 = ""
            if len(word1) > 4:
                for word2 in words2:
                    d = damerau_levenshtein_distance(unicode(word1), unicode(word2))
                    if d < min_d:
                        min_d = d
                        sword1 = word2

            if (min_d/len(word1)) <= DIST_THRESHOLD:
                print "The following word was substituted ", word1, 'by', sword1
                swords1.append(sword1)
            else:
                swords1.append(word1)

    return " ".join(swords1)


def create_substituted_features(df_all):
    """ Creature new
    :param df_all:
    :return:
    """

    time_start = time.time()

    print "Creating substituted features, it may take some time"

    df_temp = df_all[['query_uid', 'query', 'title']].groupby(['query_uid', 'query']).agg(lambda x: " ".join(x)).reset_index()
    df_temp = df_temp.rename(columns={'title': 'gtitle'})
    df_temp['squery'] = df_temp.apply(lambda x: replace_with_closest(x['query'], x['gtitle']), axis=1)

    print "Finished creating substituted query, time elapsed " + str(time.time()-time_start) + " sec."
    df_all = pd.merge(df_all, df_temp, how='left', on='query_uid')

    df = create_query_features(df_all, columns=['title', 'description'], qcol='squery')

    print "Substituted features are created, time elapsed " + str(time.time()-time_start) + " sec."

    return df