from __future__ import division
import time
from pandas import DataFrame
import numpy as np
from _count import *


def create_query_features(df_all, columns, qcol):

    time_start = time.time()

    print "Creating query features, it may take some time"
    df = DataFrame()
    df['id'] = df_all['id']

    # create length of query features, len - number of words, length - number of letters
    df['len_of_' + qcol] = df_all.apply(lambda x: count_words(x[qcol]), axis=1).astype(np.int64)
    df['length_of_' + qcol] = df_all.apply(lambda x: length_words(x[qcol]), axis=1).astype(np.int64)

    #return df
    for c in columns:

        df[qcol + '_in_' + c + '1'] = df_all.apply(lambda x: count_common_words(x[qcol], x[c]), axis=1)
        df[qcol + '_in_' + c + '2'] = df_all.apply(lambda x: count_whole_words(x[qcol], x[c]), axis=1)

        df[qcol + '_in_' + c + '3'] = df_all.apply(lambda x: length_common_words(x[qcol], x[c]), axis=1)
        df[qcol + '_in_' + c + '4'] = df_all.apply(lambda x: length_whole_words(x[qcol], x[c]), axis=1)

        # create length features
        df['len_of_' + c] = df_all.apply(lambda x: count_words(x[c]), axis=1).astype(np.int64)
        df['length_of_' + c] = df_all.apply(lambda x: length_words(x[c]), axis=1).astype(np.int64)

        # create ration features
        df[qcol + 'ratio_in_' + c + '1'] = df[qcol + '_in_' + c + '1'] / df['len_of_' + c]
        df[qcol + 'ratio_in_' + c + '2'] = df[qcol + '_in_' + c + '2'] / df['len_of_' + c]
        df[qcol + 'ratio_in_' + c + '3'] = df[qcol + '_in_' + c + '3'] / df['length_of_' + c]
        df[qcol + 'ratio_in_' + c + '4'] = df[qcol + '_in_' + c + '4'] / df['length_of_' + c]

        # create ratio of found words
        df[qcol + 'fratio_in_' + c + '1'] = df[qcol + '_in_' + c + '1'] / df['len_of_' + qcol]
        df[qcol + 'fratio_in_' + c + '2'] = df[qcol + '_in_' + c + '2'] / df['len_of_' + qcol]
        df[qcol + 'fratio_in_' + c + '3'] = df[qcol + '_in_' + c + '3'] / df['length_of_' + qcol]
        df[qcol + 'fratio_in_' + c + '4'] = df[qcol + '_in_' + c + '4'] / df['length_of_' + qcol]

    print "Query features are created, time elapsed " + str(time.time()-time_start) + " sec."

    return df
