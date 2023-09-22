from __future__ import division
from pandas import DataFrame
from _count import *
import numpy as np
import time

def extract_digits(s):

    words = s.split()
    digits = []

    for word in words:
        digit = "".join([l for l in word if l.isdigit() or l == "."])

        find_dot = digit.find(".")
        if find_dot > -1:
            digit = digit[:find_dot+1] + digit[find_dot+1:].replace(".", "")

        if (len(digit) > 0) and (digit[len(digit)-1] == "."):
            digit = digit[:len(digit)-1]

        if len(digit) > 0:
            digits.append(digit)

    return " ".join(digits)


def create_digit_features(df_all, columns, qcol):

    time_start = time.time()

    print "Creating digit features, it may take some time"
    df = DataFrame()
    df['id'] = df_all['id']

    df_all['d_' + qcol] = df_all.apply(lambda x: extract_digits(x[qcol]), axis=1)
    df_all['d_' + qcol] = df_all.apply(lambda x: extract_digits(x[qcol]), axis=1)

    # create length of query features, len - number of words, length - number of letters
    df['d_' + 'len_of_' + qcol] = df_all.apply(lambda x: count_words(x['d_' + qcol]), axis=1).astype(np.int64)
    df['d_' + 'length_of_' + qcol] = df_all.apply(lambda x: length_words(x['d_' + qcol]), axis=1).astype(np.int64)

    for c in columns:

        df_all['d_' + c] = df_all[c].map(lambda x: extract_digits(x))

        #df['d_' + qcol + '_in_' + c + '1'] = df_all.apply(lambda x: count_common_words(x['d_' + qcol], x['d_'+c]), axis=1)
        df['d_' + qcol + '_in_' + c + '2'] = df_all.apply(lambda x: count_whole_words(x['d_' + qcol], x['d_'+c]), axis=1)

        #df['d_' + qcol + '_in_' + c + '3'] = df_all.apply(lambda x: length_common_words(x['d_' + qcol], x['d_'+c]), axis=1)
        df['d_' + qcol + '_in_' + c + '4'] = df_all.apply(lambda x: length_whole_words(x['d_' + qcol], x['d_'+c]), axis=1)

        # create length features
        df['d_' + 'len_of_' + c] = df_all.apply(lambda x: count_words(x['d_'+c]), axis=1).astype(np.int64)
        df['d_' + 'length_of_' + c] = df_all.apply(lambda x: length_words(x['d_'+c]), axis=1).astype(np.int64)

        # create ration features
        #df['d_'+qcol + 'ratio_in_' + c + '1'] = df['d_'+qcol + '_in_' + c + '1'] / df['d_'+'len_of_' + c]
        df['d_'+qcol + 'ratio_in_' + c + '2'] = df['d_'+qcol + '_in_' + c + '2'] / df['d_'+'len_of_' + c]
        #df['d_'+qcol + 'ratio_in_' + c + '3'] = df['d_'+qcol + '_in_' + c + '3'] / df['d_'+'length_of_' + c]
        df['d_'+qcol + 'ratio_in_' + c + '4'] = df['d_'+qcol + '_in_' + c + '4'] / df['d_'+'length_of_' + c]

        # create ratio of found words
        #df['d_'+qcol + 'fratio_in_' + c + '1'] = df['d_'+qcol + '_in_' + c + '1'] / df['d_'+'len_of_' + qcol]
        df['d_'+qcol + 'fratio_in_' + c + '2'] = df['d_'+qcol + '_in_' + c + '2'] / df['d_'+'len_of_' + qcol]
        #df['d_'+qcol + 'fratio_in_' + c + '3'] = df['d_'+qcol + '_in_' + c + '3'] / df['d_'+'length_of_' + qcol]
        df['d_'+qcol + 'fratio_in_' + c + '4'] = df['d_'+qcol + '_in_' + c + '4'] / df['d_'+'length_of_' + qcol]

    print "Digit features are created, time elapsed " + str(time.time()-time_start) + " sec."

    return df





