import time
from pandas import DataFrame
from _tfidf import *

TFIDF_ANALYSIS = True
TFIDF2 = True


def create_tfidf_features(df_all, columns, qcol, unique=False):

    df = DataFrame()
    df['id'] = df_all['id']

    if TFIDF_ANALYSIS:

        for c in columns:

            time_start = time.time()

            print "Doing TFIDF Analysis, it may take some time"
            if unique:
                create_idf(df_all[c].unique())
            else:
                create_idf(df_all[c])

            # different types of tfidf
            types = ('binary', 'freq', 'log_freq', 'dnorm')

            # different ways of aggregating term tfidf in query
            indexes, prefixes = (0, 1, 2), ('s', 'm', 'a')

            # two different functions - one exact match, other - common words
            funcs, suffixes = [tfidf1, tfidf2], ('1', '2')

            for (func, suffix) in zip(funcs, suffixes):
                if (func == tfidf2) and (not TFIDF2):
                    continue

                df['temp'] = df_all.apply(lambda x: func(x[qcol], x[c], type='all'), axis=1)

                ind = 0
                for t in types:
                    for prefix in prefixes:
                        name = qcol + prefix + t + '_tfidf_' + c + suffix
                        df[name] = df['temp'].map(lambda x: x[ind])
                        ind += 1

            df.drop(['temp'], axis=1, inplace=True)

            print "TFIDF analysis finished, time elapsed " + str(time.time() - time_start) + " sec."

    return df