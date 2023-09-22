import numpy as np
from pandas import DataFrame
import time


def position_in(str1, str2):

    str1, str2 = str(str1), str(str2)
    words = str1.split()
    positions = []
    for word in words:
        positions.append(str2.find(word))
    positions = np.array(positions)
    return np.min(positions), np.max(positions), np.std(positions), np.mean(positions)


def word_position_in(str1, str2):

    str1, str2 = str(str1), str(str2)
    words1, words2 = str1.split(), str2.split()
    positions = []
    for word1 in words1:
        cur_pos, pos = -1, -1
        for word2 in words2:
            cur_pos += 1
            if word1 in word2:
                pos = cur_pos
                break

        positions.append(pos)

    positions = np.array(positions)
    return np.min(positions), np.max(positions), np.std(positions), np.mean(positions)


def create_position_features(df_all, columns, qcol):

    time_start = time.time()

    print "Creating position features, it may take some time"
    df = DataFrame()
    df['id'] = df_all['id']

    for c in columns:

        df['temp1'] = df_all.apply(lambda x: position_in(x[qcol], x[c]), axis=1)
        df['temp2'] = df_all.apply(lambda x: word_position_in(x[qcol], x[c]), axis=1)

        prefixes, indexes = ('min', 'max', 'std', 'mean'), (0, 1, 2, 3)

        for (prefix, i) in zip(prefixes, indexes):
            name = prefix + '_' + 'pos' + qcol + '_in_' + c
            df[name + '1'] = df['temp1'].map(lambda x: x[i])
            df[name + '2'] = df['temp2'].map(lambda x: x[i])

        df.drop(['temp1', 'temp2'], axis=1, inplace=True)

    print "Position features are created, time elapsed " + str(time.time() - time_start) + " sec."

    return df




