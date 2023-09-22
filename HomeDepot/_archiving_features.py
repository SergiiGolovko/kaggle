from pandas import DataFrame
import zlib
import time


def create_archiving_features(df_all):

    time_start = time.time()

    print "Creating Archiving Features, It may take some time"

    df_temp = DataFrame()
    lev = 5
    # 'q' -> query, 't' -> title, 'd' -> description
    df_temp['q'] = df_all['query'].map(lambda x: zlib.compress(x.encode("utf-8"), lev))
    df_temp['t'] = df_all['title'].map(lambda x: zlib.compress(x.encode("utf-8"), lev))
    df_temp['d'] = df_all['description'].map(lambda x: zlib.compress(x.encode("utf-8"), lev))

    df_temp['qt'] = df_all.apply(lambda x: zlib.compress((x['query'] + x['title']).encode("utf-8"), lev), axis=1)
    df_temp['qd'] = df_all.apply(lambda x: zlib.compress((x['query'] + x['description']).encode("utf-8"), lev), axis=1)
    df_temp['td'] = df_all.apply(lambda x: zlib.compress((x['title'] + x['description']).encode("utf-8"), lev), axis=1)

    df = DataFrame()
    df['id'] = df_all['id']

    # length features, compr -> compressed
    df['len_q_compr'] = df_temp['q'].map(lambda x: len(x))
    df['len_t_compr'] = df_temp['t'].map(lambda x: len(x))
    df['len_d_compr'] = df_temp['d'].map(lambda x: len(x))
    df['len_qt_compr'] = df_temp['qt'].map(lambda x: len(x))
    df['len_qd_compr'] = df_temp['qd'].map(lambda x: len(x))
    df['len_td_compr'] = df_temp['td'].map(lambda x: len(x))

    # ration of lengths
    df['ratio_qt_compr'] = df.apply(lambda x: (x['len_qt_compr']-min(x['len_q_compr'], x['len_t_compr']))/x['len_q_compr'], axis=1)
    df['ratio_qd_compr'] = df.apply(lambda x: (x['len_qd_compr']-min(x['len_q_compr'], x['len_d_compr']))/x['len_q_compr'], axis=1)
    df['ratio_td_compr'] = df.apply(lambda x: (x['len_td_compr']-min(x['len_t_compr'], x['len_d_compr']))/x['len_t_compr'], axis=1)

    print "Archiving features are created, time elapsed " + str(time.time()-time_start) + " sec."

    return df