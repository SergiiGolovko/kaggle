from __future__ import division
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import time
from _query_features import create_query_features
from _tfidf_features import create_tfidf_features


def create_grouped_features(df_all):

    time_start = time.time()

    print "Creating grouped features, it may take some time"

    df_temp = df_all[['query_uid', 'query', 'title']].groupby(['query_uid', 'query']).agg(lambda x: " ".join(x)).reset_index()
    df_temp = df_temp.rename(columns={'title': 'gtitle', 'query_uid': 'id'})

    df = DataFrame()
    df['id'], df['query_uid'] = df_all['id'], df_all['query_uid']

    df1 = create_query_features(df_temp, columns=['gtitle'], qcol='query')
    df2 = create_tfidf_features(df_temp, columns=['gtitle'], qcol='query')
    df3 = create_tfidf_features(df_temp, columns=['gtitle'], qcol='query', unique=True)

    df1 = df1.rename(columns={'id': 'query_uid'})
    df2 = df2.rename(columns={'id': 'query_uid'})
    df3 = df3.rename(columns={'id': 'query_uid'})

    df = pd.merge(df, df1, how='left', on='query_uid')
    df = pd.merge(df, df2, how='left', on='query_uid')
    df = pd.merge(df, df3, how='left', on='query_uid')

    df.drop(['query_uid'], axis=1, inplace=True)

    print "Grouped features are created, time elapsed " + str(time.time()-time_start) + " sec."

    return df


#def check_df_for_nulls(df):

#    print sum(pd.)

def create_stat_grouped_features(df_all, query_uid):

    time_start = time.time()

    print "Creating stat grouped features, it may take some time"

    df = DataFrame()
    df['id'], df['query_uid'] = df_all['id'], query_uid

    df_temp = DataFrame()
    df_temp['query_uid'] = query_uid

    columns = ['title', 'description']
    suffices = ['1', '2', '3', '4']

    # copy features from df_all to df_temp
    for c in columns:
        for suffix in suffices:
            name = 'query_in_' + c + suffix
            df_temp[name] = df_all[name]

    df_grouped = df_temp.groupby('query_uid')

    # max features
    df_max = df_grouped.max()
    df_max.columns = "max" + df_max.columns
    df_max = df_max.reset_index()

    # min features
    df_min = df_grouped.min()
    df_min.columns = "min" + df_min.columns
    df_min = df_min.reset_index()

    # std features
    df_std = df_grouped.std()
    df_std.columns = "std" + df_std.columns
    df_std = df_std.reset_index()

    # add to df
    for df1 in [df_max, df_min, df_std]:
        df = pd.merge(df, df1, how='left', on='query_uid')

    # print some statistic to get a sense whether features could be useful
    print "Number of records " + str(len(df_all))
    print "Number of query in title that equal to max " + str(sum(df_all['query_in_title1'] == df['maxquery_in_title1']))
    print "Number of query in title that equal to min " + str(sum(df_all['query_in_title1'] == df['minquery_in_title1']))
    print "Number of query in title with std less that 0.0001 " + str(sum(df['stdquery_in_title1'].map(lambda x: 1 if x < 0.0001 else 0)))

    # relative to max features
    for c in df_max.columns:
        # skip query_uid features
        if c == 'query_uid':
            continue

        name = 'reltomax' + c[3:]
        df[name] = df_all[c[3:]] / df[c].map(lambda x: -1 if x == 0 else x)

    # relative to min features
    for c in df_min.columns:
        # skip query_uid features
        if c == 'query_uid':
            continue

        name = 'reltomin' + c[3:]
        df[name] = df_all[c[3:]] / df[c].map(lambda x: -1 if x == 0 else x)

    df.drop('query_uid', axis=1, inplace=True)

    # should be replaced later - there is some bug, should not be any nan values
    df.fillna(-1, inplace=True)

    print "Grouped stat features are created, time elapsed " + str(time.time()-time_start) + " sec."

    return df


def create_stat_tfidf_grouped_features(df_all, query_uid, columns):

    time_start = time.time()

    print "Creating stat tfidf grouped features, it may take some time"

    df = DataFrame()
    df['id'], df['query_uid'] = df_all['id'], query_uid

    df_temp = DataFrame()
    df_temp['query_uid'] = query_uid

    prefixes, suffices = ('s', 'm', 'a'), ('1', '2')

    # different types of tfidf
    types = ('binary', 'freq', 'log_freq', 'dnorm')

    # copy features from df_all to df_temp
    for c in columns:
        for suffix in suffices:
            for prefix in prefixes:
                for t in types:
                    # do not forget to remove 'set3' suffix once add to create_all_features file
                    name = 'query' + prefix + t + '_tfidf_' + c + suffix
                    df_temp[name] = df_all[name]

    df_grouped = df_temp.groupby('query_uid')

    # max features
    df_max = df_grouped.max()
    df_max.columns = "max" + df_max.columns
    df_max = df_max.reset_index()

    # min features
    df_min = df_grouped.min()
    df_min.columns = "min" + df_min.columns
    df_min = df_min.reset_index()

    # std features
    df_std = df_grouped.std()
    df_std.columns = "std" + df_std.columns
    df_std = df_std.reset_index()

    # add to df
    for df1 in [df_max, df_min, df_std]:
        df = pd.merge(df, df1, how='left', on='query_uid')

    # print some statistic to get a sense whether features could be useful
    print "Number of records " + str(len(df_all))

    # relative to max features
    for c in df_max.columns:
        # skip query_uid features
        if c == 'query_uid':
            continue

        name = 'reltomax' + c[3:]
        df[name] = df_all[c[3:]] / df[c].map(lambda x: -1 if abs(x) < 1.e-4 else x)

    # relative to min features
    for c in df_min.columns:
        # skip query_uid features
        if c == 'query_uid':
            continue

        name = 'reltomin' + c[3:]
        df[name] = df_all[c[3:]] / df[c].map(lambda x: -1 if x == 0 else x)

    df.drop('query_uid', axis=1, inplace=True)

    # should be replaced later - there is some bug, should not be any nan values
    df.fillna(-1, inplace=True)

    print "Grouped stat tfidf features are created, time elapsed " + str(time.time()-time_start) + " sec."

    return df

