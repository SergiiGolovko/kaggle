#import pandas as pd
from pandas import DataFrame
from _count import count_whole_words

def create_attribute_features(df_all):

    #df = DataFrame()
    #df['id'] = df_all['id']
    #df_all['material'] = df_all['material'].map(lambda x: str(x))
    #return create_query_features(df_all, columns=['material'], qcol='query')

    df = DataFrame()
    df['id'] = df_all['id']

    columns = ['material'] #, 'width', 'height', 'depth', 'weight', 'as_height', 'as_width', 'as_depth', 'length')
    # create c in query type of features
    for c in columns:
        df_all[c] = df_all[c].map(lambda x: str(x))
        name = c + '_in_query'
        df[name] = df_all.apply(lambda x: count_whole_words(x[c], x['query']), axis=1)
        print df[name].value_counts()

    # create c in title type of features
    for c in columns:
        df_all[c] = df_all[c].map(lambda x: str(x))
        name = c + '_in_title'
        df[name] = df_all.apply(lambda x: count_whole_words(x[c], x['title']), axis=1)
        print df[name].value_counts()

    return df

    #return df
