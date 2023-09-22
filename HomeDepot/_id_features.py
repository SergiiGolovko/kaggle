import pandas as pd
from pandas import DataFrame


def create_id_features(df_all):

    df = DataFrame()

    df['id'] = df_all['id']

    # create number of query_uid feature
    grouped_df = df_all.groupby('query_uid', as_index=False)['id'].count()
    grouped_df.rename(columns={'id': 'count_query'}, inplace=True)
    df_all = pd.merge(df_all, grouped_df, how='left', on='query_uid')
    df['count_query'] = df_all['count_query']
    print df['count_query'].value_counts()

    # create number of product_uid feature
    grouped_df = df_all.groupby('product_uid', as_index=False)['id'].count()
    grouped_df.rename(columns={'id': 'count_product'}, inplace=True)
    df_all = pd.merge(df_all, grouped_df, how='left', on='product_uid')
    df['count_product'] = df_all['count_product']

    # create number of brand_uid feature
    # 0.create brand uid
    df_brand = DataFrame()
    df_brand['brand'] = df_all['brand'].unique()
    df_brand['brand_uid'] = range(1, len(df_brand)+1)
    df_all = pd.merge(df_all, df_brand, how='left', on='brand')

    # 1.create number of brand_uid feature
    grouped_df = df_all.groupby('brand_uid', as_index=False)['id'].count()
    grouped_df.rename(columns={'id': 'count_brand'}, inplace=True)
    df_all = pd.merge(df_all, grouped_df, how='left', on='brand_uid')
    df['count_brand'] = df_all['count_brand']
    print df['count_brand'].value_counts()

    # create has_(certain attribute) type of features
    attribute_list = ('brand', 'width', 'height', 'depth', 'weight', 'as_height', 'as_width', 'as_depth', 'length', 'material')
    for attribute in attribute_list:
        name = 'has_' + attribute
        df[name] = df_all[attribute].map(lambda x: 1 if x == '__null__' else 0)
        print df[name].value_counts()

    #return df
    # create word in title type of features
    word_list = (' xbi ', '(', ' h ', ' l ', ' r ', ' w ', 'in.', 'gal.', 'cu.', 'oz.', 'cm.', 'mm.', 'deg.', 'volt.',
                 'watt.', 'amp.')

    for word in word_list:
        name = word + '_in_title'
        df[name] = df_all['title'].map(lambda x: x.count(word))
        print df[name].value_counts()

    # create word in query type of features
    for word in word_list:
        name = word + '_in_query'
        df[name] = df_all['query'].map(lambda x: x.count(word))
        df[word +'ratio'] = df[name] / df[word + '_in_title'].map(lambda x: max(x, 1))
        print df[name].value_counts()

    return df