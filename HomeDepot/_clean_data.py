import sys
import time
import numpy as np
import pandas as pd
from pandas import DataFrame
from _stemming import clean_df, stem_df, replace_df
from _spelling import spell_check


TEST_MODE = False


def read_attributes():

    # read attributes.csv file
    attributes = pd.read_csv('input/attributes.csv', encoding="ISO-8859-1")
    #attributes = attributes[0:1000]

    names = ('Material', 'MFG Brand Name', 'Product Width (in.)', 'Product Height (in.)', 'Product Depth (in.)',
             'Product Weight (lb.)', 'Assembled Height (in.)', 'Assembled Width (in.)', 'Assembled Depth (in.)',
             'Product Length (in.)')
    columns = ('material', 'brand', 'width', 'height', 'depth', 'weight', 'as_height', 'as_width', 'as_depth', 'length')

    df = DataFrame()
    for (name, column) in zip(names, columns):

        df1 = attributes[attributes['name'] == name]
        df1.drop(['name'], axis=1, inplace=True)
        df1.rename(columns={'value': column}, inplace=True)

        print df1.head()
        df1[column] = df1[column].map(lambda x: str(x))
        if column == "material":
            df1 = df1.groupby('product_uid').agg(lambda x: " ".join(x)).reset_index()
        assert len(df1) == len(df1['product_uid'].unique()), "Number of elements in df and product ids are not the same"
        print df1.head()

        if len(df) > 0:
            df = pd.merge(df, df1, how='outer', on='product_uid')
        else:
            df = df1

        print "Number of " + column + " records: " + str(len(df1))
        assert len(df) == len(df['product_uid'].unique()), "Number of elements in df and product ids are not the same"

    assert len(df) == len(df['product_uid'].unique()), "Number of elements in df and product ids are not the same"

    return df


def find_brand(s, brands, startswith=True):

    if startswith:
        for brand in brands:
            if s.startswith(brand):
                return brand
    else:
        for brand in brands:
            if brand in s:
                return brand

    # if brand is not found
    return "__null__"


def extract_brands(df):
    """
    :param df: data frame
    :return: df with brands recovered
    """

    time_start = time.time()
    print "Extracting brands, it may take some time "

    brands = list((df['brand'].unique()))
    brands.sort(key=len, reverse=True)

    nnobrand = len(df[df['brand']=='__null__'])

    print "Number of non-defined brands: " + str(nnobrand)

    # replace unbrand products
    df['brand'] = df.apply(lambda x: 'unbrand' if x['brand']=='__null__' and 'unbrand' in x['title'] else x['brand'], axis=1)
    temp = len(df[df['brand']=='__null__'])
    print "Number of non-defined brands replaced by unbrand: " + str(nnobrand - temp)
    nnobrand = temp

    # try to find a brand
    df['brand'] = df.apply(lambda x: find_brand(x['title'], brands) if x['brand'] == '__null__' else x['brand'], axis=1)
    temp = len(df[df['brand']=='__null__'])
    print "Number of null brands found as the beginning of title: " + str(nnobrand - temp)
    nnobrand = temp

    # second try among brands that not go at the beginning
    brands = list(df.apply(lambda x: "__null__" if x['title'].split()[0] == x['brand'].split()[0] else x['brand'],
                           axis=1).unique())
    brands.sort(key=len, reverse=True)
    df['brand'] = df.apply(lambda x: find_brand(x['title'], brands, startswith=False) if x['brand'] == '__null__' \
                                                                                      else x['brand'], axis=1)
    temp = len(df[df['brand']=='__null__'])
    print "Number of null brands found at the middle of title: " + str(nnobrand - temp)

    print "Brands are extracted, time elapsed " + str(time.time() - time_start) + " sec."

    return df


def clean_data():
    """ Reads data from train and test csv files
    Stemm (and lemmatize) the results
    Returns:
    Data frame with columns:
    id, query, title, brand, description"""

    print "################CLEANING DATA - IT MAY TAKE SOME TIME - ####################"

    time_start = time.time()

    reload(sys)
    sys.setdefaultencoding('utf8')

    # read train and test data sets
    df_train, df_test = pd.read_csv('input/train.csv', encoding="ISO-8859-1"), pd.read_csv('input/test.csv', encoding="ISO-8859-1")

    # concatenate train and test data frames
    df_all = pd.concat([df_train, df_test], ignore_index=True)

    # temporary for testing
    if TEST_MODE:
        df_all = df_all[0:1000]

    # query df
    df_query = DataFrame()
    df_query['search_term'] = df_all['search_term'].unique()
    df_query['query_uid'] = range(1, len(df_query)+1)
    df_query['query'] = df_query['search_term'].map(lambda s: spell_check(s))
    df_replaced = replace_df(df_query, columns=['query'], names=['query'])
    df_cleaned = stem_df(df_replaced, columns=['query'], names=['query'])
    df_query['query'] = df_cleaned['query']
    df_query['not_lemma_query'] = df_replaced['query']

    # title df
    df_title = DataFrame()
    df_title['product_title'] = df_all['product_title'].unique()
    df_replaced = replace_df(df_title, columns=['product_title'], names=['title'])
    df_cleaned = stem_df(df_replaced, columns=['title'], names=['title'])
    df_title['title'] = df_cleaned['title']
    df_title['not_lemma_title'] = df_replaced['title']

    print "Unique search term values: " + str(len(df_query))
    print "Unique title names: " + str(len(df_title))

    # read descriptions csv file
    descriptions = pd.read_csv('input/product_descriptions.csv')

    print "Unique product id values: " + str(len(descriptions))
    print "Unique product description values: " + str(len(descriptions["product_description"].unique()))

    assert len(df_all) == len(df_all["id"].unique()), "Number of elements and unique ids are not the same"

    # temporary for testing
    if TEST_MODE:
        descriptions = descriptions[0:1000]

    # product df
    attributes = read_attributes()
    df_product = pd.merge(descriptions, attributes, how='left', on='product_uid')
    df_product = df_product.rename(columns={'product_description': 'description'})
    df_product.fillna('__null__', inplace=True)
    # temporaly commented line, must be uncommented and two consequent lines commented
    df_cleaned = clean_df(df_product, columns=['description', 'brand', 'material'], names=['description', 'brand', 'material'])
    #df_cleaned['material'] = df_product['material']
    #df_cleaned = clean_df(df_product, columns=['brand'], names=['brand'])
    #df_cleaned['description'] = df_product['description']
    df_product[['description', 'brand', 'material']] = df_cleaned[['description', 'brand', 'material']]

    # merge df_all, df_query and df_product
    df_all = pd.merge(df_all, df_query, how='left', on='search_term')
    assert len(df_all) == len(df_all["id"].unique()), "Number of elements and unique ids are not the same"
    df_all.drop(['search_term'], axis=1, inplace=True)
    df_all = pd.merge(df_all, df_title, how='left', on='product_title')
    assert len(df_all) == len(df_all["id"].unique()), "Number of elements and unique ids are not the same"
    df_all.drop(['product_title'], axis=1, inplace=True)
    df_all = pd.merge(df_all, df_product[['product_uid', 'brand', 'width', 'height', 'depth', 'weight', 'as_height',
                                          'as_width', 'as_depth', 'length', 'material']], how='left', on='product_uid')
    assert len(df_all) == len(df_all["id"].unique()), "Number of elements and unique ids are not the same"
    df_all['brand'].replace('na', 'unbrand', inplace=True)
    df_all = extract_brands(df_all)
    df_all.to_csv("input/all_cleaned.csv", index=False,
                  columns=['id', 'query_uid', 'product_uid', 'not_lemma_query', 'not_lemma_title',
                           'query', 'title', 'brand', 'relevance',
                           'width', 'height', 'depth', 'weight', 'as_height', 'as_width', 'as_depth',
                           'length', 'material'])
    df_product[['product_uid', 'description']].to_csv("input/description_cleaned.csv", index=False)

    # print some statistics
    print "Number of records: " + str(len(df_all))
    print "Number of records without brand: " + str(len(df_all[df_all['brand'] == "__null__"]))
    print "Number of unbranded records: " + str(len(df_all[df_all['brand'] == 'unbrand']))
    brand_title = df_all.apply(lambda x: 1 if x['title'].split()[0] == x['brand'].split()[0] else 0, axis=1)
    print "Number of records with title starting with brand name: " + \
          str(sum(brand_title))
    print "10 first records with title starting not with brand name: "
    print df_all[['brand', 'product_uid']][(brand_title == 0) & (df_all['brand'] != 'unbrand')].head(10)
    print "Value counts for length of brand: "
    print df_all['brand'].map(lambda x: len(x.split())).value_counts()
    print "Number of unique brands: " + str(len(df_all['brand'].unique()))
    print "Number of first words in brands: " + str(len(df_all['brand'].map(lambda x: x.split()[0]).unique()))
    ##print "Number of first two words in brands: " + str(len(df_all['brand']).map(lambda x: x.split()[0] + " " + x.split(1)))


    print "Time elapsed " + str(time.time() - time_start) + " sec."
    print "################CLEANING DATA - END - ####################"

if __name__ =='__main__':
    clean_data()