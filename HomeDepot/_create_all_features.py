import pandas as pd
import sys
import time
from _tfidf_features import *
from _query_features import *
from _position_features import *
from _additional_features import create_additional_features
from _id_features import create_id_features
from _attribute_features import create_attribute_features
from _grouped_features import create_grouped_features, create_stat_grouped_features, create_stat_tfidf_grouped_features
from _tfidf_svd import create_cosine_similarity_features, create_ctfidf_svd_features, create_tfidf_svd_features
from _substitute_features import create_substituted_features
from _count_single_words import *
from _archiving_features import create_archiving_features
from _word2vec_features import create_word2vec_features

from _ngram import *

# groups of features
groups = dict()

TEST_MODE = False
ADD_NEW_FEATURES_MODE = False


def count_digits(s):
    s = str(s)
    return len([l for l in s if l.isdigit()])


def create_all_features(df_all):

    # create data frame with a single column "id"
    df = DataFrame()
    df['id'] = df_all['id']

    # create query features
    df1 = create_query_features(df_all, ['title', 'description', 'brand'], qcol='query')
    df1['unbranded'] = 1 * ((df_all['brand'] == 'unbrand') | (df_all['brand'] == '__null__'))
    df = add_new_features(df, df1, group='set1')
    assert len(df) == len(df["id"].unique()), "Length of df and number of ids are not the same"

    # create position features
    df2 = create_position_features(df_all, ['title', 'description', 'brand'], qcol='query')
    df = add_new_features(df, df2, group='set2')
    assert len(df) == len(df["id"].unique()), "Length of df and number of ids are not the same"

    # create tfidf features
    df3 = create_tfidf_features(df_all, ['title', 'description'], qcol='query')
    df = add_new_features(df, df3, group='set3')
    assert len(df) == len(df["id"].unique()), "Length of df and number of ids are not the same"

    # create tfidf unique features
    df_all['utitle'], df_all['udescription'] = df_all['title'], df_all['description']
    df4 = create_tfidf_features(df_all, ['utitle', 'udescription'], qcol='query', unique=True)
    df = add_new_features(df, df4, group='set4')
    assert len(df) == len(df["id"].unique()), "Length of df and number of ids are not the same"

    # create id features
    df5 = create_id_features(df_all)
    df = add_new_features(df, df5, group='set5')
    assert len(df) == len(df["id"].unique()), "Length of df and number of ids are not the same"

    # create bigram query features
    columns = ['query', 'title', 'description', 'brand']
    prefix = 'bi_'
    for col in columns:
       df_all[prefix+col] = df_all[col].apply(lambda s: getBigram(s))

    df6 = create_query_features(df_all[df['len_of_query'] > 1], ['bi_title'], qcol='bi_query')
    df = add_new_features(df, df6, group='set6')
    df.fillna(-1, inplace=True)
    assert len(df) == len(df["id"].unique()), "Length of df and number of ids are not the same"

    # create count digits features
    df7 = DataFrame()
    df7['id'] = df_all['id']
    columns = ['query', 'title', 'description', 'brand']
    for col in columns:
        df7['digits_in_' + col] = df_all[col].map(lambda s: count_digits(s))
    df = add_new_features(df, df7, group='set7')
    assert len(df) == len(df["id"].unique()), "Length of df and number of ids are not the same"

    # create additional features
    df8 = create_additional_features(df_all)
    df = add_new_features(df, df8, group='set8')
    assert len(df) == len(df["id"].unique()), "Length of df and number of ids are not the same"

    # create grouped features
    df9 = create_grouped_features(df_all)
    df = add_new_features(df, df9, group='set9')
    assert len(df) == len(df["id"].unique()), "Length of df and number of ids are not the same"

    # create attribute features
    df10 = create_attribute_features(df_all)
    df = add_new_features(df, df10, group='set10')
    assert len(df) == len(df["id"].unique()), "Length of df and number of ids are not the same"

    # create tfidf svd features, with common vocabulary
    df11 = create_ctfidf_svd_features(df_all)
    df = add_new_features(df, df11, group='set11')
    assert len(df) == len(df["id"].unique()), "Length of df and number of ids are not the same"

    # create tfidf svd features, using different vocabulary for query and title
    df12 = create_tfidf_svd_features(df_all, columns=['query', 'title'])
    df = add_new_features(df, df12, group='set12')
    assert len(df) == len(df["id"].unique()), "Length of df and number of ids are not the same"

    # create substituted features
    df13 = create_substituted_features(df_all)
    df = add_new_features(df, df13, group='set13')
    assert len(df) == len(df["id"].unique()), "Length of df and number of ids are not the same"

    # create query stat features
    query_uid = df_all['query_uid']
    df14 = create_stat_grouped_features(df, query_uid)
    df = add_new_features(df, df14, group='set14')
    assert len(df) == len(df["id"].unique()), "Length of df and number of ids are not the same"

    # create stat tfidf features
    df15 = create_stat_tfidf_grouped_features(df, query_uid, columns=['title', 'description'])
    df = add_new_features(df, df15, group='set15')
    assert len(df) == len(df["id"].unique()), "Length of df and number of ids are not the same"

    # create most common color material features
    df16 = create_most_common_color_material_features(df_all)
    df = add_new_features(df, df16, group='set16')
    assert len(df) == len(df['id'].unique()), "Length of df and number of ids are not the same"

    # create n most common unit features
    df17 = create_most_common_unit_features(df_all)
    df = add_new_features(df, df17, group='set17')
    assert len(df) == len(df['id'].unique()), "Length of df and number of ids are not the same"

    # create n most common word features
    df18 = create_n_most_common_word_features(df_all)
    df = add_new_features(df, df18, group='set18')
    assert len(df) == len(df['id'].unique()), "Length of df and number of ids are not the same"

    # create archiving features
    df19 = create_archiving_features(df_all)
    df = add_new_features(df, df19, group='set19')
    assert len(df) == len(df['id'].unique()), "Length of df and number of ids are not the same"

    # add relevance
    df['relevance'] = df_all['relevance']
    assert len(df) == len(df["id"].unique()), "Length of df and number of ids are not the same"

    # rename features
    df = rename_features(df)

    return df


def add_new_features(df_old, df_new, group="undefined"):

    columns_old, columns_new = set(df_old.columns), set(df_new.columns)

    # drop columns that are in both df_new and df_old from new data frame
    columns_drop = list(columns_new.intersection(columns_old.difference({'id'})))
    df_new.drop(columns_drop, axis=1, inplace=True)

    # merge new and old data frames
    df = pd.merge(df_old, df_new, how='left', on='id')

    # adding new features to a dictionary
    groups[group] = list(set(df_new.columns).difference({'id'}))

    return df


def rename_features(df):
    """ Rename columns in data frame by adding suffices
    :param df: data frame
    :return: data frame with renamed columns
    """
    for group in groups:
        # adding suffix group to all features
        columns = dict(zip(groups[group], [s + group for s in groups[group]]))
        df = df.rename(columns=columns)

    return df


def remove_new_features(df_old, df_new):

    columns_old, columns_new = set(df_old.columns), set(df_new.columns)

    # drop columns that are in both df_new and df_old
    columns_drop = list(columns_new.intersection(columns_old.difference({'id', 'relevance'})))
    df = df_old.drop(columns_drop, axis=1)

    return df


if __name__ == '__main__':

    time_start = time.time()

    print "################CREATING FEATURES - IT MAY TAKE SOME TIME - ####################"

    reload(sys)
    sys.setdefaultencoding('utf8')

    # read all cleaned
    df_all = pd.read_csv('input/all_cleaned.csv', encoding="ISO-8859-1")

    # read descriptions
    descriptions = pd.read_csv('input/description_cleaned.csv', encoding="ISO-8859-1")

    # merge all data together
    df_all = pd.merge(df_all, descriptions, how='left', on='product_uid')

    assert len(df_all) == len(df_all["id"].unique()), "Length of df and number of ids are not the same"

    if TEST_MODE:
        df_all = df_all[0:1000]

    assert len(df_all) == len(df_all["id"].unique()), "Length of df and number of ids are not the same"

    # create all features
    if ADD_NEW_FEATURES_MODE:

        # df = create_substituted_features(df_all)
        query_uid = df_all['query_uid']

        #df = create_word2vec_features(df_all)

        # reading previously saved train and test sets
        df_train, df_test = pd.read_csv('input/train_clean.csv'), pd.read_csv('input/test_clean.csv')

        # rewriting df_all, not going to use previous df_all anymore
        df_all = pd.concat([df_train, df_test], ignore_index=True)

        # merge new features with old features
        #df = add_new_features(df_all, df)
        #df = df_all
    else:
        df = create_all_features(df_all)

    assert len(df) == len(df["id"].unique()), "Length of df and number of ids are not the same"

    # split df data frame back into training and testing set
    df_train = df[~df['relevance'].isnull()]
    df_test = df[df['relevance'].isnull()]

    print "Length of df train after create_all features: " + str(len(df_train))
    print "Length of df test after create_all features: " + str(len(df_test))
    #print df_test.head()

    df_train.to_csv('input/train_clean.csv', index=False)
    df_test.to_csv('input/test_clean.csv',  index=False)

    print "Time elapsed " + str(time.time() - time_start) + " sec."

    print "################FEATURES ARE CREATED####################"
