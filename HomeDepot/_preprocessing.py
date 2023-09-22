from __future__ import division
from __future__ import unicode_literals

import nltk
from _stemming import *
from _spelling import *
from _create_all_features import *

import sys
import time
import operator


STEMMING = False
SPELL_CHECK = True
NUMBER_OF_NULLS = 0
NUMBER_OF_SPELL_CHECKS = 0
TEST_MODE = False
CREATE_DICT = False

stopwords = set(nltk.corpus.stopwords.words('english'))

synonyms_dict = {}

split_dict = {}

split_words = set()

one_letter_words = {}

except_words = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'x')


def print_info(df):

    train_product_ids = set(df['product_uid'][~df['relevance'].isnull()])
    test_product_ids = set(df['product_uid'][df['relevance'].isnull()])
    common_product_ids = train_product_ids.intersection(test_product_ids)

    print "number of product ids in train set %d" % len(train_product_ids)
    print "number of product ids in test set %d" % len(test_product_ids)
    print "number of product ids in intersection %d" % len(common_product_ids)


def preprocessing():

    global NUMBER_OF_NULLS
    global one_letter_words
    global synonyms_dict, split_dict

    print "################PREPROCESSING - IT MAY TAKE SOME TIME - ####################"

    reload(sys)
    sys.setdefaultencoding('utf8')

    # read train and test data sets
    df_train, df_test = pd.read_csv('input/train.csv', encoding="ISO-8859-1"), pd.read_csv('input/test.csv', encoding="ISO-8859-1")

    # temporary for testing
    if TEST_MODE:
        df_train = df_train[0:1000]
        df_test = df_test[0:1000]

    # concatenate train and test data frames
    df_all = pd.concat([df_train, df_test], ignore_index=True)

    if STEMMING:
        descriptions = pd.read_csv('input/product_descriptions.csv')

    # print some information
    print df_train.info()
    print df_test.info()
    if STEMMING:
        print descriptions.info()

    print_info(df_all)

    if SPELL_CHECK:

        print "Checking spelling and stemming, it may take some time"

        time_start = time.time()

        df_all['search_term_checked'] = df_all['search_term'].map(lambda s: spell_check(s))
        ind = (df_all['search_term'] != df_all['search_term_checked'])

        print "In total %d search queries are misspelled" % int(sum(ind))

        #print df_all[ind].head()

        df_all = lemmatize_df(df_all, ['search_term_checked'], ['search_term_checked'])
        df_all = stem_df(df_all, ['search_term_checked'], ['search_term_checked'])

        df_all['search_term_checked'].to_csv('input/search_term_checked.csv', index=False)

        print "Spelling check and stemming finished, time elapsed " + str(time.time()-time_start) + " sec."
    else:
        df_all['search_term_checked'] = pd.read_csv('input/search_term_checked.csv', header=None)

    df_all['search_term'] = df_all['search_term_checked']
    df_all['search_term'] = df_all['search_term'].map(lambda x: str(x))

    #print df_all['search_term'].head()

    # stemming
    if STEMMING:
        print "Stemming, it may take some time"
        time_start = time.time()

        if TEST_MODE:
            descriptions = descriptions[0:1000]

        descriptions = lemmatize_df(descriptions, ['product_description'], ['description'])
        print descriptions.head(2)

        df_all = lemmatize_df(df_all, ['product_title', 'search_term'], ['title', 'search_term'])

        descriptions = stem_df(descriptions, ['description'], ['description'])
        print descriptions.head(2)

        df_all = stem_df(df_all, ['title', 'search_term'], ['title', 'search_term'])

        # output stemmed files
        descriptions.to_csv('input/description_stemmed.csv', index=False)

        df_all['title'].to_csv('input/title_stemmed.csv', index=False)
        df_all['search_term'].to_csv('input/search_term_stemmed.csv', index=False)

        print "Stemming finished, time elapsed " + str(time.time()-time_start) + " sec."
    else:
        # read stemmed files
        descriptions = pd.read_csv('input/description_stemmed.csv')
        df_all['title'] = pd.read_csv('input/title_stemmed.csv', header=None)
        # df_all['search_term'] = pd.read_csv('input/search_term_stemmed.csv', header=None)

        df_all['title'] = df_all['title'].map(lambda x: str(x))
        # df_all['search_term'] = df_all['search_term'].map(lambda x: str(x))

    if TEST_MODE:
        descriptions = descriptions[0:1000]

    # read attributes.csv file
    attributes = pd.read_csv('input/attributes.csv', encoding="ISO-8859-1")
    attributes = attributes[attributes['name'] == "MFG Brand Name"]
    attributes.drop(['name'], axis=1, inplace=True)
    attributes.rename(columns={'value': 'brand'}, inplace=True)
    attributes = lemmatize_df(attributes, ['brand'], ['brand'])
    attributes = stem_df(attributes, ['brand'], ['brand'])

    # print df_all.head(2)
    # merge all_df and descriptions data frames
    df_all = pd.merge(df_all, descriptions, how='left', on='product_uid')
    # print df_all.head(2)
    # merge all df and attributes
    df_all = pd.merge(df_all, attributes, how='left', on='product_uid')
    # print "There are %s entries with non-specified brand" % str(df_all['brand'].isnull().sum())
    # print df_all.head(2)

    df_all.loc[df_all['brand'].isnull(), 'brand'] = 'null'
    df_all['unbranded'] = 1*((df_all['brand'] == 'null') | (df_all['brand']=='unbranded'))

    # remove stopwords
    # df_all = remove_stopwords_df(df_all, columns=['search_term'])

    #if SPELL_CHECK:
    #    df_all['search_term'] = df_all['search_term_checked']

    # remove stopwords
    # df_all = remove_stopwords_df(df_all, columns=['search_term'])

    # remove brand from title
    # df_all = remove_brand_from_title(df_all)

    if CREATE_DICT:
        df_all['term_title_description'] = df_all['search_term'] + "/t" + df_all['title'] + " " + df_all['description']
        df_all['term_title_description'].apply(lambda s: create_dicts(s.split('/t')[0], s.split('/t')[1]))
        df_all.drop(['term_title_description'], axis=1, inplace=True)

        print len(synonyms_dict)
        print len(split_dict)

        df_split = DataFrame(zip(split_dict.keys(), split_dict.values()), columns=["word", "split"])
        df_split.to_csv("input/split_dict.csv", index=False)
    #else: # commented for now, does not seem to be an improvement
        # df_split = pd.read_csv("input/split_dict.csv")
        # split_dict = df_split.set_index("word")["split"].to_dict()
        # df_all['description'] = df_all['description'].apply(lambda s: replace_split_words(s))
        # df_all['title'] = df_all['title'].apply(lambda s: replace_split_words(s))

    df = create_all_features(df_all)

    # outliers with high scores
    ind = (df['search_term_in_title1'] == 0) & (df['search_term_in_title2']==0) & (df_all['relevance'] > 2.99)
    print "number of potential outliers with high score is " + str(sum(ind))
    #print df_all[ind].head(10)
    outliers = DataFrame(df_all[ind][['id', 'title', 'search_term']].values,
                         columns=['id', 'title', 'search_term'])
    outliers.to_csv("input/outliers3.csv", index=False)

    # outliers with low scores
    ind = (df['search_term_in_title1'] > 0) & (df['search_term_in_title2'] > 0) & (df_all['relevance'] < 1.01)
    print "number of potential outliers with low score is " + str(sum(ind))
    #print df_all[ind].head(10)
    outliers = DataFrame(df_all[ind][['id', 'title', 'search_term']].values,
                         columns=['id', 'title', 'search_term'])
    outliers.to_csv("input/outliers1.csv", index=False)


    # add id and relevance columns to data frame df
    df['relevance'] = df_all['relevance']

    # split df data frame back into training and testing set
    df_train = df[~df['relevance'].isnull()]
    df_test = df[df['relevance'].isnull()]

    #print df_test.head()

    df_train.to_csv('input/train_clean.csv', index=False)
    df_test.to_csv('input/test_clean.csv',  index=False)

    print "Number of NULL entries during stemming"
    print NUMBER_OF_NULLS
    #one_letter_words = sorted(one_letter_words.items(), key=operator.itemgetter(1), reverse=True)
    #print one_letter_words
    print "################PREPROCESSING - END - ######################################"    #read all files


if __name__ =='__main__':
    preprocessing()

