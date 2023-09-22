from _query_features import create_query_features
from _tfidf_features import create_tfidf_features
from _digit_features import create_digit_features
import pandas as pd
from pandas import DataFrame
import time
import re

connectors = {' for ', ' with '}

replacements = {
' ac ': ' air condition ',
' satellietedigit ': ' satellit digit ',
' tv ': ' televis ',
' televisionsatellit ': ' televis satellit',
' ciel ': ' ceil '
#' bathroom ': ' bath room ',
#' bedroom ': ' bed room ',
#' fiberglass ': ' fiber glass '
}

stopwords = ['xbi', 'in.', 'ft.', 'oz.', 'gal.', 'mm.', 'cm.', 'deg.', 'volt.', 'watt.', 'amp.', 'lb.', 'deg.',
             ' and ', ' in ', ' that ', ' l ', ' r ', ".", " & ", " h ", " w ", " a ", " d "]

TEST_MODE = False

# def remove_digits(s):

#    c

def replace_str(s):

    s = " " + s + " "
    for r in replacements:
        s = s.replace(r, replacements[r])

    if not s.split():
        s = "_null_"

    return s


def clean_digits_stopwords(s):

    s = " " + s + " "

    # temporary under testing
    s = re.sub("\d+", "", s)

    for word in stopwords:
        s = s.replace(word, " ")

    if not s.split():
        s = "_null_"

    return " ".join(s.split())


def clean_str(s):

    for c in connectors:
        ind = s.find(c)
        if ind > -1:
            s = s[0:ind]

    if not s.split():
        s = "_null_"

    return s


def output_outliers(df):

    # outliers with high scores
    ind = (df['cl_search_term_in_cl_title2']==0) & (df['relevance'] > 2.99)
    print "number of potential outliers with high score is " + str(sum(ind))

    outliers = DataFrame(df[ind][['id', 'title', 'search_term']].values,
                         columns=['id', 'title', 'search_term'])
    outliers.to_csv("input/outliers13.csv", index=False)

    # outliers with low scores
    ind = (df['cl_search_term_in_cl_title2'] > 0) & (df['relevance'] < 1.01)
    print "number of potential outliers with low score is " + str(sum(ind))
    #print df_all[ind].head(10)
    outliers = DataFrame(df[ind][['id', 'title', 'search_term']].values,
                         columns=['id', 'title', 'search_term'])
    outliers.to_csv("input/outliers11.csv", index=False)


def clean_digits_stopwords_df(df_all, columns, new_columns=None,  file='input/temp.csv'):

    time_start = time.time()
    print "Cleaning data frame, it may take some time"

    df = DataFrame()
    df["id"] = df_all["id"]

    if not new_columns:
        new_columns = ["cl_"+c for c in columns]

    for (c, new_c) in zip(columns, new_columns):
        df[new_c] = df_all[c].map(lambda x: clean_digits_stopwords(x))

    # output results into csv file
    df.to_csv(file, index=False)

    print "Data frame is cleaned, time elapsed " + str(time.time() - time_start) + " sec. "
    return df


def clean_df(df_all, columns, file='input/temp.csv'):

    time_start = time.time()
    print "Cleaning data frame, it may take some time"

    df = DataFrame()
    df["id"] = df_all["id"]

    for c in columns:
        df["cl_" + c] = df_all[c].map(lambda x: clean_str(x))
        #df["cl_" + c] = df["cl_" + c].map(lambda x: replace_str(x))

    # output results into csv file
    df.to_csv(file, index=False)

    print "Data frame is cleaned, time elapsed " + str(time.time() - time_start) + " sec. "
    return df


def create_additional_features(df_all):

    # clean search term and title columns
    df_clean = clean_df(df_all, ["query", "title"])

    # add description and brand
    #df_clean["description"] = df_all["description"]
    # df_clean["brand"] = df_all["brand"]

    # create tfidf features
    #df4 = create_tfidf_features(df_clean, columns=['cl_title'], qcol='cl_query')

    # create digit features
    df1 = create_digit_features(df_clean, columns=['cl_title'], qcol='cl_query')


    # add description and brand
    df_clean["description"] = df_all["description"]

    # clean digits and stopwods - currently under testing
    df_clean = clean_digits_stopwords_df(df_clean, columns=['cl_title', 'cl_query', 'description'],
                                         new_columns=['cl_title', 'cl_query', 'description'])

    df_clean['f_query'] = df_clean['cl_query'].map(lambda x: x.split()[0])
    df_clean['l_query'] = df_clean['cl_query'].map(lambda x: x.split()[-1])

    # create query features
    df = create_query_features(df_clean, columns=['cl_title', 'description'], qcol='cl_query')
    df2 = create_query_features(df_clean, columns=['cl_title'], qcol='f_query')
    df3 = create_query_features(df_clean, columns=['cl_title'], qcol='l_query')

    # null search
    df['query_null'] = 1*(df_clean['cl_query'] == "_null_")

    # value counts
    print "relevance among null search"
    print df_all[df['query_null']==1]['relevance'].value_counts()
    print "entries with score 3.00"
    print df_all[(df['query_null']==1) & (df_all['relevance'] > 2.99)]['query']
    print "entries with score 2.67"
    print df_all[(df['query_null']==1) & (df_all['relevance'] > 2.66) & (df_all['relevance'] < 2.68)]['query']
    print "entries with score 2.33"
    print df_all[(df['query_null']==1) & (df_all['relevance'] > 2.32) & (df_all['relevance'] < 2.34)]['query']

    # merge with df
    df = pd.merge(df, df1, how='left', on='id')
    df = pd.merge(df, df2, how='left', on='id')
    df = pd.merge(df, df3, how='left', on='id')

    return df


def load_description():

    # read train and test files
    df_train = pd.read_csv('input/train.csv')
    df_test = pd.read_csv('input/test.csv')

    # concatenate train and test sets together
    df_all = pd.concat([df_train, df_test], ignore_index=True)

    df = DataFrame()
    df["id"] = df_all["id"]
    df["product_uid"] = df_all["product_uid"]

    # download descriptions
    description = pd.read_csv('input/description_stemmed.csv')

    # merge with df_all data frame
    df = pd.merge(df, description[["product_uid", "description"]], how='left', on='product_uid')

    return df


def add_additional_features():

    # read train and test files
    df_train = pd.read_csv('input/train_clean.csv')
    df_test = pd.read_csv('input/test_clean.csv')

    # concatenate train and test sets together
    df_all = pd.concat([df_train, df_test], ignore_index=True)

    if TEST_MODE:
        df_all = df_all[0:1000]

    # read title and search term
    df_all['title'] = pd.read_csv('input/title_stemmed.csv', header=None)
    df_all['search_term'] = pd.read_csv('input/search_term_checked.csv', header=None)
    #df_all['description'] = pd.read_csv('input/description_stemmed.csv', header=None)

    # read descriptions
    descriptions = load_description()
    df_all = pd.merge(df_all, descriptions, how='left', on='id')

    # create additional features
    df = create_additional_features(df_all)

    # merge with df_all data frame
    df_all = pd.merge(df_all, df, how='left', on='id')

    # output outliers
    output_outliers(df_all)

    # drop title and search term
    df_all.drop(["title", "search_term", "product_uid", "description"], axis=1, inplace=True)

    # output

    # split df data frame back into training and testing set
    df_train = df_all[~df_all['relevance'].isnull()]
    df_test = df_all[df_all['relevance'].isnull()]

    df_train.to_csv('input/train_clean1.csv', index=False)
    df_test.to_csv('input/test_clean1.csv',  index=False)

if __name__ == "__main__":
    add_additional_features()