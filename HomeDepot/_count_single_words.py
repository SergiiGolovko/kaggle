from __future__ import division
from collections import Counter
from pandas import DataFrame
from nltk.corpus import stopwords
from _stemming import str_stem
from _query_features import create_query_features
import time


def create_vocabulary(corpus, stop_words=[]):
    #global vocabulary

    vocabulary = Counter()
    for s in corpus:
        vocabulary.update(set(s.split()))

    for word in stop_words:
        if word in vocabulary:
            del vocabulary[word]

    del_words = []
    add_words = dict()
    for word in vocabulary:
        if len(word) < 3:
            del_words.append(word)
        if word.endswith(")"):
            add_words[word[:-1]] = vocabulary[word]
            del_words.append(word)

    for word in del_words:
        del vocabulary[word]

    for word in add_words:
        vocabulary[word] = add_words[word]

    return vocabulary


def create_count_single_words_features(df_all, word_list):
    """ Creates
    :param df:
    :param columns:
    :param l:
    :return:
    """

    df = DataFrame()
    df['id'] = df_all['id']

    for word in word_list:
        name = word + '_in_title'
        df[name] = df_all['title'].map(lambda x: x.count(word))
        print "word " + word + " counts in title "
        print df[name].value_counts()

    # create word in query type of features
    for word in word_list:
        name = word + '_in_query'
        df[name] = df_all['query'].map(lambda x: x.count(word))
        print "word " + word + " counts in query "
        df[word +'ratio'] = df[name] / df[word + '_in_title'].map(lambda x: max(x, 1))
        print df[name].value_counts()

    return df


def create_n_most_common_word_features(df_all, n=25):

    time_start = time.time()

    print "Creating n most common word features, it may take some time"

    corpus = list(df_all['query'].unique()) + list(df_all['title'].unique())

    stop_words = stopwords.words('english') + ["xbi", "12in.", '14in.', '34in.', '6in.', '18in.', '4in.']
    vocabulary = create_vocabulary(corpus, stop_words=stop_words)
    most_common_words = vocabulary.most_common(n)

    print 'Most common words in joint vocabulary for title and query'
    print most_common_words

    word_list = [word[0] for word in most_common_words]

    df = create_count_single_words_features(df_all, word_list)
    df['id'] = df_all['id']

    print "n most common word features are created, time elapsed " + str(time.time()-time_start) + " sec."

    return df


def create_most_common_unit_features(df_all):

    time_start = time.time()

    print "Creating most common units features, it may take some time"

    units = ['in.', 'lb.', 'watts', 'volts', 'ft.', 'hours', 'amp.', 'min', 'psi', 'mm.', 'oz.', 'gal.', 'sq.ft.',
             'cu.ft.']

    df = create_count_single_words_features(df_all, units)
    df['id'] = df_all['id']

    print "Most common unit features are created, time elapsed " + str(time.time()-time_start) + " sec."

    return df


def create_most_common_color_material_features(df_all):

    time_start = time.time()

    print "Creating most common color material features, it may take some time"

    color_materials = ['white', 'black', 'brown', 'gray', 'chrome', 'stainless', 'steel', 'red', 'brown', 'silve',
                       'blue', 'nickel', 'metal', 'clear']

    df = create_count_single_words_features(df_all, color_materials)
    df['id'] = df_all['id']

    print "Most common color material features are created, time elapsed " + str(time.time()-time_start) + " sec."

    return df


def create_valuable_words_features(df_all, n=25):

    time_start = time.time()

    print "Creating most valuable words features, it may take some time"

    corpus = list(df_all['query'].unique()) + list(df_all['title'].unique())

    stop_words = stopwords.words('english') + ["xbi", "12in.", '14in.', '34in.', '6in.', '18in.', '4in.']
    vocabulary = create_vocabulary(corpus, stop_words=stop_words)
    most_common_words = vocabulary.most_common(n)

    most_common_words = [word[0] for word in most_common_words]

    valuable_words = ['tv','downrod', 'sillcock', 'shelving', 'luminaire', 'paracord', 'ducting', \
    'recyclamat', 'rebar', 'spackling', 'hoodie', 'placemat', 'innoculant', 'protectant', \
    'colorant', 'penetrant', 'attractant', 'bibb', 'nosing', 'subflooring', 'torchiere', 'thhn',\
    'lantern','epoxy','cloth','trim','adhesive','light','lights','saw','pad','polish','nose','stove',\
    'elbow','elbows','lamp','door','doors','pipe','bulb','wood','woods','wire','sink','hose','tile','bath','table','duct',\
    'windows','mesh','rug','rugs','shower','showers','wheels','fan','lock','rod','mirror','cabinet','shelves','paint',\
    'plier','pliers','set','screw','lever','bathtub','vacuum','nut', 'nipple','straw','saddle','pouch','underlayment',\
    'shade','top', 'bulb', 'bulbs', 'paint', 'oven', 'ranges', 'sharpie', 'shed', 'faucet',\
    'finish','microwave', 'can', 'nozzle', 'grabber', 'tub', 'angles','showerhead', 'dehumidifier', \
    'shelving', 'urinal', 'mdf']

    # stemming valuable words
    valuable_words = [str_stem(s) for s in valuable_words]
    valuable_words += most_common_words
    valuable_words = list(set(valuable_words))

    columns = ['query', 'title']

    df_temp = DataFrame()
    df_temp['id'] = df_all['id']
    for c in columns:
        df_temp['v' + c] = df_all[c].map(lambda x: " ".join(word for word in valuable_words if word in x))

    df = create_query_features(df_temp, columns=['vtitle'], qcol='vquery')

    print "Most valuable word features are created, time elapsed " + str(time.time()-time_start) + " sec."

    return df




