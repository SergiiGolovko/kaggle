import time
import re
import unicodedata
import dask.dataframe as dd
from pandas import DataFrame

from nltk.stem.snowball import SnowballStemmer # 0.003 improvement but takes twice as long as PorterStemmer
stemmer = SnowballStemmer('english')

from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

strNum = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}
LEMMATISATION = False


def str_lemma(s):

    s = unicodedata.normalize('NFD', unicode(s)).encode('ascii', 'ignore')
    s = str(s)
    s = s.lower()
    return " ".join([lemmatizer.lemmatize(word) for word in s.lower().split()])

def str_replace(s):

    s = unicodedata.normalize('NFD', unicode(s)).encode('ascii', 'ignore')
    s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) #Split words with a.A
    s = s.lower()

    s = s.replace("  ", " ")
    s = s.replace(",", "")  # could be number / segment later
    s = s.replace("$", " ")
    s = s.replace("?", " ")
    s = s.replace("-", " ")
    s = s.replace("//", "/")
    s = s.replace("..", ".")
    s = s.replace(" / ", " ")
    s = s.replace(" \\ ", " ")
    s = s.replace(".", " . ")
    s = re.sub(r"(^\.|/)", r"", s)
    s = re.sub(r"(\.|/)$", r"", s)
    s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
    s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
    s = s.replace(" x ", " xbi ")
    s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
    s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
    s = s.replace("*", " xbi ")
    s = s.replace(" by ", " xbi ")
    s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
    s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
    s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
    s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
    s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
    s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
    s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
    s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
    s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
    s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
    # s = s.replace("", " degrees ")
    s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
    s = s.replace(" v ", " volts ")
    s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
    s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
    s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
    s = s.replace("  ", " ")
    s = s.replace(" . ", " ")
    s = (" ").join([str(strNum[z]) if z in strNum else z for z in s.split(" ")])
    s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
    s = s.lower()
    s = s.replace("toliet", "toilet")
    s = s.replace("airconditioner", "air conditioner")
    s = s.replace("vinal", "vinyl")
    s = s.replace("vynal", "vinyl")
    s = s.replace("skill", "skil")
    s = s.replace("snowbl", "snow bl")
    s = s.replace("plexigla", "plexi gla")
    s = s.replace("rustoleum", "rust oleum")
    s = s.replace("whirpool", "whirlpool")
    s = s.replace("whirlpoolga", "whirlpool ga")
    s = s.replace("whirlpoolstainless", "whirlpool stainless")

    return s


def str_stem(s):

    s = (" ").join([str(strNum[z]) if z in strNum else z for z in s.split(" ")])
    s = (" ").join([stemmer.stem(z) for z in s.split(" ")])

    return s


def lemmatize_df(df, columns, names):
    """Lemmatize data frame
    :param df: original data frames
    :param columns: columns that need to be lemmatized
    :param names: new names for lemmatized collumns
    :return: data frame with columns "columns" stemmed
    """

    print "Lemmatizing data frame"
    time_start = time.time()
    #dask_df = dd.io.from_pandas(df, npartitions=8)

    for (c, name) in zip(columns, names):
        if LEMMATISATION:
            df[name] = df[c].map(lambda s: str_lemma(s))
        else:
            df[name] = df[c]
    print "Finished lemmatizing data frame, time elapsed " + str(time.time() - time_start) + " sec."
    return df


def replace_df(df, columns, names):

    print "Replacing words in data frame"
    time_start = time.time()
    #df_dask = dd.io.from_pandas(df, npartitions=2)
    for (c, name) in zip(columns, names):
        #df[name] = df_dask[c].map(lambda s: str_stem(s)).compute()
        df[name] = df[c].map(lambda s: str_replace(s))
    print "Finished replacing, time elapsed " + str(time.time() - time_start) +" sec."
    return df


def stem_df(df, columns, names):
    """ Stemmes data frame
    :param df: original data frames
    :param columns: columns that need to be stemmed
    :param names: new names for stemmed collumns
    :return: data frame with columns "columns" stemmed
    """
    print "Stemming data frame"
    time_start = time.time()
    #df_dask = dd.io.from_pandas(df, npartitions=2)
    for (c, name) in zip(columns, names):
        #df[name] = df_dask[c].map(lambda s: str_stem(s)).compute()
        df[name] = df[c].map(lambda s: str_stem(s))
    print "Finished stemming data frame, time elapsed " + str(time.time() - time_start) +" sec."
    return df


def clean_df(df, columns, names):

    print "Cleaning data frame"
    time_start = time.time()
    df = replace_df(df, columns, names)
    df = stem_df(df, columns, names)
    if LEMMATISATION:
        df = lemmatize_df(df, columns, names)
    print "Finished cleaning data frame, time elapsed " + str(time.time() - time_start) + " sec."
    return df



