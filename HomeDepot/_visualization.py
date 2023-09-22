import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


THRESHOLD = 0.95


def highly_correlated_features(df):

    df_corr = df.corr()
    indexes, columns, values = df_corr.index, df_corr.columns, df_corr.values

    for i in range(values.shape[0]):
        for j in range(i+1, values.shape[1]):
            if values[i, j] > THRESHOLD:
                print indexes[i], columns[j], values[i, j]


def print_some_info(df):

    print "Length of Search Term Value Counts:"
    print df["len_of_search_term"].value_counts()

    print "Length of Clean Search Term Value Counts:"
    print df["len_of_cl_search_term"].value_counts()

    print "Relevance, Value Counts:"
    print df["relevance"].value_counts()

    print df["id"][df["search_termratio_in_title2"]>1]


def visualize_some_features(df):

    import matplotlib.pyplot as plt

    columns = ["cl_search_termratio_in_cl_title1", "cl_search_termfratio_in_cl_title1"]
    #columns = ["search_termsbinary_tfidf_title1", "search_termsbinary_tfidf_title2",
    #           "search_termsbinary_tfidf_description1", "search_termsbinary_tfidf_description2"]
    #columns = ["bi_search_termratio_in_bi_title1", "bi_search_termfratio_in_bi_title1"]
    #columns = ["prefix_possearch_term_in_title1", "prefix_possearch_term_in_title2"]
    ind = 1
    # colors = np.random.rand(len(df))
    # area = np.pi * (15 * np.random.rand(len(df)))**2  # 0 to 15 point radiuses

    for col in columns:
        plt.figure(ind)
        x = df[col].values
        y = df["relevance"].values

        plt.scatter(x, y, alpha=0.5)
        plt.xlabel("Ratio in Title")
        plt.ylabel("Relevance")

        ind += 1

    plt.show()


def visualize():

    # read all files
    df_train = pd.read_csv('input/train_clean1.csv')
    df_test = pd.read_csv('input/test_clean1.csv')

    # concatenate train and test data frames
    df_all = pd.concat([df_train, df_test], ignore_index=True)
    #df_all.drop(["id", "relevance"], axis=1, inplace=True)

    visualize_some_features(df_train)

    print_some_info(df_all)

    highly_correlated_features(df_all)


if __name__ == "__main__":

    print "################VISUALIZATION - IT MAY TAKE SOME TIME - ####################"

    start_time = time.time()
    visualize()

    print "Visualization ended, time elapsed " + str(time.time()-start_time) + "sec."

