from __future__ import division
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from pandas import DataFrame


def average_relevance(df, n_average=1000):

    relevance = df['relevance']
    i_min, i_max = n_average, len(relevance)

    rel_df = DataFrame()
    rel_df['id'] = df['id']
    arr = np.zeros(len(relevance))

    for i in range(i_min, i_max):

        rel = relevance[i-n_average:i].mean()
        arr[i] = rel

    rel_df['relevance'] = arr

    return rel_df


def ideal_data_split(df):

    relevance = df['relevance']
    i_min, i_max = 1000, len(relevance) - 1000
    # rel_mean = relevance.mean()
    # i_middle = len(relevance) / 2
    split, id, max_dist = 1000, 1000, 0

    split_df = DataFrame()
    split_df['id'] = df['id']
    #split_df['dist'] = 0
    dists = np.zeros(len(split_df))

    for i in range(i_min, i_max):

        #if i < i_middle:
        #    dist = relevance[0:i].mean() - rel_mean
        #else:
        #    dist = relevance[i:].mean() - rel_mean

        dist = relevance[:i].mean() - relevance[i:].mean()
        dists[i] = dist
        #split_df['dist'][i] = dist
        #split_df['record'][i] = i

        # if dist > max_dist:
        #    split, max_dist = i, dist
        #    print "Better split is found, record %s, id %s, dist %s" %(str(split), str(id), str(dist))

    split_df['dist'] = dists
    record = split_df['dist'].argmax()
    id = split_df['id'][record]
    max_dist = split_df['dist'][record]

    return split_df, record, id, max_dist


def adjust_test_relevance(train, test, id_start, id_end):

    train1 = train[(train['id'] >= id_start) & (train['id'] <= id_end)]
    test1 = test[(test['id'] >= id_start) & (test['id'] <= id_end)]

    train_mean = train1['relevance'].mean()
    test_mean = test1['relevance'].mean()

    print "Average relevance in train set prior to adjustment is " + str(train_mean)
    print "Average relevance in test set prior to adjustment is " + str(test_mean)

    # adjustment formula is f(x) = a * x + b, where a and b chosen in such way to
    # 1) f(1) = 1
    # 2) f(test_mean) = train_mean

    a = (train_mean-1)/(test_mean-1)
    b = 1 - a

    print "Adjustment formula is f(x) = a * x +  b, with a = %s and b = %s " %(str(a), str(b))

    test['relevance'] = test.apply(lambda x: a * x['relevance'] + b if ((x['id'] >= id_start) and (x['id'] <= id_end)) else x['relevance'],
                                   axis=1)

    test1 = test[(test['id'] >= id_start) & (test['id'] <= id_end)]
    test_new_mean = test1['relevance'].mean()

    print "Average relevance in test set after adjustment is " + str(test_new_mean)
    print "In total %s records adjusted out of total %s" %(str(len(test1)), str(len(test)))

    score = 0.44750
    lscore = score - abs(test_mean - train_mean)
    hscore = score + abs(test_mean - train_mean)

    # calculate possible gains for train set
    n_adjusted, n_others = len(test1) * 0.3, len(test) * 0.3
    n_total = n_adjusted + n_others
    lscore = math.sqrt(1/n_total * (n_adjusted * lscore * lscore + n_others * score * score))
    hscore = math.sqrt(1/n_total * (n_adjusted * hscore * hscore + n_others * score * score))

    print "Based on adjustment the new score should be in region (%s, %s) on public LB " %(str(round(lscore, 4)),
                                                                                           str(round(hscore, 4)))

    lscore = score - abs(test_mean - train_mean)
    hscore = score + abs(test_mean - train_mean)

    # calculate possible gains for train set
    n_adjusted, n_others = len(test1) * 0.7, len(test) * 0.7
    n_total = n_adjusted + n_others
    lscore = math.sqrt(1/n_total * (n_adjusted * lscore * lscore + n_others * score * score))
    hscore = math.sqrt(1/n_total * (n_adjusted * hscore * hscore + n_others * score * score))

    print "Based on adjustment the new score should be in region (%s, %s) on private LB " %(str(round(lscore, 4)),
                                                                                           str(round(hscore, 4)))

    return test


if __name__ == "__main__":

    # train set and test sets
    train = pd.read_csv('input/train_clean.csv')
    test  = pd.read_csv('output/Combo_2nd_level_43.csv')
    #relevance = train.relevance

    # calc average relevance
    av_relevance_train, av_relevance_test = average_relevance(train), average_relevance(test)

    plt.figure(1)
    plt.title('Average Relevance')
    plt.plot(av_relevance_train['id'].values, av_relevance_train['relevance'].values, 'b-', label='Train Set')
    plt.plot(av_relevance_test['id'].values, av_relevance_test['relevance'].values, 'r-', label='Test Set (Combo3_csv)')
    plt.legend(loc='upper right')
    plt.xlabel('Id')
    plt.ylabel('Relevance')
    #plt.show()

    split_train, record, id, dist = ideal_data_split(train[['id', 'relevance']])
    print "Ideal split for train set is found, record %s, id %s, dist %s" %(str(record), str(id), str(dist))

    # id_start and id_end for adjustment
    id_start, id_end = id, train['id'][len(train)-1]

    split_test, record, id, dist = ideal_data_split(test[['id', 'relevance']])
    print "Ideal split for test set (Combo_3.csv) is found, record %s, id %s, dist %s" %(str(record), str(id), str(dist))

    # plot distances
    plt.figure(2)
    plt.title('Split Mean Distance')
    plt.plot(split_train['id'].values, split_train['dist'].values, 'b-', label='Train Set')
    plt.plot(split_test['id'].values, split_test['dist'].values, 'r-', label='Test Set (Combo3_csv)')
    plt.legend(loc='upper right')
    plt.xlabel('Id')
    plt.ylabel('Distance')
    #plt.show()

    # adjust test set relevance
    test = adjust_test_relevance(train, test, id_start, id_end)
    print "Average relevance in test set after adjustment is " + str(test['relevance'].mean())

    # re calc average relevance for test set
    av_relevance_test = average_relevance(test)

    plt.figure(3)
    plt.title('Average Relevance')
    plt.plot(av_relevance_train['id'].values, av_relevance_train['relevance'].values, 'b-', label='Train Set')
    plt.plot(av_relevance_test['id'].values, av_relevance_test['relevance'].values, 'r-', label='Test Set (Combo3_csv)')
    plt.legend(loc='upper right')
    plt.xlabel('Id')
    plt.ylabel('Relevance')
    plt.show()

    test.to_csv('output/Combo_2nd_level_43_adjusted.csv', index=False)

    # plt.scatter(split_df['id'].values, split_df['dist'].values, alpha=0.5)
    # plt.xlabel("Id")
    # plt.ylabel("Distance")
    # plt.title("Train Set Id Relevance")
    # plt.show()

    # submission






