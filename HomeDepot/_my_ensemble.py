from __future__ import division
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.stats import rankdata

types = {'AVERAGE', 'GEOM_AVERAGE', 'RANK_AVERAGE'}
def my_ensemble(files, columns, type='AVERAGE', weights=None):

    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)

    if not weights:
        weights = np.ones(len(dfs)) / len(dfs)

    if type=='RANK_AVERAGE':
        for df in dfs:
            df[columns] = rankdata(df[columns]) / len(df)

    # initialize sum_df with first data frame
    sum_df = dfs[0]
    if (type=='AVERAGE') or (type=='RANK_AVERAGE'):
        sum_df[columns] *= weights[0]
    if type=='GEOM_AVERAGE':
        print weights[0]
        sum_df[columns] = np.power(sum_df[columns], weights[0])

    dfs.remove(dfs[0])
    weights = weights[1:]

    # weighted sum of all data frames
    for (df, weight) in zip(dfs, weights):
        if (type=='AVERAGE') or (type=='RANK_AVERAGE'):
            sum_df[columns] += weight * df[columns]
        else:
            print weight
            sum_df[columns] *= np.power(df[columns], weight)

    # output to ensemble.csv file
    sum_df.to_csv('output/ensemble.csv', index=False)


def correlation_statistic(files, col_name):

    all_df = DataFrame()
    ind = 0
    for file in files:
        df = pd.read_csv(file)
        all_df[ind] = df[col_name]
        ind +=1

    df_corr = all_df.corr()

    print df_corr
    return df_corr


def normalize(file_name, col_name):

    df = DataFrame()
    df = pd.read_csv(file_name)

    max_val, min_val = df[col_name].max(), df[col_name].min()
    df[col_name] = (df[col_name] - min_val) / (max_val - min_val)

    df.to_csv('output/normalized.csv', index=False)

if __name__ =='__main__':
    files = ['output/output3.csv', 'output/output4.csv', 'output/output5.csv'] #, 'output/output3.csv', 'output/output4.csv']
    files = ['output/Combo_3 2.csv', 'output/Combo_2.csv', 'output/Combo_2nd_level_2.csv', 'output/Combo_2nd_level_43 2.csv',
             'output/Combo_2nd_level_43_adjusted.csv', 'output/Combo_2nd_level.csv', 'output/Combo_4_big_2.csv', 'output/Combo_4_big_3.csv',
             'output/Combo_4_big.csv', 'output/ensemble.csv']
    #files = ['output/Combo.csv', 'output/Combo_4.csv', 'output/Combo_with_stacking_UPD.csv', 'output/combo_xgboost 2.csv',
    #         'output/opt_xgboost.csv', 'output/stacking_2level_big.csv']
    #files = ['output/Combo.csv',
    #         'output/opt_xgboost.csv']

    # weights = [0.7, 0.3]

    files = ['output/xgb_classifier.csv', 'output/opt_xgboost.csv']

    files = ['output/Combo_2nd_level_final_1.csv', 'output/Combo_2nd_level_43 2.csv']
    #files = ['output/Combo_3 2.csv', 'output/Combo_2nd_level_43_adjusted.csv']
    #files = ['output/ensemble.csv', 'output/ensemble1.csv']
    files = ['output/santander1.csv',
             'output/santander2.csv',
             'output/santander3.csv',
             'output/santander45.csv']

    files = ['output/santander1.csv',
             'output/santander1.csv']

    files = ['output/XGBClassifier1.csv',
             'output/XGBClassifier2.csv']

    files = ['output/santander9.csv',
             'output/santander10.csv']

    files = ['output/santander_839000.csv',
             'output/santander_839587.csv',
             'output/santander_840212.csv',
             'output/santander_840204.csv',
             'output/santander_840921.csv',
             'output/simple_average.csv']

    files = ['output/santander_840921.csv',
             'output/our_ensemble.csv',
             'output/submission.csv']

    weights = [0.33, 0.34, 0.33]

    columns = ['TARGET']
    my_ensemble(files, columns, weights=weights, type='RANK_AVERAGE') #, weights=weights)
    correlation_statistic(files, 'TARGET')