from sklearn.cross_validation import train_test_split, KFold
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn import preprocessing
from xgboost import XGBRegressor
from _estimation import remove_highly_correlated_features, feature_classification, output_results

import numpy as np
import pandas as pd
import logging

from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import sys
# The path to XGBoost wrappers goes here
#sys.path.append('C:\\Users\\Amine\\Documents\\GitHub\\xgboost\\wrapper')
import xgboost as xgb


def root_mean_squared_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5

RMSE = make_scorer(root_mean_squared_error, greater_is_better=False)


def load_train():
    train = pd.read_csv('input/train_clean.csv')
    #train = train[0:1000]
    labels = train.relevance.values
    lbl_enc = preprocessing.LabelEncoder()
    labels = lbl_enc.fit_transform(labels)
    train = train.drop('id', axis=1)
    train = train.drop('relevance', axis=1)
    return train.values, labels


def load_test():
    test = pd.read_csv('input/test_clean.csv')
    test = test.drop('id', axis=1)
    return test.values


def load_train_and_test():

    train = pd.read_csv('input/train_clean.csv')
    #train = train[0:1000]
    labels = train.relevance.values
    #lbl_enc = preprocessing.LabelEncoder()
    #labels = lbl_enc.fit_transform(labels)
    train = train.drop('id', axis=1)
    train = train.drop('relevance', axis=1)

    test = pd.read_csv('input/test_clean.csv')
    ids = test["id"].apply(lambda val: int(val)).values
    ids = np.reshape(ids, (len(ids), 1))
    test = test.drop('id', axis=1)

    feature_dict = feature_classification(test)

    drop_features = ['set3', 'set11', 'set15', 'set16', 'set17', 'set18', 'set19', 'set-1']
    for name in drop_features:
        train.drop(feature_dict[name], axis=1, inplace=True, errors='ignore')
        test.drop(feature_dict[name], axis=1, inplace=True, errors='ignore')

    #train = pd.merge(train, word2vec, how='left', on='id')
    #test = pd.merge(test, word2vec, how='left', on='id')

    #train = train.drop('relevance', axis=1)
    train, test = remove_highly_correlated_features(train, test)

    return train.values, labels, test, ids


def write_submission(preds, output):
    sample = pd.read_csv('input/sampleSubmission.csv')
    preds = pd.DataFrame(
        preds, index=sample.id.values, columns=sample.columns[1:])
    preds.to_csv(output, index_label='id')


def score(params):
    print "Training with params : "
    print params
    num_round = int(params['n_estimators'])
    del params['n_estimators']
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)
    # watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    model = xgb.train(params, dtrain, num_round)
    predictions = model.predict(dvalid) #.reshape((X_test.shape[0], 9))

    # do cross validation for multiple folds here
    # for it in range():
    #     X_train, X_valid = X[foldsIdx[it]['train'], :], X[foldsIdx[it]['valid'], :]
    #     y_train, y_valid = y[foldsIdx[it]['train']], y[foldsIdx[it]['valid']]
    #
    #     for (p, alg) in enumerate(algs):
    #         alg.fit(X_train, y_train)
    #         valid_pred = alg.predict(X_valid)
    #         trainMF[p, foldsIdx[it]['valid']] = valid_pred
    #         testMF_1[p, :] += alg.predict(Xtest)
    #         logging.info("Fold = {}, Result = {}".format(it,
    #                      np.sqrt(mean_squared_error(y_valid, valid_pred))))
    #         print "Fold = {}, Result = {}".format(it,
    #                      np.sqrt(mean_squared_error(y_valid, valid_pred)))


    score = root_mean_squared_error(y_test, predictions)
    print "\tScore {0}\n\n".format(score)
    return {'loss': score, 'status': STATUS_OK}


def optimize(trials):
    # space = {
    #          #'booster' : 'gblinear',
    #          'n_estimators' : hp.quniform('n_estimators', 100, 1000, 1),
    #          'eta' : hp.quniform('eta', 0.025, 0.5, 0.025),
    #          'max_depth' : hp.quniform('max_depth', 1, 13, 1),
    #          'min_child_weight' : hp.quniform('min_child_weight', 1, 6, 1),
    #          'subsample' : hp.quniform('subsample', 0.5, 1, 0.05),
    #          'gamma' : hp.quniform('gamma', 0.5, 1, 0.05),
    #          'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.05),
    #          'num_class' : 9,
    #          'eval_metric' : 'rmse',
    #          'objective': 'reg:linear',
    #          'nthread' : -1,
    #          'silent' : 0
    #          }

    # space for gbtree
    space = {
    #'task': 'regression',
    'booster': 'gbtree',
    'objective': 'reg:linear',
    'eta': hp.quniform('eta', 0.01, 1, 0.01),
    'gamma': hp.quniform('gamma', 0, 2, 0.1),
    'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
    'max_depth': hp.quniform('max_depth', 1, 10, 1),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.1),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1, 0.1),
    'n_estimators': hp.quniform('num_round', 10, 500, 10),
    'eval_metric': 'rmse',
    'nthread': -1,
    'silent': 1,
    'seed': 2016,
    "max_evals": 200
    }

    # regression with linear booster
    # space = {
    # 'task': 'regression',
    # 'booster': 'gblinear',
    # 'objective': 'reg:linear',
    # 'eta' : hp.quniform('eta', 0.01, 1, 0.01),
    # 'lambda' : hp.quniform('lambda', 0, 5, 0.05),
    # 'alpha' : hp.quniform('alpha', 0, 0.5, 0.005),
    # 'lambda_bias' : hp.quniform('lambda_bias', 0, 3, 0.1),
    # 'n_estimators': hp.quniform('num_round', 10, 500, 10),
    # 'eval_metric': 'rmse',
    # 'nthread': -1,
    # 'silent' : 1,
    # 'seed': 2016,
    # "max_evals": 200,
    # }

    best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=250)

    print best


def CreateNFoldsIndecies(X, y, NFolds, rs):
    skf = KFold(y.shape[0], n_folds=NFolds, shuffle=True, random_state=rs)
    idx = []
    for train_index, test_index in skf:
        idx.append({'train':train_index, 'valid':test_index})
    return idx


def train_models(X, y, Xtest, Nfolds, foldsIdx, algs):
    NAlgs = len(algs)
    trainMF = np.zeros((NAlgs, X.shape[0]))
    testMF_1 = np.zeros((NAlgs, Xtest.shape[0]))
    testMF_2 = np.zeros((NAlgs, Xtest.shape[0]))

    for it in range(Nfolds):
        X_train, X_valid = X[foldsIdx[it]['train'], :], X[foldsIdx[it]['valid'], :]
        y_train, y_valid = y[foldsIdx[it]['train']], y[foldsIdx[it]['valid']]

        for (p, alg) in enumerate(algs):
            alg.fit(X_train, y_train)
            valid_pred = alg.predict(X_valid)
            trainMF[p, foldsIdx[it]['valid']] = valid_pred
            testMF_1[p, :] += alg.predict(Xtest)
            logging.info("Fold = {}, Result = {}".format(it,
                         np.sqrt(mean_squared_error(y_valid, valid_pred))))
            print "Fold = {}, Result = {}".format(it,
                         np.sqrt(mean_squared_error(y_valid, valid_pred)))


    testMF_1 /= Nfolds
    for (p, alg) in enumerate(algs):
        alg.fit(X, y)
        testMF_2[p, :] = alg.predict(Xtest)
        logging.info("Alg = {} full training finished".format(p))
        print "Alg = {} full training finished".format(p)

    return trainMF, testMF_1, testMF_2


X, y, Xtest, ids = load_train_and_test()
print "Splitting data into train and valid ...\n\n"
#X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.2, random_state=1234)

#Trials object where the history of search will be stored
#trials = Trials()

#optimize(trials)

algs = [ XGBRegressor(base_score=0.5,
                        colsample_bylevel=1,
                        colsample_bytree=0.2,
                        gamma=0.7,
                        learning_rate=0.03, #eta
                        max_delta_step=0,
                        max_depth=9,
                        min_child_weight=9.0,
                        missing=None,
                        n_estimators=430,
                        nthread=-1,
                        objective='reg:linear',
                        reg_lambda=1,
                        scale_pos_weight=1,
                        seed=2016,
                        silent=True,
                        subsample=0.9) ]

Nfolds, rs = 2, 1234

foldsIdx = CreateNFoldsIndecies(X, y, Nfolds, rs)

trainMF, testMF_1, testMF_2 = train_models(X, y, Xtest, Nfolds, foldsIdx, algs)

output_results(ids, testMF_2.transpose(), file_name='output/opt_xgboost')