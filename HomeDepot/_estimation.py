# impdas, DataFrame
from __future__ import division
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.decomposition import TruncatedSVD

# import numpy and matplotlib
import numpy as np
import matplotlib.pyplot as plt

# import sklearn
from sklearn.cross_validation import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, make_scorer
from xgboost import XGBClassifier, XGBRegressor
import xgboost as xgb

# import time
import time

CURR_FIGURE = 1
TEST_MODE = False

def root_mean_squared_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5

RMSE = make_scorer(root_mean_squared_error, greater_is_better=False)


def get_irrelevant_features(estimator, columns):

    zero_coef = estimator.coef_ < 1.e-9
    print columns[zero_coef]
    return columns[zero_coef]


def tune_parameters(estimator, name, parameters_grid, train_X, train_y):
    """Returns the best set of parameters out of ones specified in parameters_grid

    Parameters
    ----------
    estimator : estimator for which parameters are tuned
    name : name of estimator
    parameters_grid : dictionary or list of dictionaries with parameters to be tuned
    train_X : data frame with features
    train_y : data frame with labels

    Return
    ------
    best_parameter : dictionary of best parameters
    best_score : float
    """

    print "Tuning parameters for " + name

    start_time = time.time()

    cv = KFold(n=len(train_y), n_folds=2, shuffle=True, random_state=1234)

    gscv = GridSearchCV(estimator,
                        parameters_grid,
                        cv=cv,
                        scoring=RMSE,
                        verbose=100)
    gscv.fit(train_X, train_y)

    print "The mean score and all cv scores are"
    for params, mean_score, cv_scores in gscv.grid_scores_:
        print("%0.4f %s for %r" % (mean_score, np.array_str(cv_scores, precision=4), params))

    elapsed_time = time.time() - start_time

    print "The best score is %0.4f and the best parameters are %r" %(gscv.best_score_, gscv.best_params_)
    print "Finished tuning parameters, time elapsed " + str(elapsed_time) + "sec."

    return [gscv.best_params_, gscv.best_score_]


def plot_gb_performance(gb, train_X, train_y):
    """Plots the performance of gradient boosting as a function of number of trees

    Parameters
    ----------
    gb : gradient boosting estimator
    train_X : data frame with features
    train_y : data frame with labels

    Returns
    -------
    The plot of gradient boosting performance
    """

    global CURR_FIGURE

    print "Plotting gradient boosting performance"

    # split training set into training set and validation set
    train_X, val_X, train_y, val_y = train_test_split(train_X, 
                                                      train_y, 
                                                      test_size=0.5, 
                                                      random_state=1)
                                                      
    gb.fit(train_X, train_y)
    
    
    # this part of the code is adapted from scikit learn documentation
    # http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html
    # compute test set mse
    n_estimators = gb.get_params()['n_estimators']
    test_score = np.zeros((n_estimators,), dtype=np.float64)

    for i, y_pred in enumerate(gb.staged_predict(val_X)):
        test_score[i] = root_mean_squared_error(y_true=val_y, y_pred=y_pred)

    plt.figure(CURR_FIGURE)
    plt.title('Gradient Boosting Performance')
    plt.plot(np.arange(n_estimators) + 1, gb.train_score_/len(train_X), 'b-', label='Training Set')
    plt.plot(np.arange(n_estimators) + 1, test_score, 'r-', label='Test Set')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Score')
    CURR_FIGURE += 1


def plot_rf_performance(rf, train_X, train_y, at_least=10):
    """Plots the performance of random forest as a function of number of trees

    Parameters
    ----------
    rf : random forest estimator
    train_X : data frame with features
    train_y : data frame with labels
    at_least : float, plot the performance starting with at least at_least trees

    Return
    ------
    Plot of random forest performance
    """

    global CURR_FIGURE

    print "Plotting random forest performance"

    #split training set into training set and validation set
    train_X, val_X, train_y, val_y = train_test_split(train_X, 
                                                      train_y, 
                                                      test_size=0.5, 
                                                      random_state=0)
    
    rf.fit(train_X, train_y)

    # this part of the code is adapted from scikit learn documentation
    # http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html
    # compute test set log loss score
    n_estimators = rf.get_params()['n_estimators']

    # empty arrays for train and test scores
    # n_classes = rf.n_classes_
    test_score = np.zeros((n_estimators,), dtype=np.float64)
    train_score = np.zeros((n_estimators,), dtype=np.float64)
    
    for scores, y, X in zip([train_score, test_score], [train_y, val_y], [train_X, val_X]):

        # aggregate pred y for all trees up to current tree
        aggr_y = np.zeros(y.shape[0], dtype=np.float64)

        for i, tree in enumerate(rf.estimators_):
        
            pred_y = tree.predict(X)

            # update aggregate y
            aggr_y = (aggr_y * i + pred_y)/(i+1)
            scores[i] = root_mean_squared_error(y_true=y, y_pred=aggr_y)

    plt.figure(CURR_FIGURE)
    plt.title('Random Forest Performance')
    plt.plot(np.arange(n_estimators - at_least) + at_least + 1, train_score[at_least:], 'b-', label='Training Set')
    plt.plot(np.arange(n_estimators - at_least) + at_least + 1, test_score[at_least:], 'r-', label='Test Set')
    plt.legend(loc='upper right')
    plt.xlabel('Iterations')
    plt.ylabel('Score')
    CURR_FIGURE += 1

    print "The end of plotting"


def my_cross_validation(estimator, X, y, n_folds=10):
    """Does cross validation

    Parameters
    ----------
    estimator : estimator for which cross validation is dome
    X : data frame with features
    y : data frame with labels
    n_folds : float, number of folds

    Return
    ------
    np array of length n_folds
    """

    # do K-fold validation manually
    kf = KFold(n=X.shape[0], n_folds=n_folds, shuffle=True, random_state=1)
    # scores
    scores = []
    
    for train_ind, test_ind in kf:
        
        # fit the model on training set
        iX, iy = X.values[train_ind], y[train_ind]
        estimator.fit(iX, iy)
        
        # make a prediction for test set
        iX, iy = X.values[test_ind], y[test_ind]
        pred_y = estimator.predict(iX)
        
        # calculate the score
        score = root_mean_squared_error(y_true=iy, y_pred=pred_y)
        scores.append(score)
        
    return np.array(scores)


def output_results(ids, predictions, file_name='output/output'):
    """ Writes results to file

    Parameters
    ---------
    ids : data frame of ids
    predictions : data frame of predictions
    file_name : file name

    Return
    ------
    File saved in csv format with name file_name
    first column - ids, second columns - predictions"""
    predictions[predictions > 3.0] = 3.0
    predictions[predictions < 1.0] = 1.0
    predictions = np.reshape(predictions, (len(predictions), 1))
    output_df = DataFrame(np.concatenate((ids, predictions), axis=1), columns=['id', 'relevance'])
    output_df['id'] = output_df['id'].astype(int)
    output_df.to_csv(file_name + '.csv', index=False)


def plot_feature_importance(estimator, columns, n=50):
    """ Plots feature importance

    Parameters
    ----------
    estimator : estimator for which the important features needs to be drawn
    columns : names of all features
    n : int, number of important features to plot

    Return
    ------
    Plot of features importance
    """

    global CURR_FIGURE

    # check whether n is larger than number of columns
    n = min(n, len(columns))

    # extract feature importance and normalize them to sum up to 100
    feature_importance = estimator.feature_importances_
    feature_importance = (100.0 * feature_importance) / sum(feature_importance)
    index = np.argsort(feature_importance)[::-1][0:n]

    # feature names
    feature_names = columns
    
    # plot
    plt.figure(CURR_FIGURE)
    pos = (np.arange(n) + .5)[::-1]
    plt.barh(pos, feature_importance[index], align='center')
    plt.yticks(pos, feature_names[index])
    plt.xlabel('Relative Importance')
    plt.title(str(n) + ' Most Important Features')
    CURR_FIGURE += 1


def apply_truncated_svd(x_train, x_test):

    n_components = 517
    df = pd.concat([x_train, x_test], ignore_index=True)

    svd = TruncatedSVD(n_components=n_components, n_iter=15)
    X = svd.fit_transform(df.values)

    svd_columns = ['svd' + str(i) for i in range(n_components)]

    df1 = DataFrame(X, columns=svd_columns)
    x_train = df1.head(len(x_train))
    x_test = df1.tail(len(x_test))

    return x_train, x_test


THRESHOLD = 0.99
def remove_highly_correlated_features(x_train, x_test):

    df = pd.concat([x_train, x_test], ignore_index=True)

    print "Number of features prior to removing highly correlated features " + str(df.shape[1])

    df_corr = df.corr()
    indexes, columns, values = df_corr.index, df_corr.columns, df_corr.values
    drop_columns = []

    for i in range(values.shape[0]):
        for j in range(i+1, values.shape[1]):
            if values[i, j] > THRESHOLD:
                print indexes[i], columns[j], values[i, j]
                drop_columns.append(columns[j])

    drop_columns = list(set(drop_columns))
    #columns = set(df.columns).difference(drop_columns)

    x_train.drop(drop_columns, axis=1, inplace=True, errors='ignore')
    x_test.drop(drop_columns, axis=1, inplace=True, errors='ignore')

    print "Number of features after removing highly correlated features " + str(x_train.shape[1])

    #return x_train[list(columns)], x_test[list(columns)]
    return x_train, x_test


def feature_classification(df_all):

    feature_dict = dict()
    columns = df_all.columns

    names = ('tfidf_features', 'title_features', 'query_features', 'brand_features', 'has_features', 'ratio_features',
             'description_features', 'tfidf_svd_features', 'title_svd_features', 'query_svd_features', 'gtitle_features',
             'ctitle_svd_features', 'cquery_svd_features')
    parts = ('_tfidf', 'title', 'query', 'brand', 'has_', 'ratio', 'description', 'tfidf_svd', 'tfidf_svd_t',
             'tfidf_svd_q', 'gtitle','ctfidf_svd_t', 'ctfidf_svd_q')

    for (name, part) in zip(names, parts):
        feature_dict[name] = [feature for feature in columns if part in feature]
        print "There are " + str(len(feature_dict[name])) + " " + name

    # split tfidf features into unique and not unique
    feature_dict['utfidf_features'] = [feature for feature in feature_dict['tfidf_features'] if ('utitle' in feature) or
                                       ('udescription' in feature)]
    feature_dict['tfidf_features'] = list(set(feature_dict['tfidf_features']).difference(set(feature_dict['utfidf_features'])))

    # drop gtitle features
    feature_dict['tfidf_features'] = [feature for feature in feature_dict['tfidf_features'] if not ('gtitle' in feature)]

    feature_dict['title_svd_features'] = list(set(feature_dict['title_svd_features']).difference(set(feature_dict['ctitle_svd_features'])))
    feature_dict['query_svd_features'] = list(set(feature_dict['query_svd_features']).difference(set(feature_dict['cquery_svd_features'])))

    print "There are " + str(len(feature_dict['utfidf_features'])) + " " + 'unique tfidf features'
    print "There are " + str(len(feature_dict['tfidf_features'])) + " " + 'nonunique tfidf features'

    names = ['set' + str(i+1) for i in range(19)]
    features = []
    for name in names:
        feature_dict[name] = [feature for feature in columns if feature.endswith(name)]
        features += feature_dict[name]
        print "There are " + str(len(feature_dict[name])) + " " + name + " features"

    feature_dict['set-1'] = list(set(columns).difference(features + ["id"]))

    return feature_dict


def add_predicted_relavance(test_x, train_x, train_y, columns):

    # Ridge
    estimator = Ridge(random_state = 0)
    grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    name = 'Ridge'

    param, best_score = tune_parameters(estimator, name, grid, train_x[columns], train_y)

    start_time = time.time()

    print "Fitting %s model" % name

    estimator = estimator.set_params(**param)
    estimator.fit(train_x[columns], train_y)

    train_x['predicted_relevance'] = estimator.predict(train_x[columns])
    test_x['predicted_relevance'] = estimator.predict(test_x[columns])

    print "Model is fitted, time elapsed " + str(time.time() - start_time) + " sec. "

    return test_x, train_x


def estimation():
    
    print "################ESTIMATION - IT MAY TAKE SOME TIME - ####################"
    
    # read all files
    train_df = pd.read_csv('input/train_clean.csv')
    test_df = pd.read_csv('input/test_clean.csv')
    #word2vec = pd.read_csv('input/word2vec_features.csv')
    #word2vec = pd.read_csv('input/tfidf_dist_features.csv')
    #word2vec = pd.read_csv('input/word2vec_similarity_features.csv')
    #query_features = pd.read_csv('input/query_features.csv')

    # temporary
    if TEST_MODE:
        train_df = train_df[:1000]
        test_df = test_df[:1000]

    # split both train and test sets into features and labels
    train_y = train_df.relevance
    train_df.drop(["relevance"], axis=1, inplace=True)
    train_x = train_df
    ids = test_df["id"].apply(lambda val: int(val)).values
    ids = np.reshape(ids, (len(ids), 1))
    # previous_important = test_df["previous_important"]
    test_df.drop(["relevance"], axis=1, inplace=True)
    test_x = test_df

    # temporary under testing
    feature_dict = feature_classification(test_x)

    # create predicted feature
    #test_x, train_x = add_predicted_relavance(test_x,
    #                                          train_x,
    #                                          train_y,
    #                                          columns=feature_dict['title_svd_features'] + feature_dict['query_svd_features'])

    # drop tfidf features
    #drop_features = ['tfidf_features', 'title_svd_features', 'query_svd_features',
    #                 'ctitle_svd_features', 'cquery_svd_features']

    #drop_features = ['tfidf_features', 'title_svd_features', 'query_svd_features']
    drop_features = ['set3', 'set11', 'set15', 'set16', 'set17', 'set18', 'set19', 'set-1']
    for name in drop_features:
        train_x.drop(feature_dict[name], axis=1, inplace=True, errors='ignore')
        test_x.drop(feature_dict[name], axis=1, inplace=True, errors='ignore')

    #train_x = pd.merge(train_x, word2vec, how='left', on='id')
    #test_x = pd.merge(test_x, word2vec, how='left', on='id')

    #train_x, test_x = remove_highly_correlated_features(train_x, test_x)
    #train_x, test_x = apply_truncated_svd(train_x, test_x)

    #train_x = pd.merge(train_x, word2vec, how='left', on='id')
    #test_x = pd.merge(test_x, word2vec, how='left', on='id')

    #train_x.drop(["id", "brand_uid"], axis=1, inplace=True, errors='ignore')
    #test_x.drop(["id", "brand_uid"], axis=1, inplace=True, errors='ignore')
    print "Number of features after adding word 2 vec features " + str(train_x.shape[1])

    #temporaly under testing
    #train_x, test_x = apply_truncated_svd(train_x, test_x)

    # 0. extreme trees
    etr = ExtraTreesRegressor(n_estimators=200, n_jobs=-1, max_features='auto', random_state=0)
    etr_parameter_grid = {'max_features':[0.1, 0.2, 0.3]}
    #plot_rf_performance(etr, train_x, train_y)
    #plt.show()

    # 0. gradient boosting - testing to see how well we could do with it
    #xgb_reg = XGBRegressor()
    #xgb_reg_parameter_grid = {'max_depth': [6, 7, 8], 'n_estimators': [100, 150, 200, 250]} #[50, 100, 200]}
    xgb_reg = XGBRegressor(base_score=0.5,
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
                           subsample=0.9)
    xgb_reg_parameter_grid = {'max_depth': [9], 'n_estimators': [430]}

    # 1. random forest
    rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, max_features='auto', random_state=0)
    rf_parameter_grid = {'max_features': ['auto']}# [0.1, 0.2, 0.3, 0.4, 'auto']} #['auto', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
    #plot_rf_performance(rf, train_x, train_y)

    #plt.show()

    # 2. gradient boosting
    gb = GradientBoostingRegressor(n_estimators=150,
                                   subsample=0.5,
                                   max_depth=5,
                                   learning_rate=0.1,
                                   random_state=0)

    gb_parameter_grid = {'max_depth': [5]} #[3, 5]}
    #plot_gb_performance(gb, train_x, train_y)

    # plt.show()
    # 3.xgboost classifier
    xgb_clf = XGBClassifier(base_score=0.5,
                            colsample_bylevel=1,
                            colsample_bytree=0.2,
                            gamma=0.7,
                            learning_rate=0.03,
                            max_delta_step=0,
                            max_depth=9,
                            min_child_weight=9.0,
                            missing=None,
                            n_estimators=430,
                            nthread=-1,
                            objective='multi:softprob',
                            reg_alpha=0,
                            reg_lambda=1,
                            scale_pos_weight=1,
                            seed=2016,
                            silent=True,
                            subsample=0.9)

    xgb_parameter_grid = {'max_depth': [2], 'n_estimators': [50]}

    # Ridge
    rdg = Ridge(random_state=0)
    rdg_parameter_grid = {'alpha': [0.01, 0.1, 1, 10, 100, 1000]}

    # Lasso
    lasso = Lasso(random_state = 0)
    lasso_parameter_grid = {'alpha': [0.01, 0.1, 1, 10, 100, 1000]}

    # 4. estimate rf and gb and do cross validation
    estimators, names = (xgb_clf, xgb_reg, rdg, lasso, gb, rf), \
                        ("XGBClassifier", "XGBoost", "Ridge", "Lasso", "Gradient Boosting", "Random Forest")

    parameters_grid = (xgb_parameter_grid, xgb_reg_parameter_grid, rdg_parameter_grid, lasso_parameter_grid,
                       gb_parameter_grid, rf_parameter_grid)

    file_names = ("output/test", "output/xgb_output", "output/rgd_output", "output/lasso_output",
                  "output/gb_output", "output/rf_output")

    for estimator, name, grid, file in zip(estimators, names, parameters_grid, file_names):

        #param, best_score = tune_parameters(estimator, name, grid, train_x, train_y)

        start_time = time.time()

        print "Fitting %s model" % name

        # initialize y_pred
        y_pred = []

        if name == "XGBClassifier":

            print "XGBClassifier"

            # do K-fold validation manually
            kf = KFold(n=len(train_x), n_folds=2, shuffle=True, random_state=1)
            # scores
            scores = []

            ind = 0
            for train_ind, test_ind in kf:

                # fit the model on training set
                iX, iy = train_x.values[train_ind], train_y[train_ind]
                estimator.fit(iX, iy)

                # make predictions for validation set
                iX, iy = train_x.values[test_ind], train_y[test_ind]
                predict_proba = estimator.predict_proba(iX)
                classes = np.array(estimator.classes_)
                pred_y = np.dot(predict_proba, classes)

                # calculate the score
                score = root_mean_squared_error(y_true=iy, y_pred=pred_y)
                scores.append(score)

                # print results
                print "Fold = {}, Result = {}".format(ind, score)
                ind += 1

            mean_score = sum(scores) / len(scores)
            print "The mean score and all cv scores are"
            for cv_scores in scores:
                print("%0.4f %s" % (mean_score, np.array_str(cv_scores, precision=4)))

            # fit the model
            estimator.fit(train_x.values, train_y)
            predict_proba = estimator.predict_proba(test_x.values)
            classes = np.array(estimator.classes_)
            y_pred = np.dot(predict_proba, classes)

            print "XGBClassifier ends"
        else:

            param, best_score = tune_parameters(estimator, name, grid, train_x, train_y)

            estimator = estimator.set_params(**param)
            estimator.fit(train_x, train_y)

            y_pred = estimator.predict(test_x)
            y_pred = np.reshape(y_pred, (len(y_pred), 1))

        # output results
        output_results(ids, y_pred, file_name=file)

        elapsed_time = time.time() - start_time
        print "The model was successfully estimated, time elapsed " + str(elapsed_time) + "sec."

    # 5. plotting feature importance for gradient boosting
    plot_feature_importance(gb, train_x.columns)
    # plot_feature_importance(rf, train_x.columns)

    # 6. show all plots
    plt.show()


    print "################ESTIMATION - END - ######################################"

if __name__ =='__main__':
    estimation()