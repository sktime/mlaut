from mlaut.shared.static_variables import GRIDSEARCH_NUM_CV_FOLDS, GRIDSEARCH_CV_NUM_PARALLEL_JOBS
from mlaut.highlevel.strategies import TabClassifStrategy, TabRegrStrategy
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
import numpy as np

LinearRegressonStrategy = TabRegrStrategy(estimator=linear_model.LinearRegression(), name='LinearRegression')

alphas = [0.1, 1, 10.0]
max_n_alphas=1000
c_param = np.logspace(-3,3,7)
penalty=['l1','l2']
n_iter =[100, 200, 300, 400, 500]

ridge = linear_model.RidgeCV(
    alphas=alphas,
    cv=GRIDSEARCH_NUM_CV_FOLDS)

RidgeRegressionStrategy = TabRegrStrategy(estimator=ridge, name='RidgeRegression')

lasso = linear_model.LassoCV(
    alphas=alphas,
    cv=GRIDSEARCH_NUM_CV_FOLDS,
    n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS
)

LassoStrategy = TabRegrStrategy(estimator=lasso, name='Lasso')

lasso_lars = linear_model.LarsCV(
    max_n_alphas=max_n_alphas,
    cv=GRIDSEARCH_NUM_CV_FOLDS,
    n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS
)

LassoLarsStrategy = TabRegrStrategy(estimator=lasso_lars, name='LassoLars')

logistic_regression = GridSearchCV(
    estimator=linear_model.LogisticRegression(),
    param_grid={
        'C':c_param,
        'penalty':penalty,
    },
    n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
    cv=GRIDSEARCH_NUM_CV_FOLDS
)

LogisticRegressionStrategy = TabRegrStrategy(estimator=logistic_regression, name='LogisticRegression')

bayesian_ridge = GridSearchCV(
    estimator=linear_model.BayesianRidge(),
    param_grid={
        'n_iter':n_iter #todo expand param grid
    },
    n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
    cv=GRIDSEARCH_NUM_CV_FOLDS)

BayesianRidgeStrategy = TabRegrStrategy(estimator=logistic_regression, name='BayesianRidge')

passive_aggressive =  GridSearchCV(
    estimator = linear_model.PassiveAggressiveClassifier(),
    param_grid={
        'C': c_param,
        'max_iter': n_iter
    },
    n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
    cv=GRIDSEARCH_NUM_CV_FOLDS)

PassiveAggressiveClassifierStrategy=TabClassifStrategy(estimator=passive_aggressive, name='PassiveAggressiveClassifier')