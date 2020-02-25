
from mlaut.shared.static_variables import(GRIDSEARCH_NUM_CV_FOLDS,
                                      GRIDSEARCH_CV_NUM_PARALLEL_JOBS)

from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
import numpy as np

from mlaut.estimators.base import BaseClassifier
from mlaut.estimators.base  import BaseRegressor


class Linear_Regression(BaseRegressor):
    """
    Wrapper for `sklearn Linear Regression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>`_.
    """

    def __init__(self,
                regressor=linear_model.LinearRegression,
                n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS):
        self.regressor = regressor
        self.arguments = {'n_jobs':n_jobs}

class Ridge_Regression(BaseRegressor):
    """
    Wrapper for `sklearn Ridge Regression <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html>`_.
    """
    def __init__(self,
                regressor=linear_model.RidgeCV,
                alphas= {'alphas':[0.1, 1, 10.0]},
                n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
                cv=GRIDSEARCH_NUM_CV_FOLDS):
        self.regressor=regressor
        self.arguments = {
            'alphas':alphas,
            'n_jobs': n_jobs,
            'cv': cv
        }
        
class Lasso(BaseRegressor):
    
    def __init__(self, 
                 regressor=linear_model.LassoCV,
                 alphas=[0.1, 1, 10.0], 
                 cv=GRIDSEARCH_NUM_CV_FOLDS, 
                 n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS):
        self.regressor = regressor
        self.arguments = {'alphas':alphas,
                          'cv':cv,
                          'n_jobs':n_jobs}
        
class Lasso_Lars(BaseRegressor):
    """
    Wrapper for `sklearn Lasso Lars <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html>`_.
    """
    def __init__(self, 
                 regressor=linear_model.LassoLarsCV,
                 max_n_alphas=1000, 
                 cv=GRIDSEARCH_NUM_CV_FOLDS, 
                 n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS):
        self.regressor = regressor
        self.arguments = {
            'max_n_alphas':max_n_alphas,
            'cv': cv,
            'n_jobs': n_jobs
        }

class Logistic_Regression(BaseRegressor):
    """
    Wrapper for `sklearn Logistic Regression <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_.
    """
    def __init__(self, 
                 regressor=linear_model.LassoLarsCV,
                 max_n_alphas=1000, 
                 cv=GRIDSEARCH_NUM_CV_FOLDS, 
                 n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS):
        self.regressor = regressor
        self.arguments = {
            'max_n_alphas': max_n_alphas,
            'cv':cv,
            'n_jobs': n_jobs
        }
 
class Bayesian_Ridge(BaseRegressor):
    """
    Wrapper for `sklearn Bayesian Ridge Regression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html>`_.
    """
    def __init__(self,
                regressor=linear_model.BayesianRidge):
        self.regressor = regressor
        self.arguments = {}

class Passive_Aggressive_Classifier(BaseClassifier):
    """
    Wrapper for `sklearn Passive Aggressive Classifier <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html>`_.
    """
    def __init__(self,
                classifier=GridSearchCV,
                estimator=linear_model.PassiveAggressiveClassifier(),
                param_grid = {'C': np.logspace(-2, 10, 13),
                             'max_iter':[1000]},
                n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
                cv=GRIDSEARCH_NUM_CV_FOLDS):

        self.classifier=classifier(estimator=estimator, 
                                   param_grid=param_grid, 
                                   n_jobs=n_jobs,
                                   cv=cv)
