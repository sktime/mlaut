from mlaut.estimators.mlaut_estimator import MlautEstimator

from mlaut.shared.files_io import DiskOperations
from mlaut.shared.static_variables import(GENERALIZED_LINEAR_MODELS,
                                      REGRESSION, 
                                      CLASSIFICATION,
                                      GRIDSEARCH_NUM_CV_FOLDS,
                                      GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
                                      VERBOSE)
from mlaut.shared.static_variables import PICKLE_EXTENTION

from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
import numpy as np

from sktime.classifiers.base import BaseClassifier
from sktime.regressors.base import BaseRegressor

from sktime.classifiers.base import BaseClassifier
from sktime.regressors.base import BaseRegressor

from mlaut.estimators.base import MLautRegressor, MLautClassifier

class Linear_Regression(BaseRegressor):
    """
    Wrapper for `sklearn Linear Regression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>`_.
    """

    def __init__(self,
                n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS):
        self.fitted_regressor = None
        self.n_jobs = n_jobs
    def fit(self, X,y):
        regressor = linear_model.LinearRegression(n_jobs=self.n_jobs)
        self.fitted_regressor = regressor.fit(X,y)
        self.is_fitted = True
        return self
    def predict(self, X):
        return self.fitted_regressor.predict(X)

class Ridge_Regression(BaseRegressor):
    """
    Wrapper for `sklearn Ridge Regression <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html>`_.
    """
    def __init__(self,
                hyperparameters=None,
                n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
                cv=GRIDSEARCH_NUM_CV_FOLDS):
        self.fitted_regressor = None
        self.n_jobs = n_jobs
        self.cv = cv
        if hyperparameters is None:
            self.hyperparameters = {'alphas':[0.1, 1, 10.0]}
        else:
            self.hyperparameters = hyperparameters

    def fit(self, X,y):
        regressor = linear_model.RidgeCV(alphas=self.hyperparameters['alphas'], cv=self.cv)
        self.fitted_regressor = regressor.fit(X,y)
        self.is_fitted = True
        return self
    def predict(self, X):
        return self.fitted_regressor.predict(X)

Lasso = MLautRegressor(base_regressor=linear_model.LassoCV, alphas=[0.1, 1, 10.0], cv=5, n_jobs=-1)
# class Lasso(MLautRegressor):
#     """
#     Wrapper for `sklearn Lasso <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html>`_.
#     """


#     def __init__(self,
#                 estimator=None,
#                 properties=None,
#                 cv=5,
#                 n_jobs=-1):
#         if estimator is None:
#             hyperparameters = {'alphas':[0.1, 1, 10.0]}
#             self._estimator = linear_model.LassoCV(alphas=hyperparameters['alphas'], cv=cv, n_jobs=n_jobs)
#         else:
#             self._estimator = estimator

#         if properties is None:
#             self._properties = {'estimator_family':[GENERALIZED_LINEAR_MODELS], 
#                                 'tasks':[REGRESSION], 
#                                 'name':'Lasso'}
#         else:
#             self._properties=properties
  

class Lasso_Lars(MlautEstimator):
    """
    Wrapper for `sklearn Lasso Lars <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html>`_.
    """



    def __init__(self,
                estimator=None,
                properties=None,
                cv=5,
                n_jobs=-1):
        if estimator is None:
            self._estimator = linear_model.LassoLarsCV(max_n_alphas=1000, cv=cv, n_jobs=n_jobs)
        else:
            self._estimator = estimator

        if properties is None:
            self._properties = {'estimator_family':[GENERALIZED_LINEAR_MODELS], 
                                'tasks':[REGRESSION], 
                                'name':'LassoLars'}
        else:
            self._properties=properties


   

class Logistic_Regression(MlautEstimator):
    """
    Wrapper for `sklearn Logistic Regression <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_.
    """


   
    def __init__(self,
                estimator=None,
                properties=None,
                n_jobs=-1,
                cv=5):
        if estimator is None:
            hyperparameters = {
                            'C': np.linspace(2**(-5), 2**(15), 13)
                        }
            self._estimator = GridSearchCV(linear_model.LogisticRegression(), 
                                            hyperparameters,
                                            n_jobs=n_jobs,
                                            cv=cv)
        else:
            self._estimator = estimator

        if properties is None:
            self._properties = {'estimator_family':[GENERALIZED_LINEAR_MODELS], 
                                'tasks':[CLASSIFICATION], 
                                'name':'LogisticRegression'}
        else:
            self._properties=properties

 
class Bayesian_Ridge(MlautEstimator):
    """
    Wrapper for `sklearn Bayesian Ridge Regression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html>`_.
    """


    def __init__(self,
                estimator=None,
                properties=None):
        if estimator is None:
           
            self._estimator = linear_model.BayesianRidge()
        else:
            self._estimator = estimator

        if properties is None:
            self._properties = {'estimator_family':[GENERALIZED_LINEAR_MODELS], 
                                'tasks':[REGRESSION], 
                                'name':'BayesianRidge'}
        else:
            self._properties=properties




class Passive_Aggressive_Classifier(MlautEstimator):
    """
    Wrapper for `sklearn Passive Aggressive Classifier <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html>`_.
    """

    
    def __init__(self,
                estimator=None,
                properties=None,
                n_jobs=-1,
                cv=5):
        if estimator is None:
            hyperparameters = {
                'C': np.logspace(-2, 10, 13),
                'max_iter':[1000]
            }
            self._estimator = GridSearchCV(linear_model.PassiveAggressiveClassifier(),
                                          param_grid=hyperparameters,
                                          n_jobs=n_jobs,
                                          cv=cv)
        else:
            self._estimator = estimator

        if properties is None:
            self._properties = {'estimator_family':[GENERALIZED_LINEAR_MODELS], 
                                'tasks':[CLASSIFICATION], 
                                'name':'PassiveAggressiveClassifier'}
        else:
            self._properties=properties
