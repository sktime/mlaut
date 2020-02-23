
from sklearn.model_selection import GridSearchCV
from mlaut.shared.static_variables import(GRIDSEARCH_NUM_CV_FOLDS,
                                      GRIDSEARCH_CV_NUM_PARALLEL_JOBS)
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import numpy as np
from sktime.classifiers.base import BaseClassifier
from sktime.regressors.base import BaseRegressor


class Random_Forest_Classifier(BaseClassifier):
    """
    Wrapper for `sklearn Random Forest Classifier <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_.
    """
    def __init__(self, hyperparameters=None, 
                       n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS, 
                       cv=GRIDSEARCH_NUM_CV_FOLDS):
        self.fitted_classifier = None
        self.n_jobs = n_jobs
        self.cv = cv
        if hyperparameters is None:
            self.hyperparameters = hyperparameters = {"max_depth": [10,100, None],
                        "max_features": ['auto', 'sqrt','log2', None],
                        "min_samples_split": [2, 3, 10],
                        "bootstrap": [True, False],
                        "criterion": ["gini", "entropy"],
                        "n_estimators": [10, 100, 200, 500]}
        else:
            self.hyperparameters=hyperparameters

    def fit(self, X, y):
        classifier = GridSearchCV(RandomForestClassifier(), 
                            param_grid=self.hyperparameters, 
                            n_jobs=self.n_jobs,
                            cv=self.cv)
        self.fitted_classifier = classifier.fit(X,y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        return self.fitted_classifier.predict(X)

class Random_Forest_Regressor(BaseRegressor):
    """
    Wrapper for `sklearn Random Forest Regressor <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html>`_.
    """
    def __init__(self, hyperparameters=None, 
                       n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS, 
                       cv=GRIDSEARCH_NUM_CV_FOLDS):
        self.fitted_classifier = None
        self.n_jobs = n_jobs
        self.cv = cv
        if hyperparameters is None:
            self.hyperparameters = {
            'n_estimators': [10, 50, 100],
            'max_features': ['auto', 'sqrt','log2', None],
            'max_depth': [5, 15, None]
            }
        else:
            self.hyperparameters=hyperparameters

    def fit(self, X, y):
        classifier = GridSearchCV(RandomForestRegressor(), 
                                            param_grid=self.hyperparameters, 
                                            n_jobs=self.n_jobs, 
                                            cv=self.cv)
        self.fitted_classifier = classifier.fit(X,y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        return self.fitted_classifier.predict(X)

class Bagging_Classifier(BaseClassifier):
    """
    Wrapper for `sklearn Bagging Classifier <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html>`_.
    """

    def __init__(self, hyperparameters=None, 
                       n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS, 
                       cv=GRIDSEARCH_NUM_CV_FOLDS):
        self.fitted_classifier = None
        self.n_jobs = n_jobs
        self.cv = cv
        if hyperparameters is None:
            self.hyperparameters = {
            'n_estimators': [10, 100, 200, 500],
            'max_samples':[0.5, 1],
            'max_features': [0.5,1],
            'base_estimator': DecisionTreeClassifier()}
        else:
            self.hyperparameters=hyperparameters

    def fit(self, X, y):
        classifier = GridSearchCV(BaggingClassifier(), 
                                            param_grid=self.hyperparameters, 
                                            n_jobs=self.n_jobs, 
                                            cv=self.cv)
        self.fitted_classifier = classifier.fit(X,y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        return self.fitted_classifier.predict(X)

class Bagging_Regressor(BaseRegressor):
    """
    Wrapper for `sklearn Bagging Regressor <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html>`_.
    """

    def __init__(self, hyperparameters=None, 
                       n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS, 
                       cv=GRIDSEARCH_NUM_CV_FOLDS):
        self.fitted_classifier = None
        self.n_jobs = n_jobs
        self.cv = cv
        if hyperparameters is None:
            self.hyperparameters = {
            'n_estimators': [10, 100, 200, 500],
            'max_samples':[0.5, 1],
            'max_features': [0.5,1],
            'base_estimator': DecisionTreeRegressor()}
        else:
            self.hyperparameters=hyperparameters

    def fit(self, X, y):
        classifier = GridSearchCV(BaggingRegressor(), 
                                            param_grid=self.hyperparameters, 
                                            n_jobs=self.n_jobs, 
                                            cv=self.cv)
        self.fitted_classifier = classifier.fit(X,y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        return self.fitted_classifier.predict(X)

class Gradient_Boosting_Classifier(BaseClassifier):
    """
    Wrapper for `sklearn Gradient Boosting Classifier <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html>`_.
    """


    def __init__(self, hyperparameters=None, 
                       n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS, 
                       cv=GRIDSEARCH_NUM_CV_FOLDS):
        self.fitted_classifier = None
        self.n_jobs = n_jobs
        self.cv = cv
        if hyperparameters is None:
            self.hyperparameters = {
            'n_estimators': [100, 200, 500],
                    'max_depth': np.arange(1,11),
                    'learning_rate': [0.01, 0.1, 1, 10, 100]}
        else:
            self.hyperparameters=hyperparameters

    def fit(self, X, y):
        classifier = GridSearchCV(estimator=GradientBoostingClassifier(), 
                                           param_grid=self.hyperparameters,
                                           n_jobs=self.n_jobs,
                                           cv=self.cv)

        self.fitted_classifier = classifier.fit(X,y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        return self.fitted_classifier.predict(X)

class Gradient_Boosting_Regressor(BaseRegressor):
    """
    Wrapper for `sklearn Gradient Boosting Regressor <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html>`_.
    """

    def __init__(self, hyperparameters=None, 
                       n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS, 
                       cv=GRIDSEARCH_NUM_CV_FOLDS):
        self.fitted_classifier = None
        self.n_jobs = n_jobs
        self.cv = cv
        if hyperparameters is None:
            self.hyperparameters = {
           'n_estimators': [10, 50, 100],
                            'max_depth':[10,100, None]}
        else:
            self.hyperparameters=hyperparameters

    def fit(self, X, y):
        classifier = GridSearchCV(estimator=GradientBoostingRegressor(), 
                                           param_grid=self.hyperparameters,
                                           n_jobs=self.n_jobs,
                                           cv=self.cv)

        self.fitted_classifier = classifier.fit(X,y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        return self.fitted_classifier.predict(X)

