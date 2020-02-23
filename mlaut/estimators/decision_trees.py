from mlaut.estimators.mlaut_estimator import MlautEstimator
from mlaut.shared.static_variables import(DECISION_TREE_METHODS, 
                                      CLASSIFICATION,
                                      REGRESSION,
                                      GRIDSEARCH_NUM_CV_FOLDS,
                                      GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
                                      VERBOSE)
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import GridSearchCV
import numpy as np


from sktime.classifiers.base import BaseClassifier
from sktime.regressors.base import BaseRegressor

class Decision_Tree_Classifier(BaseClassifier):
    """
    Wrapper for `sklearn Decision Tree Classifier <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_.
    """
    def __init__(self, hyperparameters=None, 
                       n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS, 
                       cv=GRIDSEARCH_NUM_CV_FOLDS):
        self.fitted_classifier = None
        self.n_jobs = n_jobs
        self.cv = cv
        if hyperparameters is None:
            self.hyperparameters = {"max_depth": [10,100, None],
                            "criterion": ['gini', 'entropy'],
                            "max_features": ['auto', 'sqrt','log2'],
                            "min_samples_leaf":np.arange(1,11)}
        else:
            self.hyperparameters=hyperparameters


    def fit(self, X, y):
        classifier = GridSearchCV(DecisionTreeClassifier(), 
                            self.hyperparameters, 
                            n_jobs=self.n_jobs,
                            cv=self.cv)
        self.fitted_classifier = classifier.fit(X,y)
        self.is_fitted = True
        return self

    def predict(self, X):
        return self.fitted_classifier.predict(X)

class Decision_Tree_Regressor(BaseRegressor):
    def __init__(self, hyperparameters=None, 
                       n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS, 
                       cv=GRIDSEARCH_NUM_CV_FOLDS):
        self.fitted_classifier = None
        self.n_jobs = n_jobs
        self.cv = cv
        if hyperparameters is None:
            self.hyperparameters = {"max_depth": [10,100, None],
                                "criterion": ['mse', 'friedman_mse', 'mae'],
                                "max_features": ['auto', 'sqrt','log2'],
                                "min_samples_leaf":np.arange(1,11)}
        else:
            self.hyperparameters=hyperparameters


    def fit(self, X, y):
        classifier = GridSearchCV(DecisionTreeRegressor(), 
                            self.hyperparameters, 
                            n_jobs=self.n_jobs,
                            cv=self.cv)
        self.fitted_classifier = classifier.fit(X,y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        return self.fitted_classifier.predict(X)

    