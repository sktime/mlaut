
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from mlaut.shared.static_variables import GRIDSEARCH_CV_NUM_PARALLEL_JOBS, GRIDSEARCH_NUM_CV_FOLDS
from sktime.classifiers.base import BaseClassifier

class K_Neighbours(BaseClassifier):
    """
    Wrapper for `sklearn KNeighbours classifier <http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier>`_.
    """
    def __init__(self, hyperparameters=None, 
                       n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS, 
                       cv=GRIDSEARCH_NUM_CV_FOLDS):
        self.fitted_classifier = None
        self.n_jobs = n_jobs
        self.cv = cv
        if hyperparameters is None:
            self.hyperparameters = {
                    'n_neighbors': np.arange(1,31),
                    'p': [1, 2]
                    }
        else:
            self.hyperparameters=hyperparameters

    def fit(self, X, y):
        classifier = GridSearchCV(neighbors.KNeighborsClassifier(), 
                                            param_grid=self.hyperparameters, 
                                            n_jobs=self.n_jobs, 
                                            cv=self.cv)
        self.fitted_classifier = classifier.fit(X,y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        return self.fitted_classifier.predict(X)

 
