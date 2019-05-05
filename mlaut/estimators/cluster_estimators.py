
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from mlaut.estimators.mlaut_estimator import MlautEstimator

from mlaut.shared.files_io import DiskOperations
from mlaut.shared.static_variables import (CLUSTER, 
                                           CLASSIFICATION,
                                           PICKLE_EXTENTION, 
                                           HDF5_EXTENTION,
                                           GRIDSEARCH_NUM_CV_FOLDS,
                                           GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
                                           VERBOSE)
import numpy as np

from mlaut.estimators.generic_estimator import Generic_Estimator


class K_Neighbours(MlautEstimator):
    """
    Wrapper for `sklearn KNeighbours classifier <http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier>`_.
    """


    def __init__(self,
                estimator=None,
                properties=None,
                n_jobs=-1,
                cv=5):
        if estimator is None:
            hyperparameters = {
                    'n_neighbors': np.arange(1,31),
                    'p': [1, 2]
                    }
            self._estimator = GridSearchCV(neighbors.KNeighborsClassifier(), 
                                            param_grid=hyperparameters, 
                                            n_jobs=n_jobs, 
                                            cv=cv)
        else:
            self._estimator = estimator
        
        if properties is None:
            self._properties = {'estimator_family':[CLUSTER], 
                                'tasks':[CLASSIFICATION], 
                                'name':'K_Neighbours'}
        else:
            self._properties=properties
 
