
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
    properties = {'estimator_family':[CLUSTER], 
            'tasks':[CLASSIFICATION], 
            'name':'K_Neighbours'}

    hyperparameters = {
                    'n_neighbors': np.arange(1,31),
                    'p': [1, 2]
                    }
    def __init__(self,
                hyperparameters=hyperparameters,
                properties=properties, 
                verbose=VERBOSE,
                n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
                num_cv_folds=GRIDSEARCH_NUM_CV_FOLDS, 
                refit=True):

        self.properties = properties
        self._hyperparameters = hyperparameters
        self._verbose = verbose
        self._n_jobs = n_jobs
        self._num_cv_folds = num_cv_folds
        self._refit = refit


    def build(self, **kwargs):
        """
        Builds and returns estimator class.

        Parameters
        ----------
        hyperparameters (dictionary): dictionary with hyperparameters.
        kwargs(key-value): At a minimum the user must specify ``input_dim``, ``num_samples`` and ``num_classes``.

        Returns
        -------
        `sklearn pipeline` object
            pipeline for transforming the features and training the estimator
        """
        # input_dim=kwargs['input_dim']
        num_samples = kwargs['num_samples']
        # num_classes = kwargs['num_classes']
        #append log of num samples to n neighbours range if it is not included already in the array
        
        log_of_num_samples = int(np.log(num_samples))

        if log_of_num_samples not in self._hyperparameters['n_neighbors']:
            self._hyperparameters['n_neighbors'] = np.append(self._hyperparameters, log_of_num_samples)
        
        
        estimator = GridSearchCV(neighbors.KNeighborsClassifier(), 
                            self._hyperparameters, 
                            verbose = self._verbose,
                            n_jobs=self._n_jobs,
                            refit=self._refit,
                            cv=self._num_cv_folds)
        return self._create_pipeline(estimator=estimator)        

