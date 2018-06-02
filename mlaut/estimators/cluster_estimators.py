
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from mlaut.estimators.mlaut_estimator import properties
from mlaut.estimators.mlaut_estimator import MlautEstimator

from mlaut.shared.files_io import DiskOperations
from mlaut.shared.static_variables import (CLUSTER, 
                                           CLASSIFICATION,
                                           PICKLE_EXTENTION, 
                                           HDF5_EXTENTION)
import numpy as np


@properties(estimator_family=[CLUSTER], 
            tasks=[CLASSIFICATION], 
            name='K_Neighbours')
class K_Neighbours(MlautEstimator):
    """
    Wrapper for `sklearn KNeighbours classifier <http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier>`_.
    """
    def __init__(self, verbose=0, 
                n_jobs=-1,
                num_cv_folds=3, 
                refit=True):
        super().__init__(verbose=verbose, 
                         n_jobs=n_jobs, 
                        num_cv_folds=num_cv_folds, 
                        refit=refit)
        self._hyperparameters = {
                                'n_neighbors': np.arange(1,31),
                                'weights': ['uniform', 'distance'],
                                'n_jobs':[-1]
                        }

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
                            refit=self._refit)
        return self._create_pipeline(estimator=estimator)        

