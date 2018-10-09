from mlaut.estimators.mlaut_estimator import MlautEstimator

from mlaut.shared.files_io import DiskOperations
from sklearn.model_selection import GridSearchCV
from mlaut.shared.static_variables import(SVM,
                                      REGRESSION, 
                                      CLASSIFICATION,
                                      GRIDSEARCH_NUM_CV_FOLDS,
                                      GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
                                      VERBOSE)

from sklearn.svm import SVC
import numpy as np


class SVC_mlaut(MlautEstimator):
    """
    Wrapper for `sklearn SVC estimator <http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_.
    """
    properties = {'estimator_family':[SVM], 
            'tasks':[CLASSIFICATION], 
            'name':'SVC'}
            
    def __init__(self, verbose=VERBOSE,
                properties=properties, 
                n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
                num_cv_folds=GRIDSEARCH_NUM_CV_FOLDS, 
                refit=True):
        super().__init__(verbose=verbose,
                         n_jobs=n_jobs, 
                        num_cv_folds=num_cv_folds, 
                        refit=refit)
        c_range = np.linspace(2**(-5), 2**(15), 13)
        gamma_range = np.linspace(2**(-15), 2**3, 13)
        self._hyperparameters = {
                            'C': c_range,
                            'gamma': gamma_range
                        }
        self.properties = properties

    def build(self, **kwargs):
        """
        builds and returns estimator
        
        Args:
            hyperparameters (dictionary): Dictionary of hyperparameters to be used for tuning the estimator.
            **kwargs (key-value arguments): Ignored in this implementation. Added for compatibility with :func:`mlaut.estimators.nn_estimators.Deep_NN_Classifier`.
        
        Returns:
            `sklearn pipeline` object: pipeline for transforming the features and training the estimator
        
        """
        estimator = GridSearchCV(SVC(), 
                            self._hyperparameters, 
                            verbose = self._verbose,
                            n_jobs=self._n_jobs,
                            refit=self._refit)
        return self._create_pipeline(estimator=estimator)

       
