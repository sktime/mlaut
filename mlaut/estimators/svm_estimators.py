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
from mlaut.estimators.generic_estimator import Generic_Estimator


class SVC_mlaut(MlautEstimator):
    """
    Wrapper for `sklearn SVC estimator <http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_.
    """
    properties = {'estimator_family':[SVM], 
            'tasks':[CLASSIFICATION], 
            'name':'SVC'}
    # c_range = np.linspace(2**(-5), 2**(15), 5) #change last num to 13 for better search
    # gamma_range = np.linspace(2**(-15), 2**3, 5) #change last num to 13 for better search     
    # hyperparameters = {
    #                     'C': c_range,
    #                     'gamma': ['auto']
    #                     }

    # Set the parameters by cross-validation

    #inspired from http://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html#sphx-glr-auto-examples-model-selection-plot-grid-search-digits-py
    # hyperparameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
    #                  'C': [1, 10, 100, 1000]},
    #                 {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    hyperparameters = [{'kernel': ['rbf'], 'gamma': gamma_range,
                     'C': C_range},
                    {'kernel': ['linear'], 'C': C_range}]
                    
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

       
