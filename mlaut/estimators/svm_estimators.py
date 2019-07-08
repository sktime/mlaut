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
from sklearn.svm import SVR
import numpy as np


class SVC_mlaut(MlautEstimator):
    """
    Wrapper for `sklearn SVC estimator <http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_.
    """

                    
    def __init__(self,
                estimator=None,
                properties=None,
                n_jobs=-1,
                cv=5):

        if estimator is None:
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
            gamma_range = np.append(gamma_range,'scale')
            hyperparameters = [{'kernel': ['rbf'], 'gamma': gamma_range,
                            'C': C_range},
                            {'kernel': ['linear'], 'C': C_range}]
            self._estimator = GridSearchCV(estimator=SVC(), 
                                            param_grid=hyperparameters,
                                            n_jobs=n_jobs,
                                            cv=cv)
        else:
            self._estimator = estimator

        if properties is None:
            self._properties = {'estimator_family':[SVM], 
                                'tasks':[CLASSIFICATION], 
                                'name':'SVC'}
        else:
            self._properties=properties





class SVR_mlaut(MlautEstimator):
    """
    Wrapper for `sklearn SVC estimator <http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_.
    """

    

 
                    
    def __init__(self,
                estimator=None,
                properties=None,
                n_jobs=-1,
                cv=5):

        if estimator is None:


            C_range = np.logspace(-2, 10, 13)
            gamma_range = np.logspace(-9, 3, 13)
            hyperparameters = [{'kernel': ['rbf'], 'gamma': gamma_range,
                                'C': C_range},
                                {'kernel': ['linear'], 'C': C_range}]
            self._estimator = GridSearchCV(estimator=SVR(), 
                                            param_grid=hyperparameters,
                                            n_jobs=n_jobs,
                                            cv=cv)
        else:
            self._estimator = estimator

        if properties is None:
            self._properties =  {'estimator_family':[SVM], 
                                'tasks':[REGRESSION], 
                                'name':'SVR'}
        else:
            self._properties=properties

