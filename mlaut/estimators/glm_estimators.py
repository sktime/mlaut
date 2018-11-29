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
from mlaut.estimators.generic_estimator import Generic_Estimator


class Ridge_Regression(MlautEstimator):
    """
    Wrapper for `sklearn Ridge Regression <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html>`_.
    """
    properties= {'estimator_family':[GENERALIZED_LINEAR_MODELS], 
            'tasks':[REGRESSION], 
            'name':'RidgeRegression'}

    hyperparameters = {'alphas':[0.1, 1, 10.0],
            
            } # this is the alpha hyperparam

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
            **kwargs(key-value arguments): Ignored in this implementation. Added for compatibility with :func:`mlaut.estimators.nn_estimators.Deep_NN_Classifier`.
        
        Returns:
            (`sklearn pipeline` object): pipeline for transforming the features and training the estimator
        """
        
        
        estimator = linear_model.RidgeCV(alphas=self._hyperparameters['alphas'],
                                cv=self._num_cv_folds)

        return self._create_pipeline(estimator=estimator)
        


class Lasso(MlautEstimator):
    """
    Wrapper for `sklearn Lasso <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html>`_.
    """
    properties = {'estimator_family':[GENERALIZED_LINEAR_MODELS], 
            'tasks':[REGRESSION], 
            'name':'Lasso'}

    hyperparameters = {'alphas':[0.1, 1, 10.0]}

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
            hyperparameters(dictionary): Dictionary of hyperparameters to be used for tuning the estimator.
            **kwargs (key-value arguments): Ignored in this implementation. Added for compatibility with :func:`mlaut.estimators.nn_estimators.Deep_NN_Classifier`.
        
        Returns:
            `sklearn pipeline` object: pipeline for transforming the features and training the estimator
        """


        estimator = linear_model.LassoCV(alphas=self._hyperparameters['alphas'],
                                    cv=self._num_cv_folds,
                                    n_jobs=self._n_jobs)

        return self._create_pipeline(estimator=estimator)


class Lasso_Lars(MlautEstimator):
    """
    Wrapper for `sklearn Lasso Lars <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html>`_.
    """

    properties = {'estimator_family':[GENERALIZED_LINEAR_MODELS], 
                'tasks':[REGRESSION], 
                'name':'LassoLars'}
    
    hyperparameters = {'max_n_alphas':1000}

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



        estimator = linear_model.LassoLarsCV(max_n_alphas=self._hyperparameters['max_n_alphas'],
                                    cv=self._num_cv_folds,
                                    n_jobs=self._n_jobs)

        return self._create_pipeline(estimator=estimator)
   

class Logistic_Regression(MlautEstimator):
    """
    Wrapper for `sklearn Logistic Regression <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_.
    """
    properties = {'estimator_family':[GENERALIZED_LINEAR_MODELS], 
            'tasks':[REGRESSION], 
            'name':'LogisticRegression'}

    hyperparameters = {
                'C': np.linspace(2**(-5), 2**(15), 13)

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
        builds and returns estimator

        Args:
            hyperparameters (dictionary): Dictionary of hyperparameters to be used for tuning the estimator.
            **kwargs (key-value arguments): Ignored in this implementation. Added for compatibility with :func:`mlaut.estimators.nn_estimators.Deep_NN_Classifier`.
        
        Returns:
            `sklearn pipeline` object: pipeline for transforming the features and training the estimator

        """
        estimator = GridSearchCV(linear_model.LogisticRegression(), 
                            self._hyperparameters, 
                            verbose = self._verbose,
                            n_jobs=self._n_jobs,
                            refit=self._refit,
                            cv=self._num_cv_folds)
        return self._create_pipeline(estimator=estimator)

 


class Passive_Aggressive_Classifier(MlautEstimator):
    """
    Wrapper for `sklearn Passive Aggressive Classifier <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html>`_.
    """
    properties = {'estimator_family':[GENERALIZED_LINEAR_MODELS],
            'tasks':[CLASSIFICATION],
            'name':'PassiveAggressiveClassifier'}
    
    C_range = np.logspace(-2, 10, 13)
    hyperparameters = {
                'C': C_range,
                'max_iter':[1000]
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
        builds and returns estimator

        Args:
            hyperparameters (dictionary): Dictionary of hyperparameters to be used for tuning the estimator.
            **kwargs (key-value arguments): Ignored in this implementation. Added for compatibility with :func:`mlaut.estimators.nn_estimators.Deep_NN_Classifier`.
        
        Returns:
            `sklearn pipeline` object: pipeline for transforming the features and training the estimator
        """
        estimator = GridSearchCV(linear_model.PassiveAggressiveClassifier(), 
                            self._hyperparameters, 
                            verbose=self._verbose,
                            n_jobs=self._n_jobs,
                            refit=self._refit,
                            cv=self._num_cv_folds
                            )
        return self._create_pipeline(estimator=estimator)

