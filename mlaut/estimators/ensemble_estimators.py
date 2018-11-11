from mlaut.estimators.mlaut_estimator import MlautEstimator

from mlaut.shared.files_io import DiskOperations

from sklearn.model_selection import GridSearchCV
from mlaut.shared.static_variables import(ENSEMBLE_METHODS, 
                                      REGRESSION, 
                                      CLASSIFICATION,
                                      GRIDSEARCH_NUM_CV_FOLDS,
                                      GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
                                      VERBOSE)
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from mlaut.estimators.generic_estimator import Generic_Estimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np

class Random_Forest_Classifier(MlautEstimator):
    """
    Wrapper for `sklearn Random Forest Classifier <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_.
    """
    properties = {'estimator_family':[ENSEMBLE_METHODS], 
            'tasks':[CLASSIFICATION], 
            'name':'RandomForestClassifier'}
    # hyperparameters = {
    #                 'n_estimators': [10, 50, 100],
    #                 'max_features': ['auto', 'sqrt','log2', None],
    #                 'max_depth': [5, 15, None]
    #             }
    
    # source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html
    hyperparameters = {"max_depth": [10,100, None],
                "max_features": ['auto', 'sqrt','log2', None],
                "min_samples_split": [2, 3, 10],
                "bootstrap": [True, False],
                "criterion": ["gini", "entropy"],
                "n_estimators": [10, 100, 200, 500]}

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

        Parameters
        ----------
        hyperparameters: dictionary
            Dictionary of hyperparameters to be used for tuning the estimator.
        **kwargs : key-value arguments.
            Ignored in this implementation. Added for compatibility with :func:`mlaut.estimators.nn_estimators.Deep_NN_Classifier`.
        
        Returns
        -------
        `sklearn pipeline` object
            pipeline for transforming the features and training the estimator

        """
        estimator = GridSearchCV(RandomForestClassifier(), 
                            self._hyperparameters, 
                            verbose = self._verbose,
                            n_jobs=self._n_jobs,
                            refit=self._refit,
                            cv=self._num_cv_folds)
        return self._create_pipeline(estimator=estimator)  


        

class Random_Forest_Regressor(MlautEstimator):
    """
    Wrapper for `sklearn Random Forest Regressor <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html>`_.
    """
    properties = {'estimator_family':[ENSEMBLE_METHODS], 
            'tasks':[REGRESSION], 
            'name':'RandomForestRegressor'}
    hyperparameters = {
                'n_estimators': [10, 50, 100],
                'max_features': ['auto', 'sqrt','log2', None],
                'max_depth': [5, 15, None]
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

        Parameters
        ----------
        hyperparameters: dictionary
            Dictionary of hyperparameters to be used for tuning the estimator.
        **kwargs : key-value arguments.
            Ignored in this implementation. Added for compatibility with :func:`mlaut.estimators.nn_estimators.Deep_NN_Classifier`.
        
        Returns
        -------
        `sklearn pipeline` object
            pipeline for transforming the features and training the estimator

        """   
        estimator = GridSearchCV(RandomForestRegressor(), 
                            self._hyperparameters, 
                            verbose = self._verbose,
                            n_jobs=self._n_jobs,
                            refit=self._refit,
                            cv=self._num_cv_folds)
        return self._create_pipeline(estimator=estimator)        


class Bagging_Classifier(MlautEstimator):
    """
    Wrapper for `sklearn Bagging Classifier <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html>`_.
    """
    properties = {'estimator_family':[ENSEMBLE_METHODS], 
            'tasks':[CLASSIFICATION], 
            'name':'BaggingClassifier'}
    hyperparameters = {
            'n_estimators': [10, 100, 200, 500],
            'max_samples':[0.5, 1],
            'max_features': [0.5,1]
            # 'base_estimator': [DecisionTreeClassifier(), KNeighborsClassifier(), SVC()]
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

        Parameters
        ----------
        hyperparameters: dictionary
            Dictionary of hyperparameters to be used for tuning the estimator.
        **kwargs : key-value arguments.
            Ignored in this implementation. Added for compatibility with :func:`mlaut.estimators.nn_estimators.Deep_NN_Classifier`.
        
        Returns
        -------
        `sklearn pipeline` object
            pipeline for transforming the features and training the estimator
        """
        estimator = BaggingClassifier(base_estimator=DecisionTreeClassifier())
        estimator = GridSearchCV(estimator, 
                            self._hyperparameters, 
                            verbose = self._verbose,
                            n_jobs=self._n_jobs,
                            refit=self._refit,
                            cv=self._num_cv_folds)
        return self._create_pipeline(estimator=estimator)        



class Bagging_Regressor(MlautEstimator):
    """
    Wrapper for `sklearn Bagging Regressor <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html>`_.
    """
    properties = {'estimator_family':[ENSEMBLE_METHODS], 
            'tasks':[REGRESSION], 
            'name':'BaggingRegressor'}
    hyperparameters = {
                    'n_estimators': [10, 50, 100]
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

        Parameters
        ----------
        hyperparameters: dictionary
            Dictionary of hyperparameters to be used for tuning the estimator.
        **kwargs : key-value arguments.
            Ignored in this implementation. Added for compatibility with :func:`mlaut.estimators.nn_estimators.Deep_NN_Classifier`.
        
        Returns
        -------
        `sklearn pipeline` object
            pipeline for transforming the features and training the estimator
        """
       
        estimator = BaggingRegressor(base_estimator=DecisionTreeClassifier())
        estimator = GridSearchCV(estimator, 
                            self._hyperparameters, 
                            verbose = self._verbose,
                            n_jobs=self._n_jobs,
                            refit=self._refit,
                            cv=self._num_cv_folds)
        return self._create_pipeline(estimator=estimator)        





class Gradient_Boosting_Classifier(MlautEstimator):
    """
    Wrapper for `sklearn Gradient Boosting Classifier <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html>`_.
    """
    properties = {'estimator_family':[ENSEMBLE_METHODS], 
            'tasks':[CLASSIFICATION], 
            'name':'GradientBoostingClassifier'}
    hyperparameters = {
                    'n_estimators': [100, 200, 500],
                    'max_depth': np.arange(1,11),
                    'learning_rate': [0.01, 0.1, 1, 10, 100]
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

        Parameters
        ----------
        hyperparameters: dictionary
            Dictionary of hyperparameters to be used for tuning the estimator.
        **kwargs : key-value arguments.
            Ignored in this implementation. Added for compatibility with :func:`mlaut.estimators.nn_estimators.Deep_NN_Classifier`.
        
        Returns
        -------
        `sklearn pipeline` object
            pipeline for transforming the features and training the estimator

        """
        estimator = GridSearchCV(GradientBoostingClassifier(), 
                            self._hyperparameters, 
                            verbose = self._verbose,
                            n_jobs=self._n_jobs,
                            refit=self._refit,
                            cv=self._num_cv_folds)
        return self._create_pipeline(estimator=estimator)        



class Gradient_Boosting_Regressor(MlautEstimator):
    """
    Wrapper for `sklearn Gradient Boosting Regressor <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html>`_.
    """

    properties = {'estimator_family':[ENSEMBLE_METHODS], 
            'tasks':[REGRESSION], 
            'name':'GradientBoostingRegressor'}
    hyperparameters = {
                'n_estimators': [10, 50, 100],
                'max_depth':[10,100, None]
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

        Parameters
        ----------
        hyperparameters: dictionary
            Dictionary of hyperparameters to be used for tuning the estimator.
        **kwargs : key-value arguments.
            Ignored in this implementation. Added for compatibility with :func:`mlaut.estimators.nn_estimators.Deep_NN_Classifier`.
        
        Returns
        -------
        `sklearn pipeline` object
            pipeline for transforming the features and training the estimator

        """

        estimator = GridSearchCV(GradientBoostingRegressor(), 
                            self._hyperparameters, 
                            verbose = self._verbose,
                            n_jobs=self._n_jobs,
                            refit=self._refit,
                            cv=self._num_cv_folds)
    
        return self._create_pipeline(estimator=estimator)        
