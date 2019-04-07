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
class Decision_Tree_Classifier(MlautEstimator):
    """
    Wrapper for `sklearn Decision Tree Classifier <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_.
    """
    properties = {'estimator_family':[DECISION_TREE_METHODS], 
            'tasks':[CLASSIFICATION], 
            'name':'DecisionTreeClassifier'}

    hyperparameters = {"max_depth": [10,100, None],
                "criterion": ['gini', 'entropy'],
                "max_features": ['auto', 'sqrt','log2'],
                "min_samples_leaf":np.arange(1,11)}

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
        estimator = GridSearchCV(DecisionTreeClassifier(), 
                            self._hyperparameters, 
                            verbose = self._verbose,
                            n_jobs=self._n_jobs,
                            refit=self._refit,
                            cv=self._num_cv_folds)
        return self._create_pipeline(estimator=estimator)  


class Decision_Tree_Regressor(MlautEstimator):
    """
    Wrapper for `sklearn Decision Tree Classifier <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_.
    """
    properties = {'estimator_family':[DECISION_TREE_METHODS], 
            'tasks':[REGRESSION], 
            'name':'DecisionTreeClassifier'}

    hyperparameters = {"max_depth": [10,100, None],
                "criterion": ['mse', 'friedman_mse', 'mae'],
                "max_features": ['auto', 'sqrt','log2'],
                "min_samples_leaf":np.arange(1,11)}

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
        estimator = GridSearchCV(DecisionTreeRegressor(), 
                            self._hyperparameters, 
                            verbose = self._verbose,
                            n_jobs=self._n_jobs,
                            refit=self._refit,
                            cv=self._num_cv_folds)
        return self._create_pipeline(estimator=estimator)  