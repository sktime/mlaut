from mlaut.estimators.mlaut_estimator import properties
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

@properties(estimator_family=[GENERALIZED_LINEAR_MODELS], 
            tasks=[REGRESSION], 
            name='RidgeRegression')
class Ridge_Regression(MlautEstimator):
    """
    Wrapper for `sklearn Ridge Regression <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html>`_.
    """
    def __init__(self, verbose=VERBOSE, 
                n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
                num_cv_folds=GRIDSEARCH_NUM_CV_FOLDS, 
                refit=True):
        super().__init__(verbose=verbose, 
                         n_jobs=n_jobs, 
                        num_cv_folds=num_cv_folds, 
                        refit=refit)
        self._hyperparameters = {'alphas':[0.1, 1, 10.0],
            
            } # this is the alpha hyperparam
       
    def build(self, **kwargs):
        """
        builds and returns estimator

        Args:
            hyperparameters (dictionary): Dictionary of hyperparameters to be used for tuning the estimator.
        **kwargs(key-value arguments): Ignored in this implementation. Added for compatibility with :func:`mlaut.estimators.nn_estimators.Deep_NN_Classifier`.
        
        Returns:
            (`sklearn pipeline` object): pipeline for transforming the features and training the estimator
        """
        
        
        estimator = linear_model.RidgeCV(alphas=self._hyperparameters['alphas'],
                                cv=self._num_cv_folds)

        return self._create_pipeline(estimator=estimator)
        

@properties(estimator_family=[GENERALIZED_LINEAR_MODELS], 
            tasks=[REGRESSION], 
            name='Lasso')
class Lasso(MlautEstimator):
    """
    Wrapper for `sklearn Lasso <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html>`_.
    """
    def __init__(self, verbose=VERBOSE, 
                n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
                num_cv_folds=GRIDSEARCH_NUM_CV_FOLDS, 
                refit=True):
        super().__init__(verbose=verbose, 
                         n_jobs=n_jobs, 
                        num_cv_folds=num_cv_folds, 
                        refit=refit)
        self._hyperparameters = {'alphas':[0.1, 1, 10.0]}
    
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

    # def save(self, dataset_name):
    #     """
    #     Saves estimator on disk.

    #     :type dataset_name: string
    #     :param dataset_name: name of the dataset. Estimator will be saved under default folder structure `/data/trained_models/<dataset name>/<model name>`
    #     """
    #     disk_op = DiskOperations()
    #     disk_op.save_to_pickle(trained_model=self._trained_model,
    #                             model_name=self.properties()['name'],
    #                             dataset_name=dataset_name)
@properties(estimator_family=[GENERALIZED_LINEAR_MODELS], 
            tasks=[REGRESSION], 
            name='LassoLars')
class Lasso_Lars(MlautEstimator):
    """
    Wrapper for `sklearn Lasso Lars <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html>`_.
    """

    
    def __init__(self, verbose=VERBOSE, 
                n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
                num_cv_folds=GRIDSEARCH_NUM_CV_FOLDS, 
                refit=True):
        super().__init__(verbose=verbose, 
                         n_jobs=n_jobs, 
                        num_cv_folds=num_cv_folds, 
                        refit=refit)
        self._hyperparameters = {'max_n_alphas':1000}
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
    # def save(self, dataset_name):
    #     """
    #     Saves estimator on disk.

    #     :type dataset_name: string
    #     :param dataset_name: name of the dataset. Estimator will be saved under default folder structure `/data/trained_models/<dataset name>/<model name>`
    #     """
    #     disk_op = DiskOperations()
    #     disk_op.save_to_pickle(trained_model=self._trained_model,
    #                             model_name=self.properties()['name'],
    #                             dataset_name=dataset_name)

@properties(estimator_family=[GENERALIZED_LINEAR_MODELS], 
            tasks=[REGRESSION], 
            name='LogisticRegression')
class Logistic_Regression(MlautEstimator):
    """
    Wrapper for `sklearn Logistic Regression <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_.
    """
    def __init__(self, verbose=VERBOSE, 
                n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
                num_cv_folds=GRIDSEARCH_NUM_CV_FOLDS, 
                refit=True):
        super().__init__(verbose=verbose, 
                         n_jobs=n_jobs, 
                        num_cv_folds=num_cv_folds, 
                        refit=refit)
        self._hyperparameters = {
                'C': np.linspace(2**(-5), 2**(15), 13)

            }
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

 

@properties(estimator_family=[GENERALIZED_LINEAR_MODELS],
            tasks=[CLASSIFICATION],
            name='PassiveAggressiveClassifier')
class Passive_Aggressive_Classifier(MlautEstimator):
    """
    Wrapper for `sklearn Passive Aggressive Classifier <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html>`_.
    """

    def __init__(self, verbose=VERBOSE, 
                n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
                num_cv_folds=GRIDSEARCH_NUM_CV_FOLDS, 
                refit=True):
        super().__init__(verbose=verbose, 
                         n_jobs=n_jobs, 
                        num_cv_folds=num_cv_folds, 
                        refit=refit)
                        
        self._hyperparameters = {
                'C': np.linspace(2**(-5), 2**(15), 13),
                'max_iter':[1000]
            }
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

