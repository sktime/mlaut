from mlaut.estimators.mlaut_estimator import properties
from mlaut.estimators.mlaut_estimator import MlautEstimator

from mlaut.shared.files_io import DiskOperations

from sklearn.model_selection import GridSearchCV
from mlaut.shared.static_variables import(ENSEMBLE_METHODS, 
                                      REGRESSION, 
                                      CLASSIFICATION)
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

@properties(estimator_family=[ENSEMBLE_METHODS], 
            tasks=[CLASSIFICATION], 
            name='RandomForestClassifier')
class Random_Forest_Classifier(MlautEstimator):
    """
    Wrapper for `sklearn Random Forest Classifier <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_.
    """

    def __init__(self,  
                verbose=0, 
                n_jobs=-1,
                num_cv_folds=3, 
                refit=True):
        super().__init__( verbose=verbose, 
                n_jobs=n_jobs,
                num_cv_folds=num_cv_folds, 
                refit=refit)
        self._hyperparameters = {
                    'n_estimators': [10, 50, 100],
                    'max_features': ['auto', 'sqrt','log2', None],
                    'max_depth': [5, 15, None]
                }
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
                            refit=self._refit)
        return self._create_pipeline(estimator=estimator)  
        # return GridSearchCV(RandomForestClassifier(), 
        #                     self._hyperparameters, 
        #                     verbose = self._verbose,
        #                     n_jobs=self._n_jobs,
        #                     refit=self._refit)

        
@properties(estimator_family=[ENSEMBLE_METHODS], 
            tasks=[REGRESSION], 
            name='RandomForestRegressor')
class Random_Forest_Regressor(MlautEstimator):
    """
    Wrapper for `sklearn Random Forest Regressor <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html>`_.
    """
    def __init__(self, 
                verbose=0, 
                n_jobs=-1,
                num_cv_folds=3, 
                refit=True):
        super().__init__(verbose=verbose, 
                         n_jobs=n_jobs, 
                        num_cv_folds=num_cv_folds, 
                        refit=refit)
        self._hyperparameters = {
                'n_estimators': [10, 50, 100],
                'max_features': ['auto', 'sqrt','log2', None],
                'max_depth': [5, 15, None]
            }

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
                            refit=self._refit)
        return self._create_pipeline(estimator=estimator)        
        # return GridSearchCV(RandomForestRegressor(), 
        #                     self._hyperparameters, 
        #                     verbose = self._verbose,
        #                     n_jobs=self._n_jobs,
        #                     refit=self._refit)
    

@properties(estimator_family=[ENSEMBLE_METHODS], 
            tasks=[CLASSIFICATION], 
            name='BaggingClassifier')
class Bagging_Classifier(MlautEstimator):
    """
    Wrapper for `sklearn Bagging Classifier <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html>`_.
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
                'n_estimators': [10, 50, 100]
            }
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
                            refit=self._refit)
        return self._create_pipeline(estimator=estimator)        

        # return GridSearchCV(model, 
        #                     self._hyperparameters, 
        #                     verbose = self._verbose,
        #                     n_jobs=self._n_jobs,
        #                     refit=self._refit)
    
@properties(estimator_family=[ENSEMBLE_METHODS], 
            tasks=[REGRESSION], 
            name='BaggingRegressor')
class Bagging_Regressor(MlautEstimator):
    """
    Wrapper for `sklearn Bagging Regressor <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html>`_.
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
                'n_estimators': [10, 50, 100]
            }

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
                            refit=self._refit)
        return self._create_pipeline(estimator=estimator)        

        # return GridSearchCV(model, 
        #                     self._hyperparameters, 
        #                     verbose = self._verbose,
        #                     n_jobs=self._n_jobs,
        #                     refit=self._refit)
    


@properties(estimator_family=[ENSEMBLE_METHODS], 
            tasks=[CLASSIFICATION], 
            name='GradientBoostingClassifier')
class Gradient_Boosting_Classifier(MlautEstimator):
    """
    Wrapper for `sklearn Gradient Boosting Classifier <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html>`_.
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
                'n_estimators': [10, 50, 100],
                'max_depth':[10,100, None]
            }

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
                            refit=self._refit)
        return self._create_pipeline(estimator=estimator)        


@properties(estimator_family=[ENSEMBLE_METHODS], 
            tasks=[REGRESSION], 
            name='GradientBoostingRegressor')
class Gradient_Boosting_Regressor(MlautEstimator):
    """
    Wrapper for `sklearn Gradient Boosting Regressor <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html>`_.
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
                'n_estimators': [10, 50, 100],
                'max_depth':[10,100, None]
            }
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
                            refit=self._refit)
    
        return self._create_pipeline(estimator=estimator)        
