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
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from mlaut.estimators.generic_estimator import Generic_Estimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np

class Random_Forest_Classifier(MlautEstimator):
    """
    Wrapper for `sklearn Random Forest Classifier <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_.
    """


    def __init__(self,
                estimator=None,
                properties=None,
                n_jobs=-1,
                cv=5):
        if estimator is None:
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
            self._estimator = GridSearchCV(RandomForestClassifier(), 
                            hyperparameters, 
                            n_jobs=n_jobs,
                            cv=cv)
    
        
        if properties is None:
            self._properties = {'estimator_family':[ENSEMBLE_METHODS], 
                    'tasks':[CLASSIFICATION], 
                    'name':'RandomForestClassifier'}
       
        

class Random_Forest_Regressor(MlautEstimator):
    """
    Wrapper for `sklearn Random Forest Regressor <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html>`_.
    """

 
    def __init__(self,
                estimator=None,
                properties=None,
                n_jobs=-1,
                cv=5):
        if estimator is None:
            hyperparameters = {
            'n_estimators': [10, 50, 100],
            'max_features': ['auto', 'sqrt','log2', None],
            'max_depth': [5, 15, None]
            }
            self._estimator = GridSearchCV(estimator=RandomForestRegressor(), 
                                           param_grid=hyperparameters,
                                           n_jobs=n_jobs,
                                           cv=cv)
        else:
           self._estimator = estimator

        if properties is None:
            self._properties = {'estimator_family':[ENSEMBLE_METHODS], 
                                'tasks':[REGRESSION], 
                                'name':'RandomForestRegressor'}
        else:
            self._properties=properties


class Bagging_Classifier(MlautEstimator):
    """
    Wrapper for `sklearn Bagging Classifier <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html>`_.
    """


    def __init__(self,
                estimator=None,
                properties=None,
                n_jobs=-1,
                cv=5):

        if estimator is None:
            hyperparameters = {
            'n_estimators': [10, 100, 200, 500],
            'max_samples':[0.5, 1],
            'max_features': [0.5,1]}
            # 'base_estimator': [DecisionTreeClassifier(), KNeighborsClassifier(), SVC()]
            self._estimator = GridSearchCV(estimator=DecisionTreeClassifier(), 
                                            param_grid=hyperparameters,
                                            n_jobs=n_jobs,
                                            cv=cv)
        else:
            self._estimator = estimator

        if properties is None:
            self._properties = {'estimator_family':[ENSEMBLE_METHODS], 
                                'tasks':[CLASSIFICATION], 
                                'name':'BaggingClassifier'}
        else:
            self._properties=properties

class Bagging_Regressor(MlautEstimator):
    """
    Wrapper for `sklearn Bagging Regressor <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html>`_.
    """
   
    def __init__(self,
                estimator=None,
                properties=None,
                n_jobs=-1,
                cv=5):
        if estimator is None:
            hyperparameters ={
                    'n_estimators': [10, 50, 100]
                }
 
            self._estimator = GridSearchCV(estimator=DecisionTreeRegressor(), 
                                           param_grid=hyperparameters,
                                           n_jobs=n_jobs,
                                           cv=cv)
        else:
           self._estimator = estimator

        if properties is None:
            self._properties = {'estimator_family':[ENSEMBLE_METHODS], 
                                'tasks':[REGRESSION], 
                                'name':'BaggingRegressor'}
        else:
            self._properties=properties





class Gradient_Boosting_Classifier(MlautEstimator):
    """
    Wrapper for `sklearn Gradient Boosting Classifier <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html>`_.
    """


    def __init__(self,
                estimator=None,
                properties=None,
                n_jobs=-1,
                cv=5):
        if estimator is None:
            hyperparameters ={
                    'n_estimators': [100, 200, 500],
                    'max_depth': np.arange(1,11),
                    'learning_rate': [0.01, 0.1, 1, 10, 100]
                }
 
            self._estimator = GridSearchCV(estimator=GradientBoostingClassifier(), 
                                           param_grid=hyperparameters,
                                           n_jobs=n_jobs,
                                           cv=cv)
        else:
           self._estimator = estimator

        if properties is None:
            self._properties = {'estimator_family':[ENSEMBLE_METHODS], 
                                'tasks':[CLASSIFICATION], 
                                'name':'GradientBoostingClassifier'}
        else:
            self._properties=properties




class Gradient_Boosting_Regressor(MlautEstimator):
    """
    Wrapper for `sklearn Gradient Boosting Regressor <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html>`_.
    """


    def __init__(self,
                estimator=None,
                properties=None,
                n_jobs=-1,
                cv=5):
        if estimator is None:
            hyperparameters ={
                            'n_estimators': [10, 50, 100],
                            'max_depth':[10,100, None]
                        }        
 
            self._estimator = GridSearchCV(estimator=GradientBoostingRegressor(), 
                                           param_grid=hyperparameters,
                                           n_jobs=n_jobs,
                                           cv=cv)
        else:
           self._estimator = estimator

        if properties is None:
            self._properties ={'estimator_family':[ENSEMBLE_METHODS], 
                            'tasks':[REGRESSION], 
                            'name':'GradientBoostingRegressor'}
        else:
            self._properties=properties

