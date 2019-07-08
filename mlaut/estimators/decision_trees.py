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

    def __init__(self,
                estimator=None,
                properties=None,
                n_jobs=-1,
                cv=5):
        if estimator is None:
            hyperparameters = {"max_depth": [10,100, None],
                            "criterion": ['gini', 'entropy'],
                            "max_features": ['auto', 'sqrt','log2'],
                            "min_samples_leaf":np.arange(1,11)}
            estimator = GridSearchCV(DecisionTreeClassifier(), 
                            hyperparameters, 
                            n_jobs=n_jobs,
                            cv=cv)
        

        if properties is None:
            properties = {'estimator_family':[DECISION_TREE_METHODS], 
                                'tasks':[CLASSIFICATION], 
                                'name':'DecisionTreeClassifier'}

        self._estimator = estimator
        self._properties = properties
class Decision_Tree_Regressor(MlautEstimator):


    def __init__(self,
                estimator=None,
                properties=None,
                n_jobs=-1,
                cv=5):

        if estimator is None:
            hyperparameters = {"max_depth": [10,100, None],
                                "criterion": ['mse', 'friedman_mse', 'mae'],
                                "max_features": ['auto', 'sqrt','log2'],
                                "min_samples_leaf":np.arange(1,11)}

            self._estimator = GridSearchCV(DecisionTreeRegressor(), 
                            hyperparameters, 
                            n_jobs=n_jobs,
                            cv=cv)
        else:
            self._estimator = estimator

        if properties is None:
            self._properties = {'estimator_family':[DECISION_TREE_METHODS], 
                                'tasks':[REGRESSION], 
                                'name':'DecisionTreeRegressor'}
        else:
            self._properties = properties
