import itertools

from mlaut.estimators.glm_estimators import (Ridge_Regression, 
                                       Lasso, 
                                       Lasso_Lars, 
                                       Logistic_Regression, 
                                       Passive_Aggressive_Classifier,
                                       Bayesian_Ridge)
from mlaut.estimators.nn_estimators import  Deep_NN_Classifier, Deep_NN_Regressor

from mlaut.estimators.bayes_estimators import Gaussian_Naive_Bayes, Bernoulli_Naive_Bayes
from mlaut.estimators.ensemble_estimators import (Random_Forest_Classifier,
                                            Random_Forest_Regressor,
                                            Bagging_Classifier,
                                            Bagging_Regressor, 
                                            Gradient_Boosting_Classifier,
                                            Gradient_Boosting_Regressor)
from mlaut.estimators.svm_estimators import SVC_mlaut, SVR_mlaut
from mlaut.estimators.baseline_estimators import (Baseline_Regressor,
                                            Baseline_Classifier)
from mlaut.estimators.cluster_estimators import K_Neighbours
from mlaut.estimators.decision_trees import (Decision_Tree_Classifier, Decision_Tree_Regressor)
from mlaut.shared.static_variables import (GRIDSEARCH_NUM_CV_FOLDS, 
                                           GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
                                           VERBOSE)
"""
Each estimator is contained it its own class.
Each estimator inherits from the abstract class MlautEstimator.
Each estimator must implement the following methods:
    1. get_estimator_name
    2. build
    3. save

Each estimator is built for a specific dataset. This is particularly necessary
for Neural Network classification models where the number of nodes in the last
layer depends on the number of classes.

The following methods are implemented in the MlautEstimator Abstract class and 
are inherited by all estimators:
    1. __init__
    2. set_trained_model
    3. get_trained_model

The set/get_trained_model methods are used to facilitate saving 
of the trined models. The pipeline is the following:
    1. The Experiments class evokes the set_trained_model method.
    2. The Test Orchestrator saves the trained model to disk
    3. The Experiments class evokes get_trained_model and makes predictions 
    on the test set. 
"""
def default_estimators(estimator_family=None,
                       task=None,
                       verbose=VERBOSE, 
                       n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
                       num_cv_folds=GRIDSEARCH_NUM_CV_FOLDS, 
                       refit=True):
    """
    Returns default estimators based on criteria

    Parameters
    ----------
    estimator_family: string
        Class of estimators
    task: string
        Accepted inputs are: Classification and Regression

    Returns
    -------
    list:
        list of mlaut estimator objects
    """

    if (estimator_family is None) and (task is None):
        raise ValueError('Provide either an "estimator_familty" or "task"')
    
    all_estimators_array=[
        Logistic_Regression,
        Ridge_Regression,
        Lasso,
        Lasso_Lars,
        Bayesian_Ridge,
        Random_Forest_Classifier,
        Random_Forest_Regressor,
        Bagging_Classifier,
        Bagging_Regressor,
        Gradient_Boosting_Classifier,
        Gradient_Boosting_Regressor,
        SVC_mlaut,
        SVR_mlaut,
        Gaussian_Naive_Bayes,
        Bernoulli_Naive_Bayes,
        Deep_NN_Classifier,
        Deep_NN_Regressor,
        Passive_Aggressive_Classifier,
        Baseline_Classifier,
        Baseline_Regressor,
        K_Neighbours,
        Decision_Tree_Classifier,
        Decision_Tree_Regressor
    ]
    estimators_array = []

    for est in all_estimators_array:
        e = est()
        if estimator_family in e.properties['estimator_family']:
            estimators_array.append(est)
            continue
        
        if task in e.properties['tasks']:
            estimators_array.append(e) 
    return estimators_array