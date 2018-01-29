import itertools

from mleap.data.glm_estimators import (Ridge_Regression, 
                                       Lasso, 
                                       Lasso_Lars, 
                                       Logistic_Regression, 
                                       Passive_Aggressive_Classifier)
from mleap.data.nn_estimators import  Deep_NN_Classifier, Deep_NN_Regressor
from mleap.data.bayes_estimators import Gaussian_Naive_Bayes, Bernoulli_Naive_Bayes
from mleap.data.ensemble_estimators import (Random_Forest_Classifier,
                                            Random_Forest_Regressor,
                                            Bagging_Classifier,
                                            Bagging_Regressor, 
                                            Gradient_Boosting_Classifier,
                                            Gradient_Boosting_Regressor)
from mleap.data.svm_estimators import SVC_mleap

"""
Each estimator is coitained it its own class.
Each estimator inherits from the abstract class MleapEstimator.
Each estimator must implement the following methods:
    1. get_estimator_name
    2. build
    3. save

Each estimator is built for a specific dataset. This is particularly necessary
for Neural Network classification models where the number of nodes in the last
layer depends on the number of classes.

The following methods are implemented in the MleapEstimator Abstract class and 
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






def instantiate_default_estimators(estimators, verbose=0):

    if not isinstance(estimators, list):
        raise ValueError('Estimators parameter must be provided as an array')
    
    all_estimators_array=[
        Logistic_Regression,
        Ridge_Regression,
        Lasso,
        Lasso_Lars,
        Random_Forest_Classifier,
        Random_Forest_Regressor,
        Bagging_Classifier,
        Bagging_Regressor,
        Gradient_Boosting_Classifier,
        Gradient_Boosting_Regressor,
        SVC_mleap,
        Gaussian_Naive_Bayes,
        Bernoulli_Naive_Bayes,
        Deep_NN_Classifier,
        Deep_NN_Regressor,
        Passive_Aggressive_Classifier
    ]
    estimators_array = []



    if 'all' in estimators:
        for est in all_estimators_array:
            estimators_array.append(est())
    else:
        perms = itertools.product(estimators, all_estimators_array)
        for p in perms:
            input_estimator = p[0]
            mleap_estimator = p[1]
            mleap_estimator_prop = mleap_estimator().properties()

            if input_estimator in mleap_estimator_prop['estimator_family'] or \
                input_estimator in mleap_estimator_prop['tasks'] or \
                input_estimator in mleap_estimator_prop['name']:
                
                estimators_array.append(mleap_estimator(verbose=verbose))
    if len(estimators_array) > 0:             
        return estimators_array
    else:
        raise ValueError('Empty Estimator Array')