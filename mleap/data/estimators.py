import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GridSearchCV

from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Activation, Dropout
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from tensorflow.python.keras import optimizers

import pickle
import types
import tempfile
from tensorflow.python.keras import models as km

from .mleap_estimator import MleapEstimator

from ..shared.static_variables import GRIDSEARCH_CV_NUM_PARALLEL_JOBS
from ..shared.files_io import DiskOperations

from ..shared.static_variables import PICKLE_EXTENTION, HDF5_EXTENTION

from ..shared.static_variables import(GENERALIZED_LINEAR_MODELS,
                                      ENSEMBLE_METHODS, 
                                      SVM,
                                      NEURAL_NETWORKS,
                                      NAIVE_BAYES,
                                      REGRESSION, 
                                      CLASSIFICATION)
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

#decorator for adding properties to estimator classes
class properties(object):
    def __init__(self, estimator_family, tasks, name):
        self._estimator_family = estimator_family
        self._tasks = tasks
        self._name = name
    
    
    def __call__(self, cls):

        class Wrapped(cls):

            def properties(cls):
                #check whether the inputs are right
                if not isinstance(self._estimator_family, list) or \
                    not isinstance(self._tasks, list):
                    raise ValueError('Arguments to property_decorator must be provided as an array')
                properties_dict = {
                    'estimator_family': self._estimator_family,
                    'tasks': self._tasks,
                    'name': self._name
                }
                return properties_dict
        return Wrapped


"""
************************************************************************
BEGIN: Generalized Linear Models
************************************************************************
"""
    
@properties(estimator_family=[GENERALIZED_LINEAR_MODELS], 
            tasks=[CLASSIFICATION,REGRESSION], 
            name='RidgeRegression')
class Ridge_Regression(MleapEstimator):

    def __init__(self):
        super().__init__()
       
    def build(self, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = {'alphas':[0.1, 1, 10.0],
            
            } # this is the alpha hyperparam
        
        return linear_model.RidgeCV(alphas=hyperparameters['alphas'],
                                cv=self._num_cv_folds)
    def save(self, dataset_name):
        disk_op = DiskOperations()
        disk_op.save_to_pickle(trained_model=self._trained_model,
                                model_name=self.properties()['name'],
                                dataset_name=dataset_name)
@properties(estimator_family=[GENERALIZED_LINEAR_MODELS], 
            tasks=[CLASSIFICATION,REGRESSION], 
            name='Lasso')
class Lasso(MleapEstimator):
    def __init__(self):
        super().__init__()
    
    def build(self, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = {'alphas':[0.1, 1, 10.0]}
        return linear_model.LassoCV(alphas=hyperparameters['alphas'],
                                    cv=self._num_cv_folds)
    def save(self, dataset_name):
        disk_op = DiskOperations()
        disk_op.save_to_pickle(trained_model=self._trained_model,
                                model_name=self.properties()['name'],
                                dataset_name=dataset_name)
@properties(estimator_family=[GENERALIZED_LINEAR_MODELS], 
            tasks=[CLASSIFICATION,REGRESSION], 
            name='LassoLars')
class Lasso_Lars(MleapEstimator):
    def __init__(self):
        super().__init__()
    

    def build(self, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = {'max_n_alphas':1000}
        return linear_model.LassoLarsCV(max_n_alphas=hyperparameters['max_n_alphas'],
                                    cv=self._num_cv_folds)
    def save(self, dataset_name):
        disk_op = DiskOperations()
        disk_op.save_to_pickle(trained_model=self._trained_model,
                                model_name=self.properties()['name'],
                                dataset_name=dataset_name)

@properties(estimator_family=[GENERALIZED_LINEAR_MODELS], 
            tasks=[CLASSIFICATION,REGRESSION], 
            name='LogisticRegression')
class Logistic_Regression(MleapEstimator):
    def __init__(self):
        super().__init__()

    def build(self, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = {
                'C': [1e-6, 1] #[1e-6, 1e-5, 1e-4,1e-3, 1e-2, 1, 1e2,1e3,1e4,1e5,1e6]
            }
        return GridSearchCV(linear_model.LogisticRegression(), 
                            hyperparameters, 
                            verbose = self._verbose,
                            n_jobs=self._n_jobs,
                            refit=self._refit)
    
    def save(self, dataset_name):
        #set trained model method is implemented in the base class
        trained_model = self._trained_model
        disk_op = DiskOperations()
        disk_op.save_to_pickle(trained_model=trained_model,
                             model_name=self.properties()['name'],
                             dataset_name=dataset_name)

@properties(estimator_family=[GENERALIZED_LINEAR_MODELS],
            tasks=[CLASSIFICATION],
            name='PassiveAggressiveClassifier')
class Passive_Aggressive_Classifier(MleapEstimator):
    def __init__(self):
        super().__init__()
    def build(self, hyperparameters=None):
        hyperparameters = {
                'C': [1e-6, 1] #[1e-6, 1e-5, 1e-4,1e-3, 1e-2, 1, 1e2,1e3,1e4,1e5,1e6]
            }
        return GridSearchCV(linear_model.PassiveAggressiveClassifier(), 
                            hyperparameters, 
                            verbose=self._verbose
                            )

    
    def save(self, dataset_name):
        trained_model = self._trained_model
        disk_op = DiskOperations()
        disk_op.save_to_pickle(trained_model=trained_model,
                             model_name=self.properties()['name'],
                             dataset_name=dataset_name)
"""
************************************************************************
BEGIN: Generalized Linear Models
************************************************************************
"""


"""
************************************************************************
BEGIN: Ensemble methods
************************************************************************
"""
@properties(estimator_family=[ENSEMBLE_METHODS], 
            tasks=[CLASSIFICATION,REGRESSION], 
            name='RandomForestClassifier')
class Random_Forest_Classifier(MleapEstimator):

    def __init__(self):
        super().__init__()

    def build(self, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = {
                'n_estimators': [10, 20, 30],
                'max_features': ['auto', 'sqrt','log2', None],
                'max_depth': [5, 15, None]
            }
        return GridSearchCV(RandomForestClassifier(), 
                            hyperparameters, 
                            verbose = self._verbose,
                            n_jobs=self._n_jobs,
                            refit=self._refit)
    


    def save(self, dataset_name):
        #set trained model method is implemented in the base class
        trained_model = self._trained_model
        disk_op = DiskOperations()
        disk_op.save_to_pickle(trained_model=trained_model,
                             model_name=self.properties()['name'],
                             dataset_name=dataset_name)
        

@properties(estimator_family=[SVM], 
            tasks=[CLASSIFICATION], 
            name='SVC')
class SVC_mleap(MleapEstimator):
    def __init__(self):
        super().__init__()

    def build(self, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = {
                            'C': [1e-6, 1], #[1e-6, 1e-5, 1e-4,1e-3, 1e-2, 1, 1e2,1e3,1e4,1e5,1e6]
                            'gamma': [1e-3, 1] #[1e-3, 1e-2, 1e-1, 1]
                        }
        return GridSearchCV(SVC(), 
                            hyperparameters, 
                            verbose = self._verbose,
                            n_jobs=self._n_jobs,
                            refit=self._refit)


    def save(self, dataset_name):
        #set trained model method is implemented in the base class
        trained_model = self._trained_model
        disk_op = DiskOperations()
        disk_op.save_to_pickle(trained_model=trained_model,
                             model_name=self.properties()['name'],
                             dataset_name=dataset_name)


@properties(estimator_family=[NAIVE_BAYES], 
            tasks=[CLASSIFICATION], 
            name='GaussianNaiveBayes')
class Gaussian_Naive_Bayes(MleapEstimator):


    def save(self, dataset_name):
        #set trained model method is implemented in the base class
        trained_model = self._trained_model
        disk_op = DiskOperations()
        disk_op.save_to_pickle(trained_model=trained_model,
                             model_name=self.properties()['name'],
                             dataset_name=dataset_name)
    def build(self):
        return GaussianNB()

@properties(estimator_family=[NAIVE_BAYES], 
            tasks=[CLASSIFICATION], 
            name='BernoulliNaiveBayes')
class Bernoulli_Naive_Bayes(MleapEstimator):

    def save(self, dataset_name):
        #set trained model method is implemented in the base class
        trained_model = self._trained_model
        disk_op = DiskOperations()
        disk_op.save_to_pickle(trained_model=trained_model,
                             model_name=self.properties()['name'],
                             dataset_name=dataset_name)
    def build(self):
        return BernoulliNB()

@properties(estimator_family=[NEURAL_NETWORKS], 
            tasks=[CLASSIFICATION], 
            name='NeuralNetworkDeepClassifier')
class Deep_NN_Classifier(MleapEstimator):
    def __init__(self):
        super().__init__()

    def _nn_deep_classifier_model(self, num_classes, 
                                  input_dim,
                                  loss='mean_squared_error',
                                  optimizer = 'Adam',
                                  metrics = ['accuracy'] ):
        nn_deep_model = Sequential()
        nn_deep_model.add(Dense(288, input_dim=input_dim, activation='relu'))
        nn_deep_model.add(Dense(144, activation='relu'))
        nn_deep_model.add(Dropout(0.5))
        nn_deep_model.add(Dense(12, activation='relu'))
        nn_deep_model.add(Dense(num_classes, activation='softmax'))
        
        
        if optimizer is 'Adam':
            model_optimizer  = optimizers.Adam(lr=0.001)
        
        nn_deep_model.compile(loss=loss, optimizer=model_optimizer, metrics=metrics)
        return nn_deep_model
    
    def build(self, num_classes, input_dim, num_samples, loss='mean_squared_error', hyperparameters = None):
        model = KerasClassifier(build_fn=self._nn_deep_classifier_model, 
                                num_classes=num_classes, 
                                input_dim=input_dim,
                                verbose=self._verbose,
                                loss=loss)
        if hyperparameters is None:
            hyperparameters = {'epochs': [50,100], 'batch_size': [num_samples]}
        return model
        # return GridSearchCV(model, 
        #                     hyperparameters, 
        #                     verbose = self._verbose,
        #                     n_jobs=self._n_jobs,
        #                     refit=self._refit)


    def save(self, dataset_name):
        #set trained model method is implemented in the base class
        trained_model = self._trained_model
        disk_op = DiskOperations()
        disk_op.save_keras_model(trained_model=trained_model,
                                 model_name=self.properties()['name'],
                                 dataset_name=dataset_name)
    
    #overloading method from parent class
    def load(self, path_to_model):
        #file name could be passed with .* as extention. 
        split_path = path_to_model.split('.')
        path_to_load = split_path[0] + HDF5_EXTENTION 
        model = load_model(path_to_load)
        self.set_trained_model(model)

def instantiate_default_estimators(estimators, verbose=0):

    if not isinstance(estimators, list):
        raise ValueError('Estimators parameter must be provided as an array')
    
    all_estimators_array=[
        Logistic_Regression,
        Ridge_Regression,
        Lasso,
        Lasso_Lars,
        Random_Forest_Classifier,
        SVC_mleap,
        Gaussian_Naive_Bayes,
        Bernoulli_Naive_Bayes,
        Deep_NN_Classifier,
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
                
                estimators_array.append(mleap_estimator()) 
    return estimators_array