from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GridSearchCV

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Dropout
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from tensorflow.python.keras import optimizers


import types
import tempfile
from tensorflow.python.keras import models as km

from .mleap_estimator import MleapEstimator

from ..shared.static_variables import GRIDSEARCH_CV_NUM_PARALLEL_JOBS


class Random_Forest_Classifier(MleapEstimator):
    def __init__(self, verbose=0, n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS, refit=True):
        self._verbose=verbose
        self._n_jobs=n_jobs
        self._refit=refit

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

class SVC_mleap(MleapEstimator):
    def __init__(self, verbose=0, n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS, refit=True):
        self._verbose=verbose
        self._n_jobs=n_jobs
        self._refit=refit

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

class Logistic_Regression(MleapEstimator):
    def __init__(self, verbose=0, n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS, refit=True):
        self._verbose=verbose
        self._n_jobs=n_jobs
        self._refit=refit

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


class Gaussian_Naive_Bayes(MleapEstimator):
    def build(self):
        return GaussianNB()

class Bernoulli_Naive_Bayes(MleapEstimator):

    def build(self):
        return BernoulliNB()

class Deep_NN_Classifier(MleapEstimator):
    

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
                                loss=loss)
        if hyperparameters is None:
            hyperparameters = {'epochs': [50,100], 'batch_size': [num_samples]}
        return GridSearchCV(model, hyperparameters)


def instantiate_default_estimators(estimators, verbose=0):
    estimators_array = []
    
    if 'RandomForestClassifier' in estimators or 'all' in estimators:
        rfc_model = ['RandomForestClassifier', Random_Forest_Classifier()]
        estimators_array.append(rfc_model)
    
    if 'SVC' in estimators or 'all' in estimators:
        svc_model = ['SVC', SVC_mleap()]
        estimators_array.append(svc_model)

    if 'LogisticRegression' in estimators or 'all' in estimators:
        logisticregression_model = ['LogisticRegression', Logistic_Regression()]
        estimators_array.append(logisticregression_model)

    if 'GaussianNaiveBayes' in estimators or 'all' in estimators:
        gnb = ['GaussianNaiveBayes', Gaussian_Naive_Bayes()]
        estimators_array.append(gnb)

    if 'BernoulliNaiveBayes' in estimators or 'all' in estimators:
        bnb = ['BernoulliNaiveBayes', Bernoulli_Naive_Bayes()]
        estimators_array.append(bnb)

    if 'NeuralNetworkDeepClassifier' in estimators or 'all' in estimators:
        nnd = ['NeuralNetworkDeepClassifier', Deep_NN_Classifier()]
        estimators_array.append(nnd)
    return estimators_array