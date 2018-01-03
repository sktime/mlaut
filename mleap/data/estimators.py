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
'''
See for tutorial on decorators with parameters
http://www.artima.com/weblogs/viewpost.jsp?thread=240845
https://stackoverflow.com/questions/5929107/decorators-with-parameters
'''
def gridsearch(verbose=1, n_jobs=1, refit=True):
    def real_decorator(function):
        def wrapper(*args):
             
            model, hyperparameters = function()
            gs = GridSearchCV(model, hyperparameters, verbose=verbose, n_jobs=n_jobs, refit=refit)
            return gs
        return wrapper
    return real_decorator

def random_forest_classifier(hyperparameters=None):
    if hyperparameters is None:
        hyperparameters = {
            'n_estimators': [10, 20, 30],
            'max_features': ['auto', 'sqrt','log2', None],
            'max_depth': [5, 15, None]
        }
    return RandomForestClassifier(), hyperparameters

def svc(hyperparameters=None):
    if hyperparameters is None:
        hyperparameters = {
                        'C': [1e-6, 1], #[1e-6, 1e-5, 1e-4,1e-3, 1e-2, 1, 1e2,1e3,1e4,1e5,1e6]
                        'gamma': [1e-3, 1] #[1e-3, 1e-2, 1e-1, 1]
                    }
    return SVC(), hyperparameters

def logisticregression(hyperparameters=None):
    if hyperparameters is None:
        hyperparameters = {
            'C': [1e-6, 1] #[1e-6, 1e-5, 1e-4,1e-3, 1e-2, 1, 1e2,1e3,1e4,1e5,1e6]
        }
    return linear_model.LogisticRegression(), hyperparameters

def gaussian_naive_bayes():
    return GaussianNB()

def multinomial_naive_bayes():
    return MultinomialNB()

def bernoulli_naive_bayes():
    return BernoulliNB()

class Deep_NN_Classifier(MleapEstimator):
    def make_keras_picklable(self):
        """
        Workaround for saving keras models as pickle.
        http://zachmoshe.com/2017/04/03/pickling-keras-models.html
        """
        def __getstate__(self):
            model_str = ""
            with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
                km.save_model(self, fd.name, overwrite=True)
                model_str = fd.read()
            d = { 'model_str': model_str }
            return d

        def __setstate__(self, state):
            with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
                fd.write(state['model_str'])
                fd.flush()
                model = km.load_model(fd.name)
            self.__dict__ = model.__dict__


        cls = km.Model
        cls.__getstate__ = __getstate__
        cls.__setstate__ = __setstate__

    def _nn_deep_classifier_model(self, num_classes, 
                                  input_dim,
                                  loss='mean_squared_error',
                                  optimizer = 'Adam',
                                  metrics = ['accuracy'] ):
        self.make_keras_picklable()
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
    
    def build(self, num_classes, input_dim, loss='mean_squared_error'):
        model = KerasClassifier(build_fn=self._nn_deep_classifier_model, 
                                num_classes=num_classes, 
                                input_dim=input_dim,
                                loss=loss)
        #TODO dynamically set hyperparameters
        hyperparameters = {'epochs': [50,100], 'batch_size': [1865]}
        return GridSearchCV(model, hyperparameters)

def instantiate_default_estimators(estimators, verbose=0):
    estimators_array = []

    decorator_gridsearch = gridsearch(verbose=verbose)
    
    if 'RandomForestClassifier' in estimators or 'all' in estimators:
        rfc_model = ['RandomForestClassifier', decorator_gridsearch(random_forest_classifier)()]
        estimators_array.append(rfc_model)
    
    if 'SVC' in estimators or 'all' in estimators:
        svc_model = ['SVC', decorator_gridsearch(svc)()]
        estimators_array.append(svc_model)

    if 'LogisticRegression' in estimators or 'all' in estimators:
        logisticregression_model = ['LogisticRegression', decorator_gridsearch(logisticregression)()]
        estimators_array.append(logisticregression_model)

    if 'GaussianNaiveBayes' in estimators or 'all' in estimators:
        gnb = ['GaussianNaiveBayes', gaussian_naive_bayes()]
        estimators_array.append(gnb)

    if 'BernoulliNaiveBayes' in estimators or 'all' in estimators:
        bnb = ['BernoulliNaiveBayes', bernoulli_naive_bayes()]
        estimators_array.append(bnb)

    if 'NeuralNetworkDeepClassifier' in estimators or 'all' in estimators:
        nnd = ['NeuralNetworkDeepClassifier', Deep_NN_Classifier()]
        estimators_array.append(nnd)
    return estimators_array