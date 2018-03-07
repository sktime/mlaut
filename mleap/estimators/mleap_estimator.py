from abc import ABC, abstractmethod
from mleap.shared.static_variables import GRIDSEARCH_CV_NUM_PARALLEL_JOBS
import pickle
from mleap.shared.static_variables import PICKLE_EXTENTION
import wrapt
class MleapEstimator(ABC):

    def __init__(self, 
                verbose=0, 
                n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
                num_cv_folds=3, 
                refit=True):
        self._num_cv_folds=num_cv_folds
        self._verbose=verbose
        self._n_jobs=n_jobs
        self._refit=refit

    @abstractmethod
    def build(self):
        """ Returns the estimator and its hyper parameters"""
    
    @abstractmethod
    def save(self):
        """ saves the trained model to disk """
    def load(self, path_to_model):
        #file name could be passed with .* as extention. 
        # split_path = path_to_model.split('.')
        # path_to_load = split_path[0] + PICKLE_EXTENTION 
        model = pickle.load(open(path_to_model,'rb'))
        self.set_trained_model(model)
    
    def set_trained_model(self, trained_model):
        self._trained_model = trained_model
    
    def get_trained_model(self):
        return self._trained_model

    def predict(self, X):
        estimator = self.get_trained_model()
        return estimator.predict(X)


#decorator for adding properties to estimator classes

class properties(object):

    def __init__(self, estimator_family, tasks, name):
        self._estimator_family = estimator_family
        self._tasks = tasks
        self._name = name

    def properties(self):
                    
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

    @wrapt.decorator
    def __call__(self, wrapped, instance, args, kwargs):
        #add properties to class
        wrapped.properties = self.properties
        return wrapped(*args, **kwargs)
        
        # class Wrapped(cls):
 
        #     def properties(cls):
                
        #         #check whether the inputs are right
        #         if not isinstance(self._estimator_family, list) or \
        #             not isinstance(self._tasks, list):
        #             raise ValueError('Arguments to property_decorator must be provided as an array')
        #         properties_dict = {
        #             'estimator_family': self._estimator_family,
        #             'tasks': self._tasks,
        #             'name': self._name
        #         }
        #         return properties_dict
        # return Wrapped

