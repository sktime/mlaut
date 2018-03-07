from abc import ABC, abstractmethod
from mleap.shared.static_variables import GRIDSEARCH_CV_NUM_PARALLEL_JOBS
import pickle
from mleap.shared.static_variables import PICKLE_EXTENTION
import wrapt
class MleapEstimator(ABC):
    """
    Abstact base class that all mleap estimators should inherit from.
    """
    def __init__(self, 
                verbose=0, 
                n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
                num_cv_folds=3, 
                refit=True):
        self._num_cv_folds=num_cv_folds
        self._verbose=verbose
        self._n_jobs=n_jobs
        self._refit=refit
    """
    :type verbose: int
    :param verbose: Sets the amount of output in the terminal. Higher numbers mean more output.

    :type n_jobs: number
    :param n_jobs: number of CPU cores used for training the estimators. If set to -1 all available cores are used.

    :type num_cv_folds: int
    :param num_cv_folds: number of cross validation folds used by GridsearchCV.

    :type refit: Boolean
    :param refit: Refit an estimator using the best found parameters on the whole dataset.
    """
    @abstractmethod
    def build(self):
        """ 
        Abstract method that needs to be implemented by all estimators. 
        It should return an estimator object.
        """
    
    @abstractmethod
    def save(self):
        """ 
        Abstract method that needs to be implemented by all estimators.
        Saves the trained model to disk.
        """
    def load(self, path_to_model):
        """
        Loads trained estimator from disk.
        The default implemented method loads a sklearn estimator from a pickle file. 
        This method needs to be overwritten in the child estimator class if another 
        framework/procedure for saving/loading is used. 

        :type path_to_model: string
        :param path_to_model: Location of the trained estimator.
        """
        #file name could be passed with .* as extention. 
        # split_path = path_to_model.split('.')
        # path_to_load = split_path[0] + PICKLE_EXTENTION 
        model = pickle.load(open(path_to_model,'rb'))
        self.set_trained_model(model)
    
    def set_trained_model(self, trained_model):
        """
        setter method for storing trained estimator in memory

        :type trained_model: estimator object
        :param trained_model: Trained sklearn, keras, etc. estimator object.
        """
        self._trained_model = trained_model
    
    def get_trained_model(self):
        """
        Getter method.

        :rtype: `estimator object`
        """
        return self._trained_model

    # def predict(self, X):
    #     estimator = self.get_trained_model()
    #     return estimator.predict(X)


#decorator for adding properties to estimator classes

class properties(object):
"""
Decorator class used for adding properties to mleap estimator classes.
The properties that all mleap estimator objects must have are: 
estimator family, task (classification, regression), name of estimator. 

"""
    def __init__(self, estimator_family, tasks, name):
        """
        :type estimator_family: array of strings
        :param estimator_family: family of machine learning algorithms that the estimator belongs to.

        :type tasks: array of strings
        :param tasks: array of tasks (classification and/or regression) that the estimator can be applied to.

        :type name: string
        :param name: name of estimator.
        """
        self._estimator_family = estimator_family
        self._tasks = tasks
        self._name = name

    def _properties(self):
        """
        Method attached by the decorator to the mleap estimator object.
        :rtype: `dictionary`

        """            
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
        wrapped.properties = self._properties
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

