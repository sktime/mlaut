from abc import ABC, abstractmethod
from mlaut.shared.static_variables import (GRIDSEARCH_CV_NUM_PARALLEL_JOBS, 
                                            GRIDSEARCH_NUM_CV_FOLDS,
                                            VERBOSE)
import pickle
from mlaut.shared.static_variables import PICKLE_EXTENTION
import wrapt
from mlaut.shared.files_io import DiskOperations

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
import logging


class MlautEstimator(ABC):
    """
    Abstact base class that all mlaut estimators should inherit from.
    """
    # def __init__(self, 
    #             verbose=VERBOSE, 
    #             n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
    #             num_cv_folds=GRIDSEARCH_NUM_CV_FOLDS, 
    #             refit=True):
    #     self._num_cv_folds=num_cv_folds
    #     self._verbose=verbose
    #     self._n_jobs=n_jobs
    #     self._refit=refit
    # """
    # Args:
    #     verbose(int): Sets the amount of output in the terminal. Higher numbers mean more output.
    #     n_jobs(number): number of CPU cores used for training the estimators. If set to -1 all available cores are used.
    #     num_cv_folds(int): number of cross validation folds used by GridsearchCV.
    #     refit(Boolean): Refit an estimator using the best found parameters on the whole dataset.
    # """
    @abstractmethod
    def build(self):
        """ 
        Abstract method that needs to be implemented by all estimators. 
        It should return an estimator object.
        """
    
    
    def save(self, dataset_name):
        """
        Saves estimator on disk.

        Args:
            dataset_name(string): name of the dataset. Estimator will be saved under default folder structure `/data/trained_models/<dataset name>/<model name>`
        """
        #set trained model method is implemented in the base class
        trained_model = self._trained_model
        disk_op = DiskOperations()
        disk_op.save_to_pickle(trained_model=trained_model,
                             model_name=self.properties['name'],
                             dataset_name=dataset_name)
    
    def get_params(self):
        """
        Gets the hyperparaments of the estimator.

        Returns:
            dictionary: Dictionary with the hyperparaments of the estimator
        """
        return self._hyperparameters
    
    def set_params(self, hyperparameters):
        """
        Set the hyper-parameters of the estimator.

        Args:
            hyperparameters(dictionary): Dictionary with the hyperarameters of each model.
        """
        self._hyperparameters = hyperparameters
    
    def load(self, path_to_model):
        """
        Loads trained estimator from disk.
        The default implemented method loads a sklearn estimator from a pickle file. 
        This method needs to be overwritten in the child estimator class if another 
        framework/procedure for saving/loading is used. 

        Args:
            path_to_model (str): Location of the trained estimator.
        """
        #file name could be passed with .* as extention. 
        # split_path = path_to_model.split('.')
        # path_to_load = split_path[0] + PICKLE_EXTENTION 
        model = pickle.load(open(path_to_model,'rb'))
        self.set_trained_model(model)
    
    def set_trained_model(self, trained_model):
        """
        setter method for storing trained estimator in memory
        
        Args:
            trained_model (estimator object): Trained sklearn, keras, etc. estimator object.
        """
        self._trained_model = trained_model
    
    def get_trained_model(self):
        """
        Getter method.

        Returns:
            `sklearn pipline object`: Trained sklearn model
        """

        return self._create_pipeline(self._trained_model)

    # def properties(self):
    #     return self._properties

    def _create_pipeline(self, estimator):
        """
        Creates a pipeline for transforming the features of the dataset and training the selected estimator.

        Args:
            estimator (sklearn estimator): Reference of sklearn estimator that will be used at the end of the pipeline.


        Returns:
            `estimator(sklearn pipeline or GridSerachCV)`: `sklearn` pipeline object. If no preprocessing was set 
        """

        if 'data_preprocessing' in self.properties:
            data_preprocessing = self.properties['data_preprocessing']

            if data_preprocessing['normalize_labels'] is True:
                pipe = Pipeline(
                    memory=None,
                    steps=[
                        ('standardscaler', preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True) ),
                        ('estimator', estimator)
                        ]
                )
                return pipe
        else:
            return estimator


#decorator for adding properties to estimator classes

# class properties(object):
#     """
#     Decorator class used for adding properties to mlaut estimator classes. The properties that all mlaut estimator objects must have are: estimator family, task (classification, regression), name of estimator. 

#     The decorator attached a `properties()` method to the class which invokes it 
#     """
#     def __init__(self, 
#         estimator_family, 
#         tasks, 
#         name, 
#         data_preprocessing={'normalize_features': False,
#                             'normalize_labels': False}):
#         """
#         Args:
#             estimator_family (array of strings): family of machine learning algorithms that the estimator belongs to.
            
#             tasks (array of strings): array of tasks (classification and/or regression) that the estimator can be applied to.
            
#             name (str): name of estimator.
            
#             data_preprocessing (dictionary): dictionary with data preprocessing operations to be performed on datasets before they are used in training.
#         """
#         self._estimator_family = estimator_family
#         self._tasks = tasks
#         self._name = name
#         self._data_preprocessing = data_preprocessing


#     def _properties(self):
#         """
#         Method attached by the decorator to the mlaut estimator object.

#         Returns:
#             `dictionary`: Dictionary with the propoperties of the estimator.

#         """            
#         #check whether the inputs are right
#         if not isinstance(self._estimator_family, list) or \
#             not isinstance(self._tasks, list):
#             raise ValueError('Arguments to property_decorator must be provided as an array')
#         properties_dict = {
#             'estimator_family': self._estimator_family,
#             'tasks': self._tasks,
#             'name': self._name,
#             'data_preprocessing': self._data_preprocessing
#         }
#         return properties_dict

#     def _set_properties(self,
#                         estimator_family=None, 
#                         tasks=None, 
#                         name=None, 
#                         data_preprocessing=None):
#         """
#         Alternative method for setting the properties of the estimator. Used when creating a generic estimator by inehriting from an already created class.

#         """
#         if estimator_family is not None:
#             self._estimator_family = estimator_family
#         if tasks is not None:
#             self._tasks = tasks
#         if name is not None:
#             self._name = name
#         if data_preprocessing is not None:
#             self._data_preprocessing = data_preprocessing
        
#     @wrapt.decorator
#     def __call__(self, wrapped, instance, args, kwargs):
#         #add/attach properties to class
#         wrapped.properties = self._properties
#         # wrapped.set_properties = self._set_properties
#         return wrapped(*args, **kwargs)
        
        

