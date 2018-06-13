from abc import ABC, abstractmethod
from mlaut.shared.static_variables import GRIDSEARCH_CV_NUM_PARALLEL_JOBS
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
    def __init__(self, 
                verbose=0, 
                n_jobs=-1,
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
    
    
    def save(self, dataset_name):
        """
        Saves estimator on disk.

        :type dataset_name: string
        :param dataset_name: name of the dataset. Estimator will be saved under default folder structure `/data/trained_models/<dataset name>/<model name>`
        """
        #set trained model method is implemented in the base class
        trained_model = self._trained_model
        disk_op = DiskOperations()
        disk_op.save_to_pickle(trained_model=trained_model,
                             model_name=self.properties()['name'],
                             dataset_name=dataset_name)
    
    def get_params(self):
        """
        Gets the hyperparaments of the estimator.

        Returns
        -------
            hyperparaments(dictionary): Dictionary with the hyperparaments of the estimator
        """
        return self._hyperparameters
    
    def set_params(self, hyperparameters):
        """
        Set the hyper-parameters of the estimator.

        Parameters
        ----------
        hyperparameters(dictionary): Dictionary with the hyperarameters of each model.
        """
        self._hyperparameters = hyperparameters

    # def _set_params(self, hyperparameters):
    #     """
    #     Private method for setting the hyperparaments of the estimator. It is used by the build(). If the user specified hyperparaments the default values are overwritten. 
    #     """
    #     self._hyperparameters = hyperparameters
    
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

        :rtype: `sklearn pipline object`
        """

        return self._create_pipeline(self._trained_model)

    # def predict(self, X):
    #     estimator = self.get_trained_model()
    #     return estimator.predict(X)

    def _create_pipeline(self, estimator):
        """
        Creates a pipeline for transforming the features of the dataset and training the selected estimator.

        Parameters
        ----------
        estimator: sklearn estimator
            Reference of sklearn estimator that will be used at the end of the pipeline.


        Returns
        -------
            estimator(sklearn pipeline or GridSerachCV): `sklearn` pipeline object. If no preprocessing was set 
        """


        data_preprocessing = self.properties()['data_preprocessing']

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

class properties(object):
    """
    Decorator class used for adding properties to mlaut estimator classes. The properties that all mlaut estimator objects must have are: estimator family, task (classification, regression), name of estimator. 
    """
    def __init__(self, 
        estimator_family, 
        tasks, 
        name, 
        data_preprocessing={'normalize_features': False,
                            'normalize_labels': False}):
        """
        Parameters
        ----------
        estimator_family: array of strings
            family of machine learning algorithms that the estimator belongs to.
        tasks: array of strings
            array of tasks (classification and/or regression) that the estimator can be applied to.
        name: string
            name of estimator.
        data_preprocessing: dictionary
            dictionary with data preprocessing operations to be performed on datasets before they are used in training.
        """
        self._estimator_family = estimator_family
        self._tasks = tasks
        self._name = name
        self._data_preprocessing = data_preprocessing


    def _properties(self):
        """
        Method attached by the decorator to the mlaut estimator object.
        :rtype: `dictionary`

        """            
        #check whether the inputs are right
        if not isinstance(self._estimator_family, list) or \
            not isinstance(self._tasks, list):
            raise ValueError('Arguments to property_decorator must be provided as an array')
        properties_dict = {
            'estimator_family': self._estimator_family,
            'tasks': self._tasks,
            'name': self._name,
            'data_preprocessing': self._data_preprocessing
        }
        return properties_dict

    @wrapt.decorator
    def __call__(self, wrapped, instance, args, kwargs):
        #add/attach properties to class
        wrapped.properties = self._properties
        return wrapped(*args, **kwargs)
        
        

