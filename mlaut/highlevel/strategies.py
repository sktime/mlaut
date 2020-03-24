from sktime.highlevel.strategies import CLASSIFIER_TYPES, REGRESSOR_TYPES
from joblib import dump, load
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import BaseEstimator
import pandas as pd
from mlaut.estimators.base import BaseClassifier, BaseRegressor
import inspect
from mlaut.shared.static_variables import HDF5_EXTENTION
from tensorflow.python.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import warnings

REGRESSOR_TYPES = BaseRegressor
CLASSIFIER_TYPES = BaseClassifier
ESTIMATOR_TYPES = [REGRESSOR_TYPES, CLASSIFIER_TYPES]


class BaseStrategy(BaseEstimator):
    """
    Abstract base strategy class.

    Implements attributes and operations shared by all strategies,
    including input and compatibility checks between passed estimator,
    data and task.
    """

    def __init__(self, estimator, check_input):
        self._estimator = estimator
        self.check_input = check_input
        self._task = None
        self._check_estimator_compatibility(estimator)


    @property
    def name(self):
        """Makes attribute accessible, but read-only.
        """
        return self._name

    @property
    def estimator(self):
        """Makes attribute accessible, but read-only.
        """
        return self._estimator

    def __getitem__(self, key):
        """
        Provide read only access via keys to the private traits
        """
        if key not in self._traits.keys():
            raise KeyError
        return self._traits[key]

    def fit(self, task, data):
        """
        Fit the strategy to the given task and data.

        Parameters
        ----------
        task : Task
            Task encapsualting metadata information on feature and target variables to which to fit the data to.
        data : pandas.DataFrame
            Dataframe with feature and target variables as specified in task.

        Returns
        -------
        self : an instance of the self
        """
        if self.check_input:
            self._validate_data(data)

        # Check task compatibility with strategy
        self._check_task_compatibility(task)
        self._task = task

        # Set metadata if not already set
        if self._task.metadata is None:
            self._task.set_metadata(data)

        # strategy-specific implementation
        return self._fit(data)

    def _check_task_compatibility(self, task):
        """
        Check compatibility of task with strategy
        """
        # TODO replace by task-strategy compatibility lookup registry
        if hasattr(task, '_case'):
            if self._case != task._case:
                raise ValueError("Strategy <-> task mismatch: The chosen strategy is incompatible with the given task")
        else:
            raise AttributeError("The passed case of the task is unknown")

    def _check_estimator_compatibility(self, estimator):
        """
        Check compatibility of estimator with strategy
        """

        # Determine required estimator type from strategy case
        # TODO replace with strategy - estimator type registry lookup
        if hasattr(self, '_traits'):
            required = self._traits["required_estimator_type"]
            if required not in ESTIMATOR_TYPES:
                raise AttributeError(f"Required estimator type unknown")
        else:
            raise AttributeError(f"Required estimator type not found")
        
        # TODO perform checks for passed estimator to ensure that it is compatible with the Orchestrator

        
    @staticmethod
    def _validate_data(data):
        """
        Helper function to validate input data.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"Data must be pandas DataFrame, but found: {type(data)}")

        # TODO add input checks for contents, ie all cells be pandas Series, numpy arrays or primitives,

    def save(self, path):
        dump(self, path)

    def load(self, path):
        """
        Load saved strategy
        Parameters
        ----------
        path: String
            location on disk where the strategy was saved

        Returns
        -------
        strategy:
            sktime strategy
        """
        return load(path)

    def __repr__(self):
        strategy_name = self.__class__.__name__
        estimator_name = self.estimator.__class__.__name__
        return '%s(%s(%s))' % (strategy_name, estimator_name,
                               _pprint(self.get_params(deep=False), offset=len(strategy_name), ),)


class BaseSupervisedLearningStrategy(BaseStrategy):
    """Abstract strategy class for cross section supervised learning that accepts a low-level estimator to
    perform a given task.

    Implements predict and internal fit methods for time series regression and classification.
    """

    def _fit(self, data):
        """
        Internal fit

        Parameters
        ----------
        data : pandas.DataFrame
            Dataframe with feature and target variables as specified in task.


        Returns
        -------
        self : an instance of self
        """
        # select features and target
        X = data[self._task.features]
        y = data[self._task.target]

        # fit the estimator
        return self.estimator.fit(X, y)

    def predict(self, data):
        """
        Predict using the given test data.

        Parameters
        ----------
        data : a pandas.DataFrame
            Dataframe with feature and target variables as specified in task passed to ``fit``.


        Returns
        -------
        y_pred : pandas.Series
            Returns the series of predicted values.
        """

        # select features
        X = data[self._task.features]

        # predict
        return self.estimator.predict(X)
    
    @property
    def param_grid(self):
        try:
            return self._estimator.param_grid
        except:
            raise Exception(f'{self._name} estimator has no param_grid property.')
    
    @param_grid.setter
    def param_grid(self, param_grid):
        try:
            self._estimator.param_grid = param_grid
        except:
            raise Exception(f'{self._name} estimator has no param_grid property.')

class CSCStrategy(BaseSupervisedLearningStrategy):
    """
    Cross Section Classification strategy.

    Parameters
    ----------
    estimator : an estimator
        Low-level estimator used in strategy.
    name : str, optional (default=None)
        Name of strategy. If None, class name of estimator is used.
    check_input : bool, optional (default=True)
        - If True, input are checked.
        - If False, input are not checked and assumed correct. Use with caution.
    """
    def __init__(self, name=None, **kwargs):
        self._case = "CSC"
        self._traits = {"required_estimator_type": CLASSIFIER_TYPES }
        self._name = name
        super(CSCStrategy, self).__init__(**kwargs)

    def save(self, path):
        # TODO this method is implemented in sktime.highlevel.strategies.BaseStrategy
        # however saving fails if we don't reimplement it here
        dump(self, path)
    
    

class CSRStrategy(BaseSupervisedLearningStrategy):
    """
    Cross Section Regression Strategy.

    Parameters
    ----------
    estimator : an estimator
        Low-level estimator used in strategy.
    name : str, optional (default=None)
        Name of strategy. If None, class name of estimator is used.
    check_input : bool, optional (default=True)
        - If True, input are checked.
        - If False, input are not checked and assumed correct. Use with caution.
    """
    def __init__(self, name=None, **kwargs):
        print(f'****init CSR {name}')
        self._case = "CSC"
        self._traits = {"required_estimator_type": REGRESSOR_TYPES }
        self._name = name
        super(CSRStrategy, self).__init__(**kwargs)

    def save(self, path):
        # TODO this method is implemented in sktime.highlevel.strategies.BaseStrategy
        # however saving fails if we don't reimplement it here
        dump(self, path)


    
class CSCKerasStrategy(BaseStrategy):
    """
    Cross Section Classification strategy for kears classifiers.

    Parameters
    ----------
    build_fn : keras model
        keras model 
    param_grid : dict
        dictionary with the hyper parameters
    name : str, optional (default=None)
        Name of strategy. If None, class name of estimator is used.
    check_input : bool, optional (default=True)
        - If True, input are checked.
        - If False, input are not checked and assumed correct. Use with caution.
    """
    _case = "CSC"
    _traits = {"required_estimator_type": CLASSIFIER_TYPES }
    


    def __init__(self, estimator, build_fn, param_grid, name, check_input):

        self._build_fn = build_fn
        self._param_grid = param_grid
        self._name = name
        super().__init__(estimator=estimator, check_input=check_input)
    
    @classmethod
    def _get_param_names(cls):
        """ Overwriting BaseEstimator_get_param_names()
        Needed in order to perform clone of the strategy in the orchestrator
        TODO figure out a way to either call the BaseEstimator method here or automate the inpsection of the signature of __init__ in this class.
        """
        
        return ['build_fn', 'check_input', 'estimator', 'name', 'param_grid']
    
    def get_params(self, deep=True):
        """Overwriting BaseEstimator.get_params()
        Needed in order to perform clone of the strategy in the orchestrator
        TODO figure out a way to either call the BaseEstimator method here or automate the return of the dictinary based on the output of self._get_param_names()
        """
        
        params = {
            'build_fn': self._build_fn,
            'check_input': self.check_input,
            'estimator': self._estimator,
            'name': self._name,
            'param_grid': self._param_grid
        }

        return params
        
    def set_params(self, **params):
        """Overwriting BsaeEstimator.set_params()
        Needed in order to perform clone of the strategy in the orchestrator
        
        """
        self._build_fn = params['build_fn']
        self.check_input = params['check_input']
        self._estimator = params['estimator']
        self._name = params['name']
        self._param_grid = params['param_grid']

        return self
    
    def _fit(self,data):
        X = data[self._task.features]
        y = data[self._task.target]
        onehot_encoder = OneHotEncoder(sparse=False)
        reshaped_y = np.array(y).reshape(len(y), 1)
        y_onehot_encoded = onehot_encoder.fit_transform(reshaped_y)
        input_dim = X.shape[1]
        num_classes = len(y.unique())
        keras_classifier = self.estimator(build_fn=self._build_fn, 
                                num_classes=num_classes, 
                                input_dim=input_dim,
                                batch_size=self._param_grid['batch_size'], 
                                epochs=self._param_grid['epochs'])
        self._keras_classifier = keras_classifier
        self._fitted_keras_classifier =  keras_classifier.fit(X,y_onehot_encoded)
        return self._fitted_keras_classifier

    def predict(self, data):
        """
        Predict using the given test data.

        Parameters
        ----------
        data : a pandas.DataFrame
            Dataframe with feature and target variables as specified in task passed to ``fit``.


        Returns
        -------
        y_pred : pandas.Series
            Returns the series of predicted values.
        """

        # select features
        X = data[self._task.features]

        # predict
        return self._keras_classifier.predict(X)

    def save(self, path):
        split_path = path.split('.')
        path_to_save = split_path[0] + HDF5_EXTENTION 
        self._fitted_keras_classifier.model.save(path_to_save)

    def load(self, path):
        split_path = path.split('.')
        path_to_load = split_path[0] + HDF5_EXTENTION
        loaded_keras_estimator = load_model(path_to_load)
        self._fitted_keras_classifier = loaded_keras_estimator

