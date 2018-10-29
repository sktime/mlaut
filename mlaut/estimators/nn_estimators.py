from mlaut.estimators.mlaut_estimator import MlautEstimator

from mlaut.shared.files_io import DiskOperations
from mlaut.shared.static_variables import(GENERALIZED_LINEAR_MODELS,
                                      ENSEMBLE_METHODS, 
                                      SVM,
                                      NEURAL_NETWORKS,
                                      NAIVE_BAYES,
                                      REGRESSION, 
                                      CLASSIFICATION,
                                      GRIDSEARCH_NUM_CV_FOLDS,
                                      GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
                                      VERBOSE)
from mlaut.shared.static_variables import PICKLE_EXTENTION, HDF5_EXTENTION

from tensorflow.python.keras.models import Sequential, load_model, model_from_json
from tensorflow.python.keras.layers import Dense, Activation, Dropout
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from tensorflow.python.keras import optimizers
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
import numpy as np
import wrapt

import tensorflow as tf



class Deep_NN_Classifier(MlautEstimator):
    # """
    # Wrapper for a `keras sequential model <https://keras.io/getting-started/sequential-model-guide/>`_. 
    # """
    properties = {'estimator_family':[NEURAL_NETWORKS], 
                'tasks':[CLASSIFICATION], 
                'name':'NeuralNetworkDeepClassifier'}
    hyperparameters = {'epochs': 1, 
                        'batch_size': None}
    def keras_model(num_classes, input_dim):
        nn_deep_model = OverwrittenSequentialClassifier()
        nn_deep_model.add(Dense(288, input_dim=input_dim, activation='relu'))
        nn_deep_model.add(Dense(144, activation='relu'))
        nn_deep_model.add(Dropout(0.5))
        nn_deep_model.add(Dense(12, activation='relu'))
        nn_deep_model.add(Dense(num_classes, activation='softmax'))

        model_optimizer = optimizers.Adam(lr=0.001)
        nn_deep_model.compile(loss='mean_squared_error', optimizer=model_optimizer, metrics=['accuracy'])

        return nn_deep_model
        
    def __init__(self,
                properties=properties,
                hyperparameters=hyperparameters, 
                keras_model=keras_model, 
                verbose=VERBOSE, 
                n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
                num_cv_folds=GRIDSEARCH_NUM_CV_FOLDS, 
                refit=True):
        
        self._hyperparameters = hyperparameters
        self._keras_model = keras_model
        self.properties = properties
        self._verbose = verbose
        self._n_jobs = n_jobs
        self._num_cv_folds = num_cv_folds
        self._refit = refit

        if 'epochs' not in self._hyperparameters.keys():
            raise ValueError('You need to specify number of epochs as hyperparameter to keras models')
        if 'batch_size' not in self._hyperparameters.keys():
            raise ValueError('You need to specify batch_size as hyperparameter to keras models')
        

    
    def set_properties(self,
                        estimator_family=None, 
                        tasks=None, 
                        name=None, 
                        data_preprocessing=None):
        """
        Alternative method for setting the properties of the estimator. Used when creating a generic estimator by inehriting from an already created class.

        """
        if estimator_family is not None:
            self._estimator_family = estimator_family
        if tasks is not None:
            self._tasks = tasks
        if name is not None:
            self._name = name
        if data_preprocessing is not None:
            self._data_preprocessing = data_preprocessing
        
    def get_name(self):
        return self._name
    def build(self, **kwargs):
        """
        Builds and returns estimator.
        
        Args:
            kwargs (key-value(int)): The user must specify ``input_dim`` and ``num_samples``.

        Returns:
            `sklearn pipeline` object: pipeline for transforming the features and training the estimator
        """


        if 'input_dim' not in kwargs:
            raise ValueError('You need to specify input dimensions when building the model.')
        if 'num_samples' not in kwargs:
            raise ValueError('You need to specify num_samples when building the keras model.')
        if 'num_classes' not in kwargs:
            raise ValueError('You need to specify num_classes when building the keras model.')

         
        input_dim=kwargs['input_dim']
        num_samples = kwargs['num_samples']
        num_classes = kwargs['num_classes']
        
        
        #the arguments of ``build_fn`` are not passed directly. Instead they should be passed as arguments to ``KerasClassifier``.
        estimator = KerasClassifier(build_fn=self._keras_model, 
                                num_classes=num_classes, 
                                input_dim=input_dim,
                                batch_size=self._hyperparameters['batch_size'], 
                                epochs=self._hyperparameters['epochs'])

        # grid = GridSearchCV(estimator=estimator, 
        #                     param_grid=self._hyperparameters, 
        #                     cv=self._num_cv_folds, 
        #                     refit=self._refit,
        #                     verbose=self._verbose)
        return self._create_pipeline(estimator=estimator)

    

        
    def save(self, dataset_name):
        """
        Saves estimator on disk.

        Args:
            dataset_name (str): name of the dataset. Estimator will be saved under default folder structure `/data/trained_models/<dataset name>/<model name>`
        """
        #set trained model method is implemented in the base class
        trained_model = self._trained_model
        disk_op = DiskOperations()
        disk_op.save_keras_model(trained_model=trained_model,
                                 model_name=self.properties['name'],
                                 dataset_name=dataset_name)
    
    #overloading method from parent class
    def load(self, path_to_model):
        """
        Loads saved keras model from disk.

        Args:
            path_to_model (str): path on disk where the object is saved.
        """
        #file name could be passed with .* as extention. 
        #split_path = path_to_model.split('.')
        #path_to_load = split_path[0] + HDF5_EXTENTION 
        model = load_model(path_to_model,
                           custom_objects={
                               'OverwrittenSequentialClassifier':OverwrittenSequentialClassifier
                               })
        self.set_trained_model(model)

    def get_trained_model(self):
        """
        Getter method.

        Returns:
            `keras object`: Trained keras model.
        """

        return self._trained_model.model





class Deep_NN_Regressor(Deep_NN_Classifier):
    """
    Wrapper for a `keras sequential model <https://keras.io/getting-started/sequential-model-guide/>`_. 
    """
    properties = {'estimator_family':[NEURAL_NETWORKS], 
                'tasks':[REGRESSION], 
                'name':'NeuralNetworkDeepRegressor'}

    def nn_deep_classifier_model(self, input_dim):
        nn_deep_model = Sequential()
        nn_deep_model.add(Dense(288, input_dim=input_dim, activation='relu'))
        nn_deep_model.add(Dense(144, activation='relu'))
        nn_deep_model.add(Dropout(0.5))
        nn_deep_model.add(Dense(12, activation='relu'))
        nn_deep_model.add(Dense(1, activation='sigmoid'))
        

        model_optimizer  = optimizers.Adam(lr=self._hyperparameters['learning_rate'])
        nn_deep_model.compile(loss='mean_squared_error', optimizer=model_optimizer, metrics=['accuracy'])


        return nn_deep_model

    def build(self, **kwargs):
        """
        Builds and returns estimator.
        
        Args:
            kwargs (key-value(int)): The user must specify ``input_dim`` and ``num_samples``.

        Returns:
            `sklearn pipeline` object: pipeline for transforming the features and training the estimator
        """


        if 'input_dim' not in kwargs:
            raise ValueError('You need to specify input dimensions when building the model.')
        if 'num_samples' not in kwargs:
            raise ValueError('You need to specify num_samples when building the keras model.')
        if 'num_classes' not in kwargs:
            raise ValueError('You need to specify num_classes when building the keras model.')

        input_dim=kwargs['input_dim']
        num_samples = kwargs['num_samples']
        num_classes = kwargs['num_classes']
        
        
        #the arguments of ``build_fn`` are not passed directly. Instead they should be passed as arguments to ``KerasClassifier``.
        estimator = KerasRegressor(build_fn=self._keras_model, 
                                input_dim=input_dim)
        grid = GridSearchCV(estimator=estimator, 
                            param_grid=self._hyperparameters, 
                            cv=self._num_cv_folds, 
                            refit=self._refit,
                            verbose=self._verbose)
        return self._create_pipeline(estimator=estimator)
        


#     def __init__(self, verbose=VERBOSE, 
#                 n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
#                 num_cv_folds=GRIDSEARCH_NUM_CV_FOLDS, 
#                 refit=True):
#         super().__init__(verbose=verbose, 
#                          n_jobs=n_jobs, 
#                         num_cv_folds=num_cv_folds, 
#                         refit=refit)
#         self._hyperparameters = {'loss':'mean_squared_error', 
#                                  'learning_rate':0.001,
#                                  'optimizer': 'Adam',
#                                  'metrics': ['accuracy'],
#                                  'epochs': [50,100], 
#                                  'batch_size': 0 }

#     def _nn_deep_classifier_model(self, input_dim):
#         nn_deep_model = Sequential()
#         nn_deep_model.add(Dense(288, input_dim=input_dim, activation='relu'))
#         nn_deep_model.add(Dense(144, activation='relu'))
#         nn_deep_model.add(Dropout(0.5))
#         nn_deep_model.add(Dense(12, activation='relu'))
#         nn_deep_model.add(Dense(1, activation='sigmoid'))
        
#         optimizer = self._hyperparameters['optimizer']
#         if optimizer is 'Adam':
#             model_optimizer  = optimizers.Adam(lr=self._hyperparameters['learning_rate'])
        
#         nn_deep_model.compile(loss=self._hyperparameters['loss'], 
#                               optimizer=model_optimizer, 
#                               metrics=self._hyperparameters['metrics'])
#         return nn_deep_model
    
#     def build(self, **kwargs):
#         """
#         Builds and returns estimator.
        
#         Args:
#             kwargs (key-value(int)): The user must specify ``input_dim`` and ``num_samples``.

#         Returns:
#             `sklearn pipeline` object: pipeline for transforming the features and training the estimator
#         """
#         if 'input_dim' not in kwargs:
#             raise ValueError('You need to specify input dimentions when building the model')
#         if 'num_samples' not in kwargs:
#             raise ValueError('You need to specify num_samples when building the keras model.')
#         input_dim=kwargs['input_dim']
#         num_samples = kwargs['num_samples']
        
#         estimator = KerasRegressor(build_fn=self._nn_deep_classifier_model, 
#                                 input_dim=input_dim,
#                                 verbose=self._verbose)

        
#         return self._create_pipeline(estimator=estimator)
#         # return GridSearchCV(model, 
#         #                     hyperparameters, 
#         #                     verbose = self._verbose,
#         #                     n_jobs=self._n_jobs,
#         #                     refit=self._refit)


#     def save(self, dataset_name):
#         """
#         Saves estimator on disk.

#         Args:
#             dataset_name (str): name of the dataset. Estimator will be saved under default folder structure `/data/trained_models/<dataset name>/<model name>`
#         """
#         #set trained model method is implemented in the base class
#         trained_model = self._trained_model
#         disk_op = DiskOperations()
#         disk_op.save_keras_model(trained_model=trained_model,
#                                  model_name=self.properties['name'],
#                                  dataset_name=dataset_name)
    
#     #overloading method from parent class
#     def load(self, path_to_model):
#         """
#         Loads saved keras model from disk.

#         Args:
#             path_to_model (string): path on disk where the object is saved.
#         """
#         #file name could be passed with .* as extention. 
#         model = load_model(path_to_model,
#                            custom_objects={
#                                'OverwrittenSequentialClassifier':OverwrittenSequentialClassifier
#                                })
#         self.set_trained_model(model)
        

class OverwrittenSequentialClassifier(Sequential):
    """
    Keras sequential model that overrides the default :func:`tensorflow.python.keras.models.fit` and :func:`tensorflow.python.keras.models.predict` methods.
    """


    def fit(self, X_train, y_train, **kwargs):
            
        """
        Overrides the default :func:`tensorflow.python.keras.models.fit` and reshapes the `y_train` in one hot array. 

        Args:
            X_train: training data
            y_train: Labels that will be converted to onehot array.


        Returns:
            :func:`tensorflow.python.keras.models.fit` object

        """
        onehot_encoder = OneHotEncoder(sparse=False)
        len_y = len(y_train)
        reshaped_y = y_train.reshape(len_y, 1)
        y_train_onehot_encoded = onehot_encoder.fit_transform(reshaped_y)
        
        # if 'epochs' not in self._hyperparameters:
        #     epochs = 1
        # else:
        #     epochs = self._hyperparameters

        return super().fit(X_train, 
                            y_train_onehot_encoded, 
                            batch_size=kwargs['batch_size'],
                            epochs=kwargs['epochs'])

        

    def predict(self, X_test, batch_size=None, verbose=VERBOSE):
        """
        Overrides the default :func:`tensorflow.python.keras.models.predict` by replacing it with a :func:`tensorflow.python.keras.models.predict_classes`  

        Returns:
            :func:`tensorflow.python.keras.models.predict_classes`
        """
        predictions = Sequential.predict(self, X_test, batch_size=batch_size, verbose=verbose)
        return predictions.argmax(axis=1)
        # return super().predict_classes(X_test)