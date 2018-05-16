from mlaut.estimators.mlaut_estimator import properties
from mlaut.estimators.mlaut_estimator import MlautEstimator

from mlaut.shared.files_io import DiskOperations
from mlaut.shared.static_variables import(GENERALIZED_LINEAR_MODELS,
                                      ENSEMBLE_METHODS, 
                                      SVM,
                                      NEURAL_NETWORKS,
                                      NAIVE_BAYES,
                                      REGRESSION, 
                                      CLASSIFICATION)
from mlaut.shared.static_variables import PICKLE_EXTENTION, HDF5_EXTENTION

from tensorflow.python.keras.models import Sequential, load_model, model_from_json
from tensorflow.python.keras.layers import Dense, Activation, Dropout
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from tensorflow.python.keras import optimizers
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import wrapt

import tensorflow as tf

@properties(estimator_family=[NEURAL_NETWORKS], 
            tasks=[CLASSIFICATION], 
            name='NeuralNetworkDeepClassifier')
class Deep_NN_Classifier(MlautEstimator):
    """
    Wrapper for a `keras sequential model <https://keras.io/getting-started/sequential-model-guide/>`_. 
    """
    @wrapt.decorator
    def classification_decorator(self, keras_model, instance, args, kwargs):
        learning_rate = self._hyperparameters['learning_rate']
        loss=self._hyperparameters['loss'] 
        metrics=self._hyperparameters['metrics']
        optimizer = self._hyperparameters['optimizer']
        if optimizer is 'Adam':
            model_optimizer = optimizers.Adam(lr=learning_rate)
        model = keras_model(num_classes=kwargs['num_classes'], input_dim=kwargs['input_dim'])
        model.compile(loss=loss, optimizer=model_optimizer, metrics=metrics)
        return model


        
    def __init__(self,
                hyperparameters=None, 
                keras_model=None, 
                verbose=0, 
                n_jobs=-1,
                num_cv_folds=3, 
                refit=True):
        super().__init__(verbose=verbose, 
                         n_jobs=n_jobs, 
                        num_cv_folds=num_cv_folds, 
                        refit=refit)
        if hyperparameters is None:
            self._hyperparameters = {'epochs': [50,100], 
                                    'batch_size': 0,  
                                    'learning_rate':0.001,
                                    'loss': 'mean_squared_error',
                                    'optimizer': 'Adam',
                                    'metrics' : ['accuracy']}
        else:
            self._hyperparameters = hyperparameters

        if keras_model is None:
            #default keras model for classification tasks
            def keras_model(num_classes, input_dim):
                nn_deep_model = OverwrittenSequentialClassifier()
                nn_deep_model.add(Dense(288, input_dim=input_dim, activation='relu'))
                nn_deep_model.add(Dense(144, activation='relu'))
                nn_deep_model.add(Dropout(0.5))
                nn_deep_model.add(Dense(12, activation='relu'))
                nn_deep_model.add(Dense(num_classes, activation='softmax'))
                return nn_deep_model
            self._keras_model = self.classification_decorator(keras_model)
        else:
            self._keras_model = self.classification_decorator(keras_model)
    

    def build(self, **kwargs):
        """
        builds and returns estimator


        :type loss: string
        :param loss: loss metric as per `keras documentation <https://keras.io/losses/>`_.

        :type kwargs: key-value
        :param kwargs: At a minimum the user must specify ``input_dim``, ``num_samples`` and ``num_classes``.
        :rtype: `keras object`
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
        
        #TODO implement cross validation and hyperameters
        # https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/
        
        #the arguments of ``build_fn`` are not passed directly. Instead they should be passed as arguments to ``KerasClassifier``.
        model = KerasClassifier(build_fn=self._keras_model, 
                                num_classes=num_classes, 
                                input_dim=input_dim)#TODO include flag for verbosity

        return model


    

        
    def save(self, dataset_name):
        """
        Saves estimator on disk.

        :type dataset_name: string
        :param dataset_name: name of the dataset. Estimator will be saved under default folder structure `/data/trained_models/<dataset name>/<model name>`
        """
        #set trained model method is implemented in the base class
        trained_model = self._trained_model
        disk_op = DiskOperations()
        disk_op.save_keras_model(trained_model=trained_model,
                                 model_name=self.properties()['name'],
                                 dataset_name=dataset_name)
    
    #overloading method from parent class
    def load(self, path_to_model):
        #TODO this does not seem to work
        """
        Loads saved keras model from disk.

        :type path_to_model: string
        :param path_to_model: path on disk where the object is saved.
        """
        #file name could be passed with .* as extention. 
        #split_path = path_to_model.split('.')
        #path_to_load = split_path[0] + HDF5_EXTENTION 
        model = load_model(path_to_model,
                           custom_objects={
                               'OverwrittenSequentialClassifier':OverwrittenSequentialClassifier
                               })
        self.set_trained_model(model)

    


@properties(estimator_family=[NEURAL_NETWORKS], 
            tasks=[REGRESSION], 
            name='NeuralNetworkDeepRegressor')
class Deep_NN_Regressor(MlautEstimator):
    """
    Wrapper for a `keras sequential model <https://keras.io/getting-started/sequential-model-guide/>`_. 
    """

    def __init__(self, verbose=0, 
                n_jobs=-1,
                num_cv_folds=3, 
                refit=True):
        super().__init__(verbose=verbose, 
                         n_jobs=n_jobs, 
                        num_cv_folds=num_cv_folds, 
                        refit=refit)
        self._hyperparameters = {'loss':'mean_squared_error', 
                                 'learning_rate':0.001,
                                 'optimizer': 'Adam',
                                 'metrics': ['accuracy'],
                                 'epochs': [50,100], 
                                 'batch_size': 0 }

    def _nn_deep_classifier_model(self, input_dim):
        nn_deep_model = Sequential()
        nn_deep_model.add(Dense(288, input_dim=input_dim, activation='relu'))
        nn_deep_model.add(Dense(144, activation='relu'))
        nn_deep_model.add(Dropout(0.5))
        nn_deep_model.add(Dense(12, activation='relu'))
        nn_deep_model.add(Dense(1, activation='sigmoid'))
        
        optimizer = self._hyperparameters['optimizer']
        if optimizer is 'Adam':
            model_optimizer  = optimizers.Adam(lr=self._hyperparameters['learning_rate'])
        
        nn_deep_model.compile(loss=self._hyperparameters['loss'], 
                              optimizer=model_optimizer, 
                              metrics=self._hyperparameters['metrics'])
        return nn_deep_model
    
    def build(self, **kwargs):
        """
        builds and returns estimator

        

        :type loss: string
        :param loss: loss metric as per `keras documentation <https://keras.io/losses/>`_.

        :type learning_rate: float
        :param learning_rate: learning rate for training the neural network.

        :type hypehyperparameters: dictionary
        :param hypehyperparameters: dictionary used for tuning the network if Gridsearch is used.

        :type kwargs: key-value(integer)
        :param kwargs: The user must specify ``input_dim`` and ``num_samples``.

        :rtype: `keras object`
        """
        if 'input_dim' not in kwargs:
            raise ValueError('You need to specify input dimentions when building the model')
        if 'num_samples' not in kwargs:
            raise ValueError('You need to specify num_samples when building the keras model.')
        input_dim=kwargs['input_dim']
        num_samples = kwargs['num_samples']
        
        model = KerasRegressor(build_fn=self._nn_deep_classifier_model, 
                                input_dim=input_dim,
                                verbose=self._verbose)

        return model
        # return GridSearchCV(model, 
        #                     hyperparameters, 
        #                     verbose = self._verbose,
        #                     n_jobs=self._n_jobs,
        #                     refit=self._refit)


    def save(self, dataset_name):
        """
        Saves estimator on disk.

        :type dataset_name: string
        :param dataset_name: name of the dataset. Estimator will be saved under default folder structure `/data/trained_models/<dataset name>/<model name>`
        """
        #set trained model method is implemented in the base class
        trained_model = self._trained_model
        disk_op = DiskOperations()
        disk_op.save_keras_model(trained_model=trained_model,
                                 model_name=self.properties()['name'],
                                 dataset_name=dataset_name)
    
    #overloading method from parent class
    def load(self, path_to_model):
        """
        Loads saved keras model from disk.

        :type path_to_model: string
        :param path_to_model: path on disk where the object is saved.
        """
        #file name could be passed with .* as extention. 

        split_path = path_to_model.split('.')
        path_to_json = split_path[0] + JSON_EXTENTION
        path_to_weights = split_path[0] + HDF5_EXTENTION
        json_file = open(path_to_json, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(path_to_weights)
        self.set_trained_model(loaded_model)

class OverwrittenSequentialClassifier(Sequential):
    """
    Keras sequential model that overrides the default :func:`tensorflow.python.keras.models.fit` and :func:`tensorflow.python.keras.models.predict` methods.
    """



    def fit(self, X_train, y_train):
            
        """
        Overrides the default :func:`tensorflow.python.keras.models.fit` and reshapes the `y_train` in one hot array. 

        Paremeters
        ----------
        X_train: training data
        y_train: Labels that will be converted to onehot array.


        Returns
        -------
        :func:`tensorflow.python.keras.models.fit` object

        """
        onehot_encoder = OneHotEncoder(sparse=False)
        len_y = len(y_train)
        reshaped_y = y_train.reshape(len_y, 1)
        y_train_onehot_encoded = onehot_encoder.fit_transform(reshaped_y)
        
        
        return super().fit(X_train, y_train_onehot_encoded)

    def predict(self, X_test, batch_size=None, verbose=0):
        """
        Overrides the default :func:`tensorflow.python.keras.models.predict` by replacing it with a :func:`tensorflow.python.keras.models.predict_classes`  

        Returns
        --------
        :func:`tensorflow.python.keras.models.predict_classes`
        """
        predictions = Sequential.predict(self, X_test, batch_size=batch_size, verbose=verbose)
        return predictions.argmax(axis=1)
        # return super().predict_classes(X_test)