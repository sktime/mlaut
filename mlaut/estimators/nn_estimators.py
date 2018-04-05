from mlaut.estimators.mlaut_estimator import properties
from mlaut.estimators.mlaut_estimator import mlautEstimator

from mlaut.shared.files_io import DiskOperations
from mlaut.shared.static_variables import(GENERALIZED_LINEAR_MODELS,
                                      ENSEMBLE_METHODS, 
                                      SVM,
                                      NEURAL_NETWORKS,
                                      NAIVE_BAYES,
                                      REGRESSION, 
                                      CLASSIFICATION)
from mlaut.shared.static_variables import PICKLE_EXTENTION, HDF5_EXTENTION

from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Activation, Dropout
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from tensorflow.python.keras import optimizers
from sklearn.preprocessing import OneHotEncoder


import numpy as np

@properties(estimator_family=[NEURAL_NETWORKS], 
            tasks=[CLASSIFICATION], 
            name='NeuralNetworkDeepClassifier')
class Deep_NN_Classifier(mlautEstimator):
    """
    Wrapper for a `keras sequential model <https://keras.io/getting-started/sequential-model-guide/>`_. 
    """

    def _nn_deep_classifier_model(self, num_classes, 
                                  input_dim,
                                  loss='mean_squared_error',
                                  optimizer = 'Adam',
                                  learning_rate=0.001,
                                  metrics = ['accuracy'] ):
        nn_deep_model = OverwrittenSequentialClassifier()
        nn_deep_model.add(Dense(288, input_dim=input_dim, activation='relu'))
        nn_deep_model.add(Dense(144, activation='relu'))
        nn_deep_model.add(Dropout(0.5))
        nn_deep_model.add(Dense(12, activation='relu'))
        nn_deep_model.add(Dense(num_classes, activation='softmax'))
        
        
        if optimizer is 'Adam':
            model_optimizer = optimizers.Adam(lr=learning_rate)
        
        nn_deep_model.compile(loss=loss, optimizer=model_optimizer, metrics=metrics)
        return nn_deep_model
    
    def build(self, 
              loss='mean_squared_error', 
              learning_rate=0.001, 
              hyperparameters = None, 
              **kwargs):
        """
        builds and returns estimator


        :type loss: string
        :param loss: loss metric as per `keras documentation <https://keras.io/losses/>`_.

        :type learning_rate: float
        :param learning_rate: learning rate for training the neural network.

        :type hypehyperparameters: dictionary
        :param hypehyperparameters: dictionary used for tuning the network if Gridsearch is used.

        :type kwargs: key-value
        :param kwargs: At a minimum the user must specify ``input_dim``, ``num_samples`` and ``num_classes``.
        :rtype: `keras object`
        """
        if 'input_dim' not in kwargs:
            raise ValueError('You need to specify input dimentions when building the model.')
        if 'num_samples' not in kwargs:
            raise ValueError('You need to specify num_samples when building the keras model.')
        if 'num_classes' not in kwargs:
            raise ValueError('You need to specify num_classes when building the keras model.')
        
        input_dim=kwargs['input_dim']
        num_samples = kwargs['num_samples']
        num_classes = kwargs['num_classes']

        model = KerasClassifier(build_fn=self._nn_deep_classifier_model, 
                                num_classes=num_classes, 
                                input_dim=input_dim,
                                verbose=self._verbose,
                                loss=loss)
        if hyperparameters is None:
            hyperparameters = {'epochs': [50,100], 'batch_size': [num_samples]}
        return model

    # def predict(self, X):
    #     estimator = self.get_trained_model()
    #     predictions = estimator.predict(X)
    #     predictions = np.array(predictions)
    #     predictions = predictions.argmax(axis=1)
    #     return predictions

        
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
        #split_path = path_to_model.split('.')
        #path_to_load = split_path[0] + HDF5_EXTENTION 
        model = load_model(path_to_model)
        self.set_trained_model(model)


@properties(estimator_family=[NEURAL_NETWORKS], 
            tasks=[REGRESSION], 
            name='NeuralNetworkDeepRegressor')
class Deep_NN_Regressor(mlautEstimator):
    """
    Wrapper for a `keras sequential model <https://keras.io/getting-started/sequential-model-guide/>`_. 
    """


    def _nn_deep_classifier_model(self,  
                                  input_dim,
                                  loss='mean_squared_error',
                                  optimizer = 'Adam',
                                  learning_rate=0.001,
                                  metrics = ['accuracy'] ):
        nn_deep_model = Sequential()
        nn_deep_model.add(Dense(288, input_dim=input_dim, activation='relu'))
        nn_deep_model.add(Dense(144, activation='relu'))
        nn_deep_model.add(Dropout(0.5))
        nn_deep_model.add(Dense(12, activation='relu'))
        nn_deep_model.add(Dense(1, activation='sigmoid'))
        
        
        if optimizer is 'Adam':
            model_optimizer  = optimizers.Adam(lr=learning_rate)
        
        nn_deep_model.compile(loss=loss, optimizer=model_optimizer, metrics=metrics)
        return nn_deep_model
    
    def build(self,
              loss='mean_squared_error', 
              learning_rate=0.001, 
              hyperparameters = None, 
              **kwargs):
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
        path_to_load = split_path[0] + HDF5_EXTENTION 
        model = load_model(path_to_load)
        self.set_trained_model(model)


class OverwrittenSequentialClassifier(Sequential):
    """
    Keras sequential model that overrides the default :func:`tensorflow.python.keras.models.fit` and :func:`tensorflow.python.keras.models.predict` methods.
    """



    def fit(self, 
            X_train, 
            y_train, 
            batch_size=None, 
            epochs=1, 
            verbose=1, 
            callbacks=None, 
            validation_split=0.0, 
            validation_data=None, 
            shuffle=True, 
            class_weight=None, 
            sample_weight=None, 
            initial_epoch=0, 
            steps_per_epoch=None, 
            validation_steps=None):
            
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

    def predict(self, X_test):
        """
        Overrides the default :func:`tensorflow.python.keras.models.predict` by replacing it with a :func:`tensorflow.python.keras.models.predict_classes`  

        Returns
        --------
        :func:`tensorflow.python.keras.models.predict_classes`
        """
        return super().predict_classes(X_test)