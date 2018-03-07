from mleap.estimators.mleap_estimator import properties
from mleap.estimators.mleap_estimator import MleapEstimator

from mleap.shared.files_io import DiskOperations
from mleap.shared.static_variables import(GENERALIZED_LINEAR_MODELS,
                                      ENSEMBLE_METHODS, 
                                      SVM,
                                      NEURAL_NETWORKS,
                                      NAIVE_BAYES,
                                      REGRESSION, 
                                      CLASSIFICATION)
from mleap.shared.static_variables import PICKLE_EXTENTION, HDF5_EXTENTION

from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Activation, Dropout
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from tensorflow.python.keras import optimizers

import numpy as np

@properties(estimator_family=[NEURAL_NETWORKS], 
            tasks=[CLASSIFICATION], 
            name='NeuralNetworkDeepClassifier')
class Deep_NN_Classifier(MleapEstimator):
    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)

    def _nn_deep_classifier_model(self, num_classes, 
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
        nn_deep_model.add(Dense(num_classes, activation='softmax'))
        
        
        if optimizer is 'Adam':
            model_optimizer = optimizers.Adam(lr=learning_rate)
        
        nn_deep_model.compile(loss=loss, optimizer=model_optimizer, metrics=metrics)
        return nn_deep_model
    
    def build(self, num_classes, input_dim, num_samples, loss='mean_squared_error', learning_rate=0.001, hyperparameters = None):
        model = KerasClassifier(build_fn=self._nn_deep_classifier_model, 
                                num_classes=num_classes, 
                                input_dim=input_dim,
                                verbose=self._verbose,
                                loss=loss)
        if hyperparameters is None:
            hyperparameters = {'epochs': [50,100], 'batch_size': [num_samples]}
        return model

    def predict(self, X):
        estimator = self.get_trained_model()
        predictions = estimator.predict(X)
        predictions = np.array(predictions)
        predictions = predictions.argmax(axis=1)
        return predictions

        
    def save(self, dataset_name):
        #set trained model method is implemented in the base class
        trained_model = self._trained_model
        disk_op = DiskOperations()
        disk_op.save_keras_model(trained_model=trained_model,
                                 model_name=self.properties()['name'],
                                 dataset_name=dataset_name)
    
    #overloading method from parent class
    def load(self, path_to_model):
        #file name could be passed with .* as extention. 
        #split_path = path_to_model.split('.')
        #path_to_load = split_path[0] + HDF5_EXTENTION 
        model = load_model(path_to_model)
        self.set_trained_model(model)


@properties(estimator_family=[NEURAL_NETWORKS], 
            tasks=[REGRESSION], 
            name='NeuralNetworkDeepRegressor')
class Deep_NN_Regressor(MleapEstimator):
    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)

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
    
    def build(self, input_dim, num_samples, loss='mean_squared_error', learning_rate=0.001, hyperparameters = None):
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
        #set trained model method is implemented in the base class
        trained_model = self._trained_model
        disk_op = DiskOperations()
        disk_op.save_keras_model(trained_model=trained_model,
                                 model_name=self.properties()['name'],
                                 dataset_name=dataset_name)
    
    #overloading method from parent class
    def load(self, path_to_model):
        #file name could be passed with .* as extention. 
        split_path = path_to_model.split('.')
        path_to_load = split_path[0] + HDF5_EXTENTION 
        model = load_model(path_to_load)
        self.set_trained_model(model)