from mlaut.shared.static_variables import GRIDSEARCH_NUM_CV_FOLDS, GRIDSEARCH_CV_NUM_PARALLEL_JOBS
from mlaut.shared.static_variables import VERBOSE

from tensorflow.python.keras.models import Sequential, load_model, model_from_json
from tensorflow.python.keras.layers import Dense, Activation, Dropout
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from tensorflow.python.keras import optimizers
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
import numpy as np
import wrapt

import tensorflow as tf

from mlaut.estimators.base import BaseClassifier, BaseRegressor
REGRESSOR_TYPES = BaseRegressor
CLASSIFIER_TYPES = BaseClassifier
ESTIMATOR_TYPES = [REGRESSOR_TYPES, CLASSIFIER_TYPES]
from mlaut.highlevel.strategies import CSCKerasStrategy, CSRKerasStrategy


# class OverwrittenSequentialClassifier(Sequential):
#     """
#     Keras sequential model that overrides the default :func:`tensorflow.python.keras.models.fit` and :func:`tensorflow.python.keras.models.predict` methods.
#     """


#     def fit(self, X_train, y_train, **kwargs):
            
#         """
#         Overrides the default :func:`tensorflow.python.keras.models.fit` and reshapes the `y_train` in one hot array. 

#         Args:
#             X_train: training data
#             y_train: Labels that will be converted to onehot array.


#         Returns:
#             :func:`tensorflow.python.keras.models.fit` object

#         """
#         onehot_encoder = OneHotEncoder(sparse=False)
#         len_y = len(y_train)
#         reshaped_y = y_train.reshape(len_y, 1)
#         y_train_onehot_encoded = onehot_encoder.fit_transform(reshaped_y)
        
#         # if 'epochs' not in self._hyperparameters:
#         #     epochs = 1
#         # else:
#         #     epochs = self._hyperparameters

#         return super().fit(X_train, 
#                             y_train_onehot_encoded, 
#                             batch_size=kwargs['batch_size'],
#                             epochs=kwargs['epochs'])

        

#     def predict(self, X_test, batch_size=None, verbose=VERBOSE):
#         """
#         Overrides the default :func:`tensorflow.python.keras.models.predict` by replacing it with a :func:`tensorflow.python.keras.models.predict_classes`  

#         Returns:
#             :func:`tensorflow.python.keras.models.predict_classes`
#         """
#         predictions = Sequential.predict(self, X_test, batch_size=batch_size, verbose=verbose)
#         return predictions.argmax(axis=1)
#         # return super().predict_classes(X_test)

# class KerasClassificationStrategy(CSCKerasStrategy):
#     def keras_model_classification(num_classes, input_dim):
#         nn_deep_model = OverwrittenSequentialClassifier()
#         nn_deep_model.add(Dense(288, input_dim=input_dim, activation='relu'))
#         nn_deep_model.add(Dense(144, activation='relu'))
#         nn_deep_model.add(Dropout(0.5))
#         nn_deep_model.add(Dense(12, activation='relu'))
#         nn_deep_model.add(Dense(num_classes, activation='softmax'))

#         model_optimizer = optimizers.Adam(lr=0.001)
#         nn_deep_model.compile(loss='mean_squared_error', optimizer=model_optimizer, metrics=['accuracy'])

#         return nn_deep_model

#     def __init__(self, 
#                  estimator=KerasClassifier, 
#                  build_fn=keras_model_classification, 
#                  param_grid={'epochs': 1, 
#                             'batch_size': None},
#                  name='Keras4Layers',
#                  check_input=False):
#         print('****************** I like to init')
#         print(f'*****Param grid: {param_grid}, {name}')
#         super().__init__(estimator=estimator, build_fn=build_fn, param_grid=param_grid, name=name, check_input=check_input)
    

def keras_model_classification(num_classes, input_dim):
    # nn_deep_model = OverwrittenSequentialClassifier()
    nn_deep_model = Sequential()

    nn_deep_model.add(Dense(288, input_dim=input_dim, activation='relu'))
    nn_deep_model.add(Dense(144, activation='relu'))
    nn_deep_model.add(Dropout(0.5))
    nn_deep_model.add(Dense(12, activation='relu'))
    nn_deep_model.add(Dense(num_classes, activation='softmax'))

    model_optimizer = optimizers.Adam(lr=0.001)
    nn_deep_model.compile(loss='mean_squared_error', optimizer=model_optimizer, metrics=['accuracy'])

    return nn_deep_model
param_grid={'epochs': 1, 
                           'batch_size': None}

KerasClassificationStrategy = CSCKerasStrategy(estimator=KerasClassifier, 
                                              build_fn=keras_model_classification,
                                              param_grid=param_grid,
                                              name='KerasClassifier4Layers',
                                              check_input=False)

                        
def keras_model_regression(input_dim):
    nn_deep_model = Sequential()
    nn_deep_model.add(Dense(288, input_dim=input_dim, activation='relu'))
    nn_deep_model.add(Dense(144, activation='relu'))
    nn_deep_model.add(Dropout(0.5))
    nn_deep_model.add(Dense(12, activation='relu'))
    nn_deep_model.add(Dense(1, activation='sigmoid'))
    

    model_optimizer  = optimizers.Adam(lr=0.001)
    nn_deep_model.compile(loss='mean_squared_error', optimizer=model_optimizer, metrics=['mae'])


    return nn_deep_model
        
KerasRegressionStrategy = CSRKerasStrategy(estimator=KerasRegressor,
                                               build_fn=keras_model_regression,
                                               param_grid=param_grid,
                                               name='KerasRegressor4Layers',
                                               check_input=False)

