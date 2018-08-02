from mlaut.estimators.nn_estimators import (OverwrittenSequentialClassifier, 
                                           Deep_NN_Classifier)
from tensorflow.python.keras.layers import Dense, Activation, Dropout

def keras_model_5_layers(num_classes, input_dim):
    nn_deep_model = OverwrittenSequentialClassifier()
    nn_deep_model.add(Dense(2500, input_dim=input_dim, activation='relu'))
    nn_deep_model.add(Dense(2000, activation='relu'))
    nn_deep_model.add(Dense(1500, activation='relu'))
    nn_deep_model.add(Dense(1000, activation='relu'))
    nn_deep_model.add(Dense(num_classes, activation='softmax'))
    return nn_deep_model


model_5_layers = Deep_NN_Classifier(keras_model=keras_model_5_layers)