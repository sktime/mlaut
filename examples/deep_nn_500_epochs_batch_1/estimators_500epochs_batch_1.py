from mlaut.estimators.nn_estimators import Deep_NN_Classifier
from mlaut.estimators.nn_estimators import OverwrittenSequentialClassifier
from tensorflow.python.keras.layers import Dense, Activation, Dropout
from tensorflow.python.keras import optimizers

hyperparameters = {'epochs': 500, 
                    'batch_size': None}

def keras_model1(num_classes, input_dim):
    model = OverwrittenSequentialClassifier()
    model.add(Dense(288, input_dim=input_dim, activation='relu'))
    model.add(Dense(144, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model_optimizer = optimizers.Adam(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=model_optimizer, metrics=['accuracy'])

    return model

deep_nn_4_layer_thin_dropout = Deep_NN_Classifier(keras_model=keras_model1, 
                            properties={'name':'NN-4-layer_thin_dropout'})


def keras_model2(num_classes, input_dim):
    nn_deep_model = OverwrittenSequentialClassifier()
    nn_deep_model.add(Dense(2500, input_dim=input_dim, activation='relu'))
    nn_deep_model.add(Dense(2000, activation='relu'))
    nn_deep_model.add(Dense(1500, activation='relu'))
    nn_deep_model.add(Dense(num_classes, activation='softmax'))

    model_optimizer = optimizers.Adam(lr=0.001)
    nn_deep_model.compile(loss='mean_squared_error', optimizer=model_optimizer, metrics=['accuracy'])
    return nn_deep_model

deep_nn_4_layer_wide_no_dropout = Deep_NN_Classifier(keras_model=keras_model2,
                            properties={'name':'NN-4-layer_wide_no_dropout'})


def keras_model3(num_classes, input_dim):
    nn_deep_model = OverwrittenSequentialClassifier()
    nn_deep_model.add(Dense(2500, input_dim=input_dim, activation='relu'))
    nn_deep_model.add(Dense(2000, activation='relu'))
    nn_deep_model.add(Dropout(0.5))
    nn_deep_model.add(Dense(1500, activation='relu'))
    nn_deep_model.add(Dense(num_classes, activation='softmax'))

    model_optimizer = optimizers.Adam(lr=0.001)
    nn_deep_model.compile(loss='mean_squared_error', optimizer=model_optimizer, metrics=['accuracy'])
    return nn_deep_model

deep_nn_4_layer_wide_with_dropout = Deep_NN_Classifier(keras_model=keras_model3,
                            properties={'name':'NN-4-layer_wide_with_dropout'})


def keras_model4(num_classes, input_dim):
    nn_deep_model = OverwrittenSequentialClassifier()
    nn_deep_model.add(Dense(5000, input_dim=input_dim, activation='relu'))
    nn_deep_model.add(Dense(4500, activation='relu'))
    nn_deep_model.add(Dense(4000, activation='relu'))
    nn_deep_model.add(Dropout(0.5))

    nn_deep_model.add(Dense(3500, activation='relu'))
    nn_deep_model.add(Dense(3000, activation='relu'))
    nn_deep_model.add(Dense(2500, activation='relu'))
    nn_deep_model.add(Dropout(0.5))


    nn_deep_model.add(Dense(2000, activation='relu'))
    nn_deep_model.add(Dense(1500, activation='relu'))
    nn_deep_model.add(Dense(1000, activation='relu'))
    nn_deep_model.add(Dropout(0.5))

    nn_deep_model.add(Dense(500, activation='relu'))
    nn_deep_model.add(Dense(250, activation='relu'))
    nn_deep_model.add(Dense(num_classes, activation='softmax'))

    model_optimizer = optimizers.Adam(lr=0.001)
    nn_deep_model.compile(loss='mean_squared_error', optimizer=model_optimizer, metrics=['accuracy'])
    return nn_deep_model

deep_nn_12_layer_wide_with_dropout = Deep_NN_Classifier(keras_model=keras_model4,
                            properties={'name':'NN-12-layer_wide_with_dropout'})



def keras_model_1_lr01(num_classes, input_dim):
    model = OverwrittenSequentialClassifier()
    model.add(Dense(288, input_dim=input_dim, activation='relu'))
    model.add(Dense(144, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model_optimizer = optimizers.Adam(lr=0.1)
    model.compile(loss='mean_squared_error', optimizer=model_optimizer, metrics=['accuracy'])

    return model

deep_nn_4_layer_thin_dropout_lr01 = Deep_NN_Classifier(keras_model=keras_model_1_lr01, 
                            properties={'name':'NN-4-layer_thin_dropout_lr01'})

def keras_model_1_lr1(num_classes, input_dim):
    model = OverwrittenSequentialClassifier()
    model.add(Dense(288, input_dim=input_dim, activation='relu'))
    model.add(Dense(144, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model_optimizer = optimizers.Adam(lr=1)
    model.compile(loss='mean_squared_error', optimizer=model_optimizer, metrics=['accuracy'])

    return model

deep_nn_4_layer_thin_dropout_lr1 = Deep_NN_Classifier(keras_model=keras_model_1_lr1, 
                            properties={'name':'NN-4-layer_thin_dropout_lr1'})


def keras_model_2_lr01(num_classes, input_dim):
    nn_deep_model = OverwrittenSequentialClassifier()
    nn_deep_model.add(Dense(2500, input_dim=input_dim, activation='relu'))
    nn_deep_model.add(Dense(2000, activation='relu'))
    nn_deep_model.add(Dense(1500, activation='relu'))
    nn_deep_model.add(Dense(num_classes, activation='softmax'))

    model_optimizer = optimizers.Adam(lr=0.1)
    nn_deep_model.compile(loss='mean_squared_error', optimizer=model_optimizer, metrics=['accuracy'])
    return nn_deep_model

deep_nn_4_layer_wide_no_dropout_lr01 = Deep_NN_Classifier(keras_model=keras_model_2_lr01,
                            properties={'name':'NN-4-layer_wide_no_dropout_lr01'})


def keras_model_2_lr1(num_classes, input_dim):
    nn_deep_model = OverwrittenSequentialClassifier()
    nn_deep_model.add(Dense(2500, input_dim=input_dim, activation='relu'))
    nn_deep_model.add(Dense(2000, activation='relu'))
    nn_deep_model.add(Dense(1500, activation='relu'))
    nn_deep_model.add(Dense(num_classes, activation='softmax'))

    model_optimizer = optimizers.Adam(lr=1)
    nn_deep_model.compile(loss='mean_squared_error', optimizer=model_optimizer, metrics=['accuracy'])
    return nn_deep_model

deep_nn_4_layer_wide_no_dropout_lr1 = Deep_NN_Classifier(keras_model=keras_model_2_lr1,
                            properties={'name':'NN-4-layer_wide_no_dropout_lr1'})



def keras_model_3_lr01(num_classes, input_dim):
    nn_deep_model = OverwrittenSequentialClassifier()
    nn_deep_model.add(Dense(2500, input_dim=input_dim, activation='relu'))
    nn_deep_model.add(Dense(2000, activation='relu'))
    nn_deep_model.add(Dropout(0.5))
    nn_deep_model.add(Dense(1500, activation='relu'))
    nn_deep_model.add(Dense(num_classes, activation='softmax'))

    model_optimizer = optimizers.Adam(lr=0.1)
    nn_deep_model.compile(loss='mean_squared_error', optimizer=model_optimizer, metrics=['accuracy'])
    return nn_deep_model

deep_nn_4_layer_wide_with_dropout_lr01 = Deep_NN_Classifier(keras_model=keras_model_3_lr01,
                            properties={'name':'NN-4-layer_wide_with_dropout_lr01'})


def keras_model_3_lr1(num_classes, input_dim):
    nn_deep_model = OverwrittenSequentialClassifier()
    nn_deep_model.add(Dense(2500, input_dim=input_dim, activation='relu'))
    nn_deep_model.add(Dense(2000, activation='relu'))
    nn_deep_model.add(Dropout(0.5))
    nn_deep_model.add(Dense(1500, activation='relu'))
    nn_deep_model.add(Dense(num_classes, activation='softmax'))

    model_optimizer = optimizers.Adam(lr=1)
    nn_deep_model.compile(loss='mean_squared_error', optimizer=model_optimizer, metrics=['accuracy'])
    return nn_deep_model

deep_nn_4_layer_wide_with_dropout_lr1 = Deep_NN_Classifier(keras_model=keras_model_3_lr1,
                            properties={'name':'NN-4-layer_wide_with_dropout_lr1'})



def keras_model_4_lr01(num_classes, input_dim):
    nn_deep_model = OverwrittenSequentialClassifier()
    nn_deep_model.add(Dense(5000, input_dim=input_dim, activation='relu'))
    nn_deep_model.add(Dense(4500, activation='relu'))
    nn_deep_model.add(Dense(4000, activation='relu'))
    nn_deep_model.add(Dropout(0.5))

    nn_deep_model.add(Dense(3500, activation='relu'))
    nn_deep_model.add(Dense(3000, activation='relu'))
    nn_deep_model.add(Dense(2500, activation='relu'))
    nn_deep_model.add(Dropout(0.5))
    
    
    nn_deep_model.add(Dense(2000, activation='relu'))
    nn_deep_model.add(Dense(1500, activation='relu'))
    nn_deep_model.add(Dense(1000, activation='relu'))
    nn_deep_model.add(Dropout(0.5))
    
    nn_deep_model.add(Dense(500, activation='relu'))
    nn_deep_model.add(Dense(250, activation='relu'))
    nn_deep_model.add(Dense(num_classes, activation='softmax'))
    
    model_optimizer = optimizers.Adam(lr=0.1)
    nn_deep_model.compile(loss='mean_squared_error', optimizer=model_optimizer, metrics=['accuracy'])
    return nn_deep_model

deep_nn_12_layer_wide_with_dropout_lr01 = Deep_NN_Classifier(keras_model=keras_model_4_lr01,
                            properties={'name':'NN-12-layer_wide_with_dropout_lr01'})

def keras_model_4_lr1(num_classes, input_dim):
    nn_deep_model = OverwrittenSequentialClassifier()
    nn_deep_model.add(Dense(5000, input_dim=input_dim, activation='relu'))
    nn_deep_model.add(Dense(4500, activation='relu'))
    nn_deep_model.add(Dense(4000, activation='relu'))
    nn_deep_model.add(Dropout(0.5))

    nn_deep_model.add(Dense(3500, activation='relu'))
    nn_deep_model.add(Dense(3000, activation='relu'))
    nn_deep_model.add(Dense(2500, activation='relu'))
    nn_deep_model.add(Dropout(0.5))
    
    
    nn_deep_model.add(Dense(2000, activation='relu'))
    nn_deep_model.add(Dense(1500, activation='relu'))
    nn_deep_model.add(Dense(1000, activation='relu'))
    nn_deep_model.add(Dropout(0.5))
    
    nn_deep_model.add(Dense(500, activation='relu'))
    nn_deep_model.add(Dense(250, activation='relu'))
    nn_deep_model.add(Dense(num_classes, activation='softmax'))
    
    model_optimizer = optimizers.Adam(lr=1)
    nn_deep_model.compile(loss='mean_squared_error', optimizer=model_optimizer, metrics=['accuracy'])
    return nn_deep_model

deep_nn_12_layer_wide_with_dropout_lr1 = Deep_NN_Classifier(keras_model=keras_model_4_lr1,
                            properties={'name':'NN-12-layer_wide_with_dropout_lr1'})

def keras_model_5_lr0001(num_classes, input_dim):
    nn_deep_model = OverwrittenSequentialClassifier()
    nn_deep_model.add(Dense(2000, input_dim=input_dim, activation='relu'))
    nn_deep_model.add(Dropout(0.5))
    nn_deep_model.add(Dense(1000, activation='relu'))
    nn_deep_model.add(Dropout(0.5))
    nn_deep_model.add(Dense(1000, activation='relu'))
    nn_deep_model.add(Dropout(0.5))
    nn_deep_model.add(Dense(50, activation='relu'))
    nn_deep_model.add(Dropout(0.5))

    nn_deep_model.add(Dense(num_classes, activation='softmax'))

    model_optimizer = optimizers.Adam(lr=0.001)
    nn_deep_model.compile(loss='mean_squared_error', optimizer=model_optimizer, metrics=['accuracy'])
    return nn_deep_model
deep_nn_4_layer_droput_each_layer_lr0001 = Deep_NN_Classifier(keras_model=keras_model_5_lr0001,
                                        properties={'name':'NN-4-layer-droput-each-layer_lr0001'})

def keras_model_5_lr01(num_classes, input_dim):
    nn_deep_model = OverwrittenSequentialClassifier()
    nn_deep_model.add(Dense(2000, input_dim=input_dim, activation='relu'))
    nn_deep_model.add(Dropout(0.5))
    nn_deep_model.add(Dense(1000, activation='relu'))
    nn_deep_model.add(Dropout(0.5))
    nn_deep_model.add(Dense(1000, activation='relu'))
    nn_deep_model.add(Dropout(0.5))
    nn_deep_model.add(Dense(50, activation='relu'))
    nn_deep_model.add(Dropout(0.5))

    nn_deep_model.add(Dense(num_classes, activation='softmax'))

    model_optimizer = optimizers.Adam(lr=0.1)
    nn_deep_model.compile(loss='mean_squared_error', optimizer=model_optimizer, metrics=['accuracy'])
    return nn_deep_model
deep_nn_4_layer_droput_each_layer_lr01 = Deep_NN_Classifier(keras_model=keras_model_5_lr01,
                                        properties={'name':'NN-4-layer-droput-each-layer_lr01'})

def keras_model_5_lr1(num_classes, input_dim):
    nn_deep_model = OverwrittenSequentialClassifier()
    nn_deep_model.add(Dense(2000, input_dim=input_dim, activation='relu'))
    nn_deep_model.add(Dropout(0.5))
    nn_deep_model.add(Dense(1000, activation='relu'))
    nn_deep_model.add(Dropout(0.5))
    nn_deep_model.add(Dense(1000, activation='relu'))
    nn_deep_model.add(Dropout(0.5))
    nn_deep_model.add(Dense(50, activation='relu'))
    nn_deep_model.add(Dropout(0.5))

    nn_deep_model.add(Dense(num_classes, activation='softmax'))

    model_optimizer = optimizers.Adam(lr=1)
    nn_deep_model.compile(loss='mean_squared_error', optimizer=model_optimizer, metrics=['accuracy'])
    return nn_deep_model
deep_nn_4_layer_droput_each_layer_lr1 = Deep_NN_Classifier(keras_model=keras_model_5_lr01,
                                        properties={'name':'NN-4-layer-droput-each-layer_lr1'})

def keras_model_6_lr001(num_classes, input_dim):
    nn_deep_model = OverwrittenSequentialClassifier()
    nn_deep_model.add(Dropout(0.7, input_shape=(input_dim,)))
    nn_deep_model.add(Dense(1024, activation='relu'))
    nn_deep_model.add(Dropout(0.5))
    nn_deep_model.add(Dense(num_classes, activation='softmax'))

    model_optimizer = optimizers.Adam(lr=0.001)
    nn_deep_model.compile(loss='mean_squared_error', optimizer=model_optimizer, metrics=['accuracy'])
    return nn_deep_model
deep_nn_2_layer_droput_input_layer_lr001 = Deep_NN_Classifier(keras_model=keras_model_6_lr001,
                                        properties={'name':'NN-2-layer-droput-input-layer_lr001'})

def keras_model_6_lr01(num_classes, input_dim):
    nn_deep_model = OverwrittenSequentialClassifier()
    nn_deep_model.add(Dropout(0.7, input_shape=(input_dim,)))
    nn_deep_model.add(Dense(1024, activation='relu'))
    nn_deep_model.add(Dropout(0.5))
    nn_deep_model.add(Dense(num_classes, activation='softmax'))

    model_optimizer = optimizers.Adam(lr=0.1)
    nn_deep_model.compile(loss='mean_squared_error', optimizer=model_optimizer, metrics=['accuracy'])
    return nn_deep_model
deep_nn_2_layer_droput_input_layer_lr01 = Deep_NN_Classifier(keras_model=keras_model_6_lr01,
                                        properties={'name':'NN-2-layer-droput-input-layer_lr01'})

def keras_model_6_lr1(num_classes, input_dim):
    nn_deep_model = OverwrittenSequentialClassifier()
    nn_deep_model.add(Dropout(0.7, input_shape=(input_dim,)))
    nn_deep_model.add(Dense(1024, activation='relu'))
    nn_deep_model.add(Dropout(0.5))
    nn_deep_model.add(Dense(num_classes, activation='softmax'))

    model_optimizer = optimizers.Adam(lr=1)
    nn_deep_model.compile(loss='mean_squared_error', optimizer=model_optimizer, metrics=['accuracy'])
    return nn_deep_model

deep_nn_2_layer_droput_input_layer_lr1 = Deep_NN_Classifier(keras_model=keras_model_6_lr1,
                                        properties={'name':'NN-2-layer-droput-input-layer_lr1'})

estimators = [deep_nn_4_layer_thin_dropout_lr01,
            deep_nn_4_layer_thin_dropout_lr1, 
            deep_nn_4_layer_wide_no_dropout_lr01,
            deep_nn_4_layer_wide_no_dropout_lr1,
            deep_nn_4_layer_wide_with_dropout_lr01,
            deep_nn_4_layer_wide_with_dropout_lr1,
            deep_nn_12_layer_wide_with_dropout_lr01,
            deep_nn_12_layer_wide_with_dropout_lr1,
            deep_nn_4_layer_droput_each_layer_lr0001,
            deep_nn_4_layer_droput_each_layer_lr01,
            deep_nn_4_layer_droput_each_layer_lr1,
            deep_nn_4_layer_thin_dropout,
            deep_nn_4_layer_wide_no_dropout, 
            deep_nn_4_layer_wide_with_dropout,
            deep_nn_12_layer_wide_with_dropout,            
              deep_nn_2_layer_droput_input_layer_lr001,
            deep_nn_2_layer_droput_input_layer_lr01,
            deep_nn_2_layer_droput_input_layer_lr1]

for e in estimators:
  e.set_params(hyperparameters)
