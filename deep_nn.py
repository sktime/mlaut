from download_delgado.delgado_datasets import DownloadAndConvertDelgadoDatasets
from mlaut.data import Data
from mlaut.estimators.estimators import instantiate_default_estimators
from mlaut.experiments import Orchestrator
from download_delgado.delgado_datasets import DownloadAndConvertDelgadoDatasets
from mlaut.analyze_results.scores import ScoreAccuracy
import pandas as pd
import numpy as np
from mlaut.estimators.nn_estimators import Deep_NN_Classifier
from mlaut.estimators.nn_estimators import OverwrittenSequentialClassifier
from tensorflow.python.keras.models import Sequential, load_model, model_from_json
from tensorflow.python.keras.layers import Dense, Activation, Dropout
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from tensorflow.python.keras import optimizers
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV

from mlaut.estimators.nn_estimators import Deep_NN_Classifier
import multiprocessing




hyperparameters = {'epochs': [50,100], 
                    'batch_size': [0, 50, 100]}
                   




if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver', force=True)


    data = Data()
    input_io = data.open_hdf5('data/delgado.hdf5', mode='a')
    out_io = data.open_hdf5('data/deep_nn_study.hdf5', mode='a')
    dts_names_list, dts_names_list_full_path = data.list_datasets(hdf5_io=input_io, hdf5_group='delgado_datasets/')
    split_dts_list = data.split_datasets(hdf5_in=input_io, hdf5_out=out_io, dataset_paths=dts_names_list_full_path)

    
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

    deep_nn_4_layer_wide_no_dropout = Deep_NN_Classifier(hyperparameters=hyperparameters,
                                keras_model=keras_model2,
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

    deep_nn_4_layer_wide_with_dropout = Deep_NN_Classifier(hyperparameters=hyperparameters,
                                keras_model=keras_model3,
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

    deep_nn_12_layer_wide_with_dropout = Deep_NN_Classifier(hyperparameters=hyperparameters,
                                keras_model=keras_model4,
                                properties={'name':'NN-12-layer_wide_with_dropout'})


    estimators = [deep_nn_4_layer_thin_dropout,
                deep_nn_4_layer_wide_no_dropout, 
                deep_nn_4_layer_wide_with_dropout,
                deep_nn_12_layer_wide_with_dropout]

    orchest = Orchestrator(hdf5_input_io=input_io, hdf5_output_io=out_io, dts_names=dts_names_list,
                    original_datasets_group_h5_path='delgado_datasets/')
    orchest.run(modelling_strategies=estimators, verbose=True)



