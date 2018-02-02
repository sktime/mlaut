import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
from datetime import datetime

import numpy as np
import os
import logging
from sklearn.preprocessing import OneHotEncoder
from ..shared.files_io import DiskOperations
from mleap.shared.static_variables import NEURAL_NETWORKS
class Experiments(object):
    def __init__(self, experiments_trained_models_dir):
        self._disk_op = DiskOperations()
        self._experiments_trained_models_dir=experiments_trained_models_dir



    trained_models = []
    trained_models_fold_result_list = []
   
    def run_experiments(self, X_train, y_train, modelling_strategies, dts_name):
        """ 
        Trains estimators contained in the model_container on the dataset.
        This is separated from the test_orchestrator class to avoid too many nested loops.
        """
        trained_models = []
        timestamps_df = pd.DataFrame()
        for modelling_strategy in modelling_strategies:
            ml_strategy_name = modelling_strategy.properties()['name']
            ml_strategy_family = modelling_strategy.properties()['estimator_family']
            begin_timestamp = datetime.now()

            #check whether the model was already trained
            path_to_check = self._experiments_trained_models_dir + os.sep + dts_name + os.sep + ml_strategy_name + '.*'
            model_exists = self._disk_op.check_path_exists(path_to_check)
            if model_exists is True:
                logging.warning(f'Estimator {ml_strategy_name} already trained on {dts_name}. Loading it from disk.')
                modelling_strategy.load(path_to_check)
            else:
                #train the model if it does not exist on disk
                if NEURAL_NETWORKS in ml_strategy_family:
                    #encode the labels 
                    onehot_encoder = OneHotEncoder(sparse=False)
                    len_y = len(y_train)
                    reshaped_y = y_train.reshape(len_y, 1)
                    y_train_onehot_encoded = onehot_encoder.fit_transform(reshaped_y)
                    num_classes = y_train_onehot_encoded.shape[1]
                    num_samples, input_dim = X_train.shape
                    #build the model with the appropriate parameters
                    if ml_strategy_name is 'NeuralNetworkDeepClassifier':
                        built_model = modelling_strategy.build(num_classes, input_dim, num_samples)
                        built_model.fit(X_train, y_train_onehot_encoded)
                    if ml_strategy_name is 'NeuralNetworkDeepRegressor':
                        built_model = modelling_strategy.build(input_dim, num_samples)
                        built_model.fit(X_train, y_train)
                        
                    trained_model = built_model
                else:
                    built_model = modelling_strategy.build()
                    trained_model = built_model.fit(X_train, y_train)
            
                timestamps_df = self.record_timestamp(ml_strategy_name, begin_timestamp, timestamps_df)
                modelling_strategy.set_trained_model(trained_model)

            trained_models.append(modelling_strategy)
        return trained_models, timestamps_df
    
    def record_timestamp(self, strategy_name, begin_timestamp, timestamps_df):
        """ Timestamp used to record the duration of training for each of the estimators"""
        STF = '%Y-%m-%d %H:%M:%S'
        end_timestamp = datetime.now()
        diff = (end_timestamp - begin_timestamp).total_seconds() 
        vals = [strategy_name, begin_timestamp.strftime(STF),end_timestamp.strftime(STF),diff]
        run_time_df = pd.DataFrame([vals], columns=['strategy_name','begin_time','end_time','total_seconds'])
        timestamps_df = timestamps_df.append(run_time_df)
        
        return timestamps_df
        
    def make_predictions(self, models, dataset_name, X_test):
        """ Makes predictions on the test set """
        predictions = []
        for model in models:
            trained_model = model.get_trained_model()
            prediction = trained_model.predict(X_test)
            predictions.append([model.properties()['name'], prediction])
        return predictions
    

        