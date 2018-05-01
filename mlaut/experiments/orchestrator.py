from mlaut.shared.static_variables import (X_TRAIN_DIR, 
                                       X_TEST_DIR, 
                                       Y_TRAIN_DIR, 
                                       Y_TEST_DIR, 
                                       TRAIN_IDX, 
                                       TEST_IDX, 
                                       EXPERIMENTS_PREDICTIONS_GROUP,
                                       SPLIT_DTS_GROUP,
                                       EXPERIMENTS_TRAINED_MODELS_DIR, 
                                       LOG_ERROR_FILE, set_logging_defaults)
import sys
import os
from sklearn import preprocessing
from mlaut.shared.files_io import FilesIO
#from mlaut.experiments.experiments import Experiments
from mlaut.data import Data
import numpy as np
import logging
from datetime import datetime
import pandas as pd
from mlaut.shared.files_io import DiskOperations

class Orchestrator:
    """
    Orchestrates the sequencing of running the machine learning experiments.

    Parameters
    ------------
    hdf5_input_io: :func:`~mlaut.shared.files_io.FilesIO`
        instance of :func:`~mlaut.shared.files_io.FilesIO` with reference to the input file

    hdf5_output_io: :func:`~mlaut.shared.files_io.FilesIO`
        instance of :func:`~mlaut.shared.files_io.FilesIO` with reference to the output file

    dts_names: array of strings
        array with the names of the datasets on which experiments will be run.

    experiments_predictions_group: string
        path in HDF5 database where predictions will be saved.

    experiments_trained_models_dir: string
        folder on disk where trained estimators will be saved.
    split_datasets_group : string
        path in HDF5 database where the splits are saved.
    train_idx : string
        folder in HDF5 database which holds the train index splits.
    test_idx : string
        folder in HDF5 database which holds the test index splits.
    """
    def __init__(self, 
                 hdf5_input_io, 
                 hdf5_output_io,
                 dts_names,
                 original_datasets_group_h5_path, 
                 experiments_predictions_group=EXPERIMENTS_PREDICTIONS_GROUP,
                 experiments_trained_models_dir=EXPERIMENTS_TRAINED_MODELS_DIR,
                 split_datasets_group=SPLIT_DTS_GROUP,
                 train_idx=TRAIN_IDX,
                 test_idx=TEST_IDX):
        self.experiments_predictions_group=experiments_predictions_group
        self._experiments_trained_models_dir=experiments_trained_models_dir
        self._input_io = hdf5_input_io
        self._output_io = hdf5_output_io
        self._dts_names=dts_names
        self._original_datasets_group_h5_path = original_datasets_group_h5_path
        #self._experiments = Experiments(self._experiments_trained_models_dir)
        self._disk_op = DiskOperations()
        self._data = Data(experiments_predictions_group, 
                          split_datasets_group,
                          train_idx,
                          test_idx) 
        set_logging_defaults()

    def run(self, modelling_strategies):
        """ 
        Main module for training the estimators. 
        The inputs of the function are: 

        1. The input and output databse files containing the datasets.

        2. The instantiated estimators.
        
        The method iterates through all datasets in the input database file 
        and trains all modelling strategies on these datasets. At the end of the process 
        we make predictions on the test sets and store them in the output database file.

        The class uses helper methods in the experiments class to avoid too many nested loops.

        :type modelling_strategies: array of :ref:`mlaut_estimator-label` objects
        :param modelling_strategies: array of estimators that will be used for training
        """ 

        try:
            #loop through all datasets
            dts_trained=0
            dts_total = len(self._dts_names)
            self._prediction_accuracies = []
            for dts_name in self._dts_names:
                logging.log(1,f'Training estimators on {dts_name}')

                dts_trained +=1
                X_train, X_test, y_train, y_test = self._data.load_test_train_dts(hdf5_out=self._output_io, 
                                                                              hdf5_in=self._input_io, 
                                                                              dts_name=dts_name, 
                                                                              dts_grp_path=self._original_datasets_group_h5_path)
                # timestamps_df = self._experiments.run_experiments(X_train, 
                #                                                                   y_train, 
                #                                                                   modelling_strategies, 
                #                                                                   dts_name)
                timestamps_df = pd.DataFrame()
                for modelling_strategy in modelling_strategies:
                    ml_strategy_name = modelling_strategy.properties()['name']
                    ml_strategy_family = modelling_strategy.properties()['estimator_family']
                    data_preprocessing = modelling_strategy.properties()['data_preprocessing']
                    begin_timestamp = datetime.now()

                    #check whether the model was already trained
                    path_to_check = self._experiments_trained_models_dir + os.sep + dts_name + os.sep + ml_strategy_name + '.*'
                    model_exists = self._disk_op.check_path_exists(path_to_check)
                    if model_exists is True:
                        logging.warning(f'Estimator {ml_strategy_name} already trained on {dts_name}. Skipping it.')
                        #modelling_strategy.load(path_to_check)
                    else:
                        #preprocess data
                        X_train, X_test, y_train, y_test = self._preprocess_dataset(data_preprocessing,
                                                                                    X_train=X_train, 
                                                                                    X_test=X_test, 
                                                                                    y_train=y_train, 
                                                                                    y_test=y_test)
                        num_samples, input_dim = X_train.shape
                        unique_labels = np.unique(y_train)
                        num_classes = len(unique_labels) 
                        built_model = modelling_strategy.build(num_classes=num_classes, 
                                                                input_dim=input_dim,
                                                                num_samples=num_samples)
                        print(f'** Training estimator: {ml_strategy_name} on dataset: {dts_name}. Datasets processed: {dts_trained}/{dts_total} **')
                        trained_model = built_model.fit(X_train, y_train)
                        
                        timestamps_df = self.record_timestamp(ml_strategy_name, begin_timestamp, timestamps_df)
                        modelling_strategy.set_trained_model(trained_model)
                        modelling_strategy.save(dts_name)
                        
                        self._output_io.save_ml_strategy_timestamps(timestamps_df, dts_name)



        except KeyboardInterrupt:
            # quit
            print('***************************')
            print('********  EXITING  ********')
            print('***************************')

            sys.exit()

    def record_timestamp(self, strategy_name, begin_timestamp, timestamps_df):
        """ Timestamp used to record the duration of training for each of the estimators"""
        STF = '%Y-%m-%d %H:%M:%S'
        end_timestamp = datetime.now()
        diff = (end_timestamp - begin_timestamp).total_seconds() 
        vals = [strategy_name, begin_timestamp.strftime(STF),end_timestamp.strftime(STF),diff]
        run_time_df = pd.DataFrame([vals], columns=['strategy_name','begin_time','end_time','total_seconds'])
        timestamps_df = timestamps_df.append(run_time_df)
        
        return timestamps_df

    def predict_all(self, trained_models_dir, estimators):
        """
        Make predictions on test sets

        :type trained_models_dir: string
        :param trained_models_dir: directory where the trained models are saved

        :type estimators: array of :ref:`mlaut_estimator-label` objects
        :param estimators: :ref:`mlaut_estimator-label` objects. The trained models are set as a property to the object.
        """
        datasets = os.listdir(trained_models_dir)
        names_all_estimators = [estimator.properties()['name'] for estimator in estimators]
        for dts in datasets:
            X_train, X_test, y_train, y_test = self._data.load_test_train_dts(hdf5_out=self._output_io, 
                                                                              hdf5_in=self._input_io, 
                                                                              dts_name=dts, 
                                                                              dts_grp_path=self._original_datasets_group_h5_path)
            saved_estimators = os.listdir(f'{trained_models_dir}/{dts}')
            for saved_estimator in saved_estimators:
                name_estimator = saved_estimator.split('.')[0]
                try:
                    idx_estimator = names_all_estimators.index(name_estimator)
                    estimator = estimators[idx_estimator]
                    #preprocess data as per what was done during training
                    data_preprocessing = estimator.properties()['data_preprocessing']
                    X_train, X_test, y_train, y_test = self._preprocess_dataset(data_preprocessing,
                                                                                    X_train=X_train, 
                                                                                    X_test=X_test, 
                                                                                    y_train=y_train, 
                                                                                    y_test=y_test)
                    
                    estimator.load(f'{trained_models_dir}/{dts}/{saved_estimator}')
                    trained_estimator = estimator.get_trained_model()
                    predictions = trained_estimator.predict(X_test)
              
                    self._output_io.save_prediction_to_db(predictions=predictions, 
                                                        dataset_name=dts, 
                                                        strategy_name=name_estimator)
                    print(f'Predictions of estimator {name_estimator} on {dts} stored in database')
                except:
                    print(f'Skipping trained estimator {name_estimator}. Saved on disk but not instantiated.')
    
    def _preprocess_dataset(self, data_preprocessing, X_train, X_test, y_train, y_test):
        """
        Preprocesses the raw dataset according to the metadata attached to the estimator class.

        Parameters
        ----------
        data_preprocessing: dictionary
            dictionary with operations that need to be performed. The available values include:
            `normalize_features` and `normalize_labels`.
        X_train: array
            training array with the dataset features
        y_train: array
            training array with the dataset labels
        X_test: array
            test array with the dataset features
        y_test: array
            test array with the dataset labels

        Returns
        -------
            x_train_transformed(array): array with transformed features of the train set.
            y_train_transformed(array): array with transformed labels of the train set.
            x_test_transformed(array): array with transformed features on the test set.
            y_test_transformed(array): array with transformed labels on the test set.
        """
        
        x_train_transformed = X_train
        x_test_transformed = X_test
        y_train_transformed = y_train
        y_test_transformed = y_train

        if data_preprocessing['normalize_features'] is True:
            scaler_features = preprocessing.StandardScaler(copy=True, 
                                                           with_mean=True, 
                                                           with_std=True)
            scaler_features.fit(X_train)
            x_train_transformed  = scaler_features.transform(X_train)
            #apply the same transformation to the test set
            x_test_transformed = scaler_features.transform(X_test)
        
        if data_preprocessing['normalize_labels'] is True:
            scaler_labels = preprocessing.StandardScaler(copy=True, 
                                                           with_mean=True, 
                                                           with_std=True)
            scaler_labels.fit(y_train)
            y_train_transformed = scaler_labels.transform(y_train)
            #apply the same transformation to the test set
            y_test_transformed = scaler_labels.transform(y_test)

        return x_train_transformed, x_test_transformed, y_train_transformed, y_test_transformed
        