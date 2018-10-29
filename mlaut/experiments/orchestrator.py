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

    Args:
        hdf5_input_io (:func:`~mlaut.shared.files_io.FilesIO`): instance of :func:`~mlaut.shared.files_io.FilesIO` with reference to the input file.
        hdf5_output_io (:func:`~mlaut.shared.files_io.FilesIO`): instance of :func:`~mlaut.shared.files_io.FilesIO` with reference to the output file.
        dts_names (array of strings): array with the names of the datasets on which experiments will be run.
        original_datasets_group_h5_path (sting): root path where the raw datasets are stored
        experiments_predictions_group (string): path in HDF5 database where predictions will be saved.
        experiments_trained_models_dir (string): folder on disk where trained estimators will be saved.
        split_datasets_group (string): path in HDF5 database where the splits are saved.
        train_idx (string): folder in HDF5 database which holds the train index splits.
        test_idx (string): folder in HDF5 database which holds the test index splits.
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
        if not isinstance(dts_names, list):
            raise ValueError('dts_names must be an array')
        self._experiments_predictions_group=experiments_predictions_group
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

    def run(self, 
            modelling_strategies, 
            overwrite_saved_models=False, 
            verbose=True, 
            predict_on_runtime=True,
            overwrite_predictions=False,
            overwrite_timestamp=False):
        """ 
        Main module for training the estimators. 
        The inputs of the function are: 

        1. The input and output databse files containing the datasets.

        2. The instantiated estimators.
        
        The method iterates through all datasets in the input database file 
        and trains all modelling strategies on these datasets. At the end of the process 
        we make predictions on the test sets and store them in the output database file.

        The class uses helper methods in the experiments class to avoid too many nested loops.

        Args:
            odelling_strategies (array of :ref:`mlaut_estimator-label` objects): Array of estimators that will be used for training
            overwrite_saved_models (Boolean): Flag whether the trained models should be overwritten if they already exist on the disk.
            verbose (Boolean): If True prints info and warning messages.
            predict_on_runtime(Boolean): Make predictions immediately after the estimators are trained.
            overwrite_predictions(Boolean): Overwrite predictions in database if they exist already.
        """ 

        try:
            #loop through all datasets
            dts_trained=0
            dts_total = len(self._dts_names)
            self._prediction_accuracies = []
            for dts_name in self._dts_names:
                logging.log(1,f'Training estimators on {dts_name}')

                dts_trained +=1
                X_train, X_test, y_train, _ = self._data.load_test_train_dts(hdf5_out=self._output_io, 
                                                                              hdf5_in=self._input_io, 
                                                                              dts_name=dts_name, 
                                                                              dts_grp_path=self._original_datasets_group_h5_path)

                timestamps_df = pd.DataFrame()
                for modelling_strategy in modelling_strategies:
                    ml_strategy_name = modelling_strategy.properties['name']
                    begin_timestamp = datetime.now()

                    #check whether the model was already trained
                    path_to_check = self._experiments_trained_models_dir + os.sep + dts_name + os.sep + ml_strategy_name + '.*'
                    model_exists, path_to_model = self._disk_op.check_path_exists(path_to_check)
                    if model_exists is True and overwrite_saved_models is False:
                        
                        # modelling_strategy.load(path_to_model)
                        if verbose is True:
                            logging.info(f'Estimator {ml_strategy_name} already trained on {dts_name}. Skipping it.')

                    else: #modelling strategy does not exist
           
                        num_samples, input_dim = X_train.shape
                        unique_labels = np.unique(y_train)
                        num_classes = len(unique_labels) 
                        built_model = modelling_strategy.build(num_classes=num_classes, 
                                                                input_dim=input_dim,
                                                                num_samples=num_samples)
                        if verbose is True:
                            logging.info(f'** Training estimator: {ml_strategy_name} on dataset: {dts_name}. Datasets processed: {dts_trained}/{dts_total} **')
                        try:
                            trained_model = built_model.fit(X_train, y_train)
                            timestamps_df = self._record_timestamp(ml_strategy_name, begin_timestamp, timestamps_df)
                            modelling_strategy.set_trained_model(trained_model)
                            modelling_strategy.save(dts_name)
                        
                            self._output_io.save_ml_strategy_timestamps(timestamps_df, dts_name, overwrite_timestamp=overwrite_timestamp)
                        
                            #make predictions
                            if predict_on_runtime is True:
                                self._predict(modelling_strategy, 
                                        X_test, 
                                        dataset_name=dts_name, 
                                        overwrite=overwrite_predictions, 
                                        verbose=verbose)
                        
                            trained_model = None
                            modelling_strategy = None
                            built_model = None
                        
                        except Exception as e:
                            print(f'Failed to train dataset {ml_strategy_name} on dataset: {dts_name}')
                            logging.error(f'Failed to train dataset {ml_strategy_name} on dataset: {dts_name}')
                            logging.error(f'*****Stack trace: {e}')
                    
                   



        except KeyboardInterrupt:
            # quit
            print('***************************')
            print('********  EXITING  ********')
            print('***************************')

            sys.exit()

    def _record_timestamp(self, strategy_name, begin_timestamp, timestamps_df):
        """ Timestamp used to record the duration of training for each of the estimators"""
        STF = '%Y-%m-%d %H:%M:%S'
        end_timestamp = datetime.now()
        diff = (end_timestamp - begin_timestamp).total_seconds() 
        vals = [strategy_name, begin_timestamp.strftime(STF),end_timestamp.strftime(STF),diff]
        run_time_df = pd.DataFrame([vals], columns=['strategy_name','begin_time','end_time','total_seconds'])
        timestamps_df = timestamps_df.append(run_time_df)
        
        return timestamps_df

    def _predict(self, modelling_strategy, X_test, dataset_name, overwrite=False, verbose=True):
        """
        Make predictions on test sets. The algorithm opens all saved estimators in the output directory and checks whether their names are specified in the estimators array. If they are it fetches the dataset splits and tries to make the predictions.

        Args:
            modelling_strategy (:ref:`mlaut_estimator-label` object): The trained estimator object.
            X_test(numpy array): Test dataset.
            dataset_name(string): name of the dataset on which the estimator was trained.
            overwrite (Boolean): If True overwrites predictions in HDF5 database.
            verbose (Boolean): If True prints info and warning messages.
        """
        trained_estimator = modelling_strategy.get_trained_model()
        name_estimator = modelling_strategy.properties['name']
        if overwrite is True:
            predictions = trained_estimator.predict(X_test)
            self._output_io.save_prediction_to_db(predictions=predictions, 
                                                        dataset_name=dataset_name, 
                                                        strategy_name=name_estimator)
            logging.info(f'Predictions for {name_estimator} on {dataset_name} Saved in database.')
        
        else:
            #check whether the prediction exists before proceeding
            path_h5_predictions = f'{self._experiments_predictions_group}/{dataset_name}/{name_estimator}'
            predictions_exist = self._output_io.check_h5_path_exists(path_h5_predictions)

            if predictions_exist is True:
                logging.info(f'Predictions for {name_estimator} on {dataset_name} already exist in the database. Set overwrite to True if you wish replace them.')
            else:
                predictions = trained_estimator.predict(X_test)
                self._output_io.save_prediction_to_db(predictions=predictions, 
                                                      dataset_name=dataset_name, 
                                                      strategy_name=name_estimator)
                logging.info(f'Predictions for {name_estimator} on {dataset_name} saved in database.')
        
        trained_estimator = None

    def predict_all(self, trained_models_dir, estimators, overwrite=False, verbose=True):
        """
        Make predictions on test sets. The algorithm opens all saved estimators in the output directory and checks whether their names are specified in the estimators array. If they are it fetches the dataset splits and tries to make the predictions.

        Args:
            trained_models_dir (string): directory where the trained models are saved.
            estimators (array of :ref:`mlaut_estimator-label` objects): The trained models are set as a property to the object.
            overwrite (Boolean): If True overwrites predictions in HDF5 database.
            verbose (Boolean): If True prints info and warning messages.
        """
        datasets = os.listdir(trained_models_dir)
        names_all_estimators = [estimator.properties['name'] for estimator in estimators]
        for dts in self._dts_names:
            X_train, X_test, y_train, y_test = self._data.load_test_train_dts(hdf5_out=self._output_io, 
                                                                              hdf5_in=self._input_io, 
                                                                              dts_name=dts, 
                                                                              dts_grp_path=self._original_datasets_group_h5_path)
            saved_estimators = os.listdir(f'{trained_models_dir}/{dts}')
            for saved_estimator in saved_estimators:
                name_estimator = saved_estimator.split('.')[0]
                # try:
                if name_estimator in names_all_estimators:
                    #check whether predictions exist in the database before continuing
                    if overwrite is False:
                        path_h5_predictions = f'{self._experiments_predictions_group}/{dts}/{name_estimator}'
                        predictions_exist = self._output_io.check_h5_path_exists(path_h5_predictions)
                        if predictions_exist is True:
                            if verbose is True:
                                logging.info(f'Predictions for {name_estimator} on {dts} already exist in the database. Set overwrite to True if you wish replace them.')
                            continue
                    
                    #if overwrite is set to True make the predictions without checking
                    idx_estimator = names_all_estimators.index(name_estimator)
                    estimator = estimators[idx_estimator]

                    estimator.load(f'{trained_models_dir}/{dts}/{saved_estimator}')
                    trained_estimator = estimator.get_trained_model()
                    predictions = trained_estimator.predict(X_test)
                    self._output_io.save_prediction_to_db(predictions=predictions, 
                                                        dataset_name=dts, 
                                                        strategy_name=name_estimator)
                   
        