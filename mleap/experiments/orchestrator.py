from mleap.shared.static_variables import (X_TRAIN_DIR, 
                                       X_TEST_DIR, 
                                       Y_TRAIN_DIR, 
                                       Y_TEST_DIR, 
                                       TRAIN_IDX, 
                                       TEST_IDX, 
                                       EXPERIMENTS_PREDICTIONS_GROUP,
                                       EXPERIMENTS_TRAINED_MODELS_DIR, 
                                       LOG_ERROR_FILE, set_logging_defaults)
import sys
import os
from mleap.shared.files_io import FilesIO
from mleap.experiments.experiments import Experiments
from mleap.shared.files_io import DiskOperations
from mleap.data import Data
import numpy as np
import logging
class Orchestrator:
    """
    Orchestrates the sequencing of running the machine learning experiments.

    :type hdf5_input_io: :func:`~mleap.shared.files_io.FilesIO`
    :param hdf5_input_io: instance of :func:`~mleap.shared.files_io.FilesIO` with reference to the input file

    :type hdf5_output_io: :func:`~mleap.shared.files_io.FilesIO`
    :param hdf5_output_io: instance of :func:`~mleap.shared.files_io.FilesIO` with reference to the output file

    :type dts_names: array of strings
    :param dts_names: array with the names of the datasets on which experiments will be run.

    :type experiments_predictions_group: string
    :param experiments_predictions_group: path in HDF5 database where predictions will be saved

    :type experiments_trained_models_dir: string
    :param experiments_trained_models_dir: folder on disk where trained estimators will be saved.
    """
    def __init__(self, 
                 hdf5_input_io, 
                 hdf5_output_io,
                 dts_names,
                 original_datasets_group_h5_path, 
                 experiments_predictions_group=EXPERIMENTS_PREDICTIONS_GROUP,
                 experiments_trained_models_dir=EXPERIMENTS_TRAINED_MODELS_DIR):
        self.experiments_predictions_group=experiments_predictions_group
        self._experiments_trained_models_dir=experiments_trained_models_dir
        self._input_io = hdf5_input_io
        self._output_io = hdf5_output_io
        self._dts_names=dts_names
        self._original_datasets_group_h5_path = original_datasets_group_h5_path
        self._experiments = Experiments(self._experiments_trained_models_dir)
        self._disk_op = DiskOperations()
        self._data = Data() #TODO need to implement a way to change the defaults.
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

        :type modelling_strategies: array of :ref:`mleap_estimator-label` objects
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
                print(f'*** Training models on dataset: {dts_name}. Total datasets processed: {dts_trained}/{dts_total} ***')
                timestamps_df = self._experiments.run_experiments(X_train, 
                                                                                  y_train, 
                                                                                  modelling_strategies, 
                                                                                  dts_name)
                self._output_io.save_ml_strategy_timestamps(timestamps_df, dts_name)



        except KeyboardInterrupt:
            # quit
            print('***************************')
            print('********  EXITING  ********')
            print('***************************')

            sys.exit()

  

    def predict_all(self, trained_models_dir, estimators):
        """
        Make predictions on test sets

        :type trained_models_dir: string
        :param trained_models_dir: directory where the trained models are saved

        :type estimators: array of :ref:`mleap_estimator-label` objects
        :param estimators: :ref:`mleap_estimator-label` objects. The trained models are set as a property to the object.
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
                    estimator.load(f'{trained_models_dir}/{dts}/{saved_estimator}')
                    trained_estimator = estimator.get_trained_model()
                    if name_estimator == 'NeuralNetworkDeepClassifier':
                        predictions = trained_estimator.predict_classes(X_test)
                    else:
                        predictions = trained_estimator.predict(X_test)
              
                    self._output_io.save_prediction_to_db(predictions=predictions, 
                                                        dataset_name=dts, 
                                                        strategy_name=name_estimator)
                    print(f'Predictions of estimator {name_estimator} on {dts} stored in database')
                except:
                    print(f'Skipping trained estimator {name_estimator}. Saved on disk but not instantiated.')
                
