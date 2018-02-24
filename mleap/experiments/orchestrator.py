from ..shared.static_variables import (X_TRAIN_DIR, 
                                       X_TEST_DIR, 
                                       Y_TRAIN_DIR, 
                                       Y_TEST_DIR, 
                                       TRAIN_IDX, 
                                       TEST_IDX, 
                                       EXPERIMENTS_PREDICTIONS_DIR,
                                       EXPERIMENTS_TRAINED_MODELS_DIR, 
                                       LOG_ERROR_FILE, set_logging_defaults)
import sys
import os
from ..shared.files_io import FilesIO
from .experiments import Experiments
from ..shared.files_io import DiskOperations
from mleap.data import Data
import numpy as np
import logging
class Orchestrator:
    def __init__(self, 
                 hdf5_input_io, 
                 hdf5_output_io,
                 input_io_datasets_loc, #to be removed
                 output_io_split_idx_loc, #to be removed
                 original_datasets_group_h5_path, 
                 experiments_predictions_dir=EXPERIMENTS_PREDICTIONS_DIR,
                 experiments_trained_models_dir=EXPERIMENTS_TRAINED_MODELS_DIR):
        self._experiments_predictions_dir=experiments_predictions_dir
        self._experiments_trained_models_dir=experiments_trained_models_dir
        self._input_io = hdf5_input_io
        self._output_io = hdf5_output_io
        self._original_datasets_group_h5_path = original_datasets_group_h5_path
        self._input_io_datasets_loc = input_io_datasets_loc
        self._output_io_split_idx_loc = output_io_split_idx_loc
        self._experiments = Experiments(self._experiments_trained_models_dir)
        self._disk_op = DiskOperations()
        self._data = Data() #TODO need to implement a way to change the defaults.
        set_logging_defaults()

    def run(self, modelling_strategies):
        """ 
        Main module for training the estimators. 
        The inputs of the function are: 
            1. The input and output databse files containing the datasets.
            2. The instantiated estimators
        
        The method iterates through all datasets in the input database file 
        and trains all modelling strategies on these datasets. At the end of the process 
        we make predictions on the test sets and store them in the output database file.

        The class uses helper methods in the experiments class to avoid too many nested loops.
        """ 

        try:
            #loop through all datasets
            #self._trained_models_all_datasets = []
            #self._predictions_all_datasets = []
            dts_trained=0
            self._prediction_accuracies = []
            for dts in zip(self._input_io_datasets_loc, self._output_io_split_idx_loc):
                logging.log(1,f'Training estimators on {dts}')
                dataset_loc = dts[0]
                idx_loc = dts[1]
                # train_idx, _ = self._output_io.load_dataset_h5(idx_loc + '/' + TRAIN_IDX)
                # test_idx, _ = self._output_io.load_dataset_h5(idx_loc + '/' + TEST_IDX)

                # orig_dts, orig_dts_meta = self._input_io.load_dataset_pd(dataset_loc)
                # label_column = orig_dts_meta['class_name']
                # y = orig_dts[label_column]
                # X = orig_dts
                # X = X.drop(label_column, axis=1)

                # X_train = np.array(X.iloc[train_idx])
                # y_train = np.array(y.iloc[train_idx])

                # X_test = np.array(X.iloc[test_idx])
                # y_test = np.array(y.iloc[test_idx])

                # #train ml strategy
                # dts_name = orig_dts_meta['dataset_name']

                #dts_trained = len(self._trained_models_all_datasets)
                dts_trained +=1
                #dts_total = len(self._input_io_datasets_loc)
                X_train, y_train, X_test, y_test, dts_name, dts_total = self._load_test_train_splits(idx_loc,dataset_loc)
                print(f'*** Training models on dataset: {dts_name}. Total datasets processed: {dts_trained}/{dts_total} ***')
                trained_models, timestamps_df = self._experiments.run_experiments(X_train, 
                                                                                  y_train, 
                                                                                  modelling_strategies, 
                                                                                  dts_name)
                #self._trained_models_all_datasets.append(trained_models)
                self._output_io.save_ml_strategy_timestamps(timestamps_df, dts_name)

                
                #make predictions

                # for trained_model in trained_models:
                #     model_name = trained_model.properties()['name']
                #     if not self._output_io.check_h5_path_exists(f'{self._experiments_predictions_dir}/{dts_name}/{model_name}'):
                #         predictions = self._experiments.make_predictions(trained_models, dts, X_test)
                #         self._output_io.save_predictions_to_db(predictions, dts_name)
                #     else:
                #         #load predictions on test dataset if they were already saved in the db
                #         logging.warning(f'Preditions for {dts_name} already exist in H5 database. '
                #         'Predictions will be loaded from the H5 database instead of generating new ones. '
                #         'Delete predictions from H5 databse if using previously made predictions is not desired behaviour.')
                #         #predictions = self._output_io.load_predictions_for_dataset(dts_name)
                #         #self._predictions_all_datasets.append(predictions)


        except KeyboardInterrupt:
            # quit
            print('***************************')
            print('********  EXITING  ********')
            print('***************************')

            sys.exit()

    def _load_test_train_splits(self, split_idx_loc, orig_dataset_loc):
        train_idx, _ = self._output_io.load_dataset_h5(split_idx_loc + '/' + TRAIN_IDX)
        test_idx, _ = self._output_io.load_dataset_h5(split_idx_loc + '/' + TEST_IDX)
        orig_dts, orig_dts_meta = self._input_io.load_dataset_pd(orig_dataset_loc)
        label_column = orig_dts_meta['class_name']
        y = orig_dts[label_column]
        X = orig_dts
        X = X.drop(label_column, axis=1)

        X_train = np.array(X.iloc[train_idx])
        y_train = np.array(y.iloc[train_idx])

        X_test = np.array(X.iloc[test_idx])
        y_test = np.array(y.iloc[test_idx])

        dts_name = orig_dts_meta['dataset_name']
        dts_total = len(self._input_io_datasets_loc)
        return X_train, y_train, X_test, y_test, dts_name, dts_total
    
    def _find_dts_loc(self, dts_name):
        all_dataset_names = [dts.split('/')[-1] for dts in self._input_io_datasets_loc]
        try:
            idx_dts = all_dataset_names.index(dts_name)
            dataset_path = self._input_io_datasets_loc[idx_dts]
            split_path = self._output_io_split_idx_loc[idx_dts]

        except:
            logging.warning(f'Dataset {dts_name} not found in database. Unable to make a prediction')

    def predict_all(self, trained_models_dir, estimators):
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
                #try:
                idx_estimator = names_all_estimators.index(name_estimator)
                estimator = estimators[idx_estimator]
                estimator.load(f'{trained_models_dir}/{dts}/{saved_estimator}')
                trained_estimator = estimator.get_trained_model()
                predictions = trained_estimator.predict(X_test)
                self._output_io.save_prediction_to_db(predictions=predictions, 
                                                      dataset_name=dts, 
                                                      strategy_name=name_estimator)
                # except:
                #     logging.warning(f'Did not find trained estimator {name_estimator} for dataset{dts}')
          #make predictions

                # for trained_model in trained_models:
                #     model_name = trained_model.properties()['name']
                #     if not self._output_io.check_h5_path_exists(f'{self._experiments_predictions_dir}/{dts_name}/{model_name}'):
                #         predictions = self._experiments.make_predictions(trained_models, dts, X_test)
                #         self._output_io.save_predictions_to_db(predictions, dts_name)
                #     else:
                #         #load predictions on test dataset if they were already saved in the db
                #         logging.warning(f'Preditions for {dts_name} already exist in H5 database. '
                #         'Predictions will be loaded from the H5 database instead of generating new ones. '
                #         'Delete predictions from H5 databse if using previously made predictions is not desired behaviour.')
                #         #predictions = self._output_io.load_predictions_for_dataset(dts_name)
                #         #self._predictions_all_datasets.append(predictions)
