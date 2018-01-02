from ..shared.static_variables import (FLAG_ML_MODEL, 
    X_TRAIN_DIR, X_TEST_DIR, Y_TRAIN_DIR, Y_TEST_DIR, TRAIN_IDX, TEST_IDX)
import sys
from ..shared.files_io import FilesIO
from .experiments import Experiments
class TestOrchestrator:
    def __init__(self, hdf5_input_io, hdf5_output_io):
        self._input_io = hdf5_input_io
        self._output_io = hdf5_output_io
        self._experiments = Experiments()

    def run(self, input_io_datasets_loc, output_io_split_idx_loc, modelling_strategies):
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
            self._trained_models_all_datasets = []
            self._predictions_all_datasets = []
            self._prediction_accuracies = []
            for dts in zip(input_io_datasets_loc, output_io_split_idx_loc):
                dataset_loc = dts[0]
                idx_loc = dts[1]
                train_idx, _ = self._output_io.load_dataset_h5(idx_loc + '/' + TRAIN_IDX)
                test_idx, _ = self._output_io.load_dataset_h5(idx_loc + '/' + TEST_IDX)

                orig_dts, orig_dts_meta = self._input_io.load_dataset_pd(dataset_loc)
                label_column = orig_dts_meta['class_name']
                y = orig_dts[label_column]
                X = orig_dts
                X = X.drop(label_column, axis=1)

                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]

                X_test = X.iloc[test_idx]
                y_test = y.iloc[test_idx]

                #train ml strategy
                dts_name = orig_dts_meta['dataset_name']
                if not self._output_io.check_file_exists(dts_name, FLAG_ML_MODEL):
                    #train model if file does not exist on disk
                    dts_trained = len(self._trained_models_all_datasets)
                    dts_total = len(input_io_datasets_loc)
                    print(f'*** Training models on dataset: {dts_name}. Total datasets processed: {dts_trained}/{dts_total} ***')
                    trained_models, timestamps_df = self._experiments.run_experiments(X_train, y_train, modelling_strategies)
                    self._trained_models_all_datasets.append(trained_models)
                    self._output_io.save_trained_models_to_disk(trained_models,dts_name)
                    self._output_io.save_ml_strategy_timestamps(timestamps_df, dts_name)
                else:
                    #if model was already trained load it from the pickle
                    trained_models = self._output_io.check_file_exists(dts,FLAG_ML_MODEL)
                    self._trained_models_all_datasets.append(trained_models)
                
                #make predictions
                if not self._output_io.check_prediction_exists(dts_name):
                    #it is used for choosing the directory in which to look for the saved file
                    predictions = self._experiments.make_predictions(trained_models, dts, X_test)
                    self._predictions_all_datasets.append(predictions)
                    self._output_io.save_predictions_to_db(predictions, dts_name)
                else:
                    #load predictions on test dataset if they were already saved in the db
                    predictions = self._output_io.load_predictions_for_dataset(dts_name)
                    self._predictions_all_datasets.append(predictions)
                    
                # #calculate accuracy of predictions
                # #TODO this needs to be moved to analyze rezults
                # model_accuracies = self._experiments.calculate_prediction_accuracy(predictions_per_ml_strategy=predictions, 
                #                                                 true_labels=y_test)
                # self._prediction_accuracies.append(model_accuracies)
                # self._output_io.save_prediction_accuracies_to_db(model_accuracies)
                
        
        except KeyboardInterrupt:
            # quit
            print('***************************')
            print('********  EXITING  ********')
            print('***************************')

            sys.exit()

            
