from ..shared.static_variables import (FLAG_ML_MODEL, 
    X_TRAIN_DIR, X_TEST_DIR, Y_TRAIN_DIR, Y_TEST_DIR)
import sys
from ..shared.files_io import FilesIO
from .run_experiments import RunExperiments
class TestOrchestrator:
    def __init__(self, hdf5_input_path, hdf5_output_path):
        self._input_io = FilesIO(hdf5_input_path)
        self._output_io = FilesIO(hdf5_output_path)
        self._experiments = RunExperiments()

    def split_datasets(self, dataset_paths, split_datasets_dir, test_size=0.33, verbose=False):
        for dts_loc in dataset_paths:
            #split
            X_train, X_test, y_train, y_test = self._input_io.split_dataset(dts_loc, test_size)
            #load metadata
            _, metadata = self._input_io.load_dataset(dts_loc)
            class_name = metadata['class_name']
            dataset_name = metadata['dataset_name']
            #save
            save_split_dataset_paths = [
                split_datasets_dir + '/' + dataset_name + X_TRAIN_DIR,
                split_datasets_dir + '/' + dataset_name + X_TEST_DIR,
                split_datasets_dir + '/' + dataset_name + Y_TRAIN_DIR,
                split_datasets_dir + '/' + dataset_name + Y_TEST_DIR,
            ]
            meta = [{'dataset_name': dataset_name}]*4
            self._output_io.save_datasets(datasets=[X_train, X_test, y_train, y_test],
                                          datasets_save_paths=save_split_dataset_paths,
                                          dts_metadata=meta,
                                          verbose=verbose
                                            )
    def run_experiments(self, datasets, modelling_strategies):
        try:
            #loop through all datasets
            self._trained_models_all_datasets = []
            self._predictions_all_datasets = []
            self._prediction_accuracies = []
            for dts in datasets:
                X_train, _ = self._output_io.load_dataset(dts + X_TRAIN_DIR)
                X_test, _  = self._output_io.load_dataset(dts + X_TEST_DIR)
                y_train, _ = self._output_io.load_dataset(dts + Y_TRAIN_DIR)
                y_test, _ = self._output_io.load_dataset(dts + Y_TEST_DIR)
                #train ml strategy
                dts_name = dts.split('/')[-1]
                if not self._output_io.check_file_exists(dts_name, FLAG_ML_MODEL):
                    #train model if file does not exist on disk
                    dts_trained = len(self._trained_models_all_datasets)
                    dts_total = len(datasets)
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
                    
                #calculate accuracy of predictions
                model_accuracies = self._experiments.calculate_prediction_accuracy(predictions_per_ml_strategy=predictions, 
                                                                true_labels=y_test)
                self._prediction_accuracies.append(model_accuracies)
                self._output_io.save_prediction_accuracies_to_db(model_accuracies)
                
        
        except KeyboardInterrupt:
            # quit
            print('***************************')
            print('********  EXITING  ********')
            print('***************************')

            sys.exit()

            
