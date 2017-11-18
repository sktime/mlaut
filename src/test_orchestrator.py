from src.static_variables import (FLAG_ML_MODEL, REFORMATTED_DATASETS_DIR, 
    T_TEST_DATASET, SIGN_TEST_DATASET, 
    BONFERRONI_CORRECTION_DATASET, WILCOXON_DATASET, 
    FRIEDMAN_DATASET, 
    X_TRAIN_DIR, X_TEST_DIR, Y_TRAIN_DIR, Y_TEST_DIR)
import sys
from src.files_io import FilesIO
class TestOrchestrator:
    def __init__(self, SPLIT_DATASETS_DIR, files_io, experiments):
        self._experiments = experiments
        self._files_io = files_io
        self.SPLIT_DATASETS_DIR = SPLIT_DATASETS_DIR
     
    def run_experiments(self, datasets, modelling_strategies):
        try:
            #loop through all datasets
            self._trained_models_all_datasets = []
            self._predictions_all_datasets = []
            self._prediction_accuracies = []
            for dts in datasets:
                X_train, _ = self._files_io.load_dataset(self.SPLIT_DATASETS_DIR + '/'+dts + X_TRAIN_DIR)
                X_test, _  = self._files_io.load_dataset(self.SPLIT_DATASETS_DIR + '/'+dts + X_TEST_DIR)
                y_train, _ = self._files_io.load_dataset(self.SPLIT_DATASETS_DIR + '/'+dts + Y_TRAIN_DIR)
                y_test, _ = self._files_io.load_dataset(self.SPLIT_DATASETS_DIR + '/'+dts + Y_TEST_DIR)
                #train ml strategy
                if not self._files_io.check_file_exists(dts, FLAG_ML_MODEL):
                    #train model if file does not exist on disk
                    dts_trained = len(self._trained_models_all_datasets)
                    dts_total = len(datasets)
                    print(f'*** Training models on dataset: {dts}. Total datasets processed: {dts_trained}/{dts_total} ***')
                    trained_models, timestamps_df = self._experiments.run_experiments(X_train, y_train, modelling_strategies)
                    self._trained_models_all_datasets.append(trained_models)
                    self._files_io.save_trained_models_to_disk(trained_models,dts)
                    self._files_io.save_ml_strategy_timestamps(timestamps_df, dts)
                else:
                    #if model was already trained load it from the pickle
                    trained_models = self._files_io.check_file_exists(dts,FLAG_ML_MODEL)
                    self._trained_models_all_datasets.append(trained_models)
                
                #make predictions
                if not self._files_io.check_prediction_exists(dts):
                    #it is used for choosing the directory in which to look for the saved file
                    predictions = self._experiments.make_predictions(trained_models, dts, X_test)
                    self._predictions_all_datasets.append(predictions)
                    self._files_io.save_predictions_to_db(predictions, dts)
                else:
                    #load predictions on test dataset if they were already saved in the db
                    predictions = self._files_io.load_predictions_for_dataset(dts)
                    self._predictions_all_datasets.append(predictions)
                    
                #calculate accuracy of predictions
                model_accuracies = self._experiments.calculate_prediction_accuracy(predictions_per_ml_strategy=predictions, 
                                                                true_labels=y_test)
                self._prediction_accuracies.append(model_accuracies)
                self._files_io.save_prediction_accuracies_to_db(model_accuracies)
                
        
        except KeyboardInterrupt:
            # quit
            print('***************************')
            print('********  EXITING  ********')
            print('***************************')

            sys.exit()

    # def perform_statistical_tests(self):
    #     pred_acc_dict = {}
    #     try:
    #         pred_acc_dict = self._analyzeResults.convert_prediction_acc_from_array_to_dict(self._prediction_accuracies)
    #     except:
    #         pred_acc_dict = self._files_io.get_prediction_accuracies_per_strategy()
        
    #     #t-test
    #     _, values_df = self._analyzeResults.perform_t_test(pred_acc_dict)
    #     self._files_io.save_stat_test_result(T_TEST_DATASET, values_df)
        
    #     #sign test
    #     _, values_df = self._analyzeResults.perform_sign_test(pred_acc_dict)
    #     self._files_io.save_stat_test_result(SIGN_TEST_DATASET, values_df)
        
    #     #t-test with Bonferroni correction test
    #     _, values_df = self._analyzeResults.perform_t_test_with_bonferroni_correction(pred_acc_dict)
    #     self._files_io.save_stat_test_result(BONFERRONI_CORRECTION_DATASET, values_df)
        
    #     #Wilcoxon test
    #     _, values_df =  self._analyzeResults.perform_wilcoxon(pred_acc_dict)
    #     self._files_io.save_stat_test_result(WILCOXON_DATASET, values_df)
        
    #     #Firedman test
    #     _, values_df = self._analyzeResults.perform_friedman_test(pred_acc_dict)
    #     self._files_io.save_stat_test_result(FRIEDMAN_DATASET, values_df)
        
                
            
