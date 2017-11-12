from src.static_variables import (FLAG_ML_MODEL, REFORMATTED_DATASETS_DIR, 
    T_TEST_DATASET, SIGN_TEST_DATASET, 
    BONFERRONI_CORRECTION_DATASET, WILCOXON_DATASET, 
    FRIEDMAN_DATASET, SPLIT_DATASETS_DIR, 
    X_TRAIN_DIR, X_TEST_DIR, Y_TRAIN_DIR, Y_TEST_DIR)
import sys

class TestOrchestrator:
    def __init__(self,data):
        self._data = data
        self._experiments = None
        self._files_io = None
        self._analyzeResults = None
     
    def prepare_data(self):
        '''
        creates directories
        get datesets and seves them in HDF5 database
        Creates train / test split
        '''
        #create directories
        self._data.create_directories()
        #delgado datasets
        datasets, dataset_names, metadata = self._data.prepare_delgado_datasets()
        #add full path to dataset name
        reformatted_dataset_names = [ REFORMATTED_DATASETS_DIR + dts_name for dts_name in dataset_names]
        self._files_io.save_datasets(datasets, reformatted_dataset_names, 
            metadata, verbose=True)
        
        #create train /test split
        for dts in zip(datasets, dataset_names, metadata):
            dts_name = dts[1]
            X_train, X_test, y_train, y_test = self._data.create_train_test_split(dts[0], dts[2])
            #create save path
            data = [X_train, X_test, y_train, y_test]
            names = [
                SPLIT_DATASETS_DIR + dts_name + X_TRAIN_DIR,
                SPLIT_DATASETS_DIR + dts_name + X_TEST_DIR,
                SPLIT_DATASETS_DIR + dts_name + Y_TRAIN_DIR,
                SPLIT_DATASETS_DIR + dts_name + Y_TEST_DIR
            ]
            #add same metadata for all datasets
            metadata = [dts[2]['source']] *4
            self._files_io.save_datasets(data, names, metadata, verbose=True)        
    
    def setExperiments(self, experiments):
        self._experiments = experiments
    
    def set_ml_models_container(self, models_container):
        self.ml_models_container = models_container
    
    def setFilesIO(self, filesIO):
        self._files_io = filesIO
        
    def setAnalyzeResults(self, analyze):
        self._analyzeResults = analyze
        
    def run_experiments(self):
        try:
            #loop through all datasets
            self._trained_models_all_datasets = []
            self._predictions_all_datasets = []
            self._prediction_accuracies = []
            dataset_containers = self.map_datasets.map()
            for dts in dataset_containers: 
                dataset_name = dts.dataset_name
                #train ml strategy
                if not self._files_io.check_file_exists(dataset_name,FLAG_ML_MODEL):
                    #train model if file does not exist on disk
                    dts_trained = len(self._trained_models_all_datasets)
                    dts_total = len(dataset_containers)
                    print('*** Training models on dataset: {0}. Total datasets processed: {1}/{2} ***'.format(dataset_name, dts_trained,dts_total))
                    trained_models, timestamps_df = self._experiments.run_experiments(dts)
                    self._trained_models_all_datasets.append(trained_models)
                    self._files_io.save_trained_models_to_disk(trained_models,dataset_name)
                    self._files_io.save_ml_strategy_timestamps(timestamps_df, dataset_name)
                else:
                    #if model was already trained load it from the pickle
                    trained_models = self._files_io.check_file_exists(dataset_name,FLAG_ML_MODEL)
                    self._trained_models_all_datasets.append(trained_models)
                
                #make predictions
                if not self._files_io.check_prediction_exists(dataset_name):
                    #it is used for choosing the directory in which to look for the saved file
                    predictions = self._experiments.make_predictions(trained_models, dataset_name)
                    self._predictions_all_datasets.append(predictions)
                    self._files_io.save_predictions_to_db(predictions, dataset_name)
                else:
                    #load predictions on test dataset if they were already saved in the db
                    predictions = self._files_io.load_predictions_for_dataset(dataset_name)
                    self._predictions_all_datasets.append(predictions)
                    
                #calculate accuracy of predictions
                model_accuracies = self._experiments.calculate_prediction_accuracy(predictions_per_ml_strategy=predictions, 
                                                                dataset_name=dataset_name)
                self._prediction_accuracies.append(model_accuracies)
                self._files_io.save_prediction_accuracies_to_db(model_accuracies)
            else:
                print('Ignoring dataset: {0}'.format(dataset_name))
        
        except KeyboardInterrupt:
            # quit
            print('***************************')
            print('********  EXITING  ********')
            print('***************************')

            sys.exit()

    def perform_statistical_tests(self):
        pred_acc_dict = {}
        try:
            pred_acc_dict = self._analyzeResults.convert_prediction_acc_from_array_to_dict(self._prediction_accuracies)
        except:
            pred_acc_dict = self._files_io.get_prediction_accuracies_per_strategy()
        
        #t-test
        _, values_df = self._analyzeResults.perform_t_test(pred_acc_dict)
        self._files_io.save_stat_test_result(self, T_TEST_DATASET, values_df)
        
        #sign test
        _, values_df = self._analyzeResults.perform_sign_test(pred_acc_dict)
        self._files_io.save_stat_test_result(self, SIGN_TEST_DATASET, values_df)
        
        #t-test with Bonferroni correction test
        _, values_df = self._analyzeResults.perform_t_test_with_bonferroni_correction(pred_acc_dict)
        self._files_io.save_stat_test_result(self, BONFERRONI_CORRECTION_DATASET, values_df)
        
        #Wilcoxon test
        _, values_df =  self._analyzeResults.perform_wilcoxon(pred_acc_dict)
        self._files_io.save_stat_test_result(self, WILCOXON_DATASET, values_df)
        
        #Firedman test
        _, values_df = self._analyzeResults.perform_friedman_test(pred_acc_dict)
        self._files_io.save_stat_test_result(self, FRIEDMAN_DATASET, values_df)
        
                
            
