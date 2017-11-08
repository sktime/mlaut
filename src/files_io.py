import os
from src.static_variables import EXPERIMENTS_TRAINED_MODELS_DIR, EXPERIMENTS_PREDICTIONS_DIR, EXPERIMENTS_MODEL_ACCURACY_DIR
from src.static_variables import PICKLE_EXTENTION
from src.static_variables import FLAG_ML_MODEL,FLAG_PREDICTIONS
from src.static_variables import HDF5_DATA_FILENAME, DATA_DIR
from src.static_variables import RUNTIMES_GROUP
from src.static_variables import RESULTS_DIR
import pickle
import h5py, tables
import numpy as np
import pandas as pd

class FilesIO:
    
    def check_file_exists(self, dataset_name, file_type):
        filename = ''
        if file_type == FLAG_ML_MODEL:
            filename = EXPERIMENTS_TRAINED_MODELS_DIR + dataset_name + PICKLE_EXTENTION
        elif file_type == FLAG_PREDICTIONS:
            filename = EXPERIMENTS_PREDICTIONS_DIR + dataset_name + PICKLE_EXTENTION
        else:
            raise ValueError('Please specify supported file type')
            
        file_exists = os.path.isfile(filename)
        if file_exists:
            return pickle.load(open(filename, 'rb'))
        else:
            return False
    
    def check_prediction_exists(self,dataset_name):
        f = h5py.File(DATA_DIR + HDF5_DATA_FILENAME)
        is_present = EXPERIMENTS_PREDICTIONS_DIR +  dataset_name in f
        f.close()
        return is_present
    def load_predictions_for_dataset(self, dataset_name):
        f = h5py.File(DATA_DIR + HDF5_DATA_FILENAME)
        predictions = f['/'+EXPERIMENTS_PREDICTIONS_DIR + dataset_name]
        
        predictions_for_dataset = []
        for strategy in list(predictions):
            predictions_for_dataset.append([strategy, predictions[strategy][...]])
        return predictions_for_dataset

        
    def save_trained_models_to_disk(self, trained_models, dataset_name):
        with open(EXPERIMENTS_TRAINED_MODELS_DIR + dataset_name + PICKLE_EXTENTION,'wb') as f:
            pickle.dump(trained_models,f)
    
    def save_predictions_to_db(self, predictions, dataset_name):
        f = h5py.File(DATA_DIR + HDF5_DATA_FILENAME)
        for prediction in predictions:
            strategy_name = prediction[0]
            strategy_predictions = np.array(prediction[1])
            f[EXPERIMENTS_PREDICTIONS_DIR + dataset_name  + '/' + strategy_name] = strategy_predictions
        f.close()

    
    def save_prediction_accuracies_to_db(self, model_accuracies):
        for model in model_accuracies:
            model_name = model[0]
            model_accuracy = np.array([model[1]])
            
            #create group if necessary
            f =  h5py.File(DATA_DIR + HDF5_DATA_FILENAME, 'a')
            if not '/' + EXPERIMENTS_MODEL_ACCURACY_DIR in f:
                f =  h5py.File(DATA_DIR + HDF5_DATA_FILENAME, 'a')
                f.create_group('/' + EXPERIMENTS_MODEL_ACCURACY_DIR)
            f.close()            
            #create array with accuracies or append it
            f = tables.open_file(DATA_DIR + HDF5_DATA_FILENAME, 'a')
            if not  '/' + EXPERIMENTS_MODEL_ACCURACY_DIR + model_name in f: 
                f.create_earray('/' +EXPERIMENTS_MODEL_ACCURACY_DIR, name=model_name, obj=model_accuracy)
            else:
                mdl_acc = getattr(f.root.experiments.trained_models_accuracies, model_name)
                mdl_acc.append(model_accuracy)
            f.close()
            
    def get_prediction_accuracies_per_strategy(self):
        pred_accuracies = {}
        f = h5py.File(DATA_DIR + HDF5_DATA_FILENAME)
        strategies = f[EXPERIMENTS_MODEL_ACCURACY_DIR]
        for strategy in strategies:
            pred_accuracies[strategy] = strategies[strategy][...]
        return pred_accuracies
    
    def save_ml_strategy_timestamps(self, timestamps_df, dataset_name):
        store = pd.HDFStore(DATA_DIR + HDF5_DATA_FILENAME)
        store[RUNTIMES_GROUP + '/' + dataset_name ] = timestamps_df
        store.close()
    
    def save_stat_test_result(self, stat_test_dataset, values):
        store = pd.HDFStore(DATA_DIR + HDF5_DATA_FILENAME)
        store['/' + RESULTS_DIR + stat_test_dataset] = values
        store.close()