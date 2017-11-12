from src.delgado_datasets import DownloadAndConvertDelgadoDatasets
import os
from src.static_variables import DELGADO_DIR, DATA_DIR, EXPERIMENTS_TRAINED_MODELS_DIR
from sklearn.model_selection import train_test_split

class Data:
    '''
    Create the necessary directories.
    Download the data for the delgado datasets.
    Split dataset in test and train.
    Create HDF5 database and put all the datasets in it.
    '''
    def __init__(self):
        self._delgado = DownloadAndConvertDelgadoDatasets()
    
    def create_directories(self):
        if not os.path.exists(DATA_DIR):
            print('Creating directory:{0}'.format(DATA_DIR))
            os.makedirs(DATA_DIR)

        if not os.path.exists(DELGADO_DIR):
            print('Creating directory:{0}'.format(DELGADO_DIR))
            os.makedirs(DELGADO_DIR)

        if not os.path.exists(EXPERIMENTS_TRAINED_MODELS_DIR):
            print('Creating directory:{0}'.format(EXPERIMENTS_TRAINED_MODELS_DIR))
            os.makedirs(EXPERIMENTS_TRAINED_MODELS_DIR)

    def prepare_delgado_datasets(self): 
        datasets, dataset_names, metadata = self._delgado.download_and_extract_datasets()
        return datasets, dataset_names, metadata
        
    def create_train_test_split(self, dataset, metadata):
        class_name = metadata['class_name']
        y = dataset[class_name]
        X = dataset.loc[:, dataset.columns != class_name]
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.33, random_state=42)
        return X_train, X_test, y_train, y_test


