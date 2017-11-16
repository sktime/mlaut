from src.delgado_datasets import DownloadAndConvertDelgadoDatasets
import os
from src.static_variables import (DELGADO_DIR, DATA_DIR, 
    EXPERIMENTS_TRAINED_MODELS_DIR, SPLIT_DATASETS_DIR, REFORMATTED_DATASETS_DIR,
    X_TRAIN_DIR, X_TEST_DIR, Y_TRAIN_DIR, Y_TEST_DIR)
from sklearn.model_selection import train_test_split

class Data:
    '''
    Create the necessary directories.
    Download the data for the delgado datasets.
    Split dataset in test and train.
    Create HDF5 database and put all the datasets in it.
    '''
    def __init__(self, files_io):
        self._delgado = DownloadAndConvertDelgadoDatasets()
        self._files_io = files_io
    
    def _create_directories(self):
        if not os.path.exists(DATA_DIR):
            print('Creating directory:{0}'.format(DATA_DIR))
            os.makedirs(DATA_DIR)

        if not os.path.exists(DELGADO_DIR):
            print('Creating directory:{0}'.format(DELGADO_DIR))
            os.makedirs(DELGADO_DIR)

        if not os.path.exists(EXPERIMENTS_TRAINED_MODELS_DIR):
            print('Creating directory:{0}'.format(EXPERIMENTS_TRAINED_MODELS_DIR))
            os.makedirs(EXPERIMENTS_TRAINED_MODELS_DIR)

    def _prepare_delgado_datasets(self): 
        datasets, dataset_names, metadata = self._delgado.download_and_extract_datasets()
        return datasets, dataset_names, metadata
        
    def _create_train_test_split(self, dataset, metadata):
        class_name = metadata['class_name']
        y = dataset[class_name]
        X = dataset.loc[:, dataset.columns != class_name]
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.33, random_state=42)
        return X_train, X_test, y_train, y_test

    def prepare_data(self):
        '''
        creates directories
        get datesets and seves them in HDF5 database
        Creates train / test split
        '''
        #create directories
        self._create_directories()
        #delgado datasets
        datasets, dataset_names, metadata = self._prepare_delgado_datasets()
        #add full path to dataset name
        reformatted_dataset_names = [ REFORMATTED_DATASETS_DIR + dts_name for dts_name in dataset_names]
        self._files_io.save_datasets(datasets, reformatted_dataset_names, 
            metadata, verbose=True)
        
        #create train /test split
        for dts in zip(datasets, dataset_names, metadata):
            dts_name = dts[1]
            X_train, X_test, y_train, y_test = self._create_train_test_split(dts[0], dts[2])
            #create save path
            data = [X_train, X_test, y_train, y_test]
            names = [
                SPLIT_DATASETS_DIR + '/' + dts_name + X_TRAIN_DIR,
                SPLIT_DATASETS_DIR + '/' + dts_name + X_TEST_DIR,
                SPLIT_DATASETS_DIR + '/' + dts_name + Y_TRAIN_DIR,
                SPLIT_DATASETS_DIR + '/' + dts_name + Y_TEST_DIR
            ]
            #add same metadata for all datasets
            metadata = [dts[2]['source']] *4
            self._files_io.save_datasets(data, names, metadata, verbose=True)  

