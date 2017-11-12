from src.delgado_datasets import DownloadAndConvertDelgadoDatasets
from src.create_test_train_dataset_split import CreateTestTrainDatasets
import os
from src.static_variables import DELGADO_DIR, DATA_DIR, EXPERIMENTS_TRAINED_MODELS_DIR

class Data:
    def __init__(self):
        self._delgado = DownloadAndConvertDelgadoDatasets()
        self._create_test_train_datasets = CreateTestTrainDatasets()
    
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

    
    
    def prepare_data(self): 
            self._create_directories()
            self._delgado.download_and_extract_datasets()
            self._create_test_train_datasets.create_test_train()
   

