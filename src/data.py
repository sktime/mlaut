from src.create_directories import CreateDirs
from src.delgado_datasets import DownloadAndConvertDelgadoDatasets
from src.create_test_train_dataset_split import CreateTestTrainDatasets


class Data:

    def __init__(self):
        self._create_dirs = CreateDirs()
        self._delgado = DownloadAndConvertDelgadoDatasets()
        self._create_test_train_datasets = CreateTestTrainDatasets()
    
    def prepare_data(self): 
            self._create_dirs.create_directories()
            self._delgado.download_and_extract_datasets()
            self._create_test_train_datasets.create_test_train()
   

