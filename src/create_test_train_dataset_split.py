import pandas as pd
from sklearn.utils import shuffle
from src.static_variables import REFORMATTED_DATASETS_DIR, TRAIN_DIR
from src.static_variables import TEST_DIR, TEST_TRAIN_SPLIT
from src.static_variables import DATA_DIR, HDF5_DATA_FILENAME, SPLIT_DATASETS_DIR
import h5py

'''
TODO This will probably have to go
Test/train split will be performed before each experiment is run

'''


class CreateTestTrainDatasets:

    datasets = []
    def _load_datasets(self):
         f = h5py.File(DATA_DIR + HDF5_DATA_FILENAME)
         for i in f[REFORMATTED_DATASETS_DIR].items():
             self.datasets.append( REFORMATTED_DATASETS_DIR + i[0])
         f.close()
         
    def _split_dataset(self,store, dataset_name):
        datasetDf = store[dataset_name]
        num_samples = datasetDf.shape[0]
        
        #split the data in train and test
        split = TEST_TRAIN_SPLIT
        
        #shuffle
        #TODO: Remove random_state
        datasetDf = shuffle(datasetDf, random_state=100)
        
        test_samples = int(num_samples * split)
        test = datasetDf.iloc[0:test_samples,:]
        train = datasetDf.iloc[test_samples:num_samples,:]

        return test, train
    
    
    def create_test_train(self):
        self._load_datasets()
        store = pd.HDFStore(DATA_DIR + HDF5_DATA_FILENAME)         
        for dataset in self.datasets:
            dataset_name = dataset.split('/')[-1]
            test, train = self._split_dataset(store, dataset)
            print('Saving dataset: {0}'.format(dataset))
            store[SPLIT_DATASETS_DIR + TEST_DIR + dataset_name] = test
            store[SPLIT_DATASETS_DIR + TRAIN_DIR  + dataset_name] = train

        store.close()

