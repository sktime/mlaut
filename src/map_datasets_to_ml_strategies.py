from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from src.static_variables import TRAIN_DIR, TEST_DIR, DATA_DIR, HDF5_DATA_FILENAME, SPLIT_DATASETS_DIR 
import h5py


class DatasetContainer(object):
    test_dataset_path = ''
    train_dataset_path =''
    dataset_name = ''
    column_label_name = ''
    strategy_names = []
    strategies = []
    hyper_parameters = []
    def __init__(self, train_dataset_path, test_dataset_path, dataset_name, column_label_name, strategy_names, strategies, hyper_parameters ):
        self.test_dataset_path = test_dataset_path
        self.train_dataset_path = train_dataset_path
        self.dataset_name = dataset_name
        self.column_label_name =column_label_name
        self.strategy_names =strategy_names
        self.strategies =strategies
        self.hyper_parameters = hyper_parameters

class MapDatasets(object):
    #array with datasets that need to be ignored
    SKIP_DATASETS = ['balloons', 'fertility', 'lenses']
    
    def _get_dataset_names(self):
        f = h5py.File(DATA_DIR + HDF5_DATA_FILENAME,'r')
        dataset_names = list(f[SPLIT_DATASETS_DIR + TEST_DIR])
        f.close()
        test_dataset_paths = [SPLIT_DATASETS_DIR + TEST_DIR + dts_name for dts_name in  dataset_names]
        train_dataset_paths = [SPLIT_DATASETS_DIR + TRAIN_DIR + dts_name for dts_name in  dataset_names]

        return train_dataset_paths, test_dataset_paths, dataset_names
    
    def map(self):
        train_dataset_paths, test_dataset_paths, dataset_names = self._get_dataset_names()
        mapped_datasets = []
        
        clf_rfc = RandomForestClassifier()
        rfc_params = {
                'n_estimators': [10, 20, 30],
                'max_features': ['auto', 'sqrt','log2', None],
                'max_depth': [5,15,None]
                }
        
        clf_svm = SVC()
        svm_params = {
                'C': [1e-6,  1], #[1e-6, 1e-5, 1e-4,1e-3, 1e-2, 1, 1e2,1e3,1e4,1e5,1e6]
                'gamma': [1e-3, 1] #[1e-3, 1e-2, 1e-1, 1]
                }
        clf_logisticReg = linear_model.LogisticRegression()
        logisticReg_params = {
                'C': [1e-6,  1], #[1e-6, 1e-5, 1e-4,1e-3, 1e-2, 1, 1e2,1e3,1e4,1e5,1e6]
                }
        
       
        
        for dts in zip(train_dataset_paths, test_dataset_paths,dataset_names):
            if dts[2] in self.SKIP_DATASETS:
                continue
            
            strategy_names= ['RandomForestClassifier', 'SVM', 'LogisticRegression']
            strategies = [clf_rfc, clf_svm, clf_logisticReg]
            parameters = [rfc_params, svm_params, logisticReg_params]
        
            
            container  = DatasetContainer(train_dataset_path = dts[0],
                                          test_dataset_path = dts[1],
                                          dataset_name = dts[2],
                                          column_label_name ='clase',
                                          strategy_names = strategy_names, 
                                          strategies = strategies, 
                                          hyper_parameters= parameters)
            mapped_datasets.append(container)

        
        return mapped_datasets
        