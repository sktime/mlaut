#sys.path.append('/media/viktor/Data/PhD/mleap')
from sklearn import datasets
from mleap.data import Data
from mleap.data.estimators import instantiate_default_estimators
import pandas as pd
import os
def test_save_dataset_in_hdf5():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    
    df1 = pd.DataFrame(X, columns=['f1','f2','f3','f4'])
    df2 = pd.DataFrame(y, columns=['label'])
    result = pd.concat([df1,df2], axis=1)
    metadata = {
        'class_name':'label',
        'source':'test',
        'dataset_name': 'iris'
    }
    data = Data()
    data.pandas_to_db(save_loc_hdf5='test_dataset/', datasets=[result], 
                  dts_metadata=[metadata], save_loc_hdd='iris.hdf5')

def test_default_models():
    instantiated_models = instantiate_default_estimators(estimators=['all'])
    assert len(instantiated_models) > 0
