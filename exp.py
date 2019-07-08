import pandas as pd
from mlaut.experiments.data import HDF5
from mlaut.experiments.data import DatasetHDF5
from mlaut.experiments.data import ResultHDF5
from sklearn.model_selection import PredefinedSplit
from mlaut.resampling import Single_Split
import numpy as np

# bonds = pd.read_csv('data/processed_dataset.csv')

# input_db = HDF5(hdf5_path='data/fin_study_input.h5', mode='a')

# metadata = {'target':'SPREAD_Z', 'dataset_name':'bonds'}
# input_db.save_dataset(dataset=bonds, save_path='datasets', metadata=metadata)



from mlaut.estimators import default_estimators

est = default_estimators(task='Regression')

for e in est:
    print(e.properties)

