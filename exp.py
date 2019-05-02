import pandas as pd
from mlaut.experiments.data import HDF5
from mlaut.experiments.data import DatasetHDF5
from mlaut.experiments.data import ResultHDF5

bonds = pd.read_csv('data/processed_dataset.csv')

db = HDF5(hdf5_path='data/fin_study_input.h5', mode='a')

db.save_dataset(dataset=bonds, save_path='datasets', dataset_name='bonds')
print(bonds)

dts = DatasetHDF5('data/fin_study_input.h5', dataset_path='datasets',dataset_name='bonds')

result=ResultHDF5(hdf5_path='data/fin_study_output.h5',results_save_path='predictions')

# result.save(dataset_name='bonds', strategy_name='rf',y_true=[1,1,1], y_pred=[1,1,1], cv_fold=0)

print(result.load())