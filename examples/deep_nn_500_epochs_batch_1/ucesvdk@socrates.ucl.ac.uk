from mlaut.data import Data
import os
import random
from delgado_datasets import download_and_extract_datasets
from mlaut.experiments import Orchestrator
from estimators_500epochs_batch_1 import estimators

datasets, metadata = download_and_extract_datasets()

data = Data()
input_io = data.open_hdf5('data/delgado.h5', mode='a')

for item in zip(datasets, metadata):
    dts=item[0]
    meta=item[1]
    input_io.save_pandas_dataset(dataset=dts, save_loc='/delgado_datasets', metadata=meta)

out_io = data.open_hdf5('data/delgado-deep_epochs500_batch1.h5', mode='a')
dts_names_list, dts_names_list_full_path = data.list_datasets(hdf5_io=input_io, hdf5_group='delgado_datasets/')
random.seed(7)
split_dts_list = data.split_datasets(hdf5_in=input_io, hdf5_out=out_io, dataset_paths=dts_names_list_full_path)

orchest = Orchestrator(hdf5_input_io=input_io, 
                            hdf5_output_io=out_io, 
                            dts_names=dts_names_list,
                            original_datasets_group_h5_path='delgado_datasets/')
orchest.run(modelling_strategies=estimators)