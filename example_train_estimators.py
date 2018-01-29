from download_delgado.delgado_datasets import DownloadAndConvertDelgadoDatasets
from mleap.data import Data
from mleap.data.estimators import instantiate_default_estimators
from mleap.experiments import TestOrchestrator


data = Data()

input_io = data.open_hdf5('data/delgado.hdf5', mode='r')
out_io = data.open_hdf5('data/experiments.hdf5', mode='a')
dts_names_list, dts_names_list_full_path = data.list_datasets(hdf5_io=input_io, hdf5_group='delgado_datasets/')
split_dts_list = data.split_datasets(hdf5_in=input_io, hdf5_out=out_io, dataset_paths=dts_names_list_full_path[0:1])
instantiated_models = instantiate_default_estimators(estimators=['NeuralNetworkDeepRegressor'], verbose=1)
test_o = TestOrchestrator(hdf5_input_io=input_io, hdf5_output_io=out_io)
test_o.run(input_io_datasets_loc=dts_names_list_full_path, 
           output_io_split_idx_loc=split_dts_list, 
           modelling_strategies=instantiated_models)


