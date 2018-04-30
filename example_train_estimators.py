from mlaut.data import Data
from mlaut.estimators.estimators import instantiate_default_estimators
from mlaut.experiments import Orchestrator


if __name__ == "__main__":
    data = Data()

    input_io = data.open_hdf5('data/openml.h5', mode='r')
    out_io = data.open_hdf5('data/openml-classification.h5', mode='a')
    dts_names_list, dts_names_list_full_path = data.list_datasets(hdf5_io=input_io, hdf5_group='openml/')
    split_dts_list = data.split_datasets(hdf5_in=input_io, hdf5_out=out_io, dataset_paths=dts_names_list_full_path)
    instantiated_models = instantiate_default_estimators(estimators=['Classification'], verbose=1)
    orchest = Orchestrator(hdf5_input_io=input_io, 
                           hdf5_output_io=out_io, 
                           dts_names=dts_names_list,
                           original_datasets_group_h5_path='openml/')
    orchest.run(modelling_strategies=instantiated_models)


