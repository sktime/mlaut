
from mlaut.data import Data
from mlaut.estimators.estimators import instantiate_default_estimators
from mlaut.experiments import Orchestrator
from mlaut.analyze_results import AnalyseResults
from download_delgado.delgado_datasets import DownloadAndConvertDelgadoDatasets

#download datasets
delgado = DownloadAndConvertDelgadoDatasets()
datasets, metadata = delgado.download_and_extract_datasets(verbose = False)

#Define input and output objects
data = Data()
input_io = data.open_hdf5('data/delgado.hdf5', mode='a')
out_io = data.open_hdf5('data/classification.hdf5', mode='a')

#store datasets in HDF5 database
data.pandas_to_db(save_loc_hdf5='delgado_datasets/', 
                  datasets=datasets, 
                  dts_metadata=metadata,
                  input_io=input_io) 

#split the datasets in train and test
dts_names_list, dts_names_list_full_path = data.list_datasets(hdf5_io=input_io, 
                                           hdf5_group='delgado_datasets/')
split_dts_list = data.split_datasets(hdf5_in=input_io, 
                                     hdf5_out=out_io, 
                                     dataset_paths=dts_names_list_full_path)

#Instantiate estimator objects and the experiments orchestrator class.
instantiated_models = instantiate_default_estimators(estimators=['Classification'])
orchest = Orchestrator(hdf5_input_io=input_io, 
                      hdf5_output_io=out_io, 
                      dts_names=dts_names_list,
                      original_datasets_group_h5_path='delgado_datasets/')

#Run the experiments
orchest.run(modelling_strategies=instantiated_models)

#Make predictions on the test set
orchest.predict_all(trained_models_dir='data/trained_models', estimators=instantiated_models)

#analyze results stage
analyze = AnalyseResults(hdf5_output_io=out_io, 
                        hdf5_input_io=input_io, 
                        input_h5_original_datasets_group='delgado_datasets/', 
                        output_h5_predictions_group='experiments/predictions/')
error_all_datasets = analyze.calculate_error_all_datasets(metric='mean_squared_error')
error_per_dataset, error_per_dataset_df = analyze.calculate_error_per_dataset(metric='mean_squared_error')
