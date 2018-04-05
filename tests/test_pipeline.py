#sys.path.append('/media/viktor/Data/PhD/mlaut')
from sklearn import datasets
from mlaut.data import Data
from mlaut.data.estimators import instantiate_default_estimators
from mlaut.experiments import Orchestrator
from mlaut.analyze_results import AnalyseResults
from mlaut.shared import DiskOperations
import pandas as pd
import os
import pytest

def test_create_directories():
    disk_op = DiskOperations()
    disk_op.create_directory_on_hdd('data/trained_models')

@pytest.fixture
def db_files():
    data = Data()
    input_io = data.open_hdf5('data/iris.hdf5', mode='r')
    out_io = data.open_hdf5('data/test_output.hdf5', mode='a')
    return input_io, out_io

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
                  dts_metadata=[metadata], save_loc_hdd='data/iris.hdf5')

def test_default_models():
    instantiated_models = instantiate_default_estimators(estimators=['all'])
    assert len(instantiated_models) > 0

def test_estimators_train_pipeline(db_files):
    input_io, out_io = db_files
    data = Data()

    dts_names_list, dts_names_list_full_path = data.list_datasets(hdf5_io=input_io, 
                                                                  hdf5_group='test_dataset/')
    split_dts_list = data.split_datasets(hdf5_in=input_io, hdf5_out=out_io, dataset_paths=dts_names_list_full_path)
    instantiated_models = instantiate_default_estimators(estimators=['all'], verbose=0)
    orchest = Orchestrator(hdf5_input_io=input_io, 
                              hdf5_output_io=out_io,
                              experiments_trained_models_dir='data/trained_models')
    orchest.run(input_io_datasets_loc=dts_names_list_full_path, 
            output_io_split_idx_loc=split_dts_list, 
            modelling_strategies=instantiated_models)

def test_analyze_results(db_files):
    input_io, out_io = db_files
    analyze = AnalyseResults(hdf5_output_io=out_io, hdf5_input_io=input_io)

    observations = analyze.calculate_loss_all_datasets(input_h5_original_datasets_group='test_dataset/', 
                                        output_h5_predictions_group='experiments/predictions/', 
                                        metric='mean_squared_error')

    t_test, t_test_df = analyze.t_test(observations)
    sign_test, sign_test_df = analyze.sign_test(observations)
    t_test_bonferroni, t_test_bonferroni_df = analyze.t_test_with_bonferroni_correction(observations)
    wilcoxon_test, wilcoxon_test_df = analyze.wilcoxon_test(observations)
    friedman_test, friedman_test_df = analyze.friedman_test(observations)
    nemeniy_test = analyze.nemenyi(observations)
