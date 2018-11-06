from mlaut.data import Data
from mlaut.estimators.estimators import instantiate_default_estimators
from mlaut.experiments import Orchestrator
from download_delgado.delgado_datasets import DownloadAndConvertDelgadoDatasets
from mlaut.analyze_results.scores import ScoreAccuracy
import pandas as pd
import numpy as np
from mlaut.estimators.svm_estimators import SVC_mlaut
from mlaut.estimators.ensemble_estimators import Random_Forest_Classifier

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV


from mlaut.estimators.nn_estimators import Deep_NN_Classifier
import multiprocessing

data = Data()
input_io = data.open_hdf5('data/delgado.hdf55', mode='r')
out_io = data.open_hdf5('data/delgado-classification-deep.h5', mode='a')
dts_names_list, dts_names_list_full_path = data.list_datasets(hdf5_io=input_io, hdf5_group='openml/')
split_dts_list = data.split_datasets(hdf5_in=input_io, hdf5_out=out_io, dataset_paths=dts_names_list_full_path)


svm = SVC_mlaut()
rf = Random_Forest_Classifier()

estimators = [svm, rf]
orchest = Orchestrator(hdf5_input_io=input_io, hdf5_output_io=out_io, dts_names=dts_names_list,
                original_datasets_group_h5_path='openml/')

if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver', force=True)



    orchest.run(modelling_strategies=estimators[0:2], 
                verbose=True,
                overwrite_saved_models=True, 
                predict_on_runtime=True,
                overwrite_predictions=True)