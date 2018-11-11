from mlaut.data import Data
from mlaut.estimators.estimators import instantiate_default_estimators
from mlaut.experiments import Orchestrator
from download_delgado.delgado_datasets import DownloadAndConvertDelgadoDatasets
from mlaut.analyze_results.scores import ScoreAccuracy
import pandas as pd
import numpy as np


from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV


from mlaut.estimators.nn_estimators import Deep_NN_Classifier
import multiprocessing

from mlaut.estimators.svm_estimators import SVC_mlaut

from mlaut.estimators.ensemble_estimators import Random_Forest_Classifier
from mlaut.estimators.ensemble_estimators import Bagging_Classifier
from mlaut.estimators.ensemble_estimators import Gradient_Boosting_Classifier

from mlaut.estimators.baseline_estimators import Baseline_Classifier
from mlaut.estimators.bayes_estimators import Bernoulli_Naive_Bayes
from mlaut.estimators.bayes_estimators import Gaussian_Naive_Bayes

from mlaut.estimators.glm_estimators import Passive_Aggressive_Classifier

from mlaut.estimators.cluster_estimators import K_Neighbours

data = Data()
input_io = data.open_hdf5('data/delgado.hdf5', mode='r')
out_io = data.open_hdf5('data/new_study.h5', mode='a')
dts_names_list, dts_names_list_full_path = data.list_datasets(hdf5_io=input_io, hdf5_group='delgado_datasets/')
split_dts_list = data.split_datasets(hdf5_in=input_io, hdf5_out=out_io, dataset_paths=dts_names_list_full_path)




estimators = [Random_Forest_Classifier(),
            #   Bagging_Classifier(),
              Gradient_Boosting_Classifier(),
              K_Neighbours(),
              Bernoulli_Naive_Bayes(),
              Gaussian_Naive_Bayes(),
              Passive_Aggressive_Classifier(),
              Baseline_Classifier()
]

orchest = Orchestrator(hdf5_input_io=input_io, hdf5_output_io=out_io, dts_names=dts_names_list,
                original_datasets_group_h5_path='delgado_datasets/')

if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver', force=True)



    orchest.run(modelling_strategies=estimators, 
                verbose=True,
                overwrite_saved_models=False, 
                predict_on_runtime=True,
                overwrite_predictions=False,
                overwrite_timestamp=False)