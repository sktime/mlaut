from mlaut.experiments.data import HDF5
import pydataset
from mlaut.experiments import Orchestrator
from mlaut.estimators.baseline_estimators import Baseline_Classifier
from mlaut.estimators.bayes_estimators import Gaussian_Naive_Bayes
from mlaut.estimators.decision_trees import Decision_Tree_Classifier
from mlaut.resampling import Single_Split
from sklearn.model_selection import train_test_split
from mlaut.experiments.data import DatasetHDF5
from mlaut.experiments.data import ResultHDF5

from sklearn import preprocessing

import os
import shutil

from mlaut.experiments.analysis import AnalyseResults
from mlaut.experiments.scores import ScoreAccuracy

os.mkdir('data')

#Get the data and organize it
aids = pydataset.data('aids')
uis = pydataset.data('uis')

aids_meta = {
'target': 'adult',
'source':'pydataset',
'dataset_name':'aids'
}

uis_meta = {
    'target': 'IV3',
    'source': 'pydataset',
    'dataset_name': 'uis'
}

datasets = [aids, uis]
metadata = [aids_meta, uis_meta]
data = HDF5('data/input_data.h5')
data.pandas_to_db(datasets=datasets, dts_metadata=metadata)

aids_data = DatasetHDF5(hdf5_path='data/test_input.h5',dataset_path='pydataset/aids')
uis_data = DatasetHDF5(hdf5_path='data/test_input.h5',dataset_path='pydataset/uis')

#Orchestrate the experiments
cv = Single_Split(cv=train_test_split)
datasets = [aids_data,uis_data]
strategies = [Baseline_Classifier(), Decision_Tree_Classifier(), Gaussian_Naive_Bayes()]


result = ResultHDF5(hdf5_path='data/test_result.h5', 
                    predictions_save_path='predictions', 
                    trained_strategies_save_path='data/trained_estimators')

orchestrator = Orchestrator(datasets=datasets, strategies=strategies, cv = cv, result=result)
orchestrator.run()

#Analyse the results of the experiments
analyse = AnalyseResults(result)
score_accuracy = ScoreAccuracy()
loss_dict, loss_pd = analyse.prediction_errors(score_accuracy)

t_test, t_test_df = analyse.t_test(loss_dict)
t_test_df