import pandas as pd
from mlaut.experiments.data import HDF5
from mlaut.experiments.data import DatasetHDF5
from mlaut.experiments.data import ResultHDF5
from sklearn.model_selection import train_test_split
from mlaut.resampling import Single_Split
from mlaut.estimators.baseline_estimators import Baseline_Regressor
from mlaut.experiments.data import DatasetHDF5
from mlaut.experiments import Orchestrator
from mlaut.estimators import default_estimators
from mlaut.estimators.baseline_estimators import Baseline_Regressor
from mlaut.estimators.glm_estimators import Lasso, Lasso_Lars

result = ResultHDF5(hdf5_path='data/fin_study_result.h5', 
                    predictions_save_path='predictions', 
                    trained_strategies_save_path='data/trained_estimators')

bonds = DatasetHDF5(hdf5_path='data/fin_study_input.h5',dataset_path='datasets/bonds')

cv = Single_Split(cv=train_test_split)

strategies = default_estimators(task='Regression')

orchestrator = Orchestrator(datasets=[bonds], strategies=strategies, cv = cv, result=result)
orchestrator.run()
