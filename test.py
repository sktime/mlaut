import os
from sklearn.metrics import accuracy_score
from sktime.benchmarking.data import RAMDataset
from sklearn import datasets
from mlaut.highlevel.tasks import CSCTask
import pandas as pd
import numpy as np

from mlaut.benchmarking.results import HDDResults
from mlaut.highlevel.strategies import CSCStrategy
from mlaut.estimators.baseline_estimators import BaselineTest
from mlaut.model_selection import SingleSplit
from sktime.classifiers.compose.ensemble import TimeSeriesForestClassifier
from mlaut.estimators.baseline_estimators import BaselineTest
from mlaut.benchmarking.orchestration import Orchestrator

iris = datasets.load_iris()
wine = datasets.load_wine()

iris_pd = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_pd['target'] = iris.target

wine_pd = pd.DataFrame(wine.data, columns=wine.feature_names)
wine_pd['target'] = wine.target

datasets = ([RAMDataset(iris_pd, name='iris'),
             RAMDataset(wine_pd, name='wine')])

tasks = [CSCTask(target="target") for _ in range(len(datasets))]

results = HDDResults(path='results')

strategies = [
    CSCStrategy(BaselineTest(), name="baseline")
]

results = HDDResults(path='results')

orchestrator = Orchestrator(datasets=datasets,
                            tasks=tasks,  
                            strategies=strategies, 
                            cv=SingleSplit(), 
                            results=results)
 
orchestrator.fit_predict(save_fitted_strategies=True, overwrite_predictions=True)