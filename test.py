import os
from sklearn.metrics import accuracy_score
from mlaut.benchmarking.data import RAMDataset, make_datasets
from sklearn import datasets
from mlaut.highlevel.tasks import TSCTask
import pandas as pd
import numpy as np

from mlaut.benchmarking.results import HDDResults
from mlaut.highlevel.strategies import CSCStrategy
from mlaut.estimators.baseline_estimators import Baseline_Classifier
from mlaut.model_selection import SingleSplit

iris = datasets.load_iris()
wine = datasets.load_wine()

iris_pd = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_pd['target'] = iris.target

wine_pd = pd.DataFrame(wine.data, columns=wine.feature_names)
wine_pd['target'] = wine.target

datasets = ([RAMDataset(iris_pd, name='iris'),
             RAMDataset(wine_pd, name='wine')])

tasks = [TSCTask(target="target") for _ in range(len(datasets))]

results = HDDResults(path='results')

strategies = [
    CSCStrategy(Baseline_Classifier, name="baseline")
]

# results = HDDResults(path='results')

# orchestrator = Orchestrator(datasets=datasets,
#                             tasks=tasks,  
#                             strategies=strategies, 
#                             cv=SingleSplit(), 
#                             results=results)
 
# orchestrator.fit_predict(save_fitted_strategies=False, overwrite_predictions=True)