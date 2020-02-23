import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from mlaut.benchmarking.data import RAMDataset
from sklearn import datasets
from mlaut.highlevel.tasks import CSCTask
from mlaut.estimators.baseline_estimators import Baseline_Classifier, Baseline_Regressor
from mlaut.estimators.bayes_estimators import Gaussian_Naive_Bayes, Bernoulli_Naive_Bayes
from mlaut.estimators.cluster_estimators import K_Neighbours
from mlaut.estimators.decision_trees import Decision_Tree_Classifier, Decision_Tree_Regressor
from mlaut.estimators.ensemble_estimators import (Random_Forest_Classifier, 
                                                  Random_Forest_Regressor,
                                                  Bagging_Classifier,
                                                  Bagging_Regressor,
                                                  Gradient_Boosting_Classifier,
                                                  Gradient_Boosting_Regressor)


from mlaut.benchmarking.results import HDDResults
from mlaut.highlevel.strategies import CSCStrategy
from mlaut.model_selection import SingleSplit
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
    CSCStrategy(Baseline_Classifier(), name="Baseline_Classifier"),
    CSCStrategy(Baseline_Regressor(), name="Baseline_Regressor"),
    CSCStrategy(Gaussian_Naive_Bayes(), name="Gaussian_Naive_Bayes"),
    CSCStrategy(Bernoulli_Naive_Bayes(), name="Bernoulli_Naive_Bayes"),
    CSCStrategy(K_Neighbours(), name="K_Neighbours"),
    CSCStrategy(Decision_Tree_Classifier(), name="Decision_Tree_Classifier"),
    CSCStrategy(Decision_Tree_Regressor(), name="Decision_Tree_Regressor"),
    CSCStrategy(Random_Forest_Classifier(), name="Random_Forest_Classifier"),
    CSCStrategy(Random_Forest_Regressor(), name="Random_Forest_Regressor"),
    CSCStrategy(Bagging_Classifier(), name="Bagging_Classifier"),
    CSCStrategy(Bagging_Regressor(), name="Bagging_Regressor"),
    CSCStrategy(Gradient_Boosting_Classifier(), name="Gradient_Boosting_Classifier"),
    CSCStrategy(Gradient_Boosting_Regressor(), name="Gradient_Boosting_Regressor"),


]

results = HDDResults(path='results')

orchestrator = Orchestrator(datasets=datasets,
                            tasks=tasks,  
                            strategies=strategies, 
                            cv=SingleSplit(), 
                            results=results)
 
orchestrator.fit_predict(save_fitted_strategies=True, overwrite_predictions=True)