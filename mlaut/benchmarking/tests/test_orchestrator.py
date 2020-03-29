import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from mlaut.benchmarking.data import RAMDataset
from sklearn import datasets
from mlaut.highlevel.tasks import CSCTask, CSRTask

from mlaut.benchmarking.results import HDDResults, RAMResults
from mlaut.highlevel.strategies import CSCStrategy
from mlaut.model_selection import SingleSplit
from mlaut.benchmarking.orchestration import Orchestrator

import pytest

from sklearn import linear_model
from mlaut.strategies.baseline import BaselineClassifierStrategy, BaselineRegressorStrategy
from mlaut.strategies.bayes import GaussianNaiveBayesStrategy, BernoulliNaiveBayesStrategy
from mlaut.strategies.cluster import KNeighboursStrategy
from mlaut.strategies.decision_trees import DecisionTreeClassifierStrategy, DecisionTreeRegressorStrategy
from mlaut.strategies.ensemble import (RandomForestClassifierStrategy,
                                       RandomForestRegressorStrategy,
                                       GradientBoostingClassifierStrategy,
                                       GradientBoostingRegressorStrategy, 
                                       BaggingClassifierStrategy, 
                                       BaggingRegressorStrategy)
from mlaut.strategies.glm import (LinearRegressonStrategy, 
                                  RidgeRegressionStrategy, 
                                  LassoStrategy, 
                                  LassoLarsStrategy, 
                                  LogisticRegressionStrategy, 
                                  BayesianRidgeStrategy, 
                                  PassiveAggressiveClassifierStrategy)

from mlaut.strategies.svm import SVMStrategy, SVRStrategy
from mlaut.strategies.neural_networks import KerasClassificationStrategy, KerasRegressionStrategy

from mlaut.benchmarking.evaluation import Evaluator
from mlaut.benchmarking.metrics import PairwiseMetric
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
wine = datasets.load_wine()
def test_orchestrator():
    iris_pd = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_pd['target'] = iris.target

    wine_pd = pd.DataFrame(wine.data, columns=wine.feature_names)
    wine_pd['target'] = wine.target

    datasets = ([RAMDataset(iris_pd, name='iris'),
                RAMDataset(wine_pd, name='wine')])

    tasks = [CSCTask(target="target") for _ in range(len(datasets))]
    results = RAMResults()


    strategies = [
        BaselineClassifierStrategy, 
        GaussianNaiveBayesStrategy, 
        KNeighboursStrategy,
        DecisionTreeClassifierStrategy
        # KerasClassificationStrategy
    ]

    results = HDDResults(path='results')

    orchestrator = Orchestrator(datasets=datasets,
                                tasks=tasks,  
                                strategies=strategies, 
                                cv=SingleSplit(), 
                                results=results,
                                log_file_path=None)

    orchestrator.fit_predict(save_fitted_strategies=True, overwrite_predictions=True)

    evaluator = Evaluator(results)

    metric = PairwiseMetric(func=accuracy_score, name='accuracy')
    metrics_by_strategy = evaluator.evaluate(metric=metric)



