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
from mlaut.estimators.glm_estimators import Lasso, Passive_Aggressive_Classifier

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
    # BaselineClassifierStrategy, 
    # BaselineRegressorStrategy,
    # GaussianNaiveBayesStrategy, 
    # BernoulliNaiveBayesStrategy,
    # KNeighboursStrategy,
    # DecisionTreeClassifierStrategy, 
    # DecisionTreeRegressorStrategy,
    # RandomForestClassifierStrategy, 
    # RandomForestRegressorStrategy, 
    # GradientBoostingClassifierStrategy, 
    # GradientBoostingRegressorStrategy, 
    # BaggingClassifierStrategy, 
    # BaggingRegressorStrategy,
    LinearRegressonStrategy, 
    RidgeRegressionStrategy, 
    LassoStrategy, 
    LassoLarsStrategy, 
    LogisticRegressionStrategy, 
    BayesianRidgeStrategy, 
    PassiveAggressiveClassifierStrategy,
    SVMStrategy, 
    SVRStrategy
]

results = HDDResults(path='results')

orchestrator = Orchestrator(datasets=datasets,
                            tasks=tasks,  
                            strategies=strategies, 
                            cv=SingleSplit(), 
                            results=results)
 
orchestrator.fit_predict(save_fitted_strategies=True, overwrite_predictions=True)