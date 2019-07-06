import logging
import os
import json
import re
import numpy as np
import pandas as pd
import logging

class Orchestrator:
    """
    Orchestrates the sequencing of running the machine learning experiments.
    """
    def __init__(self, datasets, strategies, cv, result):
        """
        Parameters
        ----------
        datasets: pandas dataframe
            datasets in pandas skitme format
        strategies: list of sktime strategy
            strategy as per sktime.highlevel
        cv: sklearn.model_selection cross validation
            sklearn cross validation method. Must implement split()
        result: sktime result class
            Object for saving the results
        """
        self._datasets = datasets
        self._strategies = strategies
        self._cv = cv
        self._result = result
        logging.basicConfig(level=logging.DEBUG)

    def run(self, predict_on_runtime=True, save_strategies=True, overwrite_saved_strategies=False):
        """
        Method for running the orchestrator
        
        Parameters
        ----------
        predict_on_runtime:Boolean
            If True makes predictions after the estimator is trained
        save_strategies: Boolean
            If True saves the trained strategies on the disk
        overwrite_saved_strategies : bool
            if true trains and overwrites the saved strategies
        """
        
        for data in self._datasets:

            dts_loaded, metadata = data.load()
            for strategy in self._strategies:
                for cv_fold, (train, test) in enumerate(self._cv.split(dts_loaded)):
                    logging.info(f'fitting: {strategy.properties} on dataset: {data.dataset_name}')
                    strategy_is_trained = strategy.check_strategy_exists(dataset_name=data.dataset_name, 
                                                                        cv_fold=cv_fold,
                                                                        strategy_save_dir=self._result.trained_strategies_save_path)
                    if (strategy_is_trained) and (overwrite_saved_strategies==False):
                        logging.info(f"Strategy: {strategy.properties['name']} already trained on dataset: {data.dataset_name}. Set overwrite_saved_strategies=True if you want to replace it. Skipping.")
                        continue
                    strategy.fit(metadata=metadata, data=dts_loaded.iloc[train])

                    if predict_on_runtime:
                        X_test = dts_loaded.iloc[test].drop(metadata['target'], axis=1)
                        y_pred = strategy.predict(X_test)
                        y_true = dts_loaded[metadata['target']].iloc[test].values
                        self._result.save(dataset_name=data.dataset_name, 
                                          strategy_name=strategy.properties['name'], 
                                          y_true=y_true.tolist(), 
                                          y_pred=y_pred.tolist(),
                                          cv_fold=cv_fold)
                    if save_strategies:
                        strategy.save(dataset_name=data.dataset_name, 
                                      cv_fold=cv_fold,
                                      strategy_save_dir=self._result.trained_strategies_save_path)

    def predict(self):
        """
        TODO load saved strategies and make predictions
        """
        raise NotImplementedError()