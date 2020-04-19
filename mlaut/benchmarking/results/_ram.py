
from mlaut.benchmarking.results import BaseResults, _PredictionsWrapper
import numpy as np

class RAMResults(BaseResults):

    def __init__(self):
        self.results = {}
        super(RAMResults, self).__init__()

    def save_predictions(self, strategy_name, dataset_name, y_true, y_pred, y_proba, index, cv_fold,
                         train_or_test):
        key = self._generate_key(strategy_name, dataset_name, cv_fold, train_or_test)
        index = np.asarray(index)
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        y_proba = np.asarray(y_proba)
        self.results[key] = _PredictionsWrapper(strategy_name, dataset_name, index, y_true, y_pred, y_proba)
        self._append_key(strategy_name, dataset_name)

    def load_predictions(self, cv_fold, train_or_test):
        """Loads predictions for all datasets and strategies iteratively"""
        for strategy_name, dataset_name in self._iter():
            key = self._generate_key(strategy_name, dataset_name, cv_fold, train_or_test)
            yield self.results[key]

    def check_predictions_exist(self, strategy, dataset_name, cv_fold, train_or_test):
        # for in-memory results, always false, results are always overwritten
        return False

    def save_fitted_strategy(self, strategy, dataset_name, cv_fold):
        raise NotImplementedError()

    def load_fitted_strategy(self, strategy_name, dataset_name, cv_fold):
        raise NotImplementedError()

    def check_fitted_strategy_exists(self, strategy, dataset_name, cv_fold):
        # for in-memory results, always false, results are always overwritten
        return False

    def save(self):
        # in-memory results are currently not persisted (i.e saved to the disk)
        pass

    def _generate_key(self, strategy_name, dataset_name, cv_fold, train_or_test):
        """Function to get paths for files, this basically encapsulate the storage logic of the class"""
        return f"{strategy_name}_{dataset_name}_{train_or_test}_{str(cv_fold)}"