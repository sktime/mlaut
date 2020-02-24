from sktime.highlevel.strategies import BaseSupervisedLearningStrategy
from sktime.highlevel.strategies import CLASSIFIER_TYPES, REGRESSOR_TYPES
from joblib import dump, load
from mlaut.estimators.base import MLautClassifier, MLautRegressor


ALLOWED_TYPES = (MLautClassifier + MLautRegressor)
class CSCStrategy(BaseSupervisedLearningStrategy):
    """
    Strategy for time series classification.

    Parameters
    ----------
    estimator : an estimator
        Low-level estimator used in strategy.
    name : str, optional (default=None)
        Name of strategy. If None, class name of estimator is used.
    check_input : bool, optional (default=True)
        - If True, input are checked.
        - If False, input are not checked and assumed correct. Use with caution.
    """
    def __init__(self, estimator, name=None, check_input=True):
        self._case = "CSC"
        self._traits = {"required_estimator_type": ALLOWED_TYPES }
        super(CSCStrategy, self).__init__(estimator, name=name, check_input=check_input)

    def save(self, path):
        # TODO this method is implemented in sktime.highlevel.strategies.BaseStrategy
        # however saving fails if we don't reimplement it here
        dump(self, path)
