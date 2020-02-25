from sktime.highlevel.strategies import BaseSupervisedLearningStrategy
from sktime.highlevel.strategies import CLASSIFIER_TYPES, REGRESSOR_TYPES
from joblib import dump, load


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
        self._traits = {"required_estimator_type": CLASSIFIER_TYPES + REGRESSOR_TYPES }
        super(CSCStrategy, self).__init__(estimator, name=name, check_input=check_input)

    def save(self, path):
        # TODO this method is implemented in sktime.highlevel.strategies.BaseStrategy
        # however saving fails if we don't reimplement it here
        dump(self, path)

    def _check_estimator_compatibility(self, estimator):
        """
        Check compatibility of estimator with strategy
        """
        pass
        # Determine required estimator type from strategy case
        # TODO replace with strategy - estimator type registry lookup
        # if hasattr(self, '_traits'):
        #     required = self._traits["required_estimator_type"]
        #     if any(estimator_type not in ESTIMATOR_TYPES for estimator_type in required):
        #         raise AttributeError(f"Required estimator type unknown")
        # else:
        #     raise AttributeError(f"Required estimator type not found")

        # # Check estimator compatibility with required type
        # if not isinstance(estimator, BaseEstimator):
        #     raise ValueError(f"Estimator must inherit from BaseEstimator")

        # # If pipeline, check compatibility of final estimator
        # if isinstance(estimator, Pipeline):
        #     final_estimator = estimator.steps[-1][1]
        #     if not isinstance(final_estimator, required):
        #         raise ValueError(f"Final estimator of passed pipeline estimator must be of type: {required}, "
        #                          f"but found: {type(final_estimator)}")

        # # If tuning meta-estimator, check compatibility of inner estimator
        # elif isinstance(estimator, (GridSearchCV, RandomizedSearchCV)):
        #     estimator = estimator.estimator
        #     if not isinstance(estimator, required):
        #         raise ValueError(f"Inner estimator of passed meta-estimator must be of type: {required}, "
        #                          f"but found: {type(estimator)}")

        # # Otherwise check estimator directly
        # else:
        #     if not isinstance(estimator, required):
        #         raise ValueError(f"Passed estimator has to be of type: {required}, but found: {type(estimator)}")