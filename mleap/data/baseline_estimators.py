from mleap.data.mleap_estimator import properties
from mleap.data.mleap_estimator import MleapEstimator
from mleap.shared.files_io import DiskOperations
from mleap.shared.static_variables import(BASELINE,
                                      REGRESSION, 
                                      CLASSIFICATION)

from sklearn.dummy import DummyClassifier, DummyRegressor

@properties(estimator_family=[BASELINE],
            tasks=[REGRESSION],
            name='BaselineRegressor')
class Baseline_Regressor(MleapEstimator):
    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)

    def build(self, strategy='median'):
        return DummyRegressor(strategy=strategy)

    def save(self, dataset_name):
        #set trained model method is implemented in the base class
        trained_model = self._trained_model
        disk_op = DiskOperations()
        disk_op.save_to_pickle(trained_model=trained_model,
                             model_name=self.properties()['name'],
                             dataset_name=dataset_name)


@properties(estimator_family=[BASELINE],
            tasks=[CLASSIFICATION],
            name='BaselineClassifier')
class Baseline_Classifier(MleapEstimator):
    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)

    def build(self, strategy='stratified'):
        return DummyClassifier(strategy=strategy)

    def save(self, dataset_name):
        #set trained model method is implemented in the base class
        trained_model = self._trained_model
        disk_op = DiskOperations()
        disk_op.save_to_pickle(trained_model=trained_model,
                             model_name=self.properties()['name'],
                             dataset_name=dataset_name)