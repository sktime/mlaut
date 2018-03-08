from mleap.estimators.mleap_estimator import properties
from mleap.estimators.mleap_estimator import MleapEstimator
from mleap.shared.files_io import DiskOperations
from mleap.shared.static_variables import(BASELINE,
                                      REGRESSION, 
                                      CLASSIFICATION)

from sklearn.dummy import DummyClassifier, DummyRegressor

@properties(estimator_family=[BASELINE],
            tasks=[REGRESSION],
            name='BaselineRegressor')
class Baseline_Regressor(MleapEstimator):
    """
    Wrapper for sklearn dummy regressor
    """


    def build(self, strategy='median'):
        """
        Builds and returns estimator class.

        :type strategy: string
        :param strategy: as per `scikit-learn dummy regressor documentation <http://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html>`_.
        :rtype: `sklearn object`
        """
        return DummyRegressor(strategy=strategy)

    def save(self, dataset_name):
        """
        Saves estimator on disk.

        :type dataset_name: string
        :param dataset_name: name of the dataset. Estimator will be saved under default folder structure `/data/trained_models/<dataset name>/<model name>`
        """
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
    """
    Wrapper for sklearn dummy classifier class.
    """


    def build(self, strategy='stratified'):
        """
        Builds and returns estimator class.

        :type strategy: string
        :param strategy: as per `scikit-learn dummy classifier documentation <http://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html>`_.
        """
        return DummyClassifier(strategy=strategy)

    def save(self, dataset_name):
        """
        Saves estimator on disk.

        :type dataset_name: string
        :param dataset_name: name of the dataset. Estimator will be saved under default folder structure `/data/trained_models/<dataset name>/<model name>`
        """
        #set trained model method is implemented in the base class
        trained_model = self._trained_model
        disk_op = DiskOperations()
        disk_op.save_to_pickle(trained_model=trained_model,
                             model_name=self.properties()['name'],
                             dataset_name=dataset_name)