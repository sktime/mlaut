
from sktime.highlevel.tasks import BaseTask


class CSCTask(BaseTask):
    """
    Cross section classification task.
    A task encapsulates metadata information such as the feature and target variable
    to which to fit the data to and any additional necessary instructions on how
    to fit and predict.
    Parameters
    ----------
    target : str
        The column name for the target variable to be predicted.
    features : list of str, optinal (default=None)
        The column name(s) for the feature variable. If None, every column apart from target will be used as a feature.
    metadata : pandas.DataFrame, optional (default=None)
        Contains the metadata that the task is expected to work with.
    """

    def __init__(self, target, features=None, metadata=None):
        self._case = 'CSC'
        super(CSCTask, self).__init__(target, features=features, metadata=metadata)

class CSRTask(BaseTask):
    """
    Cross section regression task.
    A task encapsulates metadata information such as the feature and target variable
    to which to fit the data to and any additional necessary instructions on how
    to fit and predict.
    Parameters
    ----------
    target : str
        The column name for the target variable to be predicted.
    features : list of str, optinal (default=None)
        The column name(s) for the feature variable. If None, every column apart from target will be used as a feature.
    metadata : pandas.DataFrame, optional (default=None)
        Contains the metadata that the task is expected to work with.
    """

    def __init__(self, target, features=None, metadata=None):
        self._case = 'CSR'
        super(CSRTask, self).__init__(target, features=features, metadata=metadata)