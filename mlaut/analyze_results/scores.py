from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.metrics import precision_score, recall_score

class MLAUTScore(abs):
    @abstractmethod
    def calculate(self, y_true, y_pred):
        """
        Main method for performing the calculations.

        Parameters
        ----------
        y_true(array): True dataset labels.
        y_pred(array): predicted labels.

        Returns
        -------
            score(float): Returns the result of the metric.
        """
class ScoreAccuracy(MLAUTScore):
    """
    Calculates the accuracy between the true and predicted lables.
    """
    def calculate(self, y_true, y_pred):
        """
        Main method for performing the calculations.

        Parameters
        ----------
        y_true(array): True dataset labels.
        y_pred(array): predicted labels.

        Returns
        -------
            accuracy_score (float): The accuracy of the prediction.
        """
        #TODO check whether rounding the predictions is actually necessary
        # y_pred = np.rint(y_pred)
        return accuracy_score(y_true, y_pred)
class ScoreMSE(MLAUTScore):
    """
    Calculates the mean squared error between the true and predicted lables.
    """

    def calculate(y_true, y_pred):
        """
        Main method for performing the calculations.

        Parameters
        ----------
        y_true(array): True dataset labels.
        y_pred(array): predicted labels.

        Returns
        -------
            float: The mean squared error of the prediction.
        """
        return mean_squared_error(y_true, y_pred)
class ScorePrecision(MLAUTScore):
    """
    Calculates precision score of classifier.

    Parameters
    ----------
    average(string): Averaging to be performated on the data. Possible parameters as per `sklearn documentation <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html>`_.

    Returns
    -------
        float: precision score
    """
    def __init__(self, average='micro'):
        """
        Parameters
        ----------
        average(string): Averaging to be performed on the data.
        """
        self._average = average
    
    def calculate(self, y_true, y_pred):
        """
        Main method for performing the calculations.

        Parameters
        ----------
        y_true(array): True dataset labels.
        y_pred(array): predicted labels.

        Returns
        -------
            float: The precision of the predictions.
        """
        return precision_score(y_true, y_pred, average=self._average)

class ScoreRecall(MLAUTScore):
    """
    Calculates recall score of classifier.

    Parameters
    ----------
    average(string): Averaging to be performated on the data. Possible parameters as per `sklearn documentation <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html>`_.

    Returns
    -------
        float: precision score
    """
    def __init__(self, average='micro'):
        """
        Parameters
        ----------
        average(string): Averaging to be performed on the data.
        """
        self._average = average
    
    def calculate(self, y_true, y_pred):
        """
        Main method for performing the calculations.

        Parameters
        ----------
        y_true(array): True dataset labels.
        y_pred(array): predicted labels.

        Returns
        -------
            float: The precision of the predictions.
        """
        return recall_score(y_true, y_pred, average=self._average)