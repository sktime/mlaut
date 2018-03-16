from sklearn.metrics import accuracy_score, mean_squared_error
import collections
class Losses(object):
    """
    Calculates prediction losses on test datasets achieved by the trained estimators. When the class is instantiated it creates a dictionary that stores the losses.

    """

    def __init__(self):

        self._losses = collections.defaultdict(list)

    def evaluate(self, metrics, predictions, true_labels):
        """
        Calculates the loss metrics on the test sets

        :type metrics: array of string
        :param metrics: Array with loss metrics that will be calculated.

        :type predictions: numpy array
        :param v: Predictions of trained estimators

        :type true_labels: numpy array
        :param true_labels: true labels of test dataset.
        """
        
        for prediction in predictions:
            estimator_name = prediction[0]
            estimator_predictions = prediction[1]
            # self._losses[estimator_name] = collections.defaultdict(list)
            for metric in metrics:
                

                if metric is 'accuracy':
                    self._losses[estimator_name][metric].append( accuracy_score(true_labels, estimator_predictions) )
                elif metric is 'mean_squared_error':
                    self._losses[estimator_name][metric].append(mean_squared_error(true_labels, estimator_predictions))   
                else:
                    raise ValueError(f'metric {metric} is not supported.')
            

    def get_losses(self):
        """
        When the Losses class is instantiated a dictionary that holds all losses is created and appended every time the evaluate() method is run. This method returns this dictionary with the losses.

        :rtype: dictionary
        """ 
        return self._losses
