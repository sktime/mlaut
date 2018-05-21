from sklearn.metrics import accuracy_score, mean_squared_error
import collections
import numpy as np
import pandas as pd
class Losses(object):
    """
    Calculates prediction losses on test datasets achieved by the trained estimators. When the class is instantiated it creates a dictionary that stores the losses.

    Parameters
    ----------
    metric(string): loss metric that will be calculated.
    round_round_predictions(Boolean): Sets whether the predictions of the estimators should be rounded to the nearest integer. This is useful when calculating the accuracy scores of the outputs of estimators that produce regression results as opposed to classification results. The default behaviour is to round the predictions if the ``accuracy`` metric is used.   

    """

    def __init__(self, metric, round_predictions=None):

        self._losses = collections.defaultdict(list)
        self._metric = metric
        self._errors_per_estimator = collections.defaultdict(list)
        self._errors_per_dataset_per_estimator = collections.defaultdict(list)

        if round_predictions is None and metric is 'accuracy':
            self._round_predictions = True
        else:
            self._round_predictions = round_predictions

    def evaluate(self, predictions, true_labels, dataset_name):
        """
        Calculates the loss metrics on the test sets.

        Parameters
        ----------
        predictions(2d numpy array): Predictions of trained estimators in the form [estimator_name, [predictions]]
        true_labels(numpy array): true labels of test dataset.
        """
        
        for prediction in predictions:
            #evaluates error per estimator
            estimator_name = prediction[0]
            estimator_predictions = prediction[1]
            if self._round_predictions is True:
                estimator_predictions = np.rint(estimator_predictions)
            
            loss=0
            loss_function = self._get_loss_function()
            loss = loss_function(true_labels, estimator_predictions)
            # if self._metric is 'accuracy':
            #     loss = accuracy_score(true_labels, estimator_predictions)
            #     # self._losses[estimator_name].append( loss )
            # elif self._metric is 'mean_squared_error':
            #     loss = mean_squared_error(true_labels, estimator_predictions)
            #     # self._losses[estimator_name].append(loss)   
            # else:
            #     raise ValueError(f'metric {self._metric} is not supported.')
            
            self._errors_per_estimator[estimator_name].append(loss)

            #evaluate errors per dataset per estimator
            errors = (estimator_predictions - true_labels)**2
            errors = np.where(errors > 0, 1, 0)
            n = len(errors)

            std_score = np.std(errors)/np.sqrt(n) 
            sum_score = np.sum(errors)
            avg_score = sum_score/n
            self._errors_per_dataset_per_estimator[dataset_name].append([estimator_name, avg_score, std_score])
    
    
    def evaluate_per_dataset(self, 
                            predictions, 
                            true_labels, 
                            dataset_name):

        """
        Calculates the error of an estimator per dataset.
        
        Parameters
        ----------
        predictions : 2d array-like in the form [estimator name, [estimator_predictions]].
        true_labels : 1d array-like
        
        """
        estimator_name = predictions[0]
        estimator_predictions = np.array(predictions[1])
        errors = (estimator_predictions - true_labels)**2
        n = len(errors)

        std_score = np.std(errors)/np.sqrt(n) 
        sum_score = np.sum(errors)
        avg_score = sum_score/n
        self._losses[dataset_name].append([estimator_name, avg_score, std_score])

    def get_losses(self):
        """
        When the Losses class is instantiated a dictionary that holds all losses is created and appended every time the evaluate() method is run. This method returns this dictionary with the losses.

        Returns
        -------
            errors_per_estimator (dictionary), errors_per_dataset_per_estimator (dictionary), errors_per_dataset_per_estimator_df (pandas DataFrame): Returns dictionaries with the errors achieved by each estimator and errors achieved by each estimator on each of the datasets.  ``errors_per_dataset_per_estimator`` and ``errors_per_dataset_per_estimator_df`` return the same results but the first object is a dictionary and the second one a pandas DataFrame. ``errors_per_dataset_per_estimator`` and ``errors_per_dataset_per_estimator_df`` contain both the mean error and deviation.
        """ 
        # return self._losses
        return (self._errors_per_estimator, 
                self._errors_per_dataset_per_estimator, 
                self._losses_to_dataframe(self._errors_per_dataset_per_estimator))

    def _get_loss_function(self):
        """
        This function returns a loss function depending on the `metric` that was provided.
        """
        if self._metric is 'accuracy':
            loss = accuracy_score
        elif self._metric is 'mean_squared_error':
            loss = mean_squared_error
        else:
            raise ValueError(f'metric {self._metric} is not supported.')
        
        return loss
    def _losses_to_dataframe(self, losses):
        """
        Reformats the output of the dictionary returned by the :func:`mlaut.analyze_results.losses.Losses.get_losses` to a pandas DataFrame. This method can only be applied to reformat the output produced by :func:`mlaut.analyze_results.Losses.evaluate_per_dataset`.

        Parameters
        ----------

        losses: dictionary returned by the :func:`mlaut.analyze_results.losses.Losses.get_losses` generated by :func:`mlaut.analyze_results.losses.Losses.evaluate_per_dataset`
        """

        df = pd.DataFrame(losses)
        #unpivot the data
        df = df.melt(var_name='dts', value_name='values')
        df['classifier'] = df.apply(lambda raw: raw.values[1][0], axis=1)
        df['loss'] = df.apply(lambda raw: raw.values[1][1], axis=1)
        df['std'] = df.apply(lambda raw: raw.values[1][2], axis=1)
        df = df.drop('values', axis=1)
        #create multilevel index dataframe
        dts = df['dts'].unique()
        estimators_list = df['classifier'].unique()
        score = df['loss'].values
        std = df['std'].values
        
        df = df.drop('dts', axis=1)
        df=df.drop('classifier', axis=1)
        
        df.index = pd.MultiIndex.from_product([dts, estimators_list])

        return df