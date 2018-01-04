import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
from datetime import datetime

import numpy as np
from sklearn.preprocessing import OneHotEncoder

class Experiments(object):

    trained_models = []
    trained_models_fold_result_list = []
   
    def run_experiments(self, X_train, y_train, modelling_strategies):
        """ 
        Trains estimators contained in the model_container on the dataset.
        This is separated from the test_orchestrator class to avoid too many nested loops.
        """
        trained_models = []
        timestamps_df = pd.DataFrame()
        for modelling_strategy in modelling_strategies:
            ml_strategy_name = modelling_strategy.get_estimator_name()
            begin_timestamp = datetime.now()

            if ml_strategy_name is 'NeuralNetworkDeepClassifier':
                #encode the labels 
                onehot_encoder = OneHotEncoder(sparse=False)
                len_y = len(y_train)
                reshaped_y = y_train.reshape(len_y, 1)
                y_train_onehot_encoded = onehot_encoder.fit_transform(reshaped_y)
                num_classes = y_train_onehot_encoded.shape[1]
                num_samples, input_dim = X_train.shape
                #build the model with the appropriate parameters
                built_model = modelling_strategy.build(num_classes, input_dim, num_samples)
                #convert from DataFrame to nupy array
                built_model.fit(X_train, y_train_onehot_encoded)
                trained_model = built_model
            else:
                built_model = modelling_strategy.build()
                trained_model = built_model.fit(X_train, y_train)
            
            timestamps_df = self.record_timestamp(ml_strategy_name, begin_timestamp, timestamps_df)
            
            modelling_strategy.set_trained_model(trained_model)
            trained_models.append(modelling_strategy)
        return trained_models, timestamps_df
    
    def record_timestamp(self, strategy_name, begin_timestamp, timestamps_df):
        """ Timestamp used to record the duration of training for each of the estimators"""
        STF = '%Y-%m-%d %H:%M:%S'
        end_timestamp = datetime.now()
        diff = (end_timestamp - begin_timestamp).total_seconds() 
        vals = [strategy_name, begin_timestamp.strftime(STF),end_timestamp.strftime(STF),diff]
        run_time_df = pd.DataFrame([vals], columns=['strategy_name','begin_time','end_time','total_seconds'])
        timestamps_df = timestamps_df.append(run_time_df)
        
        return timestamps_df
        
    def make_predictions(self, models, dataset_name, X_test):
        """ Makes predictions on the test set """
        predictions = []
        for model in models:
            trained_model = model.get_trained_model()
            prediction = trained_model.predict(X_test)
            predictions.append([model.get_estimator_name(), prediction])
        return predictions
    
    # def calculate_prediction_accuracy(self, predictions_per_ml_strategy, true_labels):
    #     prediction_accuracy = []
    #     for prediction in predictions_per_ml_strategy:
    #         ml_strategy = prediction[0]
    #         ml_predictions = prediction[1]
    #         acc_score = accuracy_score(true_labels, ml_predictions)
    #         prediction_accuracy.append([ml_strategy, acc_score])
    #     return prediction_accuracy

    # def calculate_loss(self, metric, predictions_per_ml_strategy, true_labels):
    #     score = []
    #     for prediction in predictions_per_ml_strategy:
    #         ml_strategy = prediction[0]
    #         ml_predictions = prediction[1]
    #         result = 0
    #         if metric=='accuracy':
    #             result = accuracy_score(true_labels, ml_predictions)
    #         if metric=='mean_squared_error':
    #             result = mean_squared_error(y_true=true_labels, y_pred=ml_predictions)

    #         score.append([ml_strategy, result])

    #     return score

        