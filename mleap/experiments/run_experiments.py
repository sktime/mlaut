from ..shared.static_variables import  SPLIT_DATASETS_DIR
from ..shared.static_variables import MIN_EXAMPLES_PER_CLASS, COLUMN_LABEL_NAME
from ..shared.static_variables import  DATA_DIR,HDF5_DATA_FILENAME, GRIDSEARCH_CV_NUM_PARALLEL_JOBS
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from datetime import datetime
from sklearn.model_selection import train_test_split

class RunExperiments(object):

    trained_models = []
    trained_models_fold_result_list = []
   
    def run_experiments(self, X_train, y_train, model_container):
        trained_models = []
        timestamps_df = pd.DataFrame()
        for model in model_container:
            ml_strategy_name = model[0]
            modelling_strategy = model[1]
            begin_timestamp = datetime.now()
            trained_model = modelling_strategy.fit(X_train, y_train)
            timestamps_df = self.record_timestamp(ml_strategy_name, begin_timestamp, timestamps_df)
            trained_models.append([ml_strategy_name, trained_model])
        return trained_models, timestamps_df
    
    def record_timestamp(self, strategy_name, begin_timestamp, timestamps_df):
        STF = '%Y-%m-%d %H:%M:%S'
        end_timestamp = datetime.now()
        diff = (end_timestamp - begin_timestamp).total_seconds() 
        vals = [strategy_name, begin_timestamp.strftime(STF),end_timestamp.strftime(STF),diff]
        run_time_df = pd.DataFrame([vals], columns=['strategy_name','begin_time','end_time','total_seconds'])
        timestamps_df = timestamps_df.append(run_time_df)
        
        return timestamps_df
        
    def make_predictions(self, models, dataset_name, X_test):
        predictions = []
        for model in models:
            strategy_name = model[0]
            trained_model = model[1]
            prediction = trained_model.predict(X_test)
            predictions.append([strategy_name, prediction])
        return predictions
    
    def calculate_prediction_accuracy(self, predictions_per_ml_strategy, true_labels):
        prediction_accuracy = []
        for prediction in predictions_per_ml_strategy:
            ml_strategy = prediction[0]
            ml_predictions = prediction[1]
            acc_score = accuracy_score(true_labels, ml_predictions)
            prediction_accuracy.append([ml_strategy,acc_score])
        return prediction_accuracy
        