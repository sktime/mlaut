from src.static_variables import  SPLIT_DATASETS_DIR
from src.static_variables import MIN_EXAMPLES_PER_CLASS, COLUMN_LABEL_NAME
from src.static_variables import  DATA_DIR,HDF5_DATA_FILENAME, GRIDSEARCH_CV_NUM_PARALLEL_JOBS
from src.functions import SharedFunctions
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from datetime import datetime
from sklearn.model_selection import train_test_split
class RunExperiments(object):

    trained_models = []
    trained_models_fold_result_list = []
    def __init__(self):
        self._test_orchestrator = None
        
    def setTestOrchestrator(self, test_orchestrator):
        self._test_orchestrator = test_orchestrator
    
    def _create_test_train_split(self, dataset, metadata):
        class_name = metadata['class_name']
        y = dataset[class_name]
        X = dataset.loc[:, dataset.columns != class_name]
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.33, random_state=42)
        return X_train, X_test, y_train, y_test

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

    # def run_experiments22(self, dataset_container):
    #     train_df_path = dataset_container.train_dataset_path           
    #     store = pd.HDFStore(DATA_DIR + HDF5_DATA_FILENAME)
    #     train_df = store[train_df_path]
    #     store.close()
    #     #shuffle data
    #     random_seed = 2
    #     train_df = train_df.sample(frac=1, random_state = random_seed)
    #     #process dataset to remove classes with few examples
    #     sf = SharedFunctions()
    #     train_df = sf.process_dataset(dataset = train_df, min_num_examples = MIN_EXAMPLES_PER_CLASS)
    #     #split dataset in features and labels
    #     X_train = train_df.drop(dataset_container.column_label_name,axis=1)
    #     y_train = train_df[dataset_container.column_label_name]
    #     #train modelling strategies
    #     trained_models, timestamps_df = self._grid_search(X_train, y_train, dataset_container)
    #     return trained_models, timestamps_df

        
            
    # def _grid_search(self,X_train,y_train, dataset_container):
    #     ml_strategies = dataset_container.strategies
    #     ml_strategy_names = dataset_container.strategy_names
    #     ml_strategies_hyperparameters = dataset_container.hyper_parameters

    #     models = []
    #     timestamps_df = pd.DataFrame()
    #     for strategy in zip(ml_strategies, ml_strategy_names, ml_strategies_hyperparameters):
    #         ml_strategy = strategy[0]
    #         ml_strategy_name = strategy[1]
    #         ml_strategy_hyperparameters = strategy[2]
    #         begin_timestamp = datetime.now()
    #         gs = GridSearchCV(ml_strategy, ml_strategy_hyperparameters, verbose=1, refit=True, n_jobs = GRIDSEARCH_CV_NUM_PARALLEL_JOBS)
    #         gs = gs.fit(X_train, y_train)
    #         timestamps_df = self.record_timestamp(ml_strategy_name, begin_timestamp, timestamps_df)
    #         models.append( [ml_strategy_name,  gs])
    #     return models, timestamps_df
    
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
        