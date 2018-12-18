
from mlaut.data import Data
from mlaut.estimators.estimators import instantiate_default_estimators
from mlaut.experiments import Orchestrator
from mlaut.analyze_results import AnalyseResults
from download_delgado.delgado_datasets import DownloadAndConvertDelgadoDatasets

#database
data = Data()
input_io = data.open_hdf5('data/delgado.hdf5', mode='a')
out_io = data.open_hdf5('data/classification.hdf5', mode='a')

#split the datasets
dts_names_list, dts_names_list_full_path = data.list_datasets(hdf5_io=input_io, hdf5_group='delgado_datasets/')
split_dts_list = data.split_datasets(hdf5_in=input_io, hdf5_out=out_io, dataset_paths=dts_names_list_full_path, verbose=False)

#define the estimators
est = ['RandomForestClassifier','BaggingClassifier','GradientBoostingClassifier','SVC','GaussianNaiveBayes','BernoulliNaiveBayes','NeuralNetworkDeepClassifier','PassiveAggressiveClassifier','BaselineClassifier']
estimators = instantiate_default_estimators(estimators=est)

#run the experiments
orchest = Orchestrator(hdf5_input_io=input_io, hdf5_output_io=out_io, dts_names=dts_names_list,
                 original_datasets_group_h5_path='delgado_datasets/')
orchest.run(modelling_strategies=estimators, verbose=True)

#make predictions on the test sets
orchest.predict_all(trained_models_dir='data/trained_models', estimators=estimators, verbose=True)

#Analyse results
analyze = AnalyseResults(hdf5_output_io=out_io, 
                         hdf5_input_io=input_io,
                         input_h5_original_datasets_group='delgado_datasets/', 
                         output_h5_predictions_group='experiments/predictions/')
score_accuracy = ScoreAccuracy()


(errors_per_estimator, 
 errors_per_dataset_per_estimator, 
 errors_per_dataset_per_estimator_df) = analyze.prediction_errors(score_accuracy, estimators)