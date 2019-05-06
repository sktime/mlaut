from mlaut.experiments.data import ResultHDF5
from mlaut.experiments.analysis import AnalyseResults
from mlaut.experiments.scores import ScoreMSE, ScoreMAE

result = ResultHDF5('data/fin_study_result.h5', 
                    predictions_save_path='predictions',
                    trained_strategies_save_path='data/trained_estimators')
analyse = AnalyseResults(result)
strategy_dict_mse, losses_df_mse = analyse.prediction_errors(metric=ScoreMSE())
strategy_dict_mae, losses_df_mae = analyse.prediction_errors(metric=ScoreMAE())

print('**********MSE*****************')
print(losses_df_mse.round(3))
print('**********MAE*****************')
print(losses_df_mae.round(3))