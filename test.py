from mleap.data import Data
from mleap.experiments.experiments import Experiments
from mleap.analyze_results.analyze_results import AnalyseResults

from mleap.shared.static_variables import EXPERIMENTS_PREDICTIONS_DIR
import numpy as np
data = Data()

input_io = data.open_hdf5('data/delgado.hdf5', mode='r')
output_io = data.open_hdf5('data/experiments.hdf5', mode='a')

analyze = AnalyseResults(hdf5_output_io=output_io, hdf5_input_io=input_io)


errors = analyze.calculate_loss_all_datasets(input_h5_original_datasets_group='delgado_datasets/', 
                                    output_h5_predictions_group='experiments/predictions/', 
                                    metric='mean_squared_error')