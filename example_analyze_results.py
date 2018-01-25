from mleap.analyze_results import AnalyseResults
from mleap.data import Data

import pickle

data = Data()
input_io = data.open_hdf5('data/delgado.hdf5', mode='r')
out_io = data.open_hdf5('data/experiments.hdf5', mode='a')
analyze = AnalyseResults(hdf5_output_io=out_io, hdf5_input_io=input_io)
observations = analyze.calculate_loss_all_datasets(input_h5_original_datasets_group='delgado_datasets/', 
                                    output_h5_predictions_group='experiments/predictions/', 
                                    metric='mean_squared_error')

# t_test, t_test_df = analyze.t_test(observations)
# print('******t-test******')
# print(t_test_df)

# sign_test, sign_test_df = analyze.sign_test(observations)
# print('******sign test******')
# print(sign_test_df)

# t_test_bonferroni, t_test_bonferroni_df = analyze.t_test_with_bonferroni_correction(observations)
# print('******t-test bonferroni correction******')
# print(t_test_bonferroni_df)

# wilcoxon_test, wilcoxon_test_df = analyze.wilcoxon_test(observations)
# print('******Wilcoxon test******')
# print(wilcoxon_test_df)

friedman_test, friedman_test_df = analyze.friedman_test(observations)
print('******Friedman test******')
print(friedman_test_df)

nemeniy_test = analyze.nemenyi(observations)
print('******Nemeniy test******')
print(nemeniy_test)