from ..shared.static_variables import T_TEST_FILENAME,FRIEDMAN_TEST_FILENAME, WILCOXON_TEST_FILENAME, SIGN_TEST_FILENAME, BONFERRONI_TEST_FILENAME
from ..shared.static_variables import RESULTS_DIR, T_TEST_DATASET, SIGN_TEST_DATASET, BONFERRONI_CORRECTION_DATASET, WILCOXON_DATASET, FRIEDMAN_DATASET

from ..shared.static_variables import DATA_DIR, HDF5_DATA_FILENAME
import pandas as pd
import numpy as np
import itertools
from scipy import stats
from ..shared.files_io import FilesIO
from ..data.data import Data

from sklearn.metrics import accuracy_score, mean_squared_error
class AnalyseResults(object):
    SIGNIFICANCE_LEVEL = 0.05

    def __init__(self, hdf5_output_io, hdf5_input_io):
        self._input_io = hdf5_input_io
        self._output_io = hdf5_output_io
        self._data = Data()
        #self._prediction_accuracies = files_io.get_prediction_accuracies_per_strategy()
    
    def calculate_loss_all_datasets(self, input_h5_original_datasets_group, output_h5_predictions_group, metric):
        #load all datasets
        dts_names_list, dts_names_list_full_path = self._data.list_datasets(hdf5_group=input_h5_original_datasets_group, hdf5_io=self._input_io)

        #load all predcitions
        dts_predictions_list, dts_predictions_list_full_path = self._data.list_datasets(output_h5_predictions_group, self._output_io)
        loss_arr = []
        for dts in zip(dts_predictions_list, dts_predictions_list_full_path, dts_names_list_full_path):
            predictions = self._output_io.load_predictions_for_dataset(dts[0])
            train, test, _, _ = self._data.load_train_test_split(self._output_io, dts[0])
            dataset, meta = self._input_io.load_dataset_pd(dts[2])
            true_labels = self._data.load_true_labels(hdf5_in=self._input_io, dataset_loc=dts[2], lables_idx=test)
            true_labels = np.array(true_labels)
            loss = self.calculate_prediction_loss_per_dataset(metric=metric, predictions_per_ml_strategy=predictions, true_labels=true_labels)
            loss_arr.append(loss)
    
        return self.convert_from_array_to_dict(loss_arr)
    
    def calculate_prediction_loss_per_dataset(self, metric, predictions_per_ml_strategy, true_labels):
        score = []
        for prediction in predictions_per_ml_strategy:
            ml_strategy = prediction[0]
            ml_predictions = prediction[1]
            result = 0
            if metric=='accuracy':
                result = accuracy_score(true_labels, ml_predictions)
            if metric=='mean_squared_error':
                result = mean_squared_error(y_true=true_labels, y_pred=ml_predictions)

            score.append([ml_strategy, result])

        return score
    
    def convert_from_array_to_dict(self, prediction_accuracies):
        prediction_accuracies = np.array(prediction_accuracies)
        num_datasets = prediction_accuracies.shape[0]
        num_strategies = prediction_accuracies.shape[1]
        num_key_value_pairs = prediction_accuracies.shape[2]
    
        resh = prediction_accuracies.ravel().reshape(num_datasets * num_strategies,num_key_value_pairs)
        df = pd.DataFrame(resh, columns=['strategy', 'accuracy'])
        list_strategies = df['strategy'].unique()
    
        acc_per_strat = {}
        for strat in list_strategies:
            acc_per_strat[strat] = df[df['strategy']==strat]['accuracy'].values.astype(np.float)
        return acc_per_strat

                           
    def _t_test(self,alpha, test_type, prediction_accuracies):
        t_test = {}
        perms = itertools.combinations(prediction_accuracies.keys(), r=2)
        for perm in perms:
            comb  = perm[0] + ' - ' + perm[1]
            a0 = np.array(prediction_accuracies[perm[0]]).astype(np.float)
            a1 = np.array(prediction_accuracies[perm[1]]).astype(np.float)
            d = a0-a1
            n = len(d)
            ts = np.sqrt(n) * np.average(d)/np.std(d)
            ts = round(ts,2)
            t_stat = stats.t.ppf(1 - alpha, n - 1)
            t_stat = round(t_stat,2)
            p_val = (1 - stats.t.cdf(np.abs(ts), n-1)) * 2
            p_val = round(p_val,2)
            if np.abs(ts) >= t_stat:
                passed = 1
            else:
                passed = 0
            t_test[comb] = [t_stat, p_val ]
        return t_test
    
    def perform_t_test(self, loss_per_strategy):
        t_test = self._t_test(self.SIGNIFICANCE_LEVEL, 't-test', loss_per_strategy)

        values = []
        for pair in t_test.keys():        
            values.append( [pair, t_test[pair][0], t_test[pair][1]  ])
        values_df = pd.DataFrame(values, columns=['pair','t_statistic','p_value'])

        return t_test, values_df
                        
    def perform_sign_test(self):
        sign_test = {}
        perms = itertools.combinations(self._prediction_accuracies.keys(), r=2)
        for perm in perms:
            comb  = perm[0] + ' - ' + perm[1]
            d = np.array(self._prediction_accuracies[perm[0]]) - np.array(self._prediction_accuracies[perm[1]])
            pos = sum(x > 0 for x in d)
            neg = sum(x < 0 for x in d)
            tie = sum(x == 0 for x in d)
            
            successes = 0
            failures = 0
            
            if tie % 2 == 0:
                successes = pos + tie/2
                failures = neg + tie/2
            else:
                successes = pos + (tie -1)/2
                failures = neg + (tie -1)/2 
            z = (successes - failures)/np.sqrt(len(d) * 0.5 * 0.5)
            p_val = (1-stats.norm.cdf(np.abs(z)))*2
            
            sign_test[comb] = p_val
        
        values = []
        for pair in sign_test.keys():        
            values.append( [pair, sign_test[pair] ])
        values_df = pd.DataFrame(values, columns=['pair','p_value'])

        return sign_test, values_df
        
    def perform_t_test_with_bonferroni_correction(self):
        m = len(self._prediction_accuracies.keys())
        t_test_bonferoni = self._t_test(self.SIGNIFICANCE_LEVEL/m, 't-test with Bonferroni correction', self._prediction_accuracies)
        
        values = []
        for pair in t_test_bonferoni.keys():        
            values.append( [pair, t_test_bonferoni[pair][0], t_test_bonferoni[pair][1] ])
        values_df = pd.DataFrame(values, columns=['pair','t_statistic','p_value'])
        
        return t_test_bonferoni, values_df
        
    def perform_wilcoxon(self):
        wilcoxon_test ={}
        perms = itertools.combinations(self._prediction_accuracies.keys(), r=2)
        for perm in perms:
            comb  = perm[0] + ' - ' + perm[1]
            wilcoxon_test[comb] = stats.wilcoxon(self._prediction_accuracies[perm[0]],
                         self._prediction_accuracies[perm[1]])
        
        values = []
        for pair in wilcoxon_test.keys():        
            values.append( [pair, wilcoxon_test[pair][0], wilcoxon_test[pair][1] ])
        values_df = pd.DataFrame(values, columns=['pair','statistic','p_value'])

        return wilcoxon_test, values_df
                        
    def perform_friedman_test(self):
        friedman_test = stats.friedmanchisquare(self._prediction_accuracies['SVM'], 
                                                self._prediction_accuracies['LogisticRegression'], 
                                                self._prediction_accuracies['RandomForestClassifier'])
        values = [friedman_test[0], friedman_test[1]]
        values_df = pd.DataFrame([values], columns=['statistic','p_value'])

        return friedman_test, values_df
        