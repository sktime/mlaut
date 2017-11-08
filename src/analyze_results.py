from src.static_variables import T_TEST_FILENAME,FRIEDMAN_TEST_FILENAME, WILCOXON_TEST_FILENAME, SIGN_TEST_FILENAME, BONFERRONI_TEST_FILENAME
from src.static_variables import RESULTS_DIR, T_TEST_DATASET, SIGN_TEST_DATASET, BONFERRONI_CORRECTION_DATASET, WILCOXON_DATASET, FRIEDMAN_DATASET

from src.static_variables import DATA_DIR, HDF5_DATA_FILENAME
import pandas as pd
import numpy as np
import itertools
from scipy import stats

class AnalyseResults(object):
    SIGNIFICANCE_LEVEL = 0.05

    def convert_prediction_acc_from_array_to_dict(self,prediction_accuracies):
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

                           
    def _t_test(self,alpha, test_type, modelling_strategies_accuracy):
        t_test = {}
        perms = itertools.combinations(modelling_strategies_accuracy.keys(), r=2)
        for perm in perms:
            comb  = perm[0] + ' - ' + perm[1]
            a0 = np.array(modelling_strategies_accuracy[perm[0]]).astype(np.float)
            a1 = np.array(modelling_strategies_accuracy[perm[1]]).astype(np.float)
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
            print('{0}: Modelling strategies: {1}, TS: {2}, t-stat: {3}, p-val: {4}, passed: {5}'.format(test_type, comb,ts, t_stat, p_val, passed))
            t_test[comb] = [t_stat, p_val ]
        return t_test
    
    def perform_t_test(self, modelling_strategies_accuracy):
        
        t_test = self._t_test(self.SIGNIFICANCE_LEVEL, 't-test', modelling_strategies_accuracy)

        values = []
        for pair in t_test.keys():        
            values.append( [pair, t_test[pair][0], t_test[pair][1]  ])
        values_df = pd.DataFrame(values, columns=['pair','t_statistic','p_value'])

        return t_test, values_df
                        
    def perform_sign_test(self, modelling_strategies_accuracy):
        sign_test = {}
        perms = itertools.combinations(modelling_strategies_accuracy.keys(), r=2)
        for perm in perms:
            comb  = perm[0] + ' - ' + perm[1]
            d = np.array(modelling_strategies_accuracy[perm[0]]) - np.array(modelling_strategies_accuracy[perm[1]])
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
            print('Sign test: {0}, p-value: {1}'.format(comb, p_val))
            
            sign_test[comb] = p_val
        
        values = []
        for pair in sign_test.keys():        
            values.append( [pair, sign_test[pair] ])
        values_df = pd.DataFrame(values, columns=['pair','p_value'])

        return sign_test, values_df
        
    def perform_t_test_with_bonferroni_correction(self, modelling_strategies_accuracy):
        m = len(modelling_strategies_accuracy.keys())
        t_test_bonferoni = self._t_test(self.SIGNIFICANCE_LEVEL/m, 't-test with Bonferroni correction', modelling_strategies_accuracy)
        
        values = []
        for pair in t_test_bonferoni.keys():        
            values.append( [pair, t_test_bonferoni[pair][0], t_test_bonferoni[pair][1] ])
        values_df = pd.DataFrame(values, columns=['pair','t_statistic','p_value'])
        
        return t_test_bonferoni, values_df
        
    def perform_wilcoxon(self, modelling_strategies_accuracy):
        wilcoxon_test ={}
        perms = itertools.combinations(modelling_strategies_accuracy.keys(), r=2)
        for perm in perms:
            comb  = perm[0] + ' - ' + perm[1]
            wilcoxon_test[comb] = stats.wilcoxon(modelling_strategies_accuracy[perm[0]],
                         modelling_strategies_accuracy[perm[1]])
            print('Wilcoxon test: comb {0}'.format(wilcoxon_test[comb]))
        
        values = []
        for pair in wilcoxon_test.keys():        
            values.append( [pair, wilcoxon_test[pair][0], wilcoxon_test[pair][1] ])
        values_df = pd.DataFrame(values, columns=['pair','statistic','p_value'])

        return wilcoxon_test, values_df
                        
    def perform_friedman_test(self, modelling_strategies_accuracy):
        friedman_test = stats.friedmanchisquare(modelling_strategies_accuracy['SVM'], 
                                                modelling_strategies_accuracy['LogisticRegression'], 
                                                modelling_strategies_accuracy['RandomForestClassifier'])
        print(friedman_test)
        values = [friedman_test[0], friedman_test[1]]
        values_df = pd.DataFrame([values], columns=['statistic','p_value'])

        return friedman_test, values_df
        