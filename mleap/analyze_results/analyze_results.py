from mleap.shared.static_variables import T_TEST_FILENAME,FRIEDMAN_TEST_FILENAME, WILCOXON_TEST_FILENAME, SIGN_TEST_FILENAME, BONFERRONI_TEST_FILENAME
from mleap.shared.static_variables import RESULTS_DIR, T_TEST_DATASET, SIGN_TEST_DATASET, BONFERRONI_CORRECTION_DATASET, WILCOXON_DATASET, FRIEDMAN_DATASET

from mleap.shared.static_variables import (DATA_DIR, 
                                           HDF5_DATA_FILENAME, 
                                           EXPERIMENTS_PREDICTIONS_GROUP,
                                           SPLIT_DTS_GROUP,
                                           TRAIN_IDX,
                                           TEST_IDX)
import pandas as pd
import numpy as np
import itertools
from mleap.shared.files_io import FilesIO
from mleap.data.data import Data

from scipy import stats
from scipy.stats import ttest_ind
from scipy.stats import ranksums
from statsmodels.sandbox.stats.multicomp import multipletests

from sklearn.metrics import accuracy_score, mean_squared_error
import scikit_posthocs as sp

from mleap.analyze_results.losses import Losses
class AnalyseResults(object):
    """
    Analyze results of machine learning experiments.

    :type hdf5_input_io: :func:`~mleap.shared.files_io.FilesIO`
    :param hdf5_input_io: Instance of :func:`~mleap.shared.files_io.FilesIO` class.

    :type hdf5_input_io: :func:`~mleap.shared.files_io.FilesIO`
    :param hdf5_input_io: Instance of :func:`~mleap.shared.files_io.FilesIO` class.

    :type input_h5_original_datasets_group: string
    :param input_h5_original_datasets_group: location in HDF5 database where the original datasets are stored.

    :type output_h5_predictions_group: string
    :param output_h5_predictions_group: location in HDF5 where the prediction of the estimators will be saved.

    :type split_datasets_group: string
    :param split_datasets_group: location in HDF5 database where the test/train splits are saved.

    :type train_idx: string
    :param train_idx: name of group where the train split index will be stored.

    :type test_idx: string
    :param test_idx: name of group where the test split index will be stored.
    """

    def __init__(self, 
                 hdf5_output_io, 
                 hdf5_input_io, 
                 input_h5_original_datasets_group, 
                 output_h5_predictions_group,
                 split_datasets_group=SPLIT_DTS_GROUP,
                 train_idx=TRAIN_IDX,
                 test_idx=TEST_IDX):

        self._input_io = hdf5_input_io
        self._output_io = hdf5_output_io
        self._input_h5_original_datasets_group = input_h5_original_datasets_group
        self._output_h5_predictions_group = output_h5_predictions_group
        self._split_datasets_group = split_datasets_group
        self._train_idx = test_idx
        self._test_idx = test_idx
        self._data = Data(experiments_predictions_group=self._output_h5_predictions_group,
                          split_datasets_group=self._split_datasets_group,
                          train_idx=self._train_idx,
                          test_idx=self._test_idx)
    
    def calculate_error_all_datasets(self, metrics):
        """
        Calculates the prediction error for each estimator on all test splits.

        :type metric: string
        :param metric: Loss metric. Supported values are: ``accuracy``,  ``mean_squared_error``

        :rtype: dictionary with keys representing the name of the estimator 
            and values representing the loss achieved on each dataset. 
        """
        #load all datasets
        dts_names_list, dts_names_list_full_path = self._data.list_datasets(hdf5_group=self._input_h5_original_datasets_group, hdf5_io=self._input_io)

        #load all predictions
        dts_predictions_list, dts_predictions_list_full_path = self._data.list_datasets(self._output_h5_predictions_group, self._output_io)
        losses = Losses()
        for dts in dts_predictions_list:
            predictions = self._output_io.load_predictions_for_dataset(dts)
            train, test, _, _ = self._data.load_train_test_split(self._output_io, dts)
            idx_orig_dts = dts_predictions_list.index(dts)
            path_orig_dts = dts_names_list_full_path[idx_orig_dts]
            true_labels = self._data.load_true_labels(hdf5_in=self._input_io, dataset_loc=path_orig_dts, lables_idx=test)
            true_labels = np.array(true_labels)
            losses.evaluate(metrics=metrics, 
                            predictions=predictions, 
                            true_labels=true_labels)
            # loss = self._calculate_prediction_error_per_dataset(metric=metric, predictions_per_ml_strategy=predictions, true_labels=true_labels)
            # loss_arr.append(loss)
        return losses.get_losses()
        # return self._convert_from_array_to_dict(loss_arr)
    


    def _calculate_prediction_error_per_dataset(self, metric, predictions_per_ml_strategy, true_labels):
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
    
    def calculate_error_per_dataset(self, metric):
        """
        Calculates the prediction error for each estimator on each datapoint for each dataset.

        :type metric: string
        :param metric: Loss metric. Supported values are: ``accuracy``,  ``mean_squared_error``

        :rtype: dictionary
        """
        orig_dts_names_list, orig_dts_names_list_full_path = self._data.list_datasets(hdf5_group=self._input_h5_original_datasets_group, hdf5_io=self._input_io)
        pred_dts_names_list, pred_dts_names_list_full_path = self._data.list_datasets(hdf5_group=self._output_h5_predictions_group, hdf5_io=self._output_io)
        result = {}
        for dts in pred_dts_names_list:
            result[dts] = []
            predictions_all_estimators = self._output_io.load_predictions_for_dataset(dts)
            _, test, _, _ = self._data.load_train_test_split(self._output_io, dts)
            idx_orig_dts = orig_dts_names_list.index(dts)
            path_orig_dts = orig_dts_names_list_full_path[idx_orig_dts]
            true_labels = self._data.load_true_labels(hdf5_in=self._input_io, dataset_loc=path_orig_dts, lables_idx=test)
            true_labels = np.array(true_labels)

            for est in predictions_all_estimators:
                est_name = est[0]
                est_predictions = est[1]
                score_per_label = self._calculate_error_per_datapoint(predictions=est_predictions, 
                                                                    true_labels=true_labels, 
                                                                    metric=metric)
                std_score = np.std(score_per_label)
                n = len(score_per_label)
                sum_score = np.sum(score_per_label)
                score = np.sqrt(sum_score/n)
                result[dts].append([est_name, score, std_score])

        result_df = self._reformat_error_per_dataset(result)
        return result, result_df

    def _reformat_error_per_dataset(self, error_per_dataset):
        df = pd.DataFrame(error_per_dataset)
        #unpivot the data
        df = df.melt(var_name='dts', value_name='values')
        df['classifier'] = df.apply(lambda raw: raw.values[1][0], axis=1)
        df['score'] = df.apply(lambda raw: raw.values[1][1], axis=1)
        df['std'] = df.apply(lambda raw: raw.values[1][2], axis=1)
        df = df.drop('values', axis=1)
        #create multilevel index dataframe
        dts = df['dts'].unique()
        estimators_list = df['classifier'].unique()
        score = df['score'].values
        std = df['std'].values
        
        df = df.drop('dts', axis=1)
        df=df.drop('classifier', axis=1)
        
        df.index = pd.MultiIndex.from_product([dts, estimators_list])

        return df
        
    def _calculate_error_per_datapoint(self, predictions, true_labels, metric):
        errors = []
        for pair in zip(predictions, true_labels):
            prediction = pair[0]
            true_label = pair[1]
            if metric == 'mean_squared_error':
                mse = mean_squared_error([prediction], [true_label])
                errors.append(mse)
            if metric == 'accuracy':
                accuracy = accuracy_score(true_label, prediction)
                errors.append(accuracy)

        return np.array(errors)

    def calculate_average_std(self, scores_dict):
        """
        Calculates simple average and standard deviation.

        :type scores_dict: dictionary
        :param scores_dict: Dictionary with estimators (keys) and corresponding 
            prediction accuracies on different datasets.
        
        :rtype: pandas DataFrame
        """
        result = {}
        for k in scores_dict.keys():
            average = np.average(scores_dict[k])
            std = np.std(scores_dict[k])
            result[k]=[average,std]
        
        res_df = pd.DataFrame.from_dict(result, orient='index')
        res_df.columns=['avg','std']
        res_df = res_df.sort_values(['avg','std'], ascending=[1,1])

        return res_df

    def cohens_d(self, estimator_dict):
        """
        Cohen's d is an effect size used to indicate the standardised difference
        between two means. The calculation is implemented natively (without the 
        use of third-party libraries). More information can be found here:
        `Cohen\'s d <https://en.wikiversity.org/wiki/Cohen%27s_d>`_.

        :type estimator_dict: dictionary
        :param estimator_dict: dictionay with keys `names of estimators` and 
            values `errors achieved by estimators on test datasets`.
        :rtype: pandas DataFrame.
        """
        cohens_d = {}
        comb = itertools.combinations(estimator_dict.keys(), r=2)
        for c in comb:
            pair=f'{c[0]}-{c[1]}'
            val1 = estimator_dict[c[0]]
            val2 = estimator_dict[c[1]]
            
            n1 = len(val1)
            n2 = len(val2)

            v1 = np.var(val1)
            v2 = np.var(val2)

            m1 = np.mean(val1)
            m2 = np.mean(val2)

            SDpooled = np.sqrt(((n1-1)*v1 + (n2-1)*v2)/(n1+n2-2))
            ef = (m2-m1)/SDpooled
            cohens_d[pair] = ef
        cohens_d_df = pd.DataFrame.from_dict(cohens_d, orient='index')
        cohens_d_df.columns = ['Cohen\'s d']

        #sort by absolute value
        cohens_d_df['sort']= cohens_d_df['Cohen\'s d'].abs()
        cohens_d_df = cohens_d_df.sort_values(['sort'], ascending=[0])
        cohens_d_df = cohens_d_df.drop(['sort'], axis=1)
        return cohens_d_df

    def _convert_from_array_to_dict(self, observations):
        observations = np.array(observations)
        num_datasets = observations.shape[0]
        num_strategies = observations.shape[1]
        num_key_value_pairs = observations.shape[2]
    
        resh = observations.ravel().reshape(num_datasets * num_strategies,num_key_value_pairs)
        df = pd.DataFrame(resh, columns=['strategy', 'accuracy'])
        list_strategies = df['strategy'].unique()
    
        acc_per_strat = {}
        for strat in list_strategies:
            acc_per_strat[strat] = df[df['strategy']==strat]['accuracy'].values.astype(np.float32)
        return acc_per_strat

   

    def t_test(self, observations):
        """
        Runs t-test on all possible combinations between the estimators.

        :type observations: dictionary
        :param observations: Dictionary with errors on test sets achieved by estimators.
        :rtype: tuple of dictionary, pandas DataFrame
        """
        t_test = {}
        perms = itertools.combinations(observations.keys(), r=2)
        for perm in perms:
            comb  = perm[0] + ' - ' + perm[1]
            x = np.array(observations[perm[0]])
            y = np.array(observations[perm[1]])
            t_stat, p_val = ttest_ind(x,y)
            t_test[comb] = [t_stat, p_val ]

        values = []
        for pair in t_test.keys():        
            values.append( [pair, t_test[pair][0], t_test[pair][1]  ])
        values_df = pd.DataFrame(values, columns=['pair','t_statistic','p_value'])

        return t_test, values_df
                        
    def sign_test(self, observations):
        """
        Non-parametric test for testing consistent differences between pairs of obeservations.
        The test counts the number of observations that are greater, smaller and equal to the mean
        `<https://en.wikipedia.org/wiki/Sign_test>`_.


        :type observations: dictionary
        :param observations: Dictionary with errors on test sets achieved by estimators.
        :rtype: tuple of dictionary, pandas DataFrame
        """
        sign_test = {}
        perms = itertools.combinations(observations.keys(), r=2)
        for perm in perms:
            comb  = perm[0] + ' - ' + perm[1]
            x = observations[perm[0]]
            y = observations[perm[1]]
            t_stat, p_val = ranksums(x,y)
            sign_test[comb] = [t_stat, p_val]
        
        values = []
        for pair in sign_test.keys():        
            values.append( [pair, sign_test[pair][0], sign_test[pair][1] ])
        values_df = pd.DataFrame(values, columns=['pair','t_statistic','p_value'])

        return sign_test, values_df
        
    def t_test_with_bonferroni_correction(self, observations, alpha=0.05):
        """
        correction used to counteract multiple comparissons
        https://en.wikipedia.org/wiki/Bonferroni_correction


        :type observations: dictionary
        :param observations: Dictionary with errors on test sets achieved by estimators.

        :type alpha: float
        :param alpha: confidence level.
        :rtype: tuple of dictionary, pandas DataFrame
        """
        t_test, df_t_test = self.t_test(observations)
        
        unadjusted_p_vals = np.array(df_t_test['p_value'])
        reject, p_adjusted, alphacSidak, alphacBonf = multipletests(unadjusted_p_vals, alpha=alpha, method='bonferroni')
        
        values_df = pd.concat([df_t_test['pair'], pd.Series(p_adjusted, name='p_value')], axis=1)
        t_test_bonferoni = np.array(values_df)
        return t_test_bonferoni, values_df
        
    def wilcoxon_test(self, observations):
        """
        `Wilcoxon signed-rank test <https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test>`_.
        Tests whether two  related paired samples come from the same distribution. 
        In particular, it tests whether the distribution of the differences x-y is symmetric about zero

        :type observations: dictionary
        :param observations: Dictionary with errors on test sets achieved by estimators.
        :rtype: tuple of dictionary, pandas DataFrame.
        """
        wilcoxon_test ={}
        perms = itertools.combinations(observations.keys(), r=2)
        for perm in perms:
            comb  = perm[0] + ' - ' + perm[1]
            wilcoxon_test[comb] = stats.wilcoxon(observations[perm[0]],
                         observations[perm[1]])
        
        values = []
        for pair in wilcoxon_test.keys():        
            values.append( [pair, wilcoxon_test[pair][0], wilcoxon_test[pair][1] ])
        values_df = pd.DataFrame(values, columns=['pair','statistic','p_value'])

        return wilcoxon_test, values_df
                        
    def friedman_test(self, observations):
        """
        The Friedman test is a non-parametric statistical test used to detect differences 
        in treatments across multiple test attempts. The procedure involves ranking each row (or block) together, 
        then considering the values of ranks by columns.
        Implementation used: `scipy.stats <https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.friedmanchisquare.html>`_. 

        :type observations: dictionary
        :param observations: Dictionary with errors on test sets achieved by estimators.
        :rtype: tuple of dictionary, pandas DataFrame.
        
        """

        """
        use the * operator to unpack a sequence
        https://stackoverflow.com/questions/2921847/what-does-the-star-operator-mean/2921893#2921893
        """
        friedman_test = stats.friedmanchisquare(*[observations[k] for k in observations.keys()])
        values = [friedman_test[0], friedman_test[1]]
        values_df = pd.DataFrame([values], columns=['statistic','p_value'])

        return friedman_test, values_df
    
    def nemenyi(self, obeservations):
        """
        Post-hoc test run if the `friedman_test` reveals statistical significance.
        For more information see `Nemenyi test <https://en.wikipedia.org/wiki/Nemenyi_test>`_.
        Implementation used `scikit-posthocs <https://github.com/maximtrp/scikit-posthocs>`_.

        :type observations: dictionary
        :param observations: Dictionary with errors on test sets achieved by estimators.
        :rtype: pandas DataFrame.
        """

        obeservations = pd.DataFrame(obeservations)
        obeservations = obeservations.melt(var_name='groups', value_name='values')

        return sp.posthoc_nemenyi(obeservations, val_col='values', group_col='groups')