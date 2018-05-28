from mlaut.shared.static_variables import T_TEST_FILENAME,FRIEDMAN_TEST_FILENAME, WILCOXON_TEST_FILENAME, SIGN_TEST_FILENAME, BONFERRONI_TEST_FILENAME
from mlaut.shared.static_variables import RESULTS_DIR, T_TEST_DATASET, SIGN_TEST_DATASET, BONFERRONI_CORRECTION_DATASET, WILCOXON_DATASET, FRIEDMAN_DATASET

from mlaut.shared.static_variables import (DATA_DIR, 
                                           HDF5_DATA_FILENAME, 
                                           EXPERIMENTS_PREDICTIONS_GROUP,
                                           SPLIT_DTS_GROUP,
                                           TRAIN_IDX,
                                           TEST_IDX)
import pandas as pd
import numpy as np
import itertools
from mlaut.shared.files_io import FilesIO
from mlaut.data.data import Data

from scipy import stats
from scipy.stats import ttest_ind
from scipy.stats import ranksums
from statsmodels.sandbox.stats.multicomp import multipletests

from sklearn.metrics import accuracy_score, mean_squared_error
import scikit_posthocs as sp

from mlaut.analyze_results.losses import Losses
class AnalyseResults(object):
    """
    Analyze results of machine learning experiments.

    :type hdf5_input_io: :func:`~mlaut.shared.files_io.FilesIO`
    :param hdf5_input_io: Instance of :func:`~mlaut.shared.files_io.FilesIO` class.

    :type hdf5_input_io: :func:`~mlaut.shared.files_io.FilesIO`
    :param hdf5_input_io: Instance of :func:`~mlaut.shared.files_io.FilesIO` class.

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
    
    def prediction_errors(self, metric):
        """
        Calculates the average prediction error per estimator as well as the prediction error achieved by each estimator on individual datasets.

        Args:
            metric(`mlaut.analyse_results.scores`): Error function. 

        Returns:
            estimator_avg_error, estimator_avg_error_per_dataset (pickle of pandas DataFrame): ``estimator_avg_error`` represents the average error and standard deviation achieved by each estimator. ``estimator_avg_error_per_dataset`` represents the average error and standard deviation achieved by each estimator on each dataset.
        """
        #load all predictions
        dts_predictions_list, dts_predictions_list_full_path = self._data.list_datasets(self._output_h5_predictions_group, self._output_io)
        losses = Losses(metric)
        dts_processed = []
        
        #TODO temporary fix!!!!!
        # !!!! error in code if multiple predictions for the same dataset are stored
        for dts in dts_predictions_list:
            if dts in dts_processed:
                continue
            dts_processed.append(dts)
            predictions = self._output_io.load_predictions_for_dataset(dts)
            _, _, _, y_test = self._data.load_test_train_dts(hdf5_out=self._output_io, 
                                                                              hdf5_in=self._input_io, 
                                                                              dts_name=dts, 
                                                                              dts_grp_path=self._input_h5_original_datasets_group)
            # path_orig_dts = f'{self._input_h5_original_datasets_group}/{dts}'
            # labels = np.append(y_train, y_test)
            # num_classes = len(np.unique(labels))
            losses.evaluate(predictions=predictions, 
                            true_labels=y_test,
                            dataset_name=dts)
        return losses.get_losses()



    def average_and_std_error(self, scores_dict):
        """
        Calculates simple average and standard error.

        :type scores_dict: dictionary
        :param scores_dict: Dictionary with estimators (keys) and corresponding 
            prediction accuracies on different datasets.
        
        :rtype: pandas DataFrame
        """
        result = {}
        for k in scores_dict.keys():
            average = np.average(scores_dict[k])
            n = len(scores_dict[k])
            std_error = np.std(scores_dict[k])/np.sqrt(n)
            result[k]=[average,std_error]
        
        res_df = pd.DataFrame.from_dict(result, orient='index')
        res_df.columns=['avg','std_error']
        res_df = res_df.sort_values(['avg','std_error'], ascending=[1,1])

        return res_df
    
    def ranks(self, estimator_dict, ascending):
        """
        Calculates the average ranks based on the performance of each estimator on each dataset

        Parameters
        ----------
        estimator_dict (dictionary): dictionay with keys `names of estimators` and values `errors achieved by estimators on test datasets`.
        ascending (boolean): Rank the values in ascending (True) or descending (False) order

        Returns
        -------
            ranks(DataFrame): Returns the mean peformance rank for each estimator
        """
        if not isinstance(ascending, bool):
            raise ValueError('Variable ascending needs to be boolean')
        
        df = pd.DataFrame(estimator_dict)
        ranked = df.rank(axis=1, ascending=ascending)
        mean_r = pd.DataFrame(ranked.mean(axis=0))
        mean_r.columns=['avg_rank']
        mean_r = mean_r.sort_values('avg_rank', ascending=1)
        return mean_r


    def cohens_d(self, estimator_dict):
        """
        Cohen's d is an effect size used to indicate the standardised difference between two means. The calculation is implemented natively (without the use of third-party libraries). More information can be found here: `Cohen\'s d <https://en.wikiversity.org/wiki/Cohen%27s_d>`_.

        :type estimator_dict: dictionary
        :param estimator_dict: dictionay with keys `names of estimators` and values `errors achieved by estimators on test datasets`.
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

    # def _convert_from_array_to_dict(self, observations):
    #     observations = np.array(observations)
    #     num_datasets = observations.shape[0]
    #     num_strategies = observations.shape[1]
    #     num_key_value_pairs = observations.shape[2]
    
    #     resh = observations.ravel().reshape(num_datasets * num_strategies,num_key_value_pairs)
    #     df = pd.DataFrame(resh, columns=['strategy', 'accuracy'])
    #     list_strategies = df['strategy'].unique()
    
    #     acc_per_strat = {}
    #     for strat in list_strategies:
    #         acc_per_strat[strat] = df[df['strategy']==strat]['accuracy'].values.astype(np.float32)
    #     return acc_per_strat

   

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
        `<http://en.wikipedia.org/wiki/Wilcoxon_rank-sum_test>`_.


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
        """http://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
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