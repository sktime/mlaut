import os
import logging
ORIGINAL_DATASETS_DIR = 'original_datasets/'
REFORMATTED_DATASETS_DIR = 'reformatted_datasets/'
SPLIT_DATASETS_DIR = '/split_datasets'
X_TRAIN_DIR = '/X_train'
X_TEST_DIR = '/X_test'
Y_TRAIN_DIR = '/y_train'
Y_TEST_DIR = '/y_test'
DATA_DIR = 'data' + os.sep
PREDICTIONS_DIR = 'predictions/'
HDF5_DATA_FILENAME = 'data/data.hdf5'
RESULTS_DIR = 'results_summary/'
DELGADO_DIR = 'Delgado_data/'
T_TEST_DATASET = 't_test'
SIGN_TEST_DATASET = 'sign_test'
BONFERRONI_CORRECTION_DATASET = 'bonferroni_correction_test'
WILCOXON_DATASET = 'wilcoxon_test'
FRIEDMAN_DATASET = 'friedman_test'

EXPERIMENTS_DIR = 'experiments'
EXPERIMENTS_TRAINED_MODELS_DIR = DATA_DIR + 'trained_models'
EXPERIMENTS_PREDICTIONS_GROUP = 'experiments/predictions'
EXPERIMENTS_MODEL_ACCURACY_DIR = 'experiments/trained_models_accuracies/'

RUNTIMES_GROUP = '/run_times'

DELGADO_DATASET_DOWNLOAD_URL='https://persoal.citius.usc.es/manuel.fernandez.delgado/papers/jmlr/data.tar.gz'
DELGADO_NUM_DATASETS = 120

COLUMN_LABEL_NAME = 'clase'

MIN_EXAMPLES_PER_CLASS = 15

TEST_PREDICTION_ARRAY_FILENAME = 'models_test_predictions_array.p'
T_TEST_FILENAME = 't_test.p'
FRIEDMAN_TEST_FILENAME = 'friedman_test.p'
WILCOXON_TEST_FILENAME = 'wilcoxon_test.p'
SIGN_TEST_FILENAME = 'sign_test.p'
BONFERRONI_TEST_FILENAME = 'bonferroni_correction_test.p'

PICKLE_EXTENTION = '.pkl'
HDF5_EXTENTION = '.h5'

TEST_TRAIN_SPLIT = 1/4

FLAG_PREDICTIONS = 'ml_predictions'
FLAG_ACCURACY = 'accuracy'

#gridsearch parameters
GRIDSEARCH_CV_NUM_PARALLEL_JOBS = -1 #use -1 for maximum
GRIDSEARCH_NUM_CV_FOLDS = 5

#training parameters
VERBOSE = 0

TRAIN_IDX = 'train_idx'
TEST_IDX = 'test_idx'
SPLIT_DTS_GROUP = 'split_dts_idx'
LOG_ERROR_FILE = 'mlaut_erros.log'

"""
properties for decorator to estimators
"""
GENERALIZED_LINEAR_MODELS='Generalized_Linear_Models'
ENSEMBLE_METHODS='Ensemble methods'
NAIVE_BAYES='Naive_Bayes'
NEURAL_NETWORKS='Neural_Networks'
CLASSIFICATION='Classification'
REGRESSION='Regression'
SVM='Support_Vector_Machines'
BASELINE='Baseline'
CLUSTER='Cluster'

def set_logging_defaults():
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)
    fileHandler = logging.FileHandler(LOG_ERROR_FILE)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

