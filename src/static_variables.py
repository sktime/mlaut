ORIGINAL_DATASETS_DIR = 'original_datasets/'
REFORMATTED_DATASETS_DIR = 'reformatted_datasets/'
SPLIT_DATASETS_DIR = 'split_datasets/'
X_TRAIN_DIR = '/X_train'
X_TEST_DIR = '/X_test'
Y_TRAIN_DIR = '/y_train'
Y_TEST_DIR = '/y_test'
DATA_DIR = 'data/'
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
EXPERIMENTS_TRAINED_MODELS_DIR = DATA_DIR + '/trained_models/'
EXPERIMENTS_PREDICTIONS_DIR = 'experiments/predictions/'
EXPERIMENTS_MODEL_ACCURACY_DIR = 'experiments/trained_models_accuracies/'

RUNTIMES_GROUP = '/run_times'
USE_PROXY = False

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

TEST_TRAIN_SPLIT = 1/4

FLAG_ML_MODEL = 'ml_models'
FLAG_PREDICTIONS = 'ml_predictions'
FLAG_ACCURACY = 'accuracy'

GRIDSEARCH_CV_NUM_PARALLEL_JOBS = 4