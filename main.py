from src.test_orchestrator import TestOrchestrator
from src.models_container import ModelsContainer
from src.data import Data
from src.run_experiments import RunExperiments
from src.analyze_results import AnalyseResults
from src.files_io import FilesIO
from src.static_variables import HDF5_DATA_FILENAME, SPLIT_DATASETS_DIR


if __name__ == '__main__': 
    
    experiments = RunExperiments()
    models_container = ModelsContainer()
    
    files_io = FilesIO(HDF5_DATA_FILENAME)
    analyze = AnalyseResults(files_io)

    #save data in hdf5 database
    data = Data(files_io)
    data.prepare_data()

    #instantiate models and run experiments
    instantiated_models = models_container.instantiate_models(RandomForestClassifier=None, SVM=None, LogisticRegression=None)
    testOrchestrator = TestOrchestrator(SPLIT_DATASETS_DIR, files_io, experiments) 
    datasets = files_io.list_datasets(SPLIT_DATASETS_DIR)
    testOrchestrator.run_experiments(datasets[0:5], instantiated_models)
    
    #perform statistical tests
    analyze = AnalyseResults(files_io)

    t_test, t_test_df = analyze.perform_t_test()
    print('******t-test******')
    print(t_test_df)

    sign_test, sign_test_df = analyze.perform_sign_test()
    print('******sign test******')
    print(sign_test_df)

    t_test_bonferroni, t_test_bonferroni_df = analyze.perform_t_test_with_bonferroni_correction()
    print('******t-test bonferroni correction******')
    print(t_test_bonferroni_df)
    wilcoxon_test, wilcoxon_test_df = analyze.perform_wilcoxon()
    print('******Wilcoxon test******')
    print(wilcoxon_test_df)

    friedman_test, friedman_test_df = analyze.perform_friedman_test()
    print('******Friedman test******')
    print(friedman_test_df)
    

    
    print('****************************************************************')
    print('*           Experiments were run successfully                  *')
    print('****************************************************************')
    
