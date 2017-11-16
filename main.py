from src.test_orchestrator import TestOrchestrator
from src.models_container import ModelsContainer
from src.data import Data
from src.run_experiments import RunExperiments
from src.analyze_results import AnalyseResults
from src.files_io import FilesIO
from src.static_variables import HDF5_DATA_FILENAME, SPLIT_DATASETS_DIR


if __name__ == '__main__': 
    
    experiments = RunExperiments()
    analyze = AnalyseResults()
    models_container = ModelsContainer()
    
    files_io = FilesIO(HDF5_DATA_FILENAME)
    
    #save data in hdf5 database
    data = Data(files_io)
    data.prepare_data()

    #instantiate models and run experiments
    instantiated_models = models_container.instantiate_models(RandomForestClassifier=None, SVM=None, LogisticRegression=None)
    testOrchestrator = TestOrchestrator(SPLIT_DATASETS_DIR, files_io, experiments, analyze) 
    datasets = files_io.list_datasets(SPLIT_DATASETS_DIR)
    testOrchestrator.run_experiments(datasets[0:5], instantiated_models)
    #perform tests
    testOrchestrator.perform_statistical_tests()
    

    
    print('****************************************************************')
    print('*           Experiments were run successfully                  *')
    print('****************************************************************')
    
