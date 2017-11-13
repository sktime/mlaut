from src.test_orchestrator import TestOrchestrator
from src.models_container import ModelsContainer
from src.data import Data
from src.run_experiments import RunExperiments
from src.analyze_results import AnalyseResults
from src.files_io import FilesIO
from src.static_variables import HDF5_DATA_FILENAME, SPLIT_DATASETS_DIR


if __name__ == '__main__': 
    data = Data()
    experiments = RunExperiments()
    analyze = AnalyseResults()
    testOrchestrator = TestOrchestrator(data)
    models_container = ModelsContainer()
    
    files_io = FilesIO(HDF5_DATA_FILENAME)
    testOrchestrator.setFilesIO(files_io)
    testOrchestrator.setAnalyzeResults(analyze)
    experiments.setTestOrchestrator(testOrchestrator)
   
    testOrchestrator.prepare_data()
    testOrchestrator.set_ml_models_container(models_container)
    testOrchestrator.setExperiments(experiments)
    datasets = files_io.list_datasets(SPLIT_DATASETS_DIR)
    testOrchestrator.run_experiments(datasets[0:5])
    testOrchestrator.perform_statistical_tests()
    
    
    
    print('****************************************************************')
    print('*           Experiments were run successfully                  *')
    print('****************************************************************')
    
