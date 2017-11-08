from src.test_orchestrator import TestOrchestrator
from src.map_datasets_to_ml_strategies import MapDatasets
from src.data import Data
from src.run_experiments import RunExperiments
from src.analyze_results import AnalyseResults
from src.files_io import FilesIO


if __name__ == '__main__': 
    data = Data()
    experiments = RunExperiments()
    analyze = AnalyseResults()
    testOrchestrator = TestOrchestrator(data)
    map_datasets = MapDatasets()
   
    files_io = FilesIO()
    testOrchestrator.setFilesIO(files_io)
    testOrchestrator.setAnalyzeResults(analyze)
    experiments.setTestOrchestrator(testOrchestrator)

    testOrchestrator.prepare_data()
    testOrchestrator.set_mapped_datasets(map_datasets)
    testOrchestrator.setExperiments(experiments)
    testOrchestrator.run_experiments()
    testOrchestrator.perform_statistical_tests()
    
    
    
    
    print('****************************************************************')
    print('*           Experiments were run successfully                  *')
    print('****************************************************************')
    
