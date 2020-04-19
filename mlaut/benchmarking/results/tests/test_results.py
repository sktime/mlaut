from mlaut.benchmarking.results import RAMResults, HDDResults


def test_ram_results():
    results = RAMResults()

    results.save_predictions(strategy_name='test_strategy', 
                            dataset_name='test_dataset', 
                            y_true=[1,1,1], 
                            y_pred=[1,1,1], 
                            y_proba=[1,1,1], 
                            index=[1,2,3], 
                            cv_fold=1,
                            train_or_test='train')

def test_hdd_results():
    results = HDDResults('results')
    results.save_predictions(strategy_name='test_strategy', 
                            dataset_name='test_dataset', 
                            y_true=[1,1,1], 
                            y_pred=[1,1,1], 
                            y_proba=[1,1,1], 
                            index=[1,2,3], 
                            cv_fold=1,
                            train_or_test='train')