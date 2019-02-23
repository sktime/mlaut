from mlaut.data import Data
import pydataset
from mlaut.experiments import Orchestrator
from mlaut.estimators.baseline_estimators import Baseline_Classifier
from mlaut.estimators.ensemble_estimators import (Random_Forest_Classifier,
                                                Bagging_Classifier)

acme = pydataset.data('acme')
aids = pydataset.data('aids')
iris = pydataset.data('iris')

acme_meta = {
    'class_name': 'acme',
    'source':'pydataset',
    'dataset_name':'acme'
}

aids_meta = {
    'class_name': 'adult',
    'source':'pydataset',
    'dataset_name':'aids'
}

iris_meta = {
    'class_name': 'Species',
    'source':'pydataset',
    'dataset_name':'iris'
}

data = Data(hdf5_datasets_group='pydata')
data.set_io(input_data='data/test_input.h5', output_data='data/test_output.h5')

datasets = [acme, aids, iris]
metadata = [acme_meta, aids_meta, iris_meta]
data.pandas_to_db(datasets=datasets, dts_metadata=metadata)
data.split_datasets()

orcheststrator = Orchestrator(data)
orcheststrator.run(modelling_strategies=[Baseline_Classifier(), Random_Forest_Classifier(), Bagging_Classifier()])