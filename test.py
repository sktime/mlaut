from mlaut.data import Data
import pydataset
from mlaut.experiments import Orchestrator
from mlaut.estimators.baseline_estimators import Baseline_Classifier
from mlaut.estimators.bayes_estimators import Gaussian_Naive_Bayes
from mlaut.estimators.decision_trees import Decision_Tree_Classifier

from mlaut.resampling import Single_Split

from sklearn import preprocessing
orchard = pydataset.data('OrchardSprays')
aids = pydataset.data('aids')
iris = pydataset.data('iris')

orchard_meta = {
    'class_name': 'treatment',
    'source':'pydataset',
    'dataset_name':'OrchardSprays'
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
le = preprocessing.LabelEncoder()
#transform labels for orchard
le.fit(orchard['treatment'])
orchard['treatment']=le.transform(orchard['treatment'])

#transform labels for iris
le.fit(iris['Species'])
iris['Species']=le.transform(iris['Species'])


data = Data(hdf5_datasets_group='pydata')
data.set_io(input_data='data/test_input.h5', output_data='data/test_output.h5')

datasets = [orchard, aids, iris]
metadata = [orchard_meta, aids_meta, iris_meta]
data.pandas_to_db(datasets=datasets, dts_metadata=metadata)
data.split_datasets()

orcheststrator = Orchestrator()
orcheststrator.set_data(data)
orcheststrator.set_strategies([Baseline_Classifier(), Decision_Tree_Classifier(), Gaussian_Naive_Bayes()])
orcheststrator.set_resampling(Single_Split())
orcheststrator.run()