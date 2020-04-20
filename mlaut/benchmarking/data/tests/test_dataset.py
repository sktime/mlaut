import pandas as pd
from mlaut.benchmarking.data import RAMDataset
from sklearn import datasets

def test_ram_dataset():
    iris = datasets.load_iris()
    wine = datasets.load_wine()

    iris_pd = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_pd['target'] = iris.target

    wine_pd = pd.DataFrame(wine.data, columns=wine.feature_names)
    wine_pd['target'] = wine.target

    dts = ([RAMDataset(iris_pd, name='iris'),
                RAMDataset(wine_pd, name='wine')])


