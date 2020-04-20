__all__ = ["UEADataset", "RAMDataset", "make_datasets"]
__author__ = ["Viktor Kazakov", "Markus Löning"]

import os

import pandas as pd

from mlaut.benchmarking.data import BaseDataset

class RAMDataset(BaseDataset):

    def __init__(self, dataset, name):
        """
        Container for storing a dataset in memory
        """
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError(f"Dataset must be pandas DataFrame, but found: "
                             f"{type(dataset)}")
        self._dataset = dataset
        super(RAMDataset, self).__init__(name=name)

    def load(self):
        return self._dataset

