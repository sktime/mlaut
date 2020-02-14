__all__ = ["UEADataset", "RAMDataset", "make_datasets"]
__author__ = ["Viktor Kazakov", "Markus Löning"]

import os

import pandas as pd

from mlaut.benchmarking.base import BaseDataset, HDDBaseDataset
# from mlaut.utils.load_data import load_from_tsfile_to_dataframe

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


def make_datasets(path, dataset_cls, names=None, **kwargs):
    """Make datasets"""
    # check dataset class
    # if not isinstance(dataset_cls, BaseDataset):
    #     raise ValueError(f"dataset must inherit from BaseDataset, but found:"
    #                      f"{type(dataset_cls)}")

    # check dataset names
    if names is not None:
        if not isinstance(names, list):
            raise ValueError(f"names must be a list, but found: {type(names)}")
    else:
        names = os.listdir(path)  # get names if names is not specified

    # generate datasets
    datasets = [dataset_cls(path=path, name=name, **kwargs) for name in names]
    return datasets
