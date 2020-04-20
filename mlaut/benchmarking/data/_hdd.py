
from mlaut.benchmarking.data import BaseDataset
import os
class HDDBaseDataset(BaseDataset):

    def __init__(self, path, name):
        self._path = path
        super(HDDBaseDataset, self).__init__(name=name)

    @property
    def path(self):
        return self._path

    @staticmethod
    def _validate_path(path):
        """Helper function to validate paths"""
        # check if path already exists
        if not os.path.exists(path):
            raise ValueError(f"No dataset found at path: {path}")