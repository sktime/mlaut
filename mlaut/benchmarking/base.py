from abc import ABC, abstractmethod

__author__ = ["Markus Löning", "Viktor Kazakov"]

class BaseMetric(ABC):

    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs

    @abstractmethod
    def compute(self, y_true, y_pred):
        """Compute mean and standard error of metric"""
