from abc import ABC, abstractmethod

class MleapEstimator(ABC):

    @abstractmethod
    def build(self):
        pass