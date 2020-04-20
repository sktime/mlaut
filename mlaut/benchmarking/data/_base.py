class BaseDataset:

    def __init__(self, name):
        self._name = name

    def __repr__(self):
        class_name = self.__class__.__name__
        return f"{class_name}(name={self.name})"

    def load(self):
        raise NotImplementedError("abstract method")

    @property
    def name(self):
        return self._name