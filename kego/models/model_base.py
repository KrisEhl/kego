from abc import ABC, abstractmethod


class ModelBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def train(self):
        raise NotImplementedError(f"{self.__class__} needs to implement train!")
