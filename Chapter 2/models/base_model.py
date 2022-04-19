from abc import ABC, abstractmethod
from copy import copy


class BaseModel(ABC):
    def __init__(self, **kwargs):
        self.initial_values = {}

    def reset(self):
        for item, value in self.initial_values.items():
            setattr(self, item, copy(value))

    def save_initial_values(self, **kwargs):
        for item, value in kwargs.items():
            self.initial_values[item] = copy(value)

    @abstractmethod
    def run(self, **kwargs):
        ...

    @abstractmethod
    def learn(self, **kwargs):
        ...
