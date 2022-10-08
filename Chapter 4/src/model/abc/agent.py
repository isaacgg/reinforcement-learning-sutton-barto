from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def policy_evaluation(self, threshold: float, gamma: float):
        return NotImplemented
