from abc import ABC, abstractmethod
from typing import List

from src.model.action import Action
from src.model.reward import Reward
from src.model.state import State


class Environment(ABC):
    states: List[State]
    actions: List[Action]
    rewards: List[Reward]

    @abstractmethod
    def step(self, action: Action) -> Reward:
        return NotImplemented

    def get_states(self) -> List[State]:
        return self.states

    def get_actions(self) -> List[Action]:
        return self.actions

    def get_rewards(self) -> List[Reward]:
        return self.rewards
