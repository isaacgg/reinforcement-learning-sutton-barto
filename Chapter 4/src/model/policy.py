from typing import List

import numpy as np

from src.model.action import Action
from src.model.state import State


class Policy:
    def __init__(self, policy: np.ndarray):
        self.policy = policy  # p(a|s) -> [s, a]

    def get_policy_matrix(self, states: List[State], actions: List[Action]) -> np.ndarray:
        return np.asarray([[actions.index(action), states.index(state)] for action, state in self.policy.items()])
