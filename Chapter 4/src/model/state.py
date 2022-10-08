from typing import Dict

import numpy as np

from src.model.action import Action


class State:
    def __init__(self, actions: Dict[Action, np.ndarray]):
        self.actions = actions

    def get_actions_matrix(self) -> np.ndarray:
        matrix = [probs for probs in self.actions.values()]  # [next_state, reward, action]
        return np.asarray(matrix)
