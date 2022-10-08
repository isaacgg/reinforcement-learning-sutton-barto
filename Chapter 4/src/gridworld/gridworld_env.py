from enum import Enum
from typing import List, Tuple

import numpy as np

from src.model.action import Action
from src.model.abc.environment import Environment
from src.model.reward import Reward
from src.model.state import State


class StepOptions(Enum):
    stochastic = 0
    deterministic = 1


class GridWorldEnv(Environment):
    def __init__(self, states: List[State], actions: List[Action], rewards: List[Reward], initial_state: int = 0):
        self.states = states
        self.actions = actions
        self.rewards = rewards

        self.curr_state = initial_state

    def _stochastic_choice(self, next_state_reward_prob: np.ndarray) -> Tuple[int, int]:
        choice = np.random.choice(np.size(next_state_reward_prob),
                                  p=(next_state_reward_prob / np.sum(next_state_reward_prob)).reshape([-1]))
        next_state = choice // next_state_reward_prob.shape[0]
        reward = choice % next_state_reward_prob.shape[1]

        return next_state, reward

    def _deterministic_choice(self, next_state_reward_prob: np.ndarray) -> Tuple[int, int]:
        choice = np.argmax(next_state_reward_prob)

        next_state = choice[0]
        reward = choice[1]
        return next_state, reward

    def step(self, action: Action, step_type: int = StepOptions.stochastic):
        next_state_reward_prob = np.asarray(
            self.actions[self.actions.index(action)].state_next_state_reward_probability
        )[:, :, self.curr_state]  # p(s',r|a,s): [s',r] probability matrix

        if step_type == StepOptions.stochastic:
            next_state, reward = self._stochastic_choice(next_state_reward_prob)
        elif step_type == StepOptions.deterministic:
            next_state, reward = self._deterministic_choice(next_state_reward_prob)
        else:
            return NotImplemented

        self.curr_state = next_state
        return self.rewards[reward]

    def get_current_state(self):
        return self.curr_state
