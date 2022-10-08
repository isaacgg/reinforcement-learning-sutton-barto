import math
from typing import Tuple, Optional

import numpy as np

from src.gridworld.gridworld_env import GridWorldEnv
from src.model.action import Action
from src.model.abc.agent import Agent
from src.model.abc.environment import Environment
from src.model.policy import Policy
from src.model.reward import Reward
from src.model.state import State


class GridWorldAgent(Agent):
    def __init__(self, environment: Environment, gamma: float, v0: float, policy: Policy):
        self.environment = environment
        self.state_values = np.array([v0] * len(self.environment.get_states()))
        self.policy = policy
        self.gamma = gamma

    def _get_rewards_from_next_state_reward_probability(self) -> np.ndarray:
        return np.asarray([r.reward for r in self.environment.get_rewards()])

    def _get_next_state_reward_probability(self) -> np.ndarray:
        next_state_reward_probability = np.asarray([s.get_actions_matrix()
                                               for s in self.environment.get_states()])
        return np.transpose(next_state_reward_probability, (2, 3, 0, 1))  # p(s',r|s,a)

    def _maximize_policy(self, expected_reward_by_action: np.ndarray):
        """ expected_reward_by_action: must be a matrix of shape n_states, n_actions """
        action_maxs = np.argwhere(np.isclose(expected_reward_by_action, np.max(expected_reward_by_action, axis=1)[:, None]))
        policy = np.zeros_like(expected_reward_by_action)
        policy[action_maxs[:, 0], action_maxs[:, 1]] = 1
        policy = policy / np.sum(policy, axis=1)[:, None]
        return policy

    def _policy_evaluation_step(self, policy: np.ndarray, next_state_reward_probability: np.ndarray, rewards: np.ndarray):
        new_state_values = np.einsum("sa,sa->s",
                                     policy,
                                     np.einsum("nrsa,rn->sa",
                                               next_state_reward_probability,
                                               (np.repeat(rewards[:, None], len(self.state_values), axis=1) +
                                                self.gamma * np.repeat(self.state_values[None, :], len(rewards),
                                                                       axis=0))))
        new_state_values[[0, -1]] = 0  # terminal states values are always 0
        return new_state_values

    def _policy_improvement_step(self, next_state_reward_probability: np.ndarray, rewards: np.ndarray) -> np.ndarray:
        expected_reward_by_action = np.einsum("nrsa,rn->sa",
                                              next_state_reward_probability,  # p(s',r|s,a)
                                              (np.repeat(rewards[:, None], len(self.state_values), axis=1) +
                                               self.gamma * np.repeat(self.state_values[None, :], len(rewards),
                                                                      axis=0)))
        return expected_reward_by_action

    def policy_evaluation(self, threshold: Optional[float] = None, max_steps: Optional[int] = None, **kwargs) -> None:
        assert (threshold is None or max_steps is None), "threshold or max_steps, at least one should be defined"
        threshold = -np.inf if threshold is None else threshold
        max_steps = np.inf if max_steps is None else max_steps

        policy = self.policy.policy
        next_state_reward_probability = self._get_next_state_reward_probability()
        rewards = self._get_rewards_from_next_state_reward_probability()

        delta = 100
        steps = 0
        while (delta > threshold) and ((steps := steps + 1) < max_steps):
            new_state_values = self._policy_evaluation_step(policy=policy,
                                                            next_state_reward_probability=next_state_reward_probability,
                                                            rewards=rewards)
            delta = max(abs(self.state_values - new_state_values))
            self.state_values = new_state_values

    def policy_improvement(self, **kwargs) -> None:
        policy = self.policy.policy
        next_state_reward_probability = self._get_next_state_reward_probability()  # transition dynamics
        rewards = self._get_rewards_from_next_state_reward_probability()

        self.policy_evaluation(**kwargs)
        expected_reward_by_action = self._policy_improvement_step(next_state_reward_probability=next_state_reward_probability,
                                                                  rewards=rewards)
        new_policy = self._maximize_policy(expected_reward_by_action)
        while not np.array_equal(policy, new_policy):
            policy = new_policy
            self.policy_evaluation(**kwargs)
            expected_reward_by_action = self._policy_improvement_step(next_state_reward_probability=next_state_reward_probability,
                                                                      rewards=rewards)
            new_policy = self._maximize_policy(expected_reward_by_action)

        self.policy = Policy(policy=policy)

    # Not working :'(
    def value_iteration(self, threshold: float = None, max_steps: float = None) -> None:
        assert (threshold is None or max_steps is None), "threshold or max_steps, at least one should be defined"
        threshold = -np.inf if threshold is None else threshold
        max_steps = np.inf if max_steps is None else max_steps

        next_state_reward_probability = self._get_next_state_reward_probability()
        rewards = self._get_rewards_from_next_state_reward_probability()

        delta = 100
        steps = 0
        expected_reward_by_action = None
        while (delta > threshold) and ((steps := steps + 1) < max_steps):
            expected_reward_by_action = self._policy_improvement_step(next_state_reward_probability=next_state_reward_probability,
                                                                      rewards=rewards)

            new_state_values = np.mean(expected_reward_by_action, axis=1)
            new_state_values[[0, -1]] = 0  # terminal states values are always 0

            delta = max(abs(self.state_values - new_state_values))
            self.state_values = new_state_values

        policy = self._maximize_policy(expected_reward_by_action)
        self.policy = Policy(policy=policy)


if __name__ == "__main__":
    def _create_state_reward_probabilities(set_to_one: Optional[Tuple[int, int]] = None,
                                           n_states: int = 16,
                                           n_rewards: int = 2):
        matrix = np.zeros([n_states, n_rewards])
        if set_to_one is not None:
            matrix[set_to_one[0], set_to_one[1]] = 1
        return matrix


    def create_state(coordinates: Tuple[Optional[Tuple[int, int]], ...]) -> State:
        actions = [Action("up"), Action("left"), Action("down"), Action("right")]
        state = {actions[0]: _create_state_reward_probabilities(coordinates[0]),
                 actions[1]: _create_state_reward_probabilities(coordinates[1]),
                 actions[2]: _create_state_reward_probabilities(coordinates[2]),
                 actions[3]: _create_state_reward_probabilities(coordinates[3])}
        return State(state)


    actions = [Action("up"), Action("left"), Action("down"), Action("right")]
    rewards = [Reward(0), Reward(-1)]
    states = [create_state(((0, 1), (0, 1), (4, 1), (1, 1))),
              create_state(((1, 1), (0, 1), (5, 1), (2, 1))),
              create_state(((2, 1), (1, 1), (6, 1), (3, 1))),
              create_state(((3, 1), (2, 1), (7, 1), (3, 1))),

              create_state(((0, 1), (4, 1), (8, 1), (5, 1))),
              create_state(((1, 1), (4, 1), (9, 1), (6, 1))),
              create_state(((2, 1), (5, 1), (10, 1), (7, 1))),
              create_state(((3, 1), (6, 1), (11, 1), (7, 1))),

              create_state(((4, 1), (8, 1), (12, 1), (9, 1))),
              create_state(((5, 1), (8, 1), (13, 1), (10, 1))),
              create_state(((6, 1), (9, 1), (14, 1), (11, 1))),
              create_state(((7, 1), (10, 1), (15, 1), (11, 1))),

              create_state(((8, 1), (12, 1), (12, 1), (13, 1))),
              create_state(((9, 1), (12, 1), (13, 1), (14, 1))),
              create_state(((10, 1), (13, 1), (14, 1), (15, 1))),
              create_state(((11, 1), (14, 1), (15, 1), (15, 1))), ]

    environment = GridWorldEnv(states=states, actions=actions, initial_state=0, rewards=rewards)

    policy = Policy(0.25 * np.ones([len(states), len(actions)]))

    agent_policy = GridWorldAgent(environment=environment, gamma=1, policy=policy, v0=0)
    agent_policy.policy_improvement(threshold=1e-7)

    agent_value = GridWorldAgent(environment=environment, gamma=1, policy=policy, v0=0)
    agent_value.value_iteration(threshold=1e-7)
    print(agent_policy.policy.policy)
    print(agent_value.policy.policy)
    print(np.array_equal(agent_policy.policy.policy, agent_value.policy.policy))

    print(agent_policy.state_values.reshape([4, 4]))
    print(agent_value.state_values.reshape([4, 4]))

