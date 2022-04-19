from typing import Optional

import numpy as np

from k_armed_bandit import KArmedBandit
from models.base_model import BaseModel


class NArmedTestbed:
    def __init__(self, solver: BaseModel, N: int, K: int, mean: float = 0) -> None:
        self.bandits = [KArmedBandit(K, mean=mean) for _ in range(N)]
    
        self.solver = solver

    def run(self, steps: int, alpha: Optional[float] = None):
        total_rewards = []
        total_choices = []
        for bandit in self.bandits:
            self.solver.reset()
            rewards = []
            choices = []

            for _ in range(steps):
                choice = self.solver.run()
                reward = bandit.pull(choice)
                self.solver.learn(Rn=reward, k=choice, alpha=alpha)
                
                rewards.append(reward)
                choices.append(int(choice == bandit.optimal_action))
        
            total_rewards.append(rewards)
            total_choices.append(choices)
        return np.array(total_rewards), np.array(total_choices)
