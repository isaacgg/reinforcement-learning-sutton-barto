from typing import Optional

import numpy as np

from models.base_model import BaseModel


class EpsilonGreedySampleAverage(BaseModel):
    def __init__(self, K: int, Q0: Optional[float] = 0, eps: float = 0) -> None:
        super().__init__()

        self.K = K

        self.eps = eps

        self.Q = [Q0] * K
        self.N = [1] * K

        self.save_initial_values(Q=self.Q, N=self.N)

    def learn(self, Rn: float, k: int, alpha: Optional[float]=None):
        # efficient sample-average
        # use alpha = 1/n for stationary problems
        self.N[k] += 1

        if alpha is not None:
            self.Q[k] += alpha * (Rn - self.Q[k])
        self.Q[k] += (1 / self.N[k]) * (Rn - self.Q[k])

    def explore(self):
        choice = np.random.randint(0, len(self.Q))
        return choice

    def explote(self):
        max_value = np.max(self.Q)
        argmaxs = np.where(self.Q == max_value)[0]
        return np.random.choice(argmaxs)

    def run(self):
        if np.random.rand() < self.eps:
            return self.explore()
        return self.explote()


if __name__ == "__main__":
    rl = EpsilonGreedySampleAverage(10, 0, 0.1)
    rl.learn(0.5, 0)
    rl.learn(1, 0)
    rl.learn(0, 0)

    print(rl.Q)