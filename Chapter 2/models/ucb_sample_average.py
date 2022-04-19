from typing import Optional

import numpy as np

from models.base_model import BaseModel


class UcbSampleAverage(BaseModel):
    def __init__(self, K: int, c: float, Q0: Optional[float] = 0) -> None:
        super().__init__()
        self.K = K
        self.c = c

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

    def _random_argmax(self, array):
        max_value = np.max(array)
        argmaxs = np.where(array == max_value)[0]
        return np.random.choice(argmaxs)

    def run(self):
        t = max(sum(self.N), 1)
        N = np.where(self.N == 0, 1, self.N)  # Only necessary if initial N==0
        ucb = self.Q + self.c * np.sqrt(np.log(t) / N)
        choice = self._random_argmax(ucb)
        N[choice] += 1
        return choice


if __name__ == "__main__":
    rl = UcbSampleAverage(10, 0, 0.1)
    rl.learn(0.5, 0)
    rl.learn(1, 0)
    rl.learn(0, 0)

    print(rl.Q)