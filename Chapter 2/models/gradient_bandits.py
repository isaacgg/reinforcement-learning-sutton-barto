from typing import Optional

import numpy as np

from models.base_model import BaseModel


class GradientBandits(BaseModel):
    def __init__(self, K: int, use_baseline: bool = True):
        super().__init__()
        self.use_baseline = use_baseline
        self.K = K

        self.H = np.array([0] * K, dtype=np.float)

        self.R = 0
        self.nR = 1

        self.save_initial_values(H=self.H, R=self.R, nR=self.nR)

    def softmax(self):
        return np.exp(self.H) / np.sum(np.exp(self.H))

    def update_mean_reward(self, Rn: float):
        self.nR += 1
        self.R += (1 / self.nR) * (Rn - self.R)

    def learn(self, Rn: float, k: int, alpha: float):
        if self.use_baseline:
            self.update_mean_reward(Rn=Rn)
        pi = self.softmax()

        mask = np.zeros_like(self.H, dtype=bool)
        mask[k] = True

        self.H[mask] += alpha * (Rn - self.R) * (1 - pi[mask])
        self.H[~mask] -= alpha * (Rn - self.R) * (pi[~mask])

    def run(self, **kwargs):
        return np.random.choice(np.arange(self.K).tolist(), p=self.softmax())
