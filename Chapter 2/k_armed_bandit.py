import numpy as np


class KArmedBandit:
    def __init__(self, K, mean: float = 0) -> None:
        self.K = K

        self.action_values = np.random.normal(mean, 1, K)

        self.optimal_action = np.argmax(self.action_values)

    def pull(self, action: int):
        return self.action_values[action] + np.random.normal(0, 1)


if __name__ == "__main__":
    bandit = KArmedBandit(5)
    print(bandit.pull(0))
    print(bandit.pull(0))
    print(bandit.pull(0))
    print(bandit.pull(0))
    print(bandit.pull(0))
    print(bandit.pull(0))
    print(bandit.pull(0))
