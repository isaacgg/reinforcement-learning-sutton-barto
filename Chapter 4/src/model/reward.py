from dataclasses import dataclass


@dataclass(frozen=True)
class Reward:
    reward: float
