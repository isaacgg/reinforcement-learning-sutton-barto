from dataclasses import dataclass


@dataclass(frozen=True)
class Action:
    value: str
