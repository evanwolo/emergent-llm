from __future__ import annotations

from dataclasses import dataclass


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def clamp11(v: float) -> float:
    return max(-1.0, min(1.0, v))


@dataclass(slots=True)
class Vec2:
    x: float = 0.0
    y: float = 0.0

    def validate(self) -> None:
        self.x = float(self.x)
        self.y = float(self.y)


@dataclass(slots=True)
class BodyPressureSummary:
    pain: float = 0.0
    fatigue: float = 0.0
    hunger: float = 0.0
    thirst: float = 0.0
    fear: float = 0.0
    contact_need: float = 0.0

    def validate(self) -> None:
        self.pain = clamp01(self.pain)
        self.fatigue = clamp01(self.fatigue)
        self.hunger = clamp01(self.hunger)
        self.thirst = clamp01(self.thirst)
        self.fear = clamp01(self.fear)
        self.contact_need = clamp01(self.contact_need)


@dataclass(slots=True)
class InjuryState:
    total: float = 0.0
    head: float = 0.0
    torso: float = 0.0
    limbs: float = 0.0

    def validate(self) -> None:
        self.total = clamp01(self.total)
        self.head = clamp01(self.head)
        self.torso = clamp01(self.torso)
        self.limbs = clamp01(self.limbs)
