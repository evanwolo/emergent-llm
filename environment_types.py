from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from common import Vec2, clamp01
from enums import EntityClass, Season, TimeOfDay, Weather


@dataclass(slots=True)
class EnvironmentEntity:
    entity_id: str
    entity_class: EntityClass
    entity_kind: str
    exists: bool = True
    location: Vec2 = field(default_factory=Vec2)
    size: float = 0.0
    mass: float = 0.0
    state: Dict[str, float] = field(default_factory=dict)
    properties: Dict[str, float] = field(default_factory=dict)
    affordances: Dict[str, float] = field(default_factory=dict)
    relations: Dict[str, str] = field(default_factory=dict)
    visibility: float = 1.0
    availability: float = 1.0
    danger_level: float = 0.0
    resource_value: float = 0.0

    def validate(self) -> None:
        self.location.validate()
        self.visibility = clamp01(self.visibility)
        self.availability = clamp01(self.availability)
        self.danger_level = clamp01(self.danger_level)
        self.resource_value = clamp01(self.resource_value)
        self.size = max(0.0, float(self.size))
        self.mass = max(0.0, float(self.mass))


@dataclass(slots=True)
class WorldState:
    tick: int = 0
    time_of_day: TimeOfDay = TimeOfDay.DAY
    season: Season = Season.SPRING
    temperature: float = 0.5
    weather: Weather = Weather.CLEAR
    resource_pressure: float = 0.0
    population_count: int = 0
    birth_count: int = 0
    death_count: int = 0

    def validate(self) -> None:
        self.tick = max(0, int(self.tick))
        self.temperature = clamp01(self.temperature)
        self.resource_pressure = clamp01(self.resource_pressure)
        self.population_count = max(0, int(self.population_count))
        self.birth_count = max(0, int(self.birth_count))
        self.death_count = max(0, int(self.death_count))
