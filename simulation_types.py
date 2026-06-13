from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from agent_types import AgentGraph
from common import Vec2, clamp01, clamp11
from enums import ConceptTarget, DirectionType, PhysicalActionType
from environment_types import EnvironmentEntity, WorldState


@dataclass(slots=True)
class EventRecord:
    event_id: str
    tick: int
    event_type: str
    participants: List[str] = field(default_factory=list)
    location: Vec2 = field(default_factory=Vec2)
    payload: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.tick = max(0, self.tick)
        self.location.validate()


@dataclass(slots=True)
class TickActionProposal:
    physical_action: PhysicalActionType = PhysicalActionType.NONE
    target_id: Optional[str] = None
    direction: DirectionType = DirectionType.NONE
    intensity: float = 0.0
    primary_attention: Optional[str] = None
    affective_shift: float = 0.0
    memory_triggered: bool = False
    narration_active: bool = False
    narration_intensity: float = 0.0
    proto_concept_forming: bool = False
    concept_target: ConceptTarget = ConceptTarget.NONE

    def validate(self) -> None:
        self.intensity = clamp01(self.intensity)
        self.affective_shift = clamp11(self.affective_shift)
        self.narration_intensity = clamp01(self.narration_intensity)


@dataclass(slots=True)
class World:
    world_state: WorldState = field(default_factory=WorldState)
    environment_graph: Dict[str, EnvironmentEntity] = field(default_factory=dict)
    agent_graphs: Dict[str, AgentGraph] = field(default_factory=dict)
    event_log: List[EventRecord] = field(default_factory=list)

    def validate(self) -> None:
        self.world_state.validate()
        for entity in self.environment_graph.values():
            entity.validate()
        for agent in self.agent_graphs.values():
            agent.validate()
        for event in self.event_log:
            event.validate()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
