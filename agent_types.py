from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from common import BodyPressureSummary, InjuryState, Vec2, clamp01, clamp11
from enums import (
    AgeStage,
    BeliefDomain,
    BeliefExplicitness,
    BeliefSource,
    BeliefStatus,
    ConceptTarget,
    DirectionType,
    DiscoveryStatus,
    PhysicalActionType,
    ProcessingMode,
    RevisionMode,
    RevisionOutcome,
    Sex,
    TriggerType,
)


@dataclass(slots=True)
class BaselineTemperament:
    aggression: float = 0.0
    fearfulness: float = 0.0
    curiosity: float = 0.0
    sociability: float = 0.0
    patience: float = 0.0

    def validate(self) -> None:
        self.aggression = clamp01(self.aggression)
        self.fearfulness = clamp01(self.fearfulness)
        self.curiosity = clamp01(self.curiosity)
        self.sociability = clamp01(self.sociability)
        self.patience = clamp01(self.patience)


@dataclass(slots=True)
class CognitiveProfile:
    narration_baseline: float = 0.0
    reflection_capacity: float = 0.0
    abstraction_capacity: float = 0.0
    belief_explicitness: BeliefExplicitness = BeliefExplicitness.EMBODIED
    revision_mode: RevisionMode = RevisionMode.EMBODIED

    def validate(self) -> None:
        self.narration_baseline = clamp01(self.narration_baseline)
        self.reflection_capacity = clamp01(self.reflection_capacity)
        self.abstraction_capacity = clamp01(self.abstraction_capacity)


@dataclass(slots=True)
class IdentityGraph:
    agent_id: str
    alive: bool = True
    sex: Sex = Sex.OTHER
    age_ticks: int = 0
    age_stage: AgeStage = AgeStage.INFANT
    generation: int = 0
    body_location: Vec2 = field(default_factory=Vec2)
    orientation: float = 0.0
    baseline_temperament: BaselineTemperament = field(default_factory=BaselineTemperament)
    cognitive_profile: CognitiveProfile = field(default_factory=CognitiveProfile)

    def validate(self) -> None:
        self.age_ticks = max(0, int(self.age_ticks))
        self.generation = max(0, int(self.generation))
        self.body_location.validate()
        self.baseline_temperament.validate()
        self.cognitive_profile.validate()


@dataclass(slots=True)
class SurvivalGraph:
    self_exists: bool = True
    self_can_cease: bool = True
    self_requires_continuation: bool = True
    outside_exists: bool = True
    outside_affects_self: bool = True
    self_not_sufficient: bool = True
    self_can_be_harmed: bool = True
    pattern_can_repeat: bool = True
    change_is_real: bool = True
    attention_is_limited: bool = True


@dataclass(slots=True)
class BodyStates:
    energy_level: float = 1.0
    hydration_level: float = 1.0
    body_temperature: float = 0.5
    injury_state: InjuryState = field(default_factory=InjuryState)
    illness_state: float = 0.0
    fatigue_level: float = 0.0
    pain_level: float = 0.0
    hunger_level: float = 0.0
    thirst_level: float = 0.0
    arousal_state: float = 0.0
    threat_response: float = 0.0
    reproductive_state: float = 0.0
    gestation_state: float = 0.0
    healing_rate: float = 0.0
    growth_rate: float = 0.0

    def validate(self) -> None:
        self.energy_level = clamp01(self.energy_level)
        self.hydration_level = clamp01(self.hydration_level)
        self.body_temperature = clamp01(self.body_temperature)
        self.injury_state.validate()
        self.illness_state = clamp01(self.illness_state)
        self.fatigue_level = clamp01(self.fatigue_level)
        self.pain_level = clamp01(self.pain_level)
        self.hunger_level = clamp01(self.hunger_level)
        self.thirst_level = clamp01(self.thirst_level)
        self.arousal_state = clamp01(self.arousal_state)
        self.threat_response = clamp01(self.threat_response)
        self.reproductive_state = clamp01(self.reproductive_state)
        self.gestation_state = clamp01(self.gestation_state)
        self.healing_rate = clamp01(self.healing_rate)
        self.growth_rate = clamp01(self.growth_rate)


@dataclass(slots=True)
class BodyDrives:
    drive_eat: float = 0.0
    drive_drink: float = 0.0
    drive_sleep: float = 0.0
    drive_flee: float = 0.0
    drive_reproduce: float = 0.0
    drive_warmth: float = 0.0
    drive_contact: float = 0.0

    def validate(self) -> None:
        self.drive_eat = clamp01(self.drive_eat)
        self.drive_drink = clamp01(self.drive_drink)
        self.drive_sleep = clamp01(self.drive_sleep)
        self.drive_flee = clamp01(self.drive_flee)
        self.drive_reproduce = clamp01(self.drive_reproduce)
        self.drive_warmth = clamp01(self.drive_warmth)
        self.drive_contact = clamp01(self.drive_contact)


@dataclass(slots=True)
class BodyFacilities:
    move: bool = True
    eat: bool = True
    drink: bool = True
    sleep: bool = True
    fight: bool = True
    flee: bool = True
    reproduce: bool = True
    vocalize: bool = True
    manipulate: bool = True
    build: bool = True
    rest: bool = True


@dataclass(slots=True)
class BodyLimits:
    max_speed: float = 0.0
    max_endurance: float = 0.0
    carry_capacity: float = 0.0
    reach: float = 0.0
    fertility_window: float = 0.0

    def validate(self) -> None:
        self.max_speed = max(0.0, self.max_speed)
        self.max_endurance = max(0.0, self.max_endurance)
        self.carry_capacity = max(0.0, self.carry_capacity)
        self.reach = max(0.0, self.reach)
        self.fertility_window = max(0.0, self.fertility_window)


@dataclass(slots=True)
class BodyGraph:
    states: BodyStates = field(default_factory=BodyStates)
    drives: BodyDrives = field(default_factory=BodyDrives)
    facilities: BodyFacilities = field(default_factory=BodyFacilities)
    limits: BodyLimits = field(default_factory=BodyLimits)

    def validate(self) -> None:
        self.states.validate()
        self.drives.validate()
        self.limits.validate()


@dataclass(slots=True)
class ExperienceGraph:
    primary_attention: Optional[str] = None
    secondary_attention: List[str] = field(default_factory=list)
    affective_state: float = 0.0
    salience_level: float = 0.0
    novelty_signal: float = 0.0
    recurrence_signal: float = 0.0
    help_signal: float = 0.0
    harm_signal: float = 0.0
    pattern_pressure: float = 0.0
    memory_triggered: bool = False
    triggered_memory_ids: List[str] = field(default_factory=list)
    body_pressure_summary: BodyPressureSummary = field(default_factory=BodyPressureSummary)
    proto_concept_forming: bool = False
    concept_target: ConceptTarget = ConceptTarget.NONE

    def validate(self) -> None:
        self.affective_state = clamp11(self.affective_state)
        self.salience_level = clamp01(self.salience_level)
        self.novelty_signal = clamp01(self.novelty_signal)
        self.recurrence_signal = clamp01(self.recurrence_signal)
        self.help_signal = clamp01(self.help_signal)
        self.harm_signal = clamp01(self.harm_signal)
        self.pattern_pressure = clamp01(self.pattern_pressure)
        self.body_pressure_summary.validate()


@dataclass(slots=True)
class NarrationGraph:
    narration_active: bool = False
    narration_intensity: float = 0.0
    narration_baseline: float = 0.0
    internal_processing_mode: ProcessingMode = ProcessingMode.NONVERBAL
    concept_crystallization: float = 0.0
    current_proto_belief: Optional[str] = None
    self_commentary_pressure: float = 0.0
    drift_risk: float = 0.0

    def validate(self) -> None:
        self.narration_intensity = clamp01(self.narration_intensity)
        self.narration_baseline = clamp01(self.narration_baseline)
        self.concept_crystallization = clamp01(self.concept_crystallization)
        self.self_commentary_pressure = clamp01(self.self_commentary_pressure)
        self.drift_risk = clamp01(self.drift_risk)


@dataclass(slots=True)
class DiscoveryNode:
    discovered_target: str
    discovered_relation: str
    source_history: List[str] = field(default_factory=list)
    discovery_confidence: float = 0.0
    status: DiscoveryStatus = DiscoveryStatus.PROVISIONAL
    last_confirmed_tick: int = 0
    times_confirmed: int = 0
    times_failed: int = 0

    def validate(self) -> None:
        self.discovery_confidence = clamp01(self.discovery_confidence)
        self.last_confirmed_tick = max(0, self.last_confirmed_tick)
        self.times_confirmed = max(0, self.times_confirmed)
        self.times_failed = max(0, self.times_failed)


@dataclass(slots=True)
class DiscoveryGraph:
    discoveries: Dict[str, DiscoveryNode] = field(default_factory=dict)
    stabilized_relations: List[str] = field(default_factory=list)
    failed_relations: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)

    def validate(self) -> None:
        for node in self.discoveries.values():
            node.validate()


@dataclass(slots=True)
class EpisodicMemoryNode:
    event_id: str
    tick: int
    participants: List[str] = field(default_factory=list)
    location: Vec2 = field(default_factory=Vec2)
    body_before: Dict[str, float] = field(default_factory=dict)
    action_taken: Dict[str, str] = field(default_factory=dict)
    outcome: Dict[str, float] = field(default_factory=dict)
    affective_mark: float = 0.0
    salience: float = 0.0

    def validate(self) -> None:
        self.tick = max(0, self.tick)
        self.location.validate()
        self.affective_mark = clamp11(self.affective_mark)
        self.salience = clamp01(self.salience)


@dataclass(slots=True)
class MemoryGraph:
    episodic_memory: List[EpisodicMemoryNode] = field(default_factory=list)
    pattern_memory: List[str] = field(default_factory=list)
    social_memory: Dict[str, float] = field(default_factory=dict)
    place_memory: Dict[str, float] = field(default_factory=dict)
    body_memory: Dict[str, float] = field(default_factory=dict)
    trauma_memory: List[str] = field(default_factory=list)
    beauty_memory: List[str] = field(default_factory=list)
    learning_rate: float = 1.0
    consolidation_state: float = 0.0

    def validate(self) -> None:
        for episode in self.episodic_memory:
            episode.validate()
        self.learning_rate = clamp01(self.learning_rate)
        self.consolidation_state = clamp01(self.consolidation_state)


@dataclass(slots=True)
class SocialNode:
    other_agent_id: str
    recognized: bool = False
    familiarity: float = 0.0
    trust: float = 0.0
    fear: float = 0.0
    attachment: float = 0.0
    rivalry: float = 0.0
    dependency: float = 0.0
    dominance_estimate: float = 0.0
    shared_history_ids: List[str] = field(default_factory=list)
    contact_history: float = 0.0

    def validate(self) -> None:
        self.familiarity = clamp01(self.familiarity)
        self.trust = clamp01(self.trust)
        self.fear = clamp01(self.fear)
        self.attachment = clamp01(self.attachment)
        self.rivalry = clamp01(self.rivalry)
        self.dependency = clamp01(self.dependency)
        self.dominance_estimate = clamp01(self.dominance_estimate)
        self.contact_history = clamp01(self.contact_history)


@dataclass(slots=True)
class SocialGraph:
    known_agents: Dict[str, SocialNode] = field(default_factory=dict)
    kin_relations: Dict[str, str] = field(default_factory=dict)
    dependency_relations: Dict[str, str] = field(default_factory=dict)
    dominance_relations: Dict[str, str] = field(default_factory=dict)
    cooperation_relations: Dict[str, str] = field(default_factory=dict)
    conflict_relations: Dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        for node in self.known_agents.values():
            node.validate()


@dataclass(slots=True)
class PhysicalAction:
    action_type: PhysicalActionType = PhysicalActionType.REST
    target_id: Optional[str] = None
    direction: DirectionType = DirectionType.NONE
    intensity: float = 0.0

    def validate(self) -> None:
        self.intensity = clamp01(self.intensity)


@dataclass(slots=True)
class ExperientialAction:
    primary_attention_shift: Optional[str] = None
    avoidance_bias: float = 0.0
    approach_bias: float = 0.0

    def validate(self) -> None:
        self.avoidance_bias = clamp01(self.avoidance_bias)
        self.approach_bias = clamp01(self.approach_bias)


@dataclass(slots=True)
class NarrativeAction:
    processing_active: bool = False
    processing_target: Optional[str] = None
    reframing_active: bool = False


@dataclass(slots=True)
class ActionGraph:
    current_physical_action: PhysicalAction = field(default_factory=PhysicalAction)
    current_experiential_action: ExperientialAction = field(default_factory=ExperientialAction)
    current_narrative_action: NarrativeAction = field(default_factory=NarrativeAction)
    action_history: List[str] = field(default_factory=list)

    def validate(self) -> None:
        self.current_physical_action.validate()
        self.current_experiential_action.validate()


@dataclass(slots=True)
class BeliefNode:
    belief_id: str
    content_label: str
    domain: BeliefDomain
    source: BeliefSource
    confidence: float = 0.0
    coherence: float = 0.0
    success_rate: float = 0.0
    identity_weight: float = 0.0
    status: BeliefStatus = BeliefStatus.PROTO
    supporting_discoveries: List[str] = field(default_factory=list)
    opposing_discoveries: List[str] = field(default_factory=list)
    created_tick: int = 0
    last_revised_tick: int = 0

    def validate(self) -> None:
        self.confidence = clamp01(self.confidence)
        self.coherence = clamp01(self.coherence)
        self.success_rate = clamp01(self.success_rate)
        self.identity_weight = clamp01(self.identity_weight)
        self.created_tick = max(0, self.created_tick)
        self.last_revised_tick = max(0, self.last_revised_tick)


@dataclass(slots=True)
class BeliefGraph:
    proto_beliefs: Dict[str, BeliefNode] = field(default_factory=dict)
    stable_beliefs: Dict[str, BeliefNode] = field(default_factory=dict)
    locked_convictions: Dict[str, BeliefNode] = field(default_factory=dict)
    belief_relations: List[Tuple[str, str, str]] = field(default_factory=list)

    def validate(self) -> None:
        for bucket in (self.proto_beliefs, self.stable_beliefs, self.locked_convictions):
            for node in bucket.values():
                node.validate()


@dataclass(slots=True)
class ReflectionEvent:
    tick: int
    target_belief: str
    trigger_type: TriggerType
    old_confidence: float
    new_confidence: float
    revision_outcome: RevisionOutcome

    def validate(self) -> None:
        self.tick = max(0, self.tick)
        self.old_confidence = clamp01(self.old_confidence)
        self.new_confidence = clamp01(self.new_confidence)


@dataclass(slots=True)
class ReflectionGraph:
    presupposition_detected: bool = False
    self_questioning_active: bool = False
    current_target_belief: Optional[str] = None
    belief_tension: float = 0.0
    revision_pressure: float = 0.0
    candidate_revision: Optional[str] = None
    identity_cost: float = 0.0
    locked_belief_resistance: float = 0.0
    last_reflection_tick: int = 0
    reflection_history: List[ReflectionEvent] = field(default_factory=list)

    def validate(self) -> None:
        self.belief_tension = clamp01(self.belief_tension)
        self.revision_pressure = clamp01(self.revision_pressure)
        self.identity_cost = clamp01(self.identity_cost)
        self.locked_belief_resistance = clamp01(self.locked_belief_resistance)
        self.last_reflection_tick = max(0, self.last_reflection_tick)
        for event in self.reflection_history:
            event.validate()


@dataclass(slots=True)
class ConvictionInstance:
    instance_name: str
    instance_type: str
    locked_first_principles: List[str] = field(default_factory=list)
    derived_heuristics: List[str] = field(default_factory=list)
    novelty_triggers: List[str] = field(default_factory=list)
    forbidden_inferences: List[str] = field(default_factory=list)
    stress_test_history: List[str] = field(default_factory=list)


@dataclass(slots=True)
class AgentGraph:
    identity_graph: IdentityGraph
    survival_graph: SurvivalGraph = field(default_factory=SurvivalGraph)
    body_graph: BodyGraph = field(default_factory=BodyGraph)
    experience_graph: ExperienceGraph = field(default_factory=ExperienceGraph)
    narration_graph: NarrationGraph = field(default_factory=NarrationGraph)
    discovery_graph: DiscoveryGraph = field(default_factory=DiscoveryGraph)
    memory_graph: MemoryGraph = field(default_factory=MemoryGraph)
    social_graph: SocialGraph = field(default_factory=SocialGraph)
    action_graph: ActionGraph = field(default_factory=ActionGraph)
    belief_graph: BeliefGraph = field(default_factory=BeliefGraph)
    reflection_graph: ReflectionGraph = field(default_factory=ReflectionGraph)
    conviction_instance: Optional[ConvictionInstance] = None

    def validate(self) -> None:
        self.identity_graph.validate()
        self.body_graph.validate()
        self.experience_graph.validate()
        self.narration_graph.validate()
        self.discovery_graph.validate()
        self.memory_graph.validate()
        self.social_graph.validate()
        self.action_graph.validate()
        self.belief_graph.validate()
        self.reflection_graph.validate()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
