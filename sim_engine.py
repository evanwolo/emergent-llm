from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from threading import Lock
import time
from typing import Any, Dict, List, Optional, Tuple

import ollama

try:
    import numpy as np
except ImportError:
    np = None

from agent_types import (
    AgentGraph,
    DiscoveryNode,
    DiscoveryStatus,
    EpisodicMemoryNode,
    IdentityGraph,
    SocialNode,
)
from common import Vec2, clamp01
from enums import (
    AgeStage,
    ConceptTarget,
    DirectionType,
    EntityClass,
    PhysicalActionType,
    ProcessingMode,
    Season,
    Sex,
    TimeOfDay,
    Weather,
)
from environment_types import EnvironmentEntity
from simulation_types import EventRecord, TickActionProposal, World


TIME_CYCLE = [TimeOfDay.DAWN, TimeOfDay.DAY, TimeOfDay.DUSK, TimeOfDay.NIGHT]
SEASON_CYCLE = [Season.SPRING, Season.SUMMER, Season.AUTUMN, Season.WINTER]
WEATHER_CYCLE = [Weather.CLEAR, Weather.RAIN, Weather.CLEAR, Weather.STORM, Weather.CLEAR, Weather.SNOW]


def _runtime_validation_enabled() -> bool:
    return os.getenv("SIM_VALIDATE_TICKS", "0") == "1"


def _interpreter_enabled() -> bool:
    return os.getenv("SIM_ENABLE_INTERPRETER", "0") == "1"


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return default
    return max(minimum, parsed)


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return max(minimum, parsed)


def _inference_timeout_seconds() -> float:
    return _env_float("SIM_INFERENCE_TIMEOUT_SECONDS", 12.0, minimum=1.0)


def _inference_cooldown_ticks() -> int:
    return _env_int("SIM_INFERENCE_COOLDOWN_TICKS", 2, minimum=0)


def _interpreter_timeout_seconds() -> float:
    return _env_float("SIM_INTERPRETER_TIMEOUT_SECONDS", 8.0, minimum=1.0)


def _interpreter_cooldown_ticks() -> int:
    return _env_int("SIM_INTERPRETER_COOLDOWN_TICKS", 8, minimum=0)


def _interpreter_interval_ticks() -> int:
    return _env_int("SIM_INTERPRETER_INTERVAL_TICKS", 5, minimum=1)


def _vectorized_sensorium_enabled() -> bool:
    return np is not None and os.getenv("SIM_VECTORIZE_SENSORIUM", "1") == "1"


def _sparse_inference_enabled() -> bool:
    return os.getenv("SIM_SPARSE_INFERENCE", "1") == "1"


def _sparse_novelty_threshold() -> float:
    return clamp01(_env_float("SIM_SPARSE_NOVELTY_THRESHOLD", 0.5, minimum=0.0))


INTERPRETER_BOOTSTRAP_EVENT_ID = "simulation_bootstrap"
_INFERENCE_COOLDOWN_UNTIL_TICK = -1
_INTERPRETER_COOLDOWN_UNTIL_TICK = -1
_COOLDOWN_LOCK = Lock()


def _event_log_max_records() -> int:
    return _env_int("SIM_EVENT_LOG_MAX_RECORDS", 1000, minimum=0)


def _inference_max_workers() -> int:
    return _env_int("SIM_INFERENCE_MAX_WORKERS", 4, minimum=1)


def _trim_event_log(world: World) -> None:
    max_records = _event_log_max_records()
    if max_records <= 0:
        return
    if len(world.event_log) <= max_records:
        return

    bootstrap_event: Optional[EventRecord] = None
    for event in world.event_log:
        if event.event_id == INTERPRETER_BOOTSTRAP_EVENT_ID and event.event_type == "simulation_bootstrap":
            bootstrap_event = event
            break

    retained = world.event_log[-max_records:]
    if bootstrap_event is None or bootstrap_event in retained:
        world.event_log = retained
        return

    retained = [event for event in retained if event is not bootstrap_event]
    world.event_log = [bootstrap_event] + retained


def _append_event(world: World, event: EventRecord) -> None:
    world.event_log.append(event)
    _trim_event_log(world)


def _batch_query_agent_inference(
    entries: List[Tuple[str, AgentGraph, Dict[str, Any]]],
    tick: int,
) -> Dict[str, TickActionProposal]:
    if not entries:
        return {}

    max_workers = min(len(entries), _inference_max_workers())
    proposals: Dict[str, TickActionProposal] = {}

    if max_workers <= 1:
        for agent_id, agent, perception in entries:
            proposals[agent_id] = query_agent_inference(agent, perception, tick)
        return proposals

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_context = {
            executor.submit(query_agent_inference, agent, perception, tick): (agent_id, agent, perception)
            for agent_id, agent, perception in entries
        }

        for future in as_completed(future_to_context):
            agent_id, agent, perception = future_to_context[future]
            try:
                proposals[agent_id] = future.result()
            except Exception:
                visible_map = perception.get("visible_map", {}) if isinstance(perception, dict) else {}
                proposals[agent_id] = _heuristic_proposal(agent, visible_map)

    return proposals


# =====================================================================
# 1. PHASE 1 - WORLD UPDATE (CANONICAL REALITY ADVANCES FIRST)
# =====================================================================


def _next_cycle_value(current: Any, cycle: List[Any]) -> Any:
    if current not in cycle:
        return cycle[0]
    idx = cycle.index(current)
    return cycle[(idx + 1) % len(cycle)]


def phase1_world_update(world: World) -> None:
    state = world.world_state
    state.tick += 1

    state.time_of_day = TIME_CYCLE[(state.tick - 1) % len(TIME_CYCLE)]
    if state.tick % 120 == 0:
        state.season = _next_cycle_value(state.season, SEASON_CYCLE)
    if state.tick % 35 == 0:
        state.weather = _next_cycle_value(state.weather, WEATHER_CYCLE)

    seasonal_base_temp = {
        Season.SPRING: 0.55,
        Season.SUMMER: 0.72,
        Season.AUTUMN: 0.46,
        Season.WINTER: 0.28,
    }
    tod_adjust = {
        TimeOfDay.DAWN: -0.03,
        TimeOfDay.DAY: 0.05,
        TimeOfDay.DUSK: -0.01,
        TimeOfDay.NIGHT: -0.06,
    }
    weather_adjust = {
        Weather.CLEAR: 0.0,
        Weather.RAIN: -0.04,
        Weather.STORM: -0.08,
        Weather.SNOW: -0.12,
    }

    target_temperature = clamp01(
        seasonal_base_temp[state.season] + tod_adjust[state.time_of_day] + weather_adjust[state.weather]
    )
    state.temperature = round((state.temperature * 0.70) + (target_temperature * 0.30), 3)

    all_resource_levels: List[float] = []

    for entity in world.environment_graph.values():
        if not entity.exists:
            continue

        if "growth" in entity.state:
            entity.state["growth"] = round(clamp01(float(entity.state["growth"]) + 0.015), 3)
        if "decay" in entity.state:
            entity.state["decay"] = round(clamp01(float(entity.state["decay"]) + 0.020), 3)
        if entity.entity_class == EntityClass.PROCESS and "spread" in entity.state:
            spread_gain = 0.015 if state.weather != Weather.SNOW else -0.010
            entity.state["spread"] = round(clamp01(float(entity.state["spread"]) + spread_gain), 3)

        if entity.entity_kind in ("fluid", "water", "river", "lake"):
            refill = 0.01 + (0.03 if state.weather in (Weather.RAIN, Weather.STORM, Weather.SNOW) else 0.0)
            entity.availability = round(clamp01(entity.availability + refill), 3)
            entity.resource_value = round(clamp01(max(entity.resource_value, entity.availability)), 3)

        elif entity.entity_kind in ("berry", "plant", "flora", "carcass"):
            season_growth = 0.02 if state.season in (Season.SPRING, Season.SUMMER) else -0.01
            weather_growth = 0.01 if state.weather == Weather.RAIN else (-0.01 if state.weather == Weather.STORM else 0.0)
            decay_drag = float(entity.state.get("decay", 0.0)) * 0.02
            entity.availability = round(clamp01(entity.availability + season_growth + weather_growth - decay_drag), 3)
            entity.resource_value = round(clamp01((entity.resource_value * 0.70) + (entity.availability * 0.30)), 3)

        if entity.entity_class == EntityClass.ORGANISM and entity.entity_kind not in ("berry", "plant", "flora"):
            drift_x = math.sin((state.tick * 0.15) + len(entity.entity_id)) * 0.08
            drift_y = math.cos((state.tick * 0.11) + len(entity.entity_id)) * 0.06
            entity.location.x = round(entity.location.x + drift_x, 2)
            entity.location.y = round(entity.location.y + drift_y, 2)

        all_resource_levels.append(entity.availability)
        if _runtime_validation_enabled():
            entity.validate()

    if all_resource_levels:
        mean_availability = sum(all_resource_levels) / len(all_resource_levels)
        state.resource_pressure = round(clamp01(1.0 - mean_availability), 3)
    else:
        state.resource_pressure = 0.0

    state.population_count = sum(1 for a in world.agent_graphs.values() if a.identity_graph.alive)
    if _runtime_validation_enabled():
        state.validate()


# =====================================================================
# 2. PHASE 2 - BODY UPDATE (MECHANICAL PRESSURE CASCADE)
# =====================================================================


def process_somatic_decay(agent: AgentGraph) -> None:
    states = agent.body_graph.states
    drives = agent.body_graph.drives

    states.energy_level = round(max(0.0, states.energy_level - 0.015), 3)
    states.hydration_level = round(max(0.0, states.hydration_level - 0.025), 3)
    states.fatigue_level = round(min(1.0, states.fatigue_level + 0.010), 3)

    states.hunger_level = round(1.0 - states.energy_level, 3)
    states.thirst_level = round(1.0 - states.hydration_level, 3)

    drives.drive_eat = states.hunger_level
    drives.drive_drink = states.thirst_level
    drives.drive_sleep = states.fatigue_level

    agent.experience_graph.body_pressure_summary.hunger = drives.drive_eat
    agent.experience_graph.body_pressure_summary.thirst = drives.drive_drink
    agent.experience_graph.body_pressure_summary.fatigue = drives.drive_sleep
    agent.experience_graph.body_pressure_summary.pain = states.pain_level


def _derive_age_stage(age_ticks: int) -> AgeStage:
    if age_ticks < 80:
        return AgeStage.INFANT
    if age_ticks < 220:
        return AgeStage.CHILD
    if age_ticks < 420:
        return AgeStage.ADOLESCENT
    if age_ticks < 1600:
        return AgeStage.ADULT
    return AgeStage.ELDER


def phase2_body_update(agent: AgentGraph, world: World) -> bool:
    was_alive = agent.identity_graph.alive
    if not was_alive:
        return False

    process_somatic_decay(agent)

    states = agent.body_graph.states
    drives = agent.body_graph.drives
    world_state = world.world_state

    agent.identity_graph.age_ticks += 1
    agent.identity_graph.age_stage = _derive_age_stage(agent.identity_graph.age_ticks)

    ambient = world_state.temperature
    states.body_temperature = round(clamp01(states.body_temperature + (ambient - states.body_temperature) * 0.18), 3)

    if ambient < 0.25:
        states.energy_level = round(max(0.0, states.energy_level - 0.010), 3)
        states.fatigue_level = round(min(1.0, states.fatigue_level + 0.010), 3)
        drives.drive_warmth = round(clamp01(drives.drive_warmth + 0.08), 3)
    else:
        drives.drive_warmth = round(clamp01(drives.drive_warmth * 0.90), 3)

    if ambient > 0.78:
        states.hydration_level = round(max(0.0, states.hydration_level - 0.010), 3)

    injury = states.injury_state.total
    if injury > 0.0:
        states.pain_level = round(clamp01((states.pain_level * 0.65) + (injury * 0.35)), 3)
        states.fatigue_level = round(clamp01(states.fatigue_level + (injury * 0.020)), 3)
    else:
        states.pain_level = round(clamp01(states.pain_level * 0.90), 3)

    if states.healing_rate > 0.0 and injury > 0.0:
        states.injury_state.total = round(clamp01(states.injury_state.total - (states.healing_rate * 0.01)), 3)

    if agent.identity_graph.age_stage in (AgeStage.ADULT, AgeStage.ELDER):
        states.reproductive_state = round(clamp01(states.reproductive_state + 0.010), 3)
    else:
        states.reproductive_state = 0.0

    if agent.identity_graph.age_stage == AgeStage.ELDER:
        states.growth_rate = round(clamp01(states.growth_rate * 0.95), 3)
        states.fatigue_level = round(clamp01(states.fatigue_level + 0.005), 3)

    states.hunger_level = round(1.0 - states.energy_level, 3)
    states.thirst_level = round(1.0 - states.hydration_level, 3)

    states.threat_response = round(
        clamp01(max(states.pain_level, states.injury_state.total, 1.0 - states.energy_level, states.fatigue_level * 0.6)),
        3,
    )

    drives.drive_flee = round(clamp01(max(states.threat_response, states.pain_level)), 3)
    drives.drive_contact = round(
        clamp01(agent.identity_graph.baseline_temperament.sociability * (1.0 - states.threat_response)),
        3,
    )

    pressure = agent.experience_graph.body_pressure_summary
    pressure.pain = states.pain_level
    pressure.fatigue = states.fatigue_level
    pressure.hunger = states.hunger_level
    pressure.thirst = states.thirst_level
    pressure.fear = states.threat_response
    pressure.contact_need = drives.drive_contact

    if states.energy_level <= 0.01 or states.hydration_level <= 0.01:
        agent.identity_graph.alive = False
        agent.identity_graph.age_stage = AgeStage.DEAD

    if _runtime_validation_enabled():
        agent.validate()
    return was_alive and not agent.identity_graph.alive


# =====================================================================
# 3. PHASE 3 - PERCEPTUAL SAMPLING (SALIENT, NARROW, LAGGED)
# =====================================================================


def calculate_egocentric_vectors(agent: AgentGraph, world: World) -> Dict[str, Any]:
    ax = agent.identity_graph.body_location.x
    ay = agent.identity_graph.body_location.y
    sensorium: Dict[str, Any] = {}

    if _vectorized_sensorium_enabled():
        environment_entries = [
            (entity_id, entity)
            for entity_id, entity in world.environment_graph.items()
            if entity.exists
        ]

        if environment_entries:
            env_locations = np.array(
                [(entry.location.x, entry.location.y) for _, entry in environment_entries],
                dtype=np.float64,
            )
            env_deltas = env_locations - np.array([ax, ay], dtype=np.float64)
            env_distances = np.hypot(env_deltas[:, 0], env_deltas[:, 1])
            env_bearings = np.degrees(np.arctan2(env_deltas[:, 1], env_deltas[:, 0]))

            for idx, (entity_id, entity) in enumerate(environment_entries):
                sensorium[entity_id] = {
                    "distance": round(float(env_distances[idx]), 3),
                    "bearing": round(float(env_bearings[idx]), 1),
                    "class": entity.entity_class.value,
                    "danger_level": entity.danger_level,
                    "resource_value": entity.resource_value,
                    "properties": entity.properties,
                }

        other_agent_entries = [
            (other_id, other_agent)
            for other_id, other_agent in world.agent_graphs.items()
            if other_id != agent.identity_graph.agent_id and other_agent.identity_graph.alive
        ]

        if other_agent_entries:
            other_locations = np.array(
                [
                    (other.identity_graph.body_location.x, other.identity_graph.body_location.y)
                    for _, other in other_agent_entries
                ],
                dtype=np.float64,
            )
            other_deltas = other_locations - np.array([ax, ay], dtype=np.float64)
            other_distances = np.hypot(other_deltas[:, 0], other_deltas[:, 1])
            other_bearings = np.degrees(np.arctan2(other_deltas[:, 1], other_deltas[:, 0]))

            for idx, (other_id, other_agent) in enumerate(other_agent_entries):
                sensorium[other_id] = {
                    "distance": round(float(other_distances[idx]), 3),
                    "bearing": round(float(other_bearings[idx]), 1),
                    "class": EntityClass.AGENT_BODY.value,
                    "danger_level": other_agent.body_graph.states.threat_response,
                    "resource_value": 0.0,
                    "properties": {
                        "somatic_age_ticks": float(other_agent.identity_graph.age_ticks),
                        "alive": float(1.0 if other_agent.identity_graph.alive else 0.0),
                    },
                }

        return sensorium

    for entity_id, entity in world.environment_graph.items():
        if not entity.exists:
            continue

        dx = entity.location.x - ax
        dy = entity.location.y - ay
        distance = round(math.hypot(dx, dy), 3)
        bearing = round(math.degrees(math.atan2(dy, dx)), 1)

        sensorium[entity_id] = {
            "distance": distance,
            "bearing": bearing,
            "class": entity.entity_class.value,
            "danger_level": entity.danger_level,
            "resource_value": entity.resource_value,
            "properties": entity.properties,
        }

    for other_id, other_agent in world.agent_graphs.items():
        if other_id == agent.identity_graph.agent_id or not other_agent.identity_graph.alive:
            continue

        dx = other_agent.identity_graph.body_location.x - ax
        dy = other_agent.identity_graph.body_location.y - ay
        distance = round(math.hypot(dx, dy), 3)
        bearing = round(math.degrees(math.atan2(dy, dx)), 1)

        sensorium[other_id] = {
            "distance": distance,
            "bearing": bearing,
            "class": EntityClass.AGENT_BODY.value,
            "danger_level": other_agent.body_graph.states.threat_response,
            "resource_value": 0.0,
            "properties": {
                "somatic_age_ticks": float(other_agent.identity_graph.age_ticks),
                "alive": float(1.0 if other_agent.identity_graph.alive else 0.0),
            },
        }

    return sensorium


def _parse_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return parsed


def _entity_need_bias(agent: AgentGraph, entity_id: str, payload: Dict[str, Any]) -> float:
    drives = agent.body_graph.drives
    properties = payload.get("properties", {}) if isinstance(payload.get("properties"), dict) else {}

    drinkable = _parse_float(properties.get("drinkable", 0.0), 0.0)
    edible = _parse_float(properties.get("edible", 0.0), 0.0)

    if entity_id == "water_point" or drinkable > 0.0:
        return clamp01((drives.drive_drink * 0.8) + (drinkable * 0.2))
    if entity_id == "flora_point" or edible > 0.0:
        return clamp01((drives.drive_eat * 0.8) + (edible * 0.2))

    cls = str(payload.get("class", ""))
    if cls == EntityClass.AGENT_BODY.value:
        return clamp01((drives.drive_contact * 0.5) + (drives.drive_flee * 0.5))

    return 0.0


def phase3_perceptual_sampling(agent: AgentGraph, world: World, sensorium: Dict[str, Any]) -> Dict[str, Any]:
    scored: List[Dict[str, Any]] = []
    seen_patterns = set(agent.memory_graph.pattern_memory)

    for entity_id, payload in sensorium.items():
        distance = _parse_float(payload.get("distance", 999.0), 999.0)
        proximity = clamp01(1.0 - min(distance, 12.0) / 12.0)

        danger_signal = clamp01(_parse_float(payload.get("danger_level", 0.0), 0.0))
        resource_signal = clamp01(_parse_float(payload.get("resource_value", 0.0), 0.0))
        need_bias = _entity_need_bias(agent, entity_id, payload)

        pattern_key = f"seen::{entity_id}"
        novelty_signal = 0.0 if pattern_key in seen_patterns else 1.0
        prior_node = agent.discovery_graph.discoveries.get(entity_id)
        prior_confidence = prior_node.discovery_confidence if prior_node else 0.0

        salience = clamp01(
            (0.30 * proximity)
            + (0.25 * need_bias)
            + (0.15 * danger_signal)
            + (0.10 * resource_signal)
            + (0.10 * novelty_signal)
            + (0.10 * prior_confidence)
        )

        relief_potential = clamp01(resource_signal * max(agent.body_graph.drives.drive_eat, agent.body_graph.drives.drive_drink))
        pain_potential = clamp01(danger_signal + (agent.body_graph.states.pain_level * 0.3))

        scored.append(
            {
                "entity_id": entity_id,
                "salience": round(salience, 3),
                "distance": distance,
                "novelty": novelty_signal,
                "recurrence": 1.0 - novelty_signal,
                "valence": {
                    "relief": round(relief_potential, 3),
                    "pain": round(pain_potential, 3),
                    "danger": round(danger_signal, 3),
                    "novelty": round(novelty_signal, 3),
                },
            }
        )

    scored.sort(key=lambda item: item["salience"], reverse=True)
    visible = scored[:4]

    visible_map: Dict[str, Any] = {}
    visible_social: List[str] = []
    for item in visible:
        entity_id = item["entity_id"]
        visible_map[entity_id] = sensorium[entity_id]
        if str(sensorium[entity_id].get("class", "")) == EntityClass.AGENT_BODY.value:
            visible_social.append(entity_id)

    for item in visible:
        tag = f"seen::{item['entity_id']}"
        if tag not in seen_patterns:
            agent.memory_graph.pattern_memory.append(tag)
    if len(agent.memory_graph.pattern_memory) > 256:
        agent.memory_graph.pattern_memory = agent.memory_graph.pattern_memory[-256:]

    experience = agent.experience_graph
    if visible:
        experience.primary_attention = visible[0]["entity_id"]
        experience.secondary_attention = [entry["entity_id"] for entry in visible[1:]]
        experience.salience_level = visible[0]["salience"]
        experience.novelty_signal = round(max(entry["novelty"] for entry in visible), 3)
        experience.recurrence_signal = round(sum(entry["recurrence"] for entry in visible) / len(visible), 3)
        experience.help_signal = round(max(entry["valence"]["relief"] for entry in visible), 3)
        experience.harm_signal = round(max(entry["valence"]["pain"] for entry in visible), 3)
        experience.pattern_pressure = round(clamp01((experience.salience_level + experience.recurrence_signal) / 2.0), 3)
    else:
        experience.primary_attention = None
        experience.secondary_attention = []
        experience.salience_level = 0.0
        experience.novelty_signal = 0.0
        experience.recurrence_signal = 0.0
        experience.help_signal = 0.0
        experience.harm_signal = 0.0
        experience.pattern_pressure = 0.0

    surprise_flag = any(item["novelty"] >= 1.0 and item["salience"] >= 0.45 for item in visible)
    experience.proto_concept_forming = experience.pattern_pressure >= 0.45 or surprise_flag
    experience.concept_target = ConceptTarget.PATTERN if experience.proto_concept_forming else ConceptTarget.NONE
    if _runtime_validation_enabled():
        experience.validate()

    return {
        "ordered_elements": visible,
        "visible_map": visible_map,
        "visible_social": visible_social,
        "surprise_flag": surprise_flag,
    }


# =====================================================================
# 4. LLM BOUNDARY + PHASE 4 ACTION RESOLUTION
# =====================================================================


def _strip_control_chars(text: str) -> str:
    return "".join(ch for ch in text if ch in ("\n", "\r", "\t") or ord(ch) >= 32)


def _extract_balanced_json_objects(text: str):
    in_string = False
    escaped = False
    depth = 0
    start_index = None

    for idx, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            if depth == 0:
                start_index = idx
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start_index is not None:
                    yield text[start_index : idx + 1]
                    start_index = None


def _iter_json_candidates(raw_output: str):
    if not isinstance(raw_output, str):
        return

    base = _strip_control_chars(raw_output.strip().lstrip("\ufeff"))
    if not base:
        return

    seen = set()

    def emit(candidate: str):
        cleaned = _strip_control_chars(candidate.strip().lstrip("\ufeff"))
        if not cleaned:
            return
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            yield cleaned

    for candidate in emit(base):
        yield candidate

    if "```" in base:
        for piece in base.split("```"):
            snippet = piece.strip()
            if not snippet:
                continue
            if snippet.lower().startswith("json"):
                snippet = snippet[4:].strip()
            for candidate in emit(snippet):
                yield candidate

    for candidate in _extract_balanced_json_objects(base):
        for emitted in emit(candidate):
            yield emitted


def _parse_polluted_json(raw_output: str) -> Dict[str, Any]:
    for candidate in _iter_json_candidates(raw_output):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            try:
                parsed_literal = ast.literal_eval(candidate)
                if isinstance(parsed_literal, dict):
                    return parsed_literal
            except (ValueError, SyntaxError):
                continue
    return {}


def _parse_action(value: Any) -> PhysicalActionType:
    token = str(value).strip().lower()
    try:
        return PhysicalActionType(token)
    except ValueError:
        return PhysicalActionType.REST


def _parse_direction(value: Any) -> DirectionType:
    token = str(value).strip().lower()
    try:
        return DirectionType(token)
    except ValueError:
        return DirectionType.NONE


def _parse_concept_target(value: Any) -> ConceptTarget:
    token = str(value).strip().lower()
    try:
        return ConceptTarget(token)
    except ValueError:
        return ConceptTarget.NONE


def _proposal_from_parsed(parsed: Dict[str, Any]) -> TickActionProposal:
    proposal = TickActionProposal(
        physical_action=_parse_action(parsed.get("physical_action", "rest")),
        target_id=str(parsed.get("target_id", "none") or "none"),
        direction=_parse_direction(parsed.get("direction", "none")),
        intensity=clamp01(_parse_float(parsed.get("intensity", 0.0), 0.0)),
        primary_attention=str(parsed.get("primary_attention", "none") or "none"),
        proto_concept_forming=bool(parsed.get("proto_concept_forming", False)),
        concept_target=_parse_concept_target(parsed.get("concept_target", "none")),
    )
    proposal.validate()
    return proposal


def _heuristic_proposal(agent: AgentGraph, visible_map: Dict[str, Any]) -> TickActionProposal:
    states = agent.body_graph.states

    nearest_id = "none"
    nearest_distance = 1e9
    for entity_id, payload in visible_map.items():
        distance = _parse_float(payload.get("distance", 1e9), 1e9)
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_id = entity_id

    if states.thirst_level >= 0.65 and "water_point" in visible_map:
        if _parse_float(visible_map["water_point"].get("distance", 999.0), 999.0) <= 0.45:
            return TickActionProposal(
                physical_action=PhysicalActionType.DRINK,
                target_id="water_point",
                direction=DirectionType.NONE,
                intensity=0.75,
                primary_attention="water_point",
                proto_concept_forming=True,
                concept_target=ConceptTarget.NEED,
            )
        return TickActionProposal(
            physical_action=PhysicalActionType.MOVE,
            target_id="water_point",
            direction=DirectionType.TOWARD,
            intensity=0.80,
            primary_attention="water_point",
            proto_concept_forming=True,
            concept_target=ConceptTarget.NEED,
        )

    if states.hunger_level >= 0.65 and "flora_point" in visible_map:
        if _parse_float(visible_map["flora_point"].get("distance", 999.0), 999.0) <= 0.45:
            return TickActionProposal(
                physical_action=PhysicalActionType.EAT,
                target_id="flora_point",
                direction=DirectionType.NONE,
                intensity=0.75,
                primary_attention="flora_point",
                proto_concept_forming=True,
                concept_target=ConceptTarget.NEED,
            )
        return TickActionProposal(
            physical_action=PhysicalActionType.MOVE,
            target_id="flora_point",
            direction=DirectionType.TOWARD,
            intensity=0.75,
            primary_attention="flora_point",
            proto_concept_forming=True,
            concept_target=ConceptTarget.NEED,
        )

    if states.fatigue_level >= 0.80:
        return TickActionProposal(
            physical_action=PhysicalActionType.SLEEP,
            target_id="none",
            direction=DirectionType.NONE,
            intensity=0.50,
            primary_attention="none",
            proto_concept_forming=False,
            concept_target=ConceptTarget.NONE,
        )

    if nearest_id != "none":
        return TickActionProposal(
            physical_action=PhysicalActionType.MOVE,
            target_id=nearest_id,
            direction=DirectionType.TOWARD,
            intensity=0.35,
            primary_attention=nearest_id,
            proto_concept_forming=True,
            concept_target=ConceptTarget.PATTERN,
        )

    return TickActionProposal(
        physical_action=PhysicalActionType.REST,
        target_id="none",
        direction=DirectionType.NONE,
        intensity=0.10,
        primary_attention="none",
        proto_concept_forming=False,
        concept_target=ConceptTarget.NONE,
    )


def _has_locked_target_discovery(agent: AgentGraph, target_id: str) -> bool:
    if not target_id or target_id == "none":
        return False

    target_marker = f"_{target_id.upper()}_"
    for relation in agent.discovery_graph.stabilized_relations:
        if target_marker in relation:
            return True

    return False


def _current_attention_target(agent: AgentGraph, perception: Dict[str, Any]) -> str:
    primary_attention = str(agent.experience_graph.primary_attention or "none")
    if primary_attention != "none":
        return primary_attention

    ordered = perception.get("ordered_elements", []) if isinstance(perception, dict) else []
    if ordered:
        return str(ordered[0].get("entity_id", "none") or "none")

    return "none"


def _should_use_llm_inference(agent: AgentGraph, perception: Dict[str, Any]) -> bool:
    if not _sparse_inference_enabled():
        return True

    surprise_flag = bool(perception.get("surprise_flag", False)) if isinstance(perception, dict) else False
    if surprise_flag:
        return True

    if bool(agent.experience_graph.proto_concept_forming):
        return True

    if agent.experience_graph.novelty_signal >= _sparse_novelty_threshold():
        return True

    target_id = _current_attention_target(agent, perception)
    if _has_locked_target_discovery(agent, target_id):
        return False

    return True


def query_agent_inference(agent: AgentGraph, perception: Dict[str, Any], tick: int) -> TickActionProposal:
    global _INFERENCE_COOLDOWN_UNTIL_TICK

    visible_map = perception.get("visible_map", {}) if isinstance(perception, dict) else {}

    if os.getenv("SIM_FORCE_HEURISTIC", "0") == "1":
        return _heuristic_proposal(agent, visible_map)

    if not _should_use_llm_inference(agent, perception):
        return _heuristic_proposal(agent, visible_map)

    with _COOLDOWN_LOCK:
        inference_cooldown_until_tick = _INFERENCE_COOLDOWN_UNTIL_TICK

    if tick <= inference_cooldown_until_tick:
        return _heuristic_proposal(agent, visible_map)

    system_containment_rules = """
    You are calculating one action proposal for a bounded non-linguistic organism.
    Output STRICT JSON only. No prose. No markdown. No commentary.
    The world state is partial and salience-filtered.
    """

    runtime_context = {
        "tick": tick,
        "somatic_drives": {
            "hunger": agent.body_graph.drives.drive_eat,
            "thirst": agent.body_graph.drives.drive_drink,
            "fatigue": agent.body_graph.drives.drive_sleep,
            "fear": agent.body_graph.drives.drive_flee,
        },
        "perceived_elements": perception.get("ordered_elements", []),
    }

    user_query = f"""
    Process input: {json.dumps(runtime_context)}

    Return exactly this JSON layout:
    {{
        "physical_action": "move|eat|drink|sleep|fight|flee|vocalize|manipulate|build|rest|none",
        "target_id": "water_point|flora_point|agt_001|agt_002|none",
        "direction": "toward|away|none",
        "intensity": 0.0,
        "primary_attention": "water_point|flora_point|agt_001|agt_002|none",
        "proto_concept_forming": false,
        "concept_target": "self|other|object|threat|need|pattern|place|none"
    }}
    """

    try:
        model_name = os.getenv("OLLAMA_MODEL", "mistral").strip() or "mistral"
        inference_timeout = _inference_timeout_seconds()
        host = os.getenv("OLLAMA_HOST")
        if host:
            client = ollama.Client(host=host, timeout=inference_timeout)
        else:
            client = ollama.Client(timeout=inference_timeout)

        response = client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_containment_rules},
                {"role": "user", "content": user_query},
            ],
            options={"temperature": 0.0, "top_p": 0.1, "num_predict": 90},
        )

        raw_output = str(response.get("message", {}).get("content", "")).strip()
        parsed = _parse_polluted_json(raw_output)
        if not parsed:
            return _heuristic_proposal(agent, visible_map)
        return _proposal_from_parsed(parsed)
    except Exception:
        with _COOLDOWN_LOCK:
            _INFERENCE_COOLDOWN_UNTIL_TICK = max(
                _INFERENCE_COOLDOWN_UNTIL_TICK,
                tick + _inference_cooldown_ticks(),
            )
        return _heuristic_proposal(agent, visible_map)


def apply_tick_action(agent: AgentGraph, proposal: TickActionProposal) -> None:
    proposal.validate()

    agent.action_graph.current_physical_action.action_type = proposal.physical_action
    agent.action_graph.current_physical_action.target_id = proposal.target_id
    agent.action_graph.current_physical_action.direction = proposal.direction
    agent.action_graph.current_physical_action.intensity = proposal.intensity

    agent.experience_graph.primary_attention = proposal.primary_attention
    agent.experience_graph.affective_state = max(
        -1.0,
        min(1.0, agent.experience_graph.affective_state + proposal.affective_shift),
    )
    agent.experience_graph.memory_triggered = proposal.memory_triggered
    agent.experience_graph.proto_concept_forming = proposal.proto_concept_forming
    agent.experience_graph.concept_target = proposal.concept_target

    agent.narration_graph.narration_active = proposal.narration_active
    agent.narration_graph.narration_intensity = proposal.narration_intensity


def resolve_spatial_mechanics(agent: AgentGraph, proposal: TickActionProposal, world: World) -> None:
    if proposal.physical_action != PhysicalActionType.MOVE or proposal.target_id in (None, "none"):
        return

    target_loc: Optional[Vec2] = None
    if proposal.target_id in world.environment_graph:
        target_loc = world.environment_graph[proposal.target_id].location
    elif proposal.target_id in world.agent_graphs:
        target_loc = world.agent_graphs[proposal.target_id].identity_graph.body_location

    if target_loc is None:
        return

    my_loc = agent.identity_graph.body_location
    dx = target_loc.x - my_loc.x
    dy = target_loc.y - my_loc.y
    distance = math.hypot(dx, dy)

    if distance <= 0.2:
        return

    injury_penalty = clamp01(1.0 - agent.body_graph.states.injury_state.total * 0.5)
    max_speed = agent.body_graph.limits.max_speed if agent.body_graph.limits.max_speed > 0 else 1.5
    velocity = max(0.1, min(proposal.intensity * 2.0 * injury_penalty, max_speed))

    if proposal.direction == DirectionType.TOWARD:
        my_loc.x = round(my_loc.x + (dx / distance) * velocity, 2)
        my_loc.y = round(my_loc.y + (dy / distance) * velocity, 2)
    elif proposal.direction == DirectionType.AWAY:
        my_loc.x = round(my_loc.x - (dx / distance) * velocity, 2)
        my_loc.y = round(my_loc.y - (dy / distance) * velocity, 2)


def resolve_internal_rest_effects(agent: AgentGraph, proposal: TickActionProposal) -> List[str]:
    states = agent.body_graph.states
    effects: List[str] = []

    if proposal.physical_action == PhysicalActionType.SLEEP:
        states.fatigue_level = round(clamp01(states.fatigue_level - 0.12), 3)
        states.energy_level = round(clamp01(states.energy_level + 0.04), 3)
        effects.append("sleep_recovery")
    elif proposal.physical_action == PhysicalActionType.REST:
        states.fatigue_level = round(clamp01(states.fatigue_level - 0.04), 3)
        states.energy_level = round(clamp01(states.energy_level + 0.01), 3)
        effects.append("rest_recovery")

    states.hunger_level = round(1.0 - states.energy_level, 3)
    states.thirst_level = round(1.0 - states.hydration_level, 3)
    return effects


def _target_location(world: World, target_id: Optional[str]) -> Optional[Vec2]:
    if not target_id or target_id == "none":
        return None
    if target_id in world.environment_graph:
        return world.environment_graph[target_id].location
    if target_id in world.agent_graphs:
        return world.agent_graphs[target_id].identity_graph.body_location
    return None


def _distance_to_target(agent: AgentGraph, world: World, target_id: Optional[str]) -> Optional[float]:
    target_loc = _target_location(world, target_id)
    if target_loc is None:
        return None
    dx = target_loc.x - agent.identity_graph.body_location.x
    dy = target_loc.y - agent.identity_graph.body_location.y
    return math.hypot(dx, dy)


def resolve_metabolic_intersections(
    agent: AgentGraph,
    proposal: TickActionProposal,
    world: World,
    target_distance: Optional[float],
) -> Dict[str, Any]:
    result = {
        "consumed": False,
        "effects": [],
    }

    target_id = proposal.target_id
    if not target_id or target_id == "none":
        return result
    if target_distance is None or target_distance > 0.45:
        return result

    states = agent.body_graph.states

    if proposal.physical_action == PhysicalActionType.DRINK and target_id == "water_point":
        states.hydration_level = round(min(1.0, states.hydration_level + 0.38), 3)
        states.thirst_level = round(1.0 - states.hydration_level, 3)
        result["consumed"] = True
        result["effects"].append("hydration_relief")

        entity = world.environment_graph.get(target_id)
        if entity is not None:
            entity.availability = round(clamp01(entity.availability - 0.08), 3)
            entity.resource_value = round(clamp01(entity.resource_value - 0.04), 3)

    elif proposal.physical_action == PhysicalActionType.EAT and target_id == "flora_point":
        states.energy_level = round(min(1.0, states.energy_level + 0.30), 3)
        states.hunger_level = round(1.0 - states.energy_level, 3)
        result["consumed"] = True
        result["effects"].append("energy_relief")

        entity = world.environment_graph.get(target_id)
        if entity is not None:
            entity.availability = round(clamp01(entity.availability - 0.20), 3)
            entity.resource_value = round(clamp01(entity.resource_value - 0.10), 3)

    elif proposal.physical_action == PhysicalActionType.FIGHT and target_id in world.agent_graphs:
        other = world.agent_graphs[target_id]
        if other.identity_graph.alive:
            other.body_graph.states.injury_state.total = round(
                clamp01(other.body_graph.states.injury_state.total + 0.10),
                3,
            )
            other.body_graph.states.pain_level = round(clamp01(other.body_graph.states.pain_level + 0.10), 3)
            result["effects"].append("combat_contact")

    return result


def _normalize_action_target(perception: Dict[str, Any], proposal: TickActionProposal) -> TickActionProposal:
    visible_map = perception.get("visible_map", {}) if isinstance(perception, dict) else {}

    if proposal.target_id in (None, "none"):
        return proposal

    if proposal.target_id in visible_map:
        return proposal

    ordered = perception.get("ordered_elements", []) if isinstance(perception, dict) else []
    if not ordered:
        proposal.target_id = "none"
        proposal.direction = DirectionType.NONE
        if proposal.physical_action == PhysicalActionType.MOVE:
            proposal.physical_action = PhysicalActionType.REST
        return proposal

    proposal.target_id = ordered[0]["entity_id"]
    proposal.primary_attention = proposal.target_id
    if proposal.physical_action == PhysicalActionType.FLEE:
        proposal.direction = DirectionType.AWAY
    elif proposal.physical_action in (
        PhysicalActionType.MOVE,
        PhysicalActionType.EAT,
        PhysicalActionType.DRINK,
        PhysicalActionType.FIGHT,
        PhysicalActionType.VOCALIZE,
        PhysicalActionType.MANIPULATE,
        PhysicalActionType.BUILD,
    ):
        proposal.direction = DirectionType.TOWARD
    return proposal


def phase4_action_resolution(
    agent: AgentGraph,
    world: World,
    perception: Dict[str, Any],
    tick: int,
    precomputed_proposal: Optional[TickActionProposal] = None,
) -> Tuple[TickActionProposal, Dict[str, Any]]:
    proposal = precomputed_proposal or query_agent_inference(agent, perception, tick)
    proposal = _normalize_action_target(perception, proposal)
    proposal.validate()

    before = {
        "energy": agent.body_graph.states.energy_level,
        "hydration": agent.body_graph.states.hydration_level,
        "fatigue": agent.body_graph.states.fatigue_level,
        "pain": agent.body_graph.states.pain_level,
    }

    apply_tick_action(agent, proposal)

    initial_distance = _distance_to_target(agent, world, proposal.target_id)

    resolve_spatial_mechanics(agent, proposal, world)
    final_distance = _distance_to_target(agent, world, proposal.target_id)

    passive_effects = resolve_internal_rest_effects(agent, proposal)
    metabolic_result = resolve_metabolic_intersections(agent, proposal, world, final_distance)

    after = {
        "energy": agent.body_graph.states.energy_level,
        "hydration": agent.body_graph.states.hydration_level,
        "fatigue": agent.body_graph.states.fatigue_level,
        "pain": agent.body_graph.states.pain_level,
    }

    body_delta = {
        "energy": round(after["energy"] - before["energy"], 3),
        "hydration": round(after["hydration"] - before["hydration"], 3),
        "fatigue": round(after["fatigue"] - before["fatigue"], 3),
        "pain": round(after["pain"] - before["pain"], 3),
    }

    success = False
    partial = False

    if proposal.physical_action in (PhysicalActionType.REST, PhysicalActionType.SLEEP):
        success = True
    elif proposal.physical_action == PhysicalActionType.MOVE and proposal.target_id not in (None, "none"):
        if initial_distance is None or final_distance is None:
            success = False
            partial = True
        else:
            success = final_distance < initial_distance
            partial = final_distance == initial_distance
    elif proposal.physical_action in (PhysicalActionType.DRINK, PhysicalActionType.EAT):
        success = bool(metabolic_result["consumed"])
    elif proposal.physical_action == PhysicalActionType.FIGHT:
        success = "combat_contact" in metabolic_result["effects"]

    if not success and proposal.physical_action not in (PhysicalActionType.NONE, PhysicalActionType.REST):
        partial = True

    if body_delta["hydration"] > 0.02 or body_delta["energy"] > 0.02 or body_delta["fatigue"] < 0.0:
        valence = "relief"
    elif body_delta["pain"] > 0.02 or body_delta["hydration"] < 0.0 or body_delta["energy"] < 0.0:
        valence = "harm"
    else:
        valence = "neutral"

    outcome = {
        "success": success,
        "partial_success": partial,
        "valence": valence,
        "body_delta": body_delta,
        "environment_changes": passive_effects + list(metabolic_result["effects"]),
        "unexpected": bool(perception.get("surprise_flag", False) or (not success and proposal.target_id not in (None, "none"))),
        "observed_social_agents": list(perception.get("visible_social", [])),
    }

    return proposal, outcome


# =====================================================================
# 5. PHASE 5 - DISCOVERY CONSOLIDATION
# =====================================================================


def _discovery_relation_label(proposal: TickActionProposal, outcome: Dict[str, Any]) -> str:
    delta = outcome.get("body_delta", {}) if isinstance(outcome, dict) else {}
    hydration_gain = _parse_float(delta.get("hydration", 0.0), 0.0)
    energy_gain = _parse_float(delta.get("energy", 0.0), 0.0)

    if proposal.physical_action == PhysicalActionType.DRINK and hydration_gain > 0.01:
        return "reduces_thirst"
    if proposal.physical_action == PhysicalActionType.EAT and energy_gain > 0.01:
        return "reduces_hunger"
    if proposal.physical_action == PhysicalActionType.FLEE:
        return "increases_distance_from_threat"
    if proposal.physical_action == PhysicalActionType.FIGHT:
        return "direct_conflict_contact"

    return f"interacts_with_{proposal.concept_target.value}"


def phase5_discovery_consolidation(
    agent: AgentGraph,
    perception: Dict[str, Any],
    proposal: TickActionProposal,
    outcome: Dict[str, Any],
    tick: int,
) -> None:
    discovery_graph = agent.discovery_graph

    if proposal.target_id not in (None, "none"):
        target_id = str(proposal.target_id)
    elif perception.get("ordered_elements"):
        target_id = str(perception["ordered_elements"][0]["entity_id"])
    else:
        target_id = "none"

    if target_id == "none":
        return

    relation_label = _discovery_relation_label(proposal, outcome)
    signature = f"tick={tick}|action={proposal.physical_action.value}|relation={relation_label}|valence={outcome.get('valence', 'neutral')}"
    successful = bool(outcome.get("success", False))

    node = discovery_graph.discoveries.get(target_id)

    if node is None:
        node = DiscoveryNode(
            discovered_target=target_id,
            discovered_relation=relation_label,
            source_history=[signature],
            discovery_confidence=0.15 if successful else 0.08,
            status=DiscoveryStatus.PROVISIONAL,
            last_confirmed_tick=tick,
            times_confirmed=1 if successful else 0,
            times_failed=0 if successful else 1,
        )
        discovery_graph.discoveries[target_id] = node
        if not successful:
            question = f"why_failed::{target_id}::{proposal.physical_action.value}"
            if question not in discovery_graph.open_questions:
                discovery_graph.open_questions.append(question)
    else:
        if len(node.source_history) >= 30:
            node.source_history = node.source_history[-29:]
        node.source_history.append(signature)

        consistent = node.discovered_relation == relation_label

        if successful and consistent:
            node.times_confirmed += 1
            node.last_confirmed_tick = tick
            node.discovery_confidence = round(clamp01(node.discovery_confidence + 0.16), 3)
        elif successful and not consistent:
            node.times_failed += 1
            node.discovery_confidence = round(clamp01(node.discovery_confidence - 0.06), 3)
            contradiction = f"contradiction::{target_id}::{node.discovered_relation}->{relation_label}"
            if contradiction not in discovery_graph.open_questions:
                discovery_graph.open_questions.append(contradiction)
        else:
            node.times_failed += 1
            node.discovery_confidence = round(clamp01(node.discovery_confidence - 0.10), 3)
            failed_tag = f"failed::{target_id}::{proposal.physical_action.value}"
            if failed_tag not in discovery_graph.failed_relations:
                discovery_graph.failed_relations.append(failed_tag)

    if node.times_confirmed >= 2 and node.discovery_confidence >= 0.35 and node.status == DiscoveryStatus.PROVISIONAL:
        node.status = DiscoveryStatus.STABILIZING

    if node.times_confirmed >= 4 and node.discovery_confidence >= 0.70:
        node.status = DiscoveryStatus.LOCKED
        lock_tag = f"EARNED_CORE_RELATION_{target_id.upper()}_{node.discovered_relation.upper()}"
        if lock_tag not in discovery_graph.stabilized_relations:
            discovery_graph.stabilized_relations.append(lock_tag)

    visible_social = perception.get("visible_social", []) if isinstance(perception, dict) else []
    if visible_social:
        for other_id in visible_social:
            if other_id == agent.identity_graph.agent_id:
                continue
            seed_key = f"observed::{other_id}::{target_id}"
            if seed_key not in discovery_graph.discoveries:
                discovery_graph.discoveries[seed_key] = DiscoveryNode(
                    discovered_target=seed_key,
                    discovered_relation=f"observed_relation_with_{target_id}",
                    source_history=[f"tick={tick}|observation_seed"],
                    discovery_confidence=0.08,
                    status=DiscoveryStatus.PROVISIONAL,
                    last_confirmed_tick=tick,
                    times_confirmed=1,
                    times_failed=0,
                )

    episode = EpisodicMemoryNode(
        event_id=f"{agent.identity_graph.agent_id}_episode_{tick}",
        tick=tick,
        participants=[agent.identity_graph.agent_id] + list(visible_social),
        location=Vec2(agent.identity_graph.body_location.x, agent.identity_graph.body_location.y),
        body_before={
            "hunger": agent.experience_graph.body_pressure_summary.hunger,
            "thirst": agent.experience_graph.body_pressure_summary.thirst,
            "fatigue": agent.experience_graph.body_pressure_summary.fatigue,
            "pain": agent.experience_graph.body_pressure_summary.pain,
        },
        action_taken={
            "physical_action": proposal.physical_action.value,
            "target_id": proposal.target_id or "none",
            "direction": proposal.direction.value,
        },
        outcome={
            "success": 1.0 if outcome.get("success", False) else 0.0,
            "energy_delta": _parse_float(outcome.get("body_delta", {}).get("energy", 0.0), 0.0),
            "hydration_delta": _parse_float(outcome.get("body_delta", {}).get("hydration", 0.0), 0.0),
            "pain_delta": _parse_float(outcome.get("body_delta", {}).get("pain", 0.0), 0.0),
        },
        affective_mark=agent.experience_graph.affective_state,
        salience=agent.experience_graph.salience_level,
    )

    agent.memory_graph.episodic_memory.append(episode)
    if len(agent.memory_graph.episodic_memory) > 300:
        agent.memory_graph.episodic_memory = agent.memory_graph.episodic_memory[-300:]

    if outcome.get("valence") == "harm" and target_id not in agent.memory_graph.trauma_memory:
        agent.memory_graph.trauma_memory.append(target_id)
    if outcome.get("valence") == "relief" and target_id not in agent.memory_graph.beauty_memory:
        agent.memory_graph.beauty_memory.append(target_id)

    agent.memory_graph.consolidation_state = round(
        clamp01((agent.memory_graph.consolidation_state * 0.8) + (agent.experience_graph.pattern_pressure * 0.2)),
        3,
    )


# =====================================================================
# 6. SUBORDINATE NARRATION + SOCIAL STATE UPDATES
# =====================================================================


def update_narration_and_social_state(
    agent: AgentGraph,
    perception: Dict[str, Any],
    proposal: TickActionProposal,
    outcome: Dict[str, Any],
    tick: int,
) -> None:
    profile = agent.identity_graph.cognitive_profile
    narration = agent.narration_graph
    experience = agent.experience_graph

    if profile.abstraction_capacity < 0.33:
        narration.internal_processing_mode = ProcessingMode.NONVERBAL
    elif profile.abstraction_capacity < 0.66:
        narration.internal_processing_mode = ProcessingMode.FRAGMENTARY
    else:
        narration.internal_processing_mode = ProcessingMode.VERBAL

    narration.narration_baseline = max(narration.narration_baseline, profile.narration_baseline)
    narration.narration_intensity = round(
        clamp01(
            narration.narration_baseline
            + (experience.salience_level * 0.45)
            + (experience.pattern_pressure * 0.25)
            + (0.20 if outcome.get("unexpected", False) else 0.0)
        ),
        3,
    )
    narration.narration_active = narration.narration_intensity >= 0.10

    narration.concept_crystallization = round(
        clamp01(
            (experience.recurrence_signal * 0.45)
            + (experience.pattern_pressure * 0.45)
            + (0.10 if proposal.proto_concept_forming else 0.0)
        ),
        3,
    )

    if proposal.target_id not in (None, "none"):
        narration.current_proto_belief = f"{proposal.target_id}:{proposal.physical_action.value}"
    else:
        narration.current_proto_belief = None

    narration.self_commentary_pressure = round(
        clamp01((experience.harm_signal * 0.60) + (experience.novelty_signal * 0.40)),
        3,
    )
    narration.drift_risk = round(
        clamp01((1.0 - profile.reflection_capacity) * narration.self_commentary_pressure),
        3,
    )

    social = agent.social_graph
    for other_id in perception.get("visible_social", []):
        if other_id == agent.identity_graph.agent_id:
            continue

        node = social.known_agents.get(other_id)
        if node is None:
            node = SocialNode(other_agent_id=other_id, recognized=True)
            social.known_agents[other_id] = node

        node.contact_history = clamp01(node.contact_history + 0.10)
        node.familiarity = clamp01(node.familiarity + 0.06)

        if proposal.target_id == other_id and proposal.physical_action == PhysicalActionType.FLEE:
            node.fear = clamp01(node.fear + 0.08)
            node.dominance_estimate = clamp01(node.dominance_estimate + 0.05)
        elif proposal.target_id == other_id and proposal.physical_action == PhysicalActionType.FIGHT:
            node.rivalry = clamp01(node.rivalry + 0.08)
            node.fear = clamp01(node.fear + 0.05)
        elif proposal.target_id == other_id and proposal.physical_action == PhysicalActionType.VOCALIZE:
            node.attachment = clamp01(node.attachment + 0.05)

        if outcome.get("valence") == "relief" and proposal.target_id == other_id:
            node.trust = clamp01(node.trust + 0.04)
        if outcome.get("valence") == "harm":
            node.fear = clamp01(node.fear + 0.03)

        node.validate()

        if node.trust >= 0.60:
            social.cooperation_relations[other_id] = "trusted_partner"
        if node.fear >= 0.60:
            social.conflict_relations[other_id] = "threat_source"
        if node.attachment >= 0.60:
            social.dependency_relations[other_id] = "attachment_anchor"
        if node.dominance_estimate >= 0.60:
            social.dominance_relations[other_id] = "dominant_other"

    action_marker = f"{tick}:{proposal.physical_action.value}:{proposal.target_id or 'none'}:{outcome.get('valence', 'neutral')}"
    agent.action_graph.action_history.append(action_marker)
    if len(agent.action_graph.action_history) > 300:
        agent.action_graph.action_history = agent.action_graph.action_history[-300:]

    if _runtime_validation_enabled():
        narration.validate()
        social.validate()


# =====================================================================
# 7. PIPELINE ORCHESTRATION
# =====================================================================


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (bool, int, float, str)):
        return enum_value

    return str(value)


def _serialize_world_snapshot(world: World) -> Dict[str, Any]:
    world_state = world.world_state
    serialized_entities: Dict[str, Any] = {}
    serialized_agents: Dict[str, Any] = {}

    for entity_id, entity in world.environment_graph.items():
        serialized_entities[entity_id] = {
            "entity_class": entity.entity_class.value,
            "entity_kind": entity.entity_kind,
            "exists": entity.exists,
            "location": {"x": entity.location.x, "y": entity.location.y},
            "state": _json_safe(entity.state),
            "properties": _json_safe(entity.properties),
            "affordances": _json_safe(entity.affordances),
            "availability": entity.availability,
            "danger_level": entity.danger_level,
            "resource_value": entity.resource_value,
        }

    for agent_id, agent in world.agent_graphs.items():
        states = agent.body_graph.states
        drives = agent.body_graph.drives
        identity = agent.identity_graph
        profile = identity.cognitive_profile

        serialized_agents[agent_id] = {
            "identity": {
                "agent_id": identity.agent_id,
                "alive": identity.alive,
                "sex": identity.sex.value,
                "age_ticks": identity.age_ticks,
                "age_stage": identity.age_stage.value,
                "location": {"x": identity.body_location.x, "y": identity.body_location.y},
            },
            "body": {
                "energy_level": states.energy_level,
                "hydration_level": states.hydration_level,
                "fatigue_level": states.fatigue_level,
                "pain_level": states.pain_level,
                "hunger_level": states.hunger_level,
                "thirst_level": states.thirst_level,
                "threat_response": states.threat_response,
            },
            "drives": {
                "drive_eat": drives.drive_eat,
                "drive_drink": drives.drive_drink,
                "drive_sleep": drives.drive_sleep,
                "drive_flee": drives.drive_flee,
                "drive_contact": drives.drive_contact,
            },
            "cognitive_profile": {
                "narration_baseline": profile.narration_baseline,
                "reflection_capacity": profile.reflection_capacity,
                "abstraction_capacity": profile.abstraction_capacity,
            },
        }

    return {
        "world_state": {
            "tick": world_state.tick,
            "time_of_day": world_state.time_of_day.value,
            "season": world_state.season.value,
            "temperature": world_state.temperature,
            "weather": world_state.weather.value,
            "resource_pressure": world_state.resource_pressure,
            "population_count": world_state.population_count,
            "birth_count": world_state.birth_count,
            "death_count": world_state.death_count,
        },
        "environment_graph": serialized_entities,
        "agent_graphs": serialized_agents,
    }


def _simulation_configuration_snapshot() -> Dict[str, Any]:
    inference_model = os.getenv("OLLAMA_MODEL", "mistral").strip() or "mistral"
    interpreter_model = os.getenv("OLLAMA_INTERPRETER_MODEL", "sim-interpreter").strip() or "sim-interpreter"

    return {
        "runtime": {
            "sim_force_heuristic": os.getenv("SIM_FORCE_HEURISTIC", "0") == "1",
            "sim_validate_ticks": _runtime_validation_enabled(),
            "sim_enable_interpreter": _interpreter_enabled(),
            "inference_model": inference_model,
            "interpreter_model": interpreter_model,
            "sim_inference_timeout_seconds": _inference_timeout_seconds(),
            "sim_inference_cooldown_ticks": _inference_cooldown_ticks(),
            "sim_interpreter_timeout_seconds": _interpreter_timeout_seconds(),
            "sim_interpreter_cooldown_ticks": _interpreter_cooldown_ticks(),
            "sim_interpreter_interval_ticks": _interpreter_interval_ticks(),
            "sim_event_log_max_records": _event_log_max_records(),
            "sim_inference_max_workers": _inference_max_workers(),
            "sim_vectorize_sensorium": _vectorized_sensorium_enabled(),
            "sim_sparse_inference": _sparse_inference_enabled(),
            "sim_sparse_novelty_threshold": _sparse_novelty_threshold(),
        },
        "world_cycles": {
            "time_of_day": [entry.value for entry in TIME_CYCLE],
            "season": [entry.value for entry in SEASON_CYCLE],
            "weather": [entry.value for entry in WEATHER_CYCLE],
        },
        "action_space": [action.value for action in PhysicalActionType],
        "direction_space": [direction.value for direction in DirectionType],
        "concept_space": [target.value for target in ConceptTarget],
    }


def _ensure_interpreter_bootstrap_event(world: World) -> None:
    for event in world.event_log:
        if event.event_id == INTERPRETER_BOOTSTRAP_EVENT_ID and event.event_type == "simulation_bootstrap":
            return

    _append_event(
        world,
        EventRecord(
            event_id=INTERPRETER_BOOTSTRAP_EVENT_ID,
            tick=world.world_state.tick,
            event_type="simulation_bootstrap",
            participants=list(world.agent_graphs.keys()),
            location=Vec2(0.0, 0.0),
            payload={
                "initial_sim_state": _serialize_world_snapshot(world),
                "simulation_configuration": _simulation_configuration_snapshot(),
            },
        ),
    )


def _build_interpreter_tick_payload(world: World, tick: int, tick_events: List[EventRecord]) -> Dict[str, Any]:
    initial_state: Dict[str, Any] = {}
    simulation_configuration: Dict[str, Any] = _simulation_configuration_snapshot()

    for event in world.event_log:
        if event.event_id != INTERPRETER_BOOTSTRAP_EVENT_ID or event.event_type != "simulation_bootstrap":
            continue

        if isinstance(event.payload, dict):
            if isinstance(event.payload.get("initial_sim_state"), dict):
                initial_state = _json_safe(event.payload["initial_sim_state"])
            if isinstance(event.payload.get("simulation_configuration"), dict):
                simulation_configuration = _json_safe(event.payload["simulation_configuration"])
        break

    tick_event_summaries: List[Dict[str, Any]] = []
    for event in tick_events:
        tick_event_summaries.append(
            {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "participants": list(event.participants),
                "location": {"x": event.location.x, "y": event.location.y},
                "payload": _json_safe(event.payload),
            }
        )

    return {
        "tick": tick,
        "initial_sim_state": initial_state,
        "simulation_configuration": simulation_configuration,
        "current_world_state": _serialize_world_snapshot(world),
        "tick_events": tick_event_summaries,
    }


def _heuristic_interpreter_summary(payload: Dict[str, Any]) -> str:
    tick = int(payload.get("tick", 0))
    world_state = payload.get("current_world_state", {}).get("world_state", {})
    event_count = len(payload.get("tick_events", []))

    return (
        "Reality update: "
        f"tick={tick}, time_of_day={world_state.get('time_of_day', 'unknown')}, "
        f"season={world_state.get('season', 'unknown')}, weather={world_state.get('weather', 'unknown')}. "
        "Interpreter context coverage: "
        f"initial_state={'yes' if bool(payload.get('initial_sim_state')) else 'no'}, "
        f"configuration={'yes' if bool(payload.get('simulation_configuration')) else 'no'}, "
        f"tick_events={event_count}."
    )


def _query_tick_interpreter(payload: Dict[str, Any]) -> str:
    global _INTERPRETER_COOLDOWN_UNTIL_TICK

    tick = int(payload.get("tick", 0))
    with _COOLDOWN_LOCK:
        interpreter_cooldown_until_tick = _INTERPRETER_COOLDOWN_UNTIL_TICK

    if tick <= interpreter_cooldown_until_tick:
        return _heuristic_interpreter_summary(payload)

    system_instruction = """
    You are the simulation interpreter for a deterministic, graph-based world.
    Use initial_sim_state and simulation_configuration as baseline context.
    Explain what changed this tick relative to that baseline and avoid invention.
    If required context is missing, state what is missing.
    Keep output concise plain text with labels.
    """

    user_query = f"""
    Interpret this simulation payload:
    {json.dumps(payload)}

    Use this structure:
    1) Reality update
    2) Body pressure
    3) Perception and action
    4) Discovery consolidation
    5) Social and narration
    """

    try:
        model_name = os.getenv("OLLAMA_INTERPRETER_MODEL", "sim-interpreter").strip() or "sim-interpreter"
        interpreter_timeout = _interpreter_timeout_seconds()
        host = os.getenv("OLLAMA_HOST")
        if host:
            client = ollama.Client(host=host, timeout=interpreter_timeout)
        else:
            client = ollama.Client(timeout=interpreter_timeout)

        response = client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_query},
            ],
            options={"temperature": 0.1, "top_p": 0.2, "num_predict": 280},
        )

        summary = str(response.get("message", {}).get("content", "")).strip()
        return summary or _heuristic_interpreter_summary(payload)
    except Exception:
        with _COOLDOWN_LOCK:
            _INTERPRETER_COOLDOWN_UNTIL_TICK = max(
                _INTERPRETER_COOLDOWN_UNTIL_TICK,
                tick + _interpreter_cooldown_ticks(),
            )
        return _heuristic_interpreter_summary(payload)


def _append_interpreter_summary_event(world: World, tick: int, tick_events: List[EventRecord]) -> None:
    if not _interpreter_enabled():
        return

    interval = _interpreter_interval_ticks()
    if interval > 1 and (tick % interval != 0):
        return

    payload = _build_interpreter_tick_payload(world, tick, tick_events)
    summary = _query_tick_interpreter(payload)
    _append_event(
        world,
        EventRecord(
            event_id=f"interpreter_tick_{tick}",
            tick=tick,
            event_type="interpreter_summary",
            participants=[],
            location=Vec2(0.0, 0.0),
            payload={
                "summary": summary,
                "source_event_ids": [event.event_id for event in tick_events],
                "context_coverage": {
                    "initial_sim_state": bool(payload.get("initial_sim_state")),
                    "simulation_configuration": bool(payload.get("simulation_configuration")),
                },
            },
        ),
    )


def export_agent_visible_state(agent: AgentGraph, world: World) -> Dict[str, Any]:
    if _runtime_validation_enabled():
        agent.validate()
        world.validate()
    return {
        "identity": {
            "agent_id": agent.identity_graph.agent_id,
            "alive": agent.identity_graph.alive,
            "age_stage": agent.identity_graph.age_stage.value,
        },
        "body": {
            "energy_level": agent.body_graph.states.energy_level,
            "hydration_level": agent.body_graph.states.hydration_level,
            "fatigue_level": agent.body_graph.states.fatigue_level,
            "pain_level": agent.body_graph.states.pain_level,
            "hunger_level": agent.body_graph.states.hunger_level,
            "thirst_level": agent.body_graph.states.thirst_level,
            "threat_response": agent.body_graph.states.threat_response,
        },
        "experience": {
            "primary_attention": agent.experience_graph.primary_attention,
            "salience_level": agent.experience_graph.salience_level,
            "novelty_signal": agent.experience_graph.novelty_signal,
            "recurrence_signal": agent.experience_graph.recurrence_signal,
            "proto_concept_forming": agent.experience_graph.proto_concept_forming,
            "concept_target": agent.experience_graph.concept_target.value,
        },
        "allowed_actions": [action.value for action in PhysicalActionType],
    }


def run_simulation_tick(world: World) -> None:
    _ensure_interpreter_bootstrap_event(world)

    # Phase 1: world reality advances independent of agents.
    phase1_world_update(world)
    current_tick = world.world_state.tick
    tick_events: List[EventRecord] = []

    world.world_state.birth_count = 0
    world.world_state.death_count = 0

    active_agents: List[Tuple[str, AgentGraph, Dict[str, Any]]] = []

    for agent_id, agent in world.agent_graphs.items():
        if not agent.identity_graph.alive:
            continue

        # Phase 2: body pressure updates (no choice yet).
        died_this_tick = phase2_body_update(agent, world)
        if died_this_tick:
            world.world_state.death_count += 1
            continue
        if not agent.identity_graph.alive:
            continue

        # Phase 3: narrow, salience-ordered perception sampling.
        sensorium = calculate_egocentric_vectors(agent, world)
        perception = phase3_perceptual_sampling(agent, world, sensorium)

        active_agents.append((agent_id, agent, perception))

    proposals = _batch_query_agent_inference(active_agents, current_tick)

    for agent_id, agent, perception in active_agents:
        proposal = proposals.get(agent_id)
        if proposal is None:
            visible_map = perception.get("visible_map", {}) if isinstance(perception, dict) else {}
            proposal = _heuristic_proposal(agent, visible_map)

        # Phase 4: resolve exactly one primary action against world reality.
        proposal, outcome = phase4_action_resolution(
            agent,
            world,
            perception,
            current_tick,
            precomputed_proposal=proposal,
        )

        # Phase 5: consolidate discovery from perception-action-outcome tuple.
        phase5_discovery_consolidation(agent, perception, proposal, outcome, current_tick)

        # Subordinate update: narration + social valence integration.
        update_narration_and_social_state(agent, perception, proposal, outcome, current_tick)

        tick_event = EventRecord(
            event_id=f"{agent_id}_tick_{current_tick}",
            tick=current_tick,
            event_type="tick_pipeline",
            participants=[agent_id] + list(perception.get("visible_social", [])),
            location=Vec2(agent.identity_graph.body_location.x, agent.identity_graph.body_location.y),
            payload={
                "phase_3_primary_attention": agent.experience_graph.primary_attention,
                "phase_4_action": {
                    "physical_action": proposal.physical_action.value,
                    "target_id": proposal.target_id,
                    "direction": proposal.direction.value,
                    "intensity": proposal.intensity,
                },
                "phase_4_outcome": outcome,
                "phase_5_discoveries_locked": list(agent.discovery_graph.stabilized_relations),
                "somatic": {
                    "energy": agent.body_graph.states.energy_level,
                    "hydration": agent.body_graph.states.hydration_level,
                    "fatigue": agent.body_graph.states.fatigue_level,
                    "pain": agent.body_graph.states.pain_level,
                },
            },
        )
        _append_event(world, tick_event)
        tick_events.append(tick_event)

    world.world_state.population_count = sum(1 for a in world.agent_graphs.values() if a.identity_graph.alive)
    _append_interpreter_summary_event(world, current_tick, tick_events)
    if _runtime_validation_enabled():
        world.validate()


# =====================================================================
# 8. INTEGRATION TEST RUN
# =====================================================================


if __name__ == "__main__":
    if "SIM_FORCE_HEURISTIC" not in os.environ:
        os.environ["SIM_FORCE_HEURISTIC"] = "1"

    sim_world = World()
    sim_world.environment_graph = {
        "water_point": EnvironmentEntity(
            "water_point",
            EntityClass.PLACE,
            "fluid",
            location=Vec2(1.5, -0.5),
            properties={"drinkable": 1.0},
            availability=1.0,
            resource_value=1.0,
        ),
        "flora_point": EnvironmentEntity(
            "flora_point",
            EntityClass.OBJECT,
            "berry",
            location=Vec2(-2.0, 1.0),
            properties={"edible": 1.0},
            availability=1.0,
            resource_value=1.0,
        ),
        "wind_front": EnvironmentEntity(
            "wind_front",
            EntityClass.PROCESS,
            "weather_cell",
            location=Vec2(0.0, 0.0),
            state={"spread": 0.2},
            danger_level=0.1,
        ),
    }

    alpha = AgentGraph(identity_graph=IdentityGraph(agent_id="agt_001", sex=Sex.MALE, body_location=Vec2(0.0, 0.0)))
    beta = AgentGraph(identity_graph=IdentityGraph(agent_id="agt_002", sex=Sex.FEMALE, body_location=Vec2(0.5, 0.5)))

    alpha.body_graph.limits.max_speed = 1.5
    beta.body_graph.limits.max_speed = 1.5

    alpha.identity_graph.cognitive_profile.narration_baseline = 0.15
    alpha.identity_graph.cognitive_profile.reflection_capacity = 0.35
    alpha.identity_graph.cognitive_profile.abstraction_capacity = 0.30

    beta.identity_graph.cognitive_profile.narration_baseline = 0.25
    beta.identity_graph.cognitive_profile.reflection_capacity = 0.50
    beta.identity_graph.cognitive_profile.abstraction_capacity = 0.55

    sim_world.agent_graphs = {"agt_001": alpha, "agt_002": beta}

    print("--- RUNNING MODULAR FIVE-PHASE TICK ENGINE ---")
    print(f"[Inference Mode] SIM_FORCE_HEURISTIC={os.getenv('SIM_FORCE_HEURISTIC', '0')}")
    print(
        "[Interpreter Mode] "
        f"SIM_ENABLE_INTERPRETER={os.getenv('SIM_ENABLE_INTERPRETER', '0')} "
        f"OLLAMA_INTERPRETER_MODEL={os.getenv('OLLAMA_INTERPRETER_MODEL', 'sim-interpreter')}"
    )

    for step in range(1, 6):
        if step == 2:
            sim_world.agent_graphs["agt_001"].body_graph.states.hydration_level = 0.15
            sim_world.agent_graphs["agt_001"].body_graph.states.thirst_level = 0.85
            sim_world.environment_graph["water_point"].location = Vec2(0.0, 0.1)
            print(f"\n[Tick {step}: Shock Injection -> acute dehydration + nearby water]")

        run_simulation_tick(sim_world)

        print(f"\nSimulation Report - Tick {sim_world.world_state.tick}:")
        print(
            "  * World: "
            f"time={sim_world.world_state.time_of_day.value}, "
            f"season={sim_world.world_state.season.value}, "
            f"weather={sim_world.world_state.weather.value}, "
            f"temp={sim_world.world_state.temperature}"
        )

        for agent_id, instance in sim_world.agent_graphs.items():
            loc = instance.identity_graph.body_location
            print(f"  * Agent [{agent_id}] Pos: ({loc.x}, {loc.y}) Alive={instance.identity_graph.alive}")
            print(
                "    - Somatic: "
                f"E={instance.body_graph.states.energy_level} "
                f"H={instance.body_graph.states.hydration_level} "
                f"F={instance.body_graph.states.fatigue_level} "
                f"P={instance.body_graph.states.pain_level}"
            )
            print(
                "    - Perception/Narration: "
                f"attention={instance.experience_graph.primary_attention} "
                f"salience={instance.experience_graph.salience_level} "
                f"mode={instance.narration_graph.internal_processing_mode.value}"
            )
            print(f"    - Locked Discoveries: {instance.discovery_graph.stabilized_relations}")

        if _interpreter_enabled():
            interpreter_event = next(
                (
                    event
                    for event in reversed(sim_world.event_log)
                    if event.event_type == "interpreter_summary" and event.tick == sim_world.world_state.tick
                ),
                None,
            )
            if interpreter_event is not None:
                print(f"  * Interpreter Summary: {interpreter_event.payload.get('summary', '')}")

        time.sleep(0.2)

    print("\n--- FIVE-PHASE TICK ENGINE RUN COMPLETE ---")
