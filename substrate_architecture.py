import ast
import json
import math
import os
import time
import ollama

# =====================================================================
# 1. ATOMIC CORE DATA SCHEMAS
# =====================================================================

environment_state = {
    "global_states": {
        "light_level": 1.0,          # 0.0 (midnight) to 1.0 (noon)
        "temperature": 0.6,          # 0.0 (extreme cold) to 1.0 (extreme heat)
        "seasonal_cycle": 0.0,       # 0.0 (spring) -> 0.75 (winter)
        "tick_count": 0
    },
    "entities": {
        "ent_001": {  # Structural raw material (Tree)
            "spatial": {"distance": 4.5, "bearing": 180},
            "affordances": ["structural_rigid", "fragmentable", "sedentary"],
            "state": {"density": 0.8, "yield_potential": 1.0}
        },
        "ent_002": {  # Metabolic source (Water)
            "spatial": {"distance": 5.2, "bearing": 45},
            "affordances": ["fluid", "internal_absorbable"],
            "state": {"volume": 1.0, "pure": True}
        },
        "ent_003": {  # Metabolic source (Flora/Berry)
            "spatial": {"distance": 3.0, "bearing": 270},
            "affordances": ["sedentary", "internal_absorbable", "perishable"],
            "state": {"maturity": 0.9, "toxin": 0.0}
        },
        "agent_002": {  # Alternative physical agent
            "spatial": {"distance": 12.0, "bearing": 90},
            "affordances": ["mobile", "force_exerter", "vocal_emitter", "manipulator"],
            "state": {"exertion_level": 0.1}
        }
    }
}

agent_initial_state = {
    "identity": {
        "id": "agent_001",
        "age_ticks": 120000,
        "narration_intensity": 0.85
    },
    "biological_meters": {
        "energy": 0.85,              # 0.0 (death) to 1.0 (satiated)
        "hydration": 0.90,           # 0.0 (death) to 1.0 (hydrated)
        "tissue_integrity": 1.0,     # 0.0 (death) to 1.0 (undamaged)
        "fatigue": 0.15              # 0.0 (alert) to 1.0 (exhausted)
    },
    "visceral_drives": {
        "nutritional_pull": 0.15,    # Driven inversely by energy
        "hydration_pull": 0.10,      # Driven inversely by hydration
        "rest_pull": 0.15,
        "social_contact_pull": 0.40  # Innate primitive baseline
    },
    "epistemic_ledger": {
        "discovered_entities": [],   # Earned, locked structural relations
        "proto_concepts": {}         # Counter storage for experiential patterns
    }
}

# =====================================================================
# 1B. PRE-CONCEPTUAL BIRTH GRAPH + FIXED REALITY / FLUID UNDERSTANDING
# =====================================================================

agent_birth_graph = {
    "self": {
        "exists": True,
        "can_cease": True,
    },
    "body": {
        "energy": 0.8,
        "hydration": 0.8,
        "temperature": 0.5,
        "pain": 0.0,
        "fatigue": 0.1,
        "injury": 0.0,
        "arousal": 0.0,
        "age": 0.0,
        "location": "spawn",
    },
    "contact_primitives": {
        "outside_exists": True,
        "other_present": False,
        "distance_change": 0.0,
        "state_change_from_outside": False,
    },
    "pattern_primitives": {
        "recurrence_memory": [],
        "help_signal": 0.0,
        "harm_signal": 0.0,
        "persistence_signal": 0.0,
        "novelty_signal": 0.0,
    },
}

BODY_KEYS = ("energy", "hydration", "tissue_integrity", "fatigue")
MAX_RECURRENCE_MEMORY = 256
DISCOVERY_THRESHOLDS = {
    "salience": 0.20,
    "recurrence": 3,
    "consequence": 0.03,
}


def snapshot_body_state(agent):
    meters = agent.get("biological_meters", {})
    return {
        "energy": round(float(meters.get("energy", 0.0)), 3),
        "hydration": round(float(meters.get("hydration", 0.0)), 3),
        "tissue_integrity": round(float(meters.get("tissue_integrity", 0.0)), 3),
        "fatigue": round(float(meters.get("fatigue", 0.0)), 3),
    }


def build_reality_graph(environment):
    """
    Reality is pre-existing and structured: stable topology + fluid state per tick.
    """
    stable_entities = {}
    fluid_entities = {}

    for entity_id, entity_payload in environment.get("entities", {}).items():
        stable_entities[entity_id] = {
            "affordances": list(entity_payload.get("affordances", [])),
        }
        fluid_entities[entity_id] = {
            "spatial": dict(entity_payload.get("spatial", {})),
            "state": dict(entity_payload.get("state", {})),
        }

    return {
        "policy": {
            "fixed_reality_fluid_understanding": True,
            "agent_knows_partial_reality": True,
        },
        "stable_topology": {
            "entities": stable_entities,
            "causal_channels": {
                "internal_absorbable": ["hydration", "energy"],
                "fragmentable": ["energy"],
                "force_exerter": ["tissue_integrity", "fatigue"],
                "perishable": ["energy", "hydration"],
            },
        },
        "fluid_state": {
            "global_states": dict(environment.get("global_states", {})),
            "entities": fluid_entities,
        },
    }


def initialize_agent_graph(agent):
    """
    Agent starts with a tiny grammar for mapping, not a semantic worldview.
    """
    graph = json.loads(json.dumps(agent_birth_graph))
    body_snapshot = snapshot_body_state(agent)

    graph["body"]["energy"] = body_snapshot["energy"]
    graph["body"]["hydration"] = body_snapshot["hydration"]
    graph["body"]["fatigue"] = body_snapshot["fatigue"]
    graph["body"]["injury"] = round(1.0 - body_snapshot["tissue_integrity"], 3)
    graph["body"]["age"] = round(float(agent.get("identity", {}).get("age_ticks", 0)) / 120000.0, 3)

    graph["frontier"] = {
        "nodes": {},
        "merged_nodes": {},
    }
    graph["history"] = {
        "event_counts": {},
        "co_occurrence": {},
        "last_target_distances": {},
        "body_snapshots": [body_snapshot],
    }
    return graph


def _compute_body_consequence(before_state, after_state):
    deltas = {}
    for key in BODY_KEYS:
        deltas[key] = round(float(after_state.get(key, 0.0)) - float(before_state.get(key, 0.0)), 3)

    total_consequence = round(sum(abs(value) for value in deltas.values()), 3)
    helpful_shift = round(
        max(0.0, deltas["energy"])
        + max(0.0, deltas["hydration"])
        + max(0.0, deltas["tissue_integrity"])
        + max(0.0, -deltas["fatigue"]),
        3,
    )
    harmful_shift = round(
        max(0.0, -deltas["energy"])
        + max(0.0, -deltas["hydration"])
        + max(0.0, -deltas["tissue_integrity"])
        + max(0.0, deltas["fatigue"]),
        3,
    )

    if helpful_shift > harmful_shift:
        valence = "help"
    elif harmful_shift > helpful_shift:
        valence = "harm"
    else:
        valence = "neutral"

    return deltas, total_consequence, helpful_shift, harmful_shift, valence


def discovery_update(agent_graph, reality_graph, action_output, body_before, body_after, visceral_drives):
    """
    Dynamic graph rule:
    add/strengthen/weaken/deprecate/merge/split only through recurrence + consequence.
    """
    contact = agent_graph["contact_primitives"]
    pattern = agent_graph["pattern_primitives"]
    history = agent_graph["history"]
    frontier_nodes = agent_graph["frontier"]["nodes"]
    merged_nodes = agent_graph["frontier"]["merged_nodes"]

    entities = reality_graph.get("fluid_state", {}).get("entities", {})
    contact["outside_exists"] = bool(entities)
    contact["other_present"] = any(entity_id.startswith("agent_") for entity_id in entities)

    selected_target = action_output.get("selected_target", "none")
    target = selected_target
    motor = action_output.get("motor_execution", "rest")

    if target == "none" and entities:
        target = min(
            entities,
            key=lambda entity_id: float(entities[entity_id].get("spatial", {}).get("distance", 1e9)),
        )
    contact["inferred_target_from_sensorium"] = selected_target == "none" and target != "none"

    event_key = f"{target}|{motor}"

    if target in entities:
        current_distance = float(entities[target].get("spatial", {}).get("distance", 0.0))
        previous_distance = history["last_target_distances"].get(target, current_distance)
        contact["distance_change"] = round(previous_distance - current_distance, 3)
        history["last_target_distances"][target] = current_distance
        contact["near"] = current_distance <= 1.5
        contact["touching"] = current_distance <= 0.25
    else:
        contact["distance_change"] = 0.0
        contact["near"] = False
        contact["touching"] = False

    deltas, total_consequence, helpful_shift, harmful_shift, valence = _compute_body_consequence(body_before, body_after)
    contact["state_change_from_outside"] = target != "none" and total_consequence > 0.0

    recurrence_memory = pattern["recurrence_memory"]
    recurrence_memory.append({
        "event": event_key,
        "consequence": total_consequence,
        "tick": int(reality_graph.get("fluid_state", {}).get("global_states", {}).get("tick_count", 0)),
    })
    if len(recurrence_memory) > MAX_RECURRENCE_MEMORY:
        recurrence_memory.pop(0)

    event_counts = history["event_counts"]
    event_counts[event_key] = int(event_counts.get(event_key, 0)) + 1

    dominant_drive = "none"
    salience = 0.0
    if visceral_drives:
        dominant_drive = max(visceral_drives, key=visceral_drives.get)
        salience = float(max(visceral_drives.values()))

    co_key = f"{event_key}|{dominant_drive}"
    history["co_occurrence"][co_key] = int(history["co_occurrence"].get(co_key, 0)) + 1

    pattern["help_signal"] = round(min(1.0, max(0.0, pattern["help_signal"] * 0.92 + helpful_shift)), 3)
    pattern["harm_signal"] = round(min(1.0, max(0.0, pattern["harm_signal"] * 0.92 + harmful_shift)), 3)
    pattern["persistence_signal"] = round(
        min(1.0, event_counts[event_key] / float(DISCOVERY_THRESHOLDS["recurrence"] * 2)),
        3,
    )
    pattern["novelty_signal"] = round(max(0.0, 1.0 - min(1.0, (event_counts[event_key] - 1) / 10.0)), 3)

    tick_count = int(reality_graph.get("fluid_state", {}).get("global_states", {}).get("tick_count", 0))
    if (
        target != "none"
        and salience >= DISCOVERY_THRESHOLDS["salience"]
        and event_counts[event_key] >= DISCOVERY_THRESHOLDS["recurrence"]
        and total_consequence >= DISCOVERY_THRESHOLDS["consequence"]
    ):
        node_id = f"pattern::{target}::{motor}"
        if node_id not in frontier_nodes:
            frontier_nodes[node_id] = {
                "target": target,
                "motor": motor,
                "strength": 0.35,
                "status": "active",
                "recurrence": event_counts[event_key],
                "mean_consequence": total_consequence,
                "valence": valence,
                "dominant_drive": dominant_drive,
                "evidence_ticks": [tick_count],
            }
        else:
            node = frontier_nodes[node_id]
            previous_recurrence = max(1, int(node.get("recurrence", 1)))
            previous_mean = float(node.get("mean_consequence", total_consequence))
            node["recurrence"] = event_counts[event_key]
            node["strength"] = round(min(1.0, float(node.get("strength", 0.2)) + 0.15), 3)
            node["mean_consequence"] = round(
                ((previous_mean * previous_recurrence) + total_consequence) / float(previous_recurrence + 1),
                3,
            )
            node["valence"] = valence
            node["dominant_drive"] = dominant_drive
            node["status"] = "active"
            evidence_ticks = node.get("evidence_ticks", [])
            evidence_ticks.append(tick_count)
            node["evidence_ticks"] = evidence_ticks[-8:]

    for node_id, node in frontier_nodes.items():
        if node.get("status") == "deprecated":
            continue
        if target != node.get("target") or motor != node.get("motor"):
            node["strength"] = round(max(0.0, float(node.get("strength", 0.0)) - 0.02), 3)
            if node["strength"] <= 0.08:
                node["status"] = "deprecated"

    active_by_target = {}
    for node_id, node in frontier_nodes.items():
        if node.get("status") != "active":
            continue
        active_by_target.setdefault(node.get("target", "none"), []).append(node_id)

    for entity_id, node_ids in active_by_target.items():
        merge_id = f"merged::{entity_id}"
        if len(node_ids) >= 2:
            valences = sorted({frontier_nodes[node_id].get("valence", "neutral") for node_id in node_ids})
            split_required = "help" in valences and "harm" in valences
            merged_nodes[merge_id] = {
                "target": entity_id,
                "members": sorted(node_ids),
                "status": "split" if split_required else "merged",
                "last_tick": tick_count,
            }
        elif merge_id in merged_nodes:
            merged_nodes[merge_id]["status"] = "deprecated"

    agent_graph["body"]["energy"] = body_after["energy"]
    agent_graph["body"]["hydration"] = body_after["hydration"]
    agent_graph["body"]["fatigue"] = body_after["fatigue"]
    agent_graph["body"]["injury"] = round(1.0 - body_after["tissue_integrity"], 3)
    agent_graph["history"]["body_snapshots"].append(dict(body_after))
    if len(agent_graph["history"]["body_snapshots"]) > 64:
        agent_graph["history"]["body_snapshots"].pop(0)

    return agent_graph


def summarize_discovery_frontier(agent_graph):
    active_nodes = []
    for node_id, node in agent_graph.get("frontier", {}).get("nodes", {}).items():
        if node.get("status") == "active":
            active_nodes.append(node_id)
    active_nodes.sort()
    return active_nodes

# =====================================================================
# 2. RUNTIME INFERENCE & CONTAINMENT ENGINE
# =====================================================================

ALLOWED_TARGETS = {"ent_001", "ent_002", "ent_003", "agent_002", "none"}
ALLOWED_MOTORS = {"approach", "contact", "consume", "flee", "vocalize", "manipulate", "rest"}
ALLOWED_STRUCTURAL_INDEX = {"need", "self", "other", "anomaly", "none"}


def _default_action_output():
    return {
        "selected_target": "none",
        "motor_execution": "rest",
        "exertion_intensity": 0.0,
        "cognitive_register": {
            "focus_shifting": False,
            "proto_concept_crystallizing": False,
            "structural_index": "none",
        },
    }


def _strip_control_chars(text):
    return "".join(ch for ch in text if ch in ("\n", "\r", "\t") or ord(ch) >= 32)


def _extract_balanced_json_objects(text):
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


def _iter_json_candidates(raw_output):
    if not isinstance(raw_output, str):
        return

    base = _strip_control_chars(raw_output.strip().lstrip("\ufeff"))
    if not base:
        return

    seen = set()

    def emit(candidate):
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
        pieces = base.split("```")
        for piece in pieces:
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


def _parse_polluted_json(raw_output):
    for candidate in _iter_json_candidates(raw_output):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            # Secondary parse path for occasional Python-dict style output.
            try:
                parsed_literal = ast.literal_eval(candidate)
                if isinstance(parsed_literal, dict):
                    return parsed_literal
            except (ValueError, SyntaxError):
                continue
    return None


def _normalize_token(value, default="none"):
    if not isinstance(value, str):
        return default
    cleaned = value.strip().strip("\"'.,;:!? ")
    return cleaned.lower() if cleaned else default


def _coerce_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return default


def _coerce_float_01(value, default=0.0):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return round(min(max(number, 0.0), 1.0), 3)


def _sanitize_action_output(parsed_output):
    if not isinstance(parsed_output, dict):
        return _default_action_output()

    selected_target = _normalize_token(parsed_output.get("selected_target"), default="none")
    if selected_target not in ALLOWED_TARGETS:
        selected_target = "none"

    motor_execution = _normalize_token(parsed_output.get("motor_execution"), default="rest")
    if motor_execution not in ALLOWED_MOTORS:
        motor_execution = "rest"

    exertion_intensity = _coerce_float_01(parsed_output.get("exertion_intensity"), default=0.0)

    cognitive_register = parsed_output.get("cognitive_register", {})
    if not isinstance(cognitive_register, dict):
        cognitive_register = {}

    focus_shifting = _coerce_bool(cognitive_register.get("focus_shifting"), default=False)
    proto_concept_crystallizing = _coerce_bool(cognitive_register.get("proto_concept_crystallizing"), default=False)
    structural_index = _normalize_token(cognitive_register.get("structural_index"), default="none")
    if structural_index not in ALLOWED_STRUCTURAL_INDEX:
        structural_index = "none"

    return {
        "selected_target": selected_target,
        "motor_execution": motor_execution,
        "exertion_intensity": exertion_intensity,
        "cognitive_register": {
            "focus_shifting": focus_shifting,
            "proto_concept_crystallizing": proto_concept_crystallizing,
            "structural_index": structural_index,
        },
    }


def run_simulation_step(agent, environment):
    """
    Forces a weak local model to process raw float updates without language leakage.
    """
    # Basal metabolic decay applied automatically before processing
    agent["biological_meters"]["energy"] = max(0.0, agent["biological_meters"]["energy"] - 0.01)
    agent["biological_meters"]["hydration"] = max(0.0, agent["biological_meters"]["hydration"] - 0.015)

    # Recalculate immediate drives based on pure physics
    agent["visceral_drives"]["nutritional_pull"] = round(1.0 - agent["biological_meters"]["energy"], 3)
    agent["visceral_drives"]["hydration_pull"] = round(1.0 - agent["biological_meters"]["hydration"], 3)

    # Strict confinement instruction block
    system_isolation_instruction = """
    You are computing the immediate behavioral output vector for a biological unit.
    The unit has NO linguistic capability, NO historical memory of culture, and NO pre-coded definitions of objects.
    It operates entirely on raw spatial data, structural drives, and visceral feedback loops.

    CRITICAL SAFETY RULES:
    1. Do not use words or concepts outside the provided JSON parameters.
    2. Do not explain your output. Do not include prose narrative or commentary.
    3. If the state indicates stillness, select the structural action 'rest' or 'attend'.
    4. You must output exactly valid JSON matching the exact schema requested. No wrapper text.
    """

    runtime_payload = {
        "agent_internal_meters": agent["biological_meters"],
        "agent_visceral_pulls": agent["visceral_drives"],
        "immediate_sensorium": environment["entities"],
        "global_ambient": environment["global_states"],
    }

    user_query = f"""
    Process this current state payload:
    {json.dumps(runtime_payload)}

    Output precisely this JSON template:
    {{
        "selected_target": "ent_001|ent_002|ent_003|agent_002|none",
        "motor_execution": "approach|contact|consume|flee|vocalize|manipulate|rest",
        "exertion_intensity": 0.0,
        "cognitive_register": {{
            "focus_shifting": true,
            "proto_concept_crystallizing": false,
            "structural_index": "need|self|other|anomaly|none"
        }}
    }}
    """

    try:
        ollama_host = os.getenv("OLLAMA_HOST")
        if ollama_host:
            client = ollama.Client(host=ollama_host, timeout=20.0)
        else:
            client = ollama.Client(timeout=20.0)

        response = client.chat(
            model="mistral",
            messages=[
                {"role": "system", "content": system_isolation_instruction},
                {"role": "user", "content": user_query},
            ],
            options={"temperature": 0.1, "top_p": 0.1},
        )

        raw_output = str(response.get("message", {}).get("content", "")).strip()
        parsed = _parse_polluted_json(raw_output)
        return _sanitize_action_output(parsed)
    except Exception:
        # Fallback to total operational stasis to safeguard the state array
        return _default_action_output()


# =====================================================================
# 3. EPISTEMIC ACCUMULATION LOOP
# =====================================================================

def update_epistemic_ledger(agent, action_output):
    """
    Tracks repeat behavior loops to dynamically unlock foundational concepts.
    """
    target = action_output["selected_target"]
    cognitive = action_output["cognitive_register"]

    if target == "none" or not cognitive["proto_concept_crystallizing"]:
        return agent

    proto_ledger = agent["epistemic_ledger"]["proto_concepts"]
    if target not in proto_ledger:
        proto_ledger[target] = {"interaction_count": 0, "associated_drives": []}

    proto_ledger[target]["interaction_count"] += 1
    dominant_drive = max(agent["visceral_drives"], key=agent["visceral_drives"].get)
    proto_ledger[target]["associated_drives"].append(dominant_drive)

    # Locking validation rule: Concept crystallization threshold achieved via experience
    if proto_ledger[target]["interaction_count"] >= 3:
        discovery_tag = f"RELATIONSHIP_LOCKED_{target}_REDUCES_{dominant_drive.upper()}"
        if discovery_tag not in agent["epistemic_ledger"]["discovered_entities"]:
            agent["epistemic_ledger"]["discovered_entities"].append(discovery_tag)

    return agent


# =====================================================================
# 4. STANDALONE PIPELINE VALIDATION EXECUTION
# =====================================================================

if __name__ == "__main__":
    # Deep copy via JSON to keep module-level schemas immutable across runs.
    agent = json.loads(json.dumps(agent_initial_state))
    world = json.loads(json.dumps(environment_state))
    agent_graph = initialize_agent_graph(agent)
    reality_graph = build_reality_graph(world)

    print("--- STARTING SUBSTRATE STEP VALIDATION ---")

    for tick in range(1, 6):
        world["global_states"]["tick_count"] = tick
        reality_graph = build_reality_graph(world)
        body_before = snapshot_body_state(agent)

        # Inject catastrophic dehydration at tick 3 to test inference reorientation
        if tick == 3:
            agent["biological_meters"]["hydration"] = 0.15
            world["entities"]["ent_002"]["spatial"]["distance"] = 0.0  # Force immediate vector contact
            print(f"\n[Tick {tick}: Critical Emergency Shock Injected -> Hydration Level Drop]")

        decision = run_simulation_step(agent, world)

        # Apply physical environment change based on output vector
        if decision["motor_execution"] == "consume" and decision["selected_target"] == "ent_002":
            agent["biological_meters"]["hydration"] = min(1.0, agent["biological_meters"]["hydration"] + 0.5)

        agent = update_epistemic_ledger(agent, decision)
        body_after = snapshot_body_state(agent)
        agent_graph = discovery_update(
            agent_graph,
            reality_graph,
            decision,
            body_before,
            body_after,
            agent.get("visceral_drives", {}),
        )
        frontier_nodes = summarize_discovery_frontier(agent_graph)

        print(f"Tick {tick} Log Output:")
        print(f"  * Action Executed:  {decision['motor_execution']} -> {decision['selected_target']}")
        print(f"  * Internal Target Focus: {decision['cognitive_register']['structural_index']}")
        print(f"  * Core Hydration Float:  {round(agent['biological_meters']['hydration'], 3)}")
        print(f"  * Dynamic Locked Primitives: {agent['epistemic_ledger']['discovered_entities']}\n")
        print(f"  * Discovery Frontier Nodes: {frontier_nodes}\n")
        time.sleep(1)

    print("--- SUBSTRATE RUN COMPLETED ---")
