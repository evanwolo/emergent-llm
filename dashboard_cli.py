from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, Optional

from agent_types import AgentGraph, IdentityGraph
from common import Vec2
from enums import EntityClass, Sex
from environment_types import EnvironmentEntity
from sim_engine import run_simulation_tick
from simulation_types import EventRecord, World

try:
    from rich import box
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def _build_demo_world() -> World:
    world = World()
    world.environment_graph = {
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

    world.agent_graphs = {"agt_001": alpha, "agt_002": beta}
    return world


def _apply_demo_shock(world: World) -> None:
    if "agt_001" not in world.agent_graphs or "water_point" not in world.environment_graph:
        return

    world.agent_graphs["agt_001"].body_graph.states.hydration_level = 0.15
    world.agent_graphs["agt_001"].body_graph.states.thirst_level = 0.85
    world.environment_graph["water_point"].location = Vec2(0.0, 0.1)


def _latest_tick_events(world: World, tick: int) -> Dict[str, EventRecord]:
    per_agent: Dict[str, EventRecord] = {}

    for event in reversed(world.event_log):
        if event.tick < tick:
            break
        if event.tick != tick or event.event_type != "tick_pipeline":
            continue
        if not event.participants:
            continue

        agent_id = str(event.participants[0])
        if agent_id not in per_agent:
            per_agent[agent_id] = event

    return per_agent


def _latest_interpreter_summary(world: World, tick: int) -> str:
    for event in reversed(world.event_log):
        if event.tick < tick:
            break
        if event.tick == tick and event.event_type == "interpreter_summary":
            payload = event.payload if isinstance(event.payload, dict) else {}
            summary = str(payload.get("summary", "")).strip()
            if summary:
                return summary
    return ""


def _short(text: str, limit: int = 240) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3] + "..."


def _runtime_mode_label() -> str:
    force_heuristic = os.getenv("SIM_FORCE_HEURISTIC", "0") == "1"
    interpreter = os.getenv("SIM_ENABLE_INTERPRETER", "0") == "1"
    mode = "HEURISTIC" if force_heuristic else "LLM"
    interpreter_mode = "INTERPRETER ON" if interpreter else "INTERPRETER OFF"
    return f"{mode} | {interpreter_mode}"


def _build_rich_dashboard(world: World, started_at: float) -> Panel:
    ws = world.world_state
    elapsed = max(0.0001, time.perf_counter() - started_at)
    ticks_per_second = ws.tick / elapsed if ws.tick > 0 else 0.0

    header = Table.grid(expand=True)
    header.add_column()
    header.add_column(justify="right")
    header.add_row(
        (
            f"Tick {ws.tick} | {ws.time_of_day.value}/{ws.season.value} | "
            f"weather={ws.weather.value} | temp={ws.temperature:.3f} | "
            f"resource_pressure={ws.resource_pressure:.3f}"
        ),
        f"elapsed={elapsed:.1f}s | tps={ticks_per_second:.2f} | {_runtime_mode_label()}",
    )

    events = _latest_tick_events(world, ws.tick)

    agent_table = Table(box=box.SIMPLE_HEAVY, expand=True)
    agent_table.add_column("Agent", no_wrap=True)
    agent_table.add_column("Alive", justify="center")
    agent_table.add_column("Pos", no_wrap=True)
    agent_table.add_column("E")
    agent_table.add_column("H")
    agent_table.add_column("F")
    agent_table.add_column("P")
    agent_table.add_column("Attention")
    agent_table.add_column("Action")
    agent_table.add_column("Target")
    agent_table.add_column("Outcome")
    agent_table.add_column("Locked")

    for agent_id in sorted(world.agent_graphs.keys()):
        agent = world.agent_graphs[agent_id]
        states = agent.body_graph.states
        loc = agent.identity_graph.body_location
        action = agent.action_graph.current_physical_action

        event = events.get(agent_id)
        outcome_label = "-"
        if event is not None and isinstance(event.payload, dict):
            phase_outcome = event.payload.get("phase_4_outcome", {})
            if isinstance(phase_outcome, dict):
                success = bool(phase_outcome.get("success", False))
                valence = str(phase_outcome.get("valence", "neutral"))
                outcome_label = f"{'ok' if success else 'fail'}:{valence}"

        agent_table.add_row(
            agent_id,
            "Y" if agent.identity_graph.alive else "N",
            f"({loc.x:.2f},{loc.y:.2f})",
            f"{states.energy_level:.3f}",
            f"{states.hydration_level:.3f}",
            f"{states.fatigue_level:.3f}",
            f"{states.pain_level:.3f}",
            str(agent.experience_graph.primary_attention or "none"),
            action.action_type.value,
            str(action.target_id or "none"),
            outcome_label,
            str(len(agent.discovery_graph.stabilized_relations)),
        )

    summary = _latest_interpreter_summary(world, ws.tick)
    footer_text = _short(summary, 380) if summary else "No interpreter summary on this tick."
    footer = Panel(footer_text, title="Interpreter", border_style="magenta")

    return Panel(Group(header, agent_table, footer), title="Emergent Simulation Dashboard", border_style="cyan")


def _build_plain_dashboard(world: World, started_at: float) -> str:
    ws = world.world_state
    elapsed = max(0.0001, time.perf_counter() - started_at)
    ticks_per_second = ws.tick / elapsed if ws.tick > 0 else 0.0
    events = _latest_tick_events(world, ws.tick)

    lines = [
        "=== Emergent Simulation Dashboard ===",
        (
            f"Tick {ws.tick} | {ws.time_of_day.value}/{ws.season.value} | weather={ws.weather.value} | "
            f"temp={ws.temperature:.3f} | resource_pressure={ws.resource_pressure:.3f}"
        ),
        f"elapsed={elapsed:.1f}s | tps={ticks_per_second:.2f} | {_runtime_mode_label()}",
        "",
        "Agent      Alive   Pos             E      H      F      P      Attention      Action    Target      Outcome    Locked",
        "-" * 112,
    ]

    for agent_id in sorted(world.agent_graphs.keys()):
        agent = world.agent_graphs[agent_id]
        states = agent.body_graph.states
        loc = agent.identity_graph.body_location
        action = agent.action_graph.current_physical_action

        event = events.get(agent_id)
        outcome_label = "-"
        if event is not None and isinstance(event.payload, dict):
            phase_outcome = event.payload.get("phase_4_outcome", {})
            if isinstance(phase_outcome, dict):
                success = bool(phase_outcome.get("success", False))
                valence = str(phase_outcome.get("valence", "neutral"))
                outcome_label = f"{'ok' if success else 'fail'}:{valence}"

        lines.append(
            f"{agent_id:<10} {'Y' if agent.identity_graph.alive else 'N':<6} "
            f"({loc.x:>5.2f},{loc.y:>5.2f}) "
            f"{states.energy_level:>6.3f} {states.hydration_level:>6.3f} {states.fatigue_level:>6.3f} {states.pain_level:>6.3f} "
            f"{str(agent.experience_graph.primary_attention or 'none'):<14.14} "
            f"{action.action_type.value:<8.8} {str(action.target_id or 'none'):<10.10} {outcome_label:<10.10} "
            f"{len(agent.discovery_graph.stabilized_relations)}"
        )

    summary = _latest_interpreter_summary(world, ws.tick)
    lines.append("")
    lines.append("Interpreter: " + (_short(summary, 300) if summary else "No interpreter summary on this tick."))
    lines.append("Ctrl+C to stop.")
    return "\n".join(lines)


def _configure_runtime_from_args(args: argparse.Namespace) -> None:
    if args.mode == "heuristic":
        os.environ["SIM_FORCE_HEURISTIC"] = "1"
    elif args.mode == "llm":
        os.environ["SIM_FORCE_HEURISTIC"] = "0"

    if args.interpreter is True:
        os.environ["SIM_ENABLE_INTERPRETER"] = "1"
    elif args.interpreter is False:
        os.environ["SIM_ENABLE_INTERPRETER"] = "0"


def _run_dashboard(args: argparse.Namespace) -> int:
    _configure_runtime_from_args(args)
    world = _build_demo_world()
    started_at = time.perf_counter()

    if not args.no_live and not RICH_AVAILABLE:
        print("rich is not installed; falling back to plain dashboard view.")
        print("Install rich for best live UI: py -3 -m pip install rich")

    try:
        if not args.no_live and RICH_AVAILABLE:
            console = Console()
            refresh = max(2.0, float(args.refresh_per_second))
            with Live(_build_rich_dashboard(world, started_at), console=console, refresh_per_second=refresh) as live:
                while args.ticks <= 0 or world.world_state.tick < args.ticks:
                    next_tick = world.world_state.tick + 1
                    if args.shock_tick > 0 and next_tick == args.shock_tick:
                        _apply_demo_shock(world)

                    run_simulation_tick(world)
                    live.update(_build_rich_dashboard(world, started_at), refresh=True)

                    if args.interval > 0:
                        time.sleep(args.interval)
        else:
            while args.ticks <= 0 or world.world_state.tick < args.ticks:
                next_tick = world.world_state.tick + 1
                if args.shock_tick > 0 and next_tick == args.shock_tick:
                    _apply_demo_shock(world)

                run_simulation_tick(world)
                os.system("cls" if os.name == "nt" else "clear")
                print(_build_plain_dashboard(world, started_at))

                if args.interval > 0:
                    time.sleep(args.interval)
    except KeyboardInterrupt:
        pass

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Live CLI dashboard for emergent simulation ticks.")
    parser.add_argument(
        "--ticks",
        type=int,
        default=0,
        help="Number of ticks to run. Use 0 to run continuously until Ctrl+C.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.2,
        help="Sleep interval in seconds between ticks.",
    )
    parser.add_argument(
        "--refresh-per-second",
        type=float,
        default=8.0,
        help="Dashboard refresh rate when live mode is enabled.",
    )
    parser.add_argument(
        "--shock-tick",
        type=int,
        default=2,
        help="Tick number to inject dehydration demo shock. Set 0 to disable.",
    )
    parser.add_argument(
        "--mode",
        choices=["heuristic", "llm"],
        default=None,
        help="Override inference mode. If omitted, existing environment variables are respected.",
    )

    interpreter_group = parser.add_mutually_exclusive_group()
    interpreter_group.add_argument(
        "--interpreter",
        action="store_true",
        help="Enable interpreter summaries for dashboard ticks.",
    )
    interpreter_group.add_argument(
        "--no-interpreter",
        dest="interpreter",
        action="store_false",
        help="Disable interpreter summaries for dashboard ticks.",
    )
    parser.set_defaults(interpreter=None)

    parser.add_argument(
        "--no-live",
        action="store_true",
        help="Disable rich live mode and use plain redraw output.",
    )

    args = parser.parse_args()
    return _run_dashboard(args)


if __name__ == "__main__":
    raise SystemExit(main())