from __future__ import annotations

from enum import Enum


class AgeStage(str, Enum):
    INFANT = "infant"
    CHILD = "child"
    ADOLESCENT = "adolescent"
    ADULT = "adult"
    ELDER = "elder"
    DEAD = "dead"


class Sex(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class TimeOfDay(str, Enum):
    DAWN = "dawn"
    DAY = "day"
    DUSK = "dusk"
    NIGHT = "night"


class Season(str, Enum):
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"


class Weather(str, Enum):
    CLEAR = "clear"
    RAIN = "rain"
    STORM = "storm"
    SNOW = "snow"


class EntityClass(str, Enum):
    OBJECT = "object"
    ORGANISM = "organism"
    PLACE = "place"
    PROCESS = "process"
    AGENT_BODY = "agent_body"


class ProcessingMode(str, Enum):
    NONVERBAL = "nonverbal"
    FRAGMENTARY = "fragmentary"
    VERBAL = "verbal"


class BeliefExplicitness(str, Enum):
    EMBODIED = "embodied"
    MIXED = "mixed"
    EXPLICIT = "explicit"


class RevisionMode(str, Enum):
    EMBODIED = "embodied"
    MIXED = "mixed"
    EXPLICIT = "explicit"


class DiscoveryStatus(str, Enum):
    PROVISIONAL = "provisional"
    STABILIZING = "stabilizing"
    LOCKED = "locked"


class BeliefStatus(str, Enum):
    PROTO = "proto"
    STABLE = "stable"
    LOCKED = "locked"


class TriggerType(str, Enum):
    CONTRADICTION = "contradiction"
    FAILURE = "failure"
    TRAUMA = "trauma"
    SOCIAL_CHALLENGE = "social_challenge"
    BEAUTY = "beauty"
    ANOMALY = "anomaly"


class RevisionOutcome(str, Enum):
    NONE = "none"
    WEAKENED = "weakened"
    REVISED = "revised"
    REPLACED = "replaced"
    LOCKED_HARDER = "locked_harder"


class PhysicalActionType(str, Enum):
    NONE = "none"
    MOVE = "move"
    EAT = "eat"
    DRINK = "drink"
    SLEEP = "sleep"
    FIGHT = "fight"
    FLEE = "flee"
    REPRODUCE = "reproduce"
    VOCALIZE = "vocalize"
    MANIPULATE = "manipulate"
    BUILD = "build"
    REST = "rest"


class DirectionType(str, Enum):
    TOWARD = "toward"
    AWAY = "away"
    NONE = "none"


class ConceptTarget(str, Enum):
    SELF = "self"
    OTHER = "other"
    OBJECT = "object"
    THREAT = "threat"
    NEED = "need"
    PATTERN = "pattern"
    PLACE = "place"
    NONE = "none"


class BeliefDomain(str, Enum):
    SELF = "self"
    BODY = "body"
    OTHER = "other"
    WORLD = "world"
    TIME = "time"
    VALUE = "value"
    TRANSCENDENT = "transcendent"


class BeliefSource(str, Enum):
    EXPERIENCE = "experience"
    IMITATION = "imitation"
    AUTHORITY = "authority"
    INFERENCE = "inference"
    INSTALLATION = "installation"
