"""Model-layer forward clearance; native movement permission is checked separately.

Every assertion requires fresh reviewed legal/perception evidence. Green, a lead
moving, a map speed, or a turn indicator alone cannot populate this conjunction.
stop_requirement_completed means a required stop is proved complete OR a trusted
rule explicitly establishes that no stop is required. This cannot authorize CAN.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ForwardMotionClearance:
    timestamp_s: float
    road_id: str
    movement_id: str
    qualified: bool
    reference_ids: tuple[str, ...]
    clear_path: bool | None
    right_of_way: bool | None
    crosswalk_clear: bool | None
    exit_space_clear: bool | None
    stop_requirement_completed: bool | None

    def __post_init__(self) -> None:
        if (
            type(self.timestamp_s) not in (int, float)
            or not math.isfinite(self.timestamp_s)
            or not 0 <= self.timestamp_s <= 1e12
            or type(self.qualified) is not bool
        ):
            raise ValueError("Explicit finite clearance time and qualification required")
        if not isinstance(self.reference_ids, tuple) or not 1 <= len(self.reference_ids) <= 32:
            raise ValueError("Bounded actual clearance references required")
        for value in (self.road_id, self.movement_id, *self.reference_ids):
            if (
                not isinstance(value, str)
                or not value.strip()
                or len(value) > 128
                or not value.isascii()
                or not value.isprintable()
            ):
                raise ValueError("Explicit bounded clearance identity required")
        for name in ("clear_path", "right_of_way", "crosswalk_clear", "exit_space_clear", "stop_requirement_completed"):
            if getattr(self, name) is not None and type(getattr(self, name)) is not bool:
                raise ValueError("Clearance facts must be explicit boolean or unknown")

    def eligible(self, now_s: float, road_id: str, movement_id: str, max_age_s: float = 0.05) -> bool:
        return (
            type(now_s) in (int, float)
            and math.isfinite(now_s)
            and type(max_age_s) in (int, float)
            and math.isfinite(max_age_s)
            and 0 < max_age_s <= 0.05
            and self.qualified
            and self.road_id == road_id
            and self.movement_id == movement_id
            and self.timestamp_s <= now_s <= self.timestamp_s + max_age_s
            and all(
                value is True
                for value in (
                    self.clear_path,
                    self.right_of_way,
                    self.crosswalk_clear,
                    self.exit_space_clear,
                    self.stop_requirement_completed,
                )
            )
        )
