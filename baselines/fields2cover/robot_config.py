#!/usr/bin/env python3

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "robots.json"


@dataclass(frozen=True)
class RobotConfig:
    name: str
    description: str
    body_width_m: float
    coverage_width_m: float
    safety_margin_m: float
    min_turning_radius_m: float
    max_curvature_rate_per_m2: float
    cruise_speed_mps: float
    turn_speed_mps: float

    @property
    def max_curvature_per_m(self) -> float:
        return 1.0 / self.min_turning_radius_m

    @property
    def obstacle_clearance_m(self) -> float:
        """Centerline clearance from field boundaries and obstacles."""
        return self.body_width_m / 2.0 + self.safety_margin_m

    def validate(self) -> None:
        positive = {
            "body_width_m": self.body_width_m,
            "coverage_width_m": self.coverage_width_m,
            "safety_margin_m": self.safety_margin_m,
            "min_turning_radius_m": self.min_turning_radius_m,
            "max_curvature_rate_per_m2": self.max_curvature_rate_per_m2,
            "cruise_speed_mps": self.cruise_speed_mps,
            "turn_speed_mps": self.turn_speed_mps,
        }
        for field, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be a finite positive number")

    def create_fields2cover_robot(self):
        import fields2cover as f2c

        robot = f2c.Robot(
            self.body_width_m,
            self.coverage_width_m,
            self.max_curvature_per_m,
            self.max_curvature_rate_per_m2,
        )
        robot.setName(self.name)
        robot.setCruiseVel(self.cruise_speed_mps)
        robot.setTurnVel(self.turn_speed_mps)
        return robot


def load_robot_config(
    profile: str, config_path: Path = DEFAULT_CONFIG_PATH
) -> RobotConfig:
    with config_path.open(encoding="utf-8") as config_file:
        document = json.load(config_file)

    if document.get("schema_version") != 1:
        raise ValueError("Unsupported robot configuration schema_version")

    try:
        values = document["robots"][profile]
    except KeyError as error:
        available = ", ".join(sorted(document.get("robots", {})))
        raise KeyError(
            f"Unknown robot profile '{profile}'. Available profiles: {available}"
        ) from error

    config = RobotConfig(name=profile, **values)
    config.validate()
    return config


def main() -> None:
    profile = sys.argv[1] if len(sys.argv) > 1 else "robot_baseline"
    config = load_robot_config(profile)
    print(json.dumps({
        **config.__dict__,
        "max_curvature_per_m": config.max_curvature_per_m,
    }, indent=2))


if __name__ == "__main__":
    main()
