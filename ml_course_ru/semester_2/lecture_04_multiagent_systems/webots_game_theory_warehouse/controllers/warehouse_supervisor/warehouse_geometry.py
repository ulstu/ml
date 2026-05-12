"""Shared geometric constants for the large warehouse scene.

The original classroom scene was intentionally compact.  The current version
uses the same topology but scales horizontal coordinates by 3.4, so the floor
area grows by 11.56x while robot and cargo sizes remain readable in Webots.
"""

from __future__ import annotations

from typing import Tuple

from game_models import Point


# Horizontal scene scale: 3.4 * 3.4 = 11.56 times larger floor area.
SCENE_SCALE = 3.4


def scaled_point(point: Point, scale: float = SCENE_SCALE) -> Point:
    # Scale only the horizontal Webots ENU coordinates; Z is handled by the world.
    return (round(point[0] * scale, 3), round(point[1] * scale, 3))


def scaled_deadline(deadline: float, scale: float = SCENE_SCALE) -> float:
    # Robot speed is unchanged, so deadlines grow with horizontal travel distance.
    return round(deadline * scale, 1)


def make_agent_start_positions(count: int = 4, scale: float = SCENE_SCALE) -> Tuple[Point, ...]:
    # Base coordinates match the compact scene and are scaled for the large world.
    base_positions: Tuple[Point, ...] = (
        (-6.32, -0.88),
        (-6.32, 0.88),
        (-4.08, -1.52),
        (-4.08, 1.52),
        (-6.32, -2.64),
        (-6.32, 2.64),
        (0.00, -5.15),
        (0.00, 5.15),
        (-2.48, -4.72),
        (-2.48, 4.72),
        (-0.80, -5.12),
        (-0.80, 5.12),
    )
    # The experiment may request more starts than Webots currently visualizes.
    return tuple(scaled_point(point, scale) for point in base_positions[:count])
