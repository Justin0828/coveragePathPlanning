from __future__ import annotations
from dataclasses import dataclass
import math
from PIL import Image
import numpy as np

@dataclass
class Rect: # 世界坐标系
    x1: float # 左下角x
    y1: float # 左下角y
    x2: float # 右上角x
    y2: float # 右上角y

def coordinateWorldToMap(
    point: tuple[float, float],
    pixel_size: float,
) -> tuple[int, int]:
    """
    将世界坐标系中的任意一个点映射到地图坐标系中的一个点
    两个坐标系都是左下角为原点
    使用参数pixel_size进行转换
    """
    x, y = point
    x_map = int(x / pixel_size)
    y_map = int(y / pixel_size)
    return x_map, y_map

def coordinateMapToWorld(
    point: tuple[int, int],
    pixel_size: float,
) -> tuple[float, float]:
    """
    将地图坐标系中的任意一个点映射到世界坐标系中的一个点
    两个坐标系都是左下角为原点
    使用参数pixel_size进行转换
    """
    x, y = point
    x_world = x * pixel_size
    y_world = y * pixel_size
    return x_world, y_world

def computeHeading(
    node_a: tuple[float, float],
    node_b: tuple[float, float],
) -> float:
    """
    给定两个点，计算从node_a到node_b的航向角，范围在[0, 360)
    """
    dx = node_b[0] - node_a[0]
    dy = node_b[1] - node_a[1]
    return math.degrees(math.atan2(dy, dx)) % 360

@dataclass
class Robot: # 世界坐标系
    # 静态信息
    disc_radius: float = 0.25 # 圆盘半径
    arm_length: float = 1.2 # 机械臂长度
    car_width: float = 0.8 # 车宽
    car_half_length: float = 1.0 # 车长的一半
    pivot_to_car_center: float = 0.3 # 摆臂旋转中心到车中心的距离

    speed_limit: float = 0.0 # 线速度，一秒走几米；
    angular_velocity_limit: float = 0.0 # 角速度,代表1秒转几度
    arm_angle_limit: tuple[float, float] = (-90, 90) # 摆臂角度限制
    arm_angular_velocity_limit: float = 0.0 # 摆臂角速度限制
    work_speed_policy: str = "coverage_safe" # coverage_safe 或 commanded

    # 动态信息
    heading: float = 0.0 # 航向角
    arm_angle: float = 0.0 # 摆臂角度，角度制，正向为逆时针
    arm_angular_velocity: float = 0.0 # 摆臂角速度，角度制，正向为逆时针
    car_speed: float = 0.0 # 车速度，>= 0，用 heading 确定方向
    car_angular_velocity: float = 0.0 # 车角速度，角度制，正向为逆时针
    car_position: tuple[float, float] = (0.0, 0.0) # 车中心点位置


@dataclass(frozen=True)
class MotionSegment:
    """A continuous robot motion used by sampling-independent coverage metrics."""

    phase: str
    coverage_active: bool
    duration: float
    car_position_start: tuple[float, float]
    car_position_end: tuple[float, float]
    heading_start: float
    heading_end: float
    arm_angle_start: float
    arm_angle_end: float
    arm_motion: str = "linear"
    arm_angular_velocity: float = 0.0
    arm_angle_lower: float = 0.0
    arm_angle_upper: float = 0.0


def _recordMotionSegment(
    motion_segments: list[MotionSegment] | None,
    *,
    phase: str,
    coverage_active: bool,
    duration: float,
    car_position_start: tuple[float, float],
    car_position_end: tuple[float, float],
    heading_start: float,
    heading_end: float,
    arm_angle_start: float,
    arm_angle_end: float,
    arm_motion: str = "linear",
    arm_angular_velocity: float = 0.0,
    arm_angle_lower: float = 0.0,
    arm_angle_upper: float = 0.0,
) -> None:
    if motion_segments is None or duration <= 0:
        return
    motion_segments.append(MotionSegment(
        phase=phase,
        coverage_active=coverage_active,
        duration=duration,
        car_position_start=car_position_start,
        car_position_end=car_position_end,
        heading_start=heading_start,
        heading_end=heading_end,
        arm_angle_start=arm_angle_start,
        arm_angle_end=arm_angle_end,
        arm_motion=arm_motion,
        arm_angular_velocity=arm_angular_velocity,
        arm_angle_lower=arm_angle_lower,
        arm_angle_upper=arm_angle_upper,
    ))


def _oscillatingArmAngle(
    elapsed: float,
    start_angle: float,
    signed_angular_velocity: float,
    lower_angle: float,
    upper_angle: float,
) -> tuple[float, float]:
    """Return reflected triangular-wave angle and instantaneous velocity."""
    span = upper_angle - lower_angle
    if span <= 1e-12 or abs(signed_angular_velocity) <= 1e-12:
        return start_angle, 0.0
    position = start_angle - lower_angle
    unfolded = position + signed_angular_velocity * elapsed
    period = 2.0 * span
    phase = unfolded % period
    if phase <= span:
        angle = lower_angle + phase
        velocity = abs(signed_angular_velocity)
    else:
        angle = upper_angle - (phase - span)
        velocity = -abs(signed_angular_velocity)
    if signed_angular_velocity < 0:
        # Reversing the unfolded direction reverses the triangular-wave slope.
        velocity *= -1.0
    return angle, velocity

def findNearestEdgePoint(
    rect: Rect,
    inner_point: tuple[float, float],
) -> tuple[float, float]:
    """给定矩形和一个点，返回矩形边界上离该点最近的点"""
    x, y = inner_point
    a = x - rect.x1  # 到左边的距离
    b = rect.y2 - y  # 到上边的距离
    c = rect.x2 - x  # 到右边的距离
    d = y - rect.y1  # 到下边的距离
    if a <= b and a <= c and a <= d: # 到左边的距离最近
        return (rect.x1, y)
    if b < a and b <= c and b <= d: # 到上边的距离最近
        return (x, rect.y2)
    if c < a and c < b and c <= d: # 到右边的距离最近
        return (rect.x2, y)
    if d < a and d < b and d < c: # 到下边的距离最近
        return (x, rect.y1)
    raise ValueError(f"@findNearestEdgePoint: inner_point {inner_point} 无法确定最近的矩形边界点")

def findRobotPoseByInPoint(
    rect: Rect,
    in_point: tuple[float, float], # 世界坐标系，start_point被findNearestEdgePoint作用后得到in_point,在矩形边界上，经过转换后得到start_car_center
    robot: Robot,
) -> Robot:
    """
    给定矩形和世界坐标系中的出点，确定Robot位姿信息
    重写动态信息：航向角，摆臂角度，车速度，车中心点位置
    """
    x, y = in_point
    robot.arm_angle = 0 # 停止摆臂
    robot.car_speed = 0.0 # 停止移动
    robot.car_angular_velocity = 0.0 # 停止旋转
    car_disc_distance = robot.pivot_to_car_center + robot.arm_length
    #终点在矩形边上，分4中情况，左，上，右，下 =====
    # 左
    if x == rect.x1:
        robot.heading = 180 # 车身为头，圆盘为尾；车身在左，圆盘在右
        robot.car_position = (x - car_disc_distance, y)
        return robot
    # 上
    if y == rect.y2:
        robot.heading = 90 # 车身在上，圆盘在下
        robot.car_position = (x, y + car_disc_distance)
        return robot
    # 右
    if x == rect.x2:
        robot.heading = 0 # 车身在右，圆盘在左
        robot.car_position = (x + car_disc_distance, y)
        return robot
    # 下
    if y == rect.y1:
        robot.heading = 270 # 车身在下，圆盘在上
        robot.car_position = (x, y - car_disc_distance)
        return robot
    raise ValueError(f"@findRobotPoseByInPoint: in_point {in_point} 无法确定Robot位姿信息")

def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))

def _symmetricArmLimit(robot: Robot) -> float:
    lower, upper = robot.arm_angle_limit
    if lower > upper:
        raise ValueError("arm_angle_limit 下限不能大于上限。")
    if lower > 0 or upper < 0:
        raise ValueError("arm_angle_limit 必须包含 0 度以支持回收姿态。")
    return min(abs(lower), abs(upper), 90.0)

def _discCenterForPose(
    car_position: tuple[float, float],
    heading: float,
    arm_angle: float,
    robot: Robot,
) -> tuple[float, float]:
    heading_rad = math.radians(heading)
    pivot = (
        car_position[0] - robot.pivot_to_car_center * math.cos(heading_rad),
        car_position[1] - robot.pivot_to_car_center * math.sin(heading_rad),
    )
    arm_heading = math.radians(heading + 180.0 + arm_angle)
    return (
        pivot[0] + robot.arm_length * math.cos(arm_heading),
        pivot[1] + robot.arm_length * math.sin(arm_heading),
    )

def _carPositionForDiscCenter(
    disc_center: tuple[float, float],
    heading: float,
    arm_angle: float,
    robot: Robot,
) -> tuple[float, float]:
    heading_rad = math.radians(heading)
    arm_heading = math.radians(heading + 180.0 + arm_angle)
    pivot = (
        disc_center[0] - robot.arm_length * math.cos(arm_heading),
        disc_center[1] - robot.arm_length * math.sin(arm_heading),
    )
    return (
        pivot[0] + robot.pivot_to_car_center * math.cos(heading_rad),
        pivot[1] + robot.pivot_to_car_center * math.sin(heading_rad),
    )

def _nearestCornerStartPose(
    rect: Rect,
    in_point: tuple[float, float],
    robot: Robot,
) -> tuple[tuple[float, float], int, float, float]:
    """Return a start pose whose disc center is on a nearby rectangle corner."""
    max_swing = _symmetricArmLimit(robot)
    x, y = in_point
    a = x - rect.x1
    b = rect.y2 - y
    c = rect.x2 - x
    d = y - rect.y1
    if a <= c and b <= d:
        corner = (rect.x1, rect.y2)
        start_corner = 0
        heading = 0.0
        arm_angle = -max_swing
    elif a <= c and d < b:
        corner = (rect.x1, rect.y1)
        start_corner = 1
        heading = 0.0
        arm_angle = max_swing
    elif c < a and b <= d:
        corner = (rect.x2, rect.y2)
        start_corner = 2
        heading = 180.0
        arm_angle = max_swing
    else:
        corner = (rect.x2, rect.y1)
        start_corner = 3
        heading = 180.0
        arm_angle = -max_swing
    return _carPositionForDiscCenter(corner, heading, arm_angle, robot), start_corner, heading, arm_angle

def _effectiveStripCount(cross_width: float, robot: Robot) -> int:
    if cross_width <= 0:
        return 1
    if robot.disc_radius <= 0 or robot.arm_length <= 0:
        raise ValueError("disc_radius 和 arm_length 必须大于 0。")
    max_strip_width = 2.0 * (
        robot.arm_length * math.sin(math.radians(_symmetricArmLimit(robot)))
        + robot.disc_radius
    )
    return max(1, math.ceil(cross_width / max_strip_width))

def _stripSwingAngle(strip_width: float, robot: Robot) -> float:
    """Largest symmetric swing that can cover this strip with the disc radius."""
    if strip_width <= 2.0 * robot.disc_radius:
        return 0.0
    ratio = (strip_width / 2.0 - robot.disc_radius) / robot.arm_length
    ratio = _clamp(ratio, 0.0, 1.0)
    return min(math.degrees(math.asin(ratio)), _symmetricArmLimit(robot))

def _armAngleForHorizontalOffset(heading: float, offset_y: float, robot: Robot) -> float:
    """Solve disc_y - pivot_y = offset_y for horizontal headings."""
    ratio = _clamp(offset_y / robot.arm_length, -1.0, 1.0)
    raw = math.degrees(math.asin(ratio))
    normalized = _normalizeHeadingToCardinal(heading)
    if normalized == 0:
        return -raw
    if normalized == 180:
        return raw
    raise ValueError("当前可变摆臂覆盖仅支持水平主航向。")

def _horizontalSweepSegments(
    rect: Rect,
    robot: Robot,
    start_corner: int,
) -> list[tuple[tuple[float, float], tuple[float, float], float, float, float]]:
    """Build horizontal passes with arm motion during each pass."""
    width = rect.x2 - rect.x1
    height = rect.y2 - rect.y1
    if width <= 0 or height <= 0:
        raise ValueError("矩形尺寸必须为正。")
    if height < 2.0 * robot.disc_radius:
        raise ValueError(
            "矩形短边小于圆盘直径，无法按当前可变摆臂模型保证覆盖。"
        )

    strip_count = _effectiveStripCount(height, robot)
    strip_width = height / strip_count
    top_to_bottom = start_corner in (0, 2)
    left_to_right = start_corner in (0, 1)
    segments = []

    usable_left = rect.x1 + min(robot.disc_radius, width / 2.0)
    usable_right = rect.x2 - min(robot.disc_radius, width / 2.0)

    for idx in range(strip_count):
        if top_to_bottom:
            high = rect.y2 - idx * strip_width
            low = high - strip_width
            start_offset = max(0.0, strip_width / 2.0 - robot.disc_radius)
            end_offset = -start_offset
        else:
            low = rect.y1 + idx * strip_width
            high = low + strip_width
            start_offset = -max(0.0, strip_width / 2.0 - robot.disc_radius)
            end_offset = -start_offset

        pivot_y = (low + high) / 2.0
        swing_angle = _stripSwingAngle(strip_width, robot)
        if swing_angle <= 1e-9:
            start_offset = 0.0
            end_offset = 0.0

        heading = 0.0 if left_to_right else 180.0
        start_angle = _armAngleForHorizontalOffset(heading, start_offset, robot)
        end_angle = _armAngleForHorizontalOffset(heading, end_offset, robot)

        start_disc_x = usable_left if left_to_right else usable_right
        end_disc_x = usable_right if left_to_right else usable_left
        start_car = _carPositionForDiscCenter((start_disc_x, pivot_y + start_offset), heading, start_angle, robot)
        end_car = _carPositionForDiscCenter((end_disc_x, pivot_y + end_offset), heading, end_angle, robot)
        segments.append((start_car, end_car, heading, start_angle, end_angle))
        left_to_right = not left_to_right

    return segments

def coverageStrips(
    rect: Rect,
    in_point: tuple[float, float],
    robot: Robot,
) -> list[Rect]:
    """Return continuous horizontal work strips used for coverage metrics."""
    _, start_corner, _, _ = _nearestCornerStartPose(rect, in_point, robot)
    height = rect.y2 - rect.y1
    if height <= 0 or rect.x2 <= rect.x1:
        raise ValueError("矩形尺寸必须为正。")
    if height < 2.0 * robot.disc_radius:
        raise ValueError(
            "矩形短边小于圆盘直径，无法按当前可变摆臂模型保证覆盖。"
        )

    strip_count = _effectiveStripCount(height, robot)
    strip_width = height / strip_count
    top_to_bottom = start_corner in (0, 2)
    strips = []
    for idx in range(strip_count):
        if top_to_bottom:
            high = rect.y2 - idx * strip_width
            low = high - strip_width
        else:
            low = rect.y1 + idx * strip_width
            high = low + strip_width
        strips.append(Rect(rect.x1, low, rect.x2, high))
    return strips

# ==================== 通用 Pose 生成器 ====================
def _generateCarTurnPoses(
    robot: Robot,
    car_position: tuple[float, float],
    current_heading: float,
    target_heading: float,
    arm_angle: float,
    angular_velocity_limit: float,
    sample_rate: float,
    motion_segments: list[MotionSegment] | None = None,
    phase: str = "vehicle_turn",
) -> list[dict]:
    """在原地从 current_heading 转到 target_heading，返回 pose 列表"""
    poses = []
    signed_angle = (target_heading - current_heading + 180) % 360 - 180
    if abs(signed_angle) <= 1e-9:
        robot.heading = target_heading % 360
        robot.car_angular_velocity = 0.0
        return poses
    if angular_velocity_limit <= 0:
        raise ValueError("angular_velocity_limit 必须大于 0。")
    if sample_rate <= 0:
        raise ValueError("sample_rate 必须大于 0。")
    angular_velocity = angular_velocity_limit if signed_angle >= 0 else -angular_velocity_limit
    turn_time = abs(signed_angle) / angular_velocity_limit
    _recordMotionSegment(
        motion_segments,
        phase=phase,
        coverage_active=False,
        duration=turn_time,
        car_position_start=car_position,
        car_position_end=car_position,
        heading_start=current_heading,
        heading_end=current_heading + signed_angle,
        arm_angle_start=arm_angle,
        arm_angle_end=arm_angle,
    )
    n_steps = max(1, math.ceil(turn_time * sample_rate))
    for i in range(1, n_steps + 1):
        s = i / n_steps
        poses.append({
            'heading': (current_heading + signed_angle * s) % 360,
            'arm_angle': arm_angle,
            'car_speed': 0.0,
            'car_angular_velocity': angular_velocity,
            'car_position': car_position,
        })
    robot.heading = target_heading % 360
    robot.car_angular_velocity = 0.0
    return poses

def _generateCarMovePoses(
    robot: Robot,
    node_a: tuple[float, float],
    node_b: tuple[float, float],
    heading: float,
    arm_angle: float,
    speed_limit: float,
    sample_rate: float,
    motion_segments: list[MotionSegment] | None = None,
    phase: str = "transition",
) -> list[dict]:
    """沿直线从 node_a 向 node_b 移动，返回 pose 列表"""
    poses = []
    dx = node_b[0] - node_a[0]
    dy = node_b[1] - node_a[1]
    distance = math.hypot(dx, dy)
    if distance <= 1e-9:
        robot.car_position = node_b
        robot.car_speed = 0.0
        return poses
    if speed_limit <= 0:
        raise ValueError("speed_limit 必须大于 0。")
    if sample_rate <= 0:
        raise ValueError("sample_rate 必须大于 0。")
    move_time = distance / speed_limit
    _recordMotionSegment(
        motion_segments,
        phase=phase,
        coverage_active=False,
        duration=move_time,
        car_position_start=node_a,
        car_position_end=node_b,
        heading_start=heading,
        heading_end=heading,
        arm_angle_start=arm_angle,
        arm_angle_end=arm_angle,
    )
    n_steps = max(1, math.ceil(move_time * sample_rate))
    for i in range(1, n_steps + 1):
        s = i / n_steps
        poses.append({
            'heading': heading,
            'arm_angle': arm_angle,
            'car_speed': speed_limit,
            'car_angular_velocity': 0.0,
            'car_position': (node_a[0] + dx * s, node_a[1] + dy * s),
        })
    robot.car_position = node_b
    robot.car_speed = 0.0
    robot.car_angular_velocity = 0.0
    return poses

def _generateSweepingMovePoses(
    robot: Robot,
    node_a: tuple[float, float],
    node_b: tuple[float, float],
    heading: float,
    arm_angle_start: float,
    arm_angle_end: float,
    speed_limit: float,
    arm_angular_velocity_limit: float,
    sample_rate: float,
    motion_segments: list[MotionSegment] | None = None,
) -> list[dict]:
    """Move while oscillating the arm at fixed angular speed across the strip."""
    poses = []
    dx = node_b[0] - node_a[0]
    dy = node_b[1] - node_a[1]
    distance = math.hypot(dx, dy)
    delta_arm = arm_angle_end - arm_angle_start
    if distance <= 1e-9:
        robot.car_position = node_b
        return _generateArmTurnPoses(
            robot,
            node_b,
            heading,
            arm_angle_start,
            arm_angle_end,
            arm_angular_velocity_limit,
            sample_rate,
            motion_segments=motion_segments,
            phase="work",
            coverage_active=True,
        )
    if speed_limit <= 0:
        raise ValueError("speed_limit 必须大于 0。")
    if arm_angular_velocity_limit <= 0:
        raise ValueError("arm_angular_velocity_limit 必须大于 0。")
    if sample_rate <= 0:
        raise ValueError("sample_rate 必须大于 0。")

    lower_angle = min(arm_angle_start, arm_angle_end)
    upper_angle = max(arm_angle_start, arm_angle_end)
    span = upper_angle - lower_angle
    signed_arm_speed = (
        abs(arm_angular_velocity_limit)
        if delta_arm >= 0
        else -abs(arm_angular_velocity_limit)
    )
    # Adjacent half-sweeps must overlap longitudinally by at least one disc
    # diameter; otherwise the zig-zag center curve leaves holes in the strip.
    if span > 1e-9:
        half_sweep_time = span / abs(signed_arm_speed)
        coverage_safe_speed = 2.0 * robot.disc_radius / half_sweep_time
        if robot.work_speed_policy == "coverage_safe":
            car_speed = min(speed_limit, coverage_safe_speed)
        elif robot.work_speed_policy == "commanded":
            car_speed = speed_limit
        else:
            raise ValueError(
                "work_speed_policy 必须为 'coverage_safe' 或 'commanded'。"
            )
    else:
        car_speed = speed_limit
    total_time = distance / car_speed
    n_steps = max(1, math.ceil(total_time * sample_rate))
    final_arm_angle, _ = _oscillatingArmAngle(
        total_time,
        arm_angle_start,
        signed_arm_speed,
        lower_angle,
        upper_angle,
    )
    _recordMotionSegment(
        motion_segments,
        phase="work",
        coverage_active=True,
        duration=total_time,
        car_position_start=node_a,
        car_position_end=node_b,
        heading_start=heading,
        heading_end=heading,
        arm_angle_start=arm_angle_start,
        arm_angle_end=final_arm_angle,
        arm_motion="oscillating",
        arm_angular_velocity=signed_arm_speed,
        arm_angle_lower=lower_angle,
        arm_angle_upper=upper_angle,
    )

    for i in range(1, n_steps + 1):
        elapsed = total_time * i / n_steps
        s = elapsed / total_time
        arm_angle, instantaneous_arm_speed = _oscillatingArmAngle(
            elapsed,
            arm_angle_start,
            signed_arm_speed,
            lower_angle,
            upper_angle,
        )
        poses.append({
            'heading': heading,
            'arm_angle': arm_angle,
            'arm_angular_velocity': instantaneous_arm_speed,
            'car_speed': car_speed,
            'car_angular_velocity': 0.0,
            'car_position': (node_a[0] + dx * s, node_a[1] + dy * s),
        })
    robot.car_position = node_b
    robot.heading = heading % 360
    robot.arm_angle = final_arm_angle
    robot.car_speed = 0.0
    robot.car_angular_velocity = 0.0
    robot.arm_angular_velocity = 0.0
    return poses

def _generateArmTurnPoses(
    robot: Robot,
    car_position: tuple[float, float],
    heading: float,
    arm_angle_start: float,
    arm_angle_end: float,
    arm_angular_velocity_limit: float,
    sample_rate: float,
    motion_segments: list[MotionSegment] | None = None,
    phase: str = "arm_setup",
    coverage_active: bool = False,
) -> list[dict]:
    """Rotate arm from arm_angle_start to arm_angle_end while car is stationary."""
    poses = []
    delta = arm_angle_end - arm_angle_start
    if abs(delta) < 1e-6:
        robot.arm_angle = arm_angle_end
        robot.arm_angular_velocity = 0.0
        return poses
    if arm_angular_velocity_limit <= 0:
        raise ValueError("arm_angular_velocity_limit 必须大于 0。")
    if sample_rate <= 0:
        raise ValueError("sample_rate 必须大于 0。")
    arm_angular_velocity = arm_angular_velocity_limit if delta > 0 else -arm_angular_velocity_limit
    turn_time = abs(delta) / arm_angular_velocity_limit
    _recordMotionSegment(
        motion_segments,
        phase=phase,
        coverage_active=coverage_active,
        duration=turn_time,
        car_position_start=car_position,
        car_position_end=car_position,
        heading_start=heading,
        heading_end=heading,
        arm_angle_start=arm_angle_start,
        arm_angle_end=arm_angle_end,
    )
    n_steps = max(1, math.ceil(turn_time * sample_rate))
    for i in range(1, n_steps + 1):
        s = i / n_steps
        poses.append({
            'heading': heading,
            'arm_angle': arm_angle_start + delta * s,
            'car_speed': 0.0,
            'car_angular_velocity': 0.0,
            'car_position': car_position,
        })
    robot.arm_angle = arm_angle_end
    robot.arm_angular_velocity = 0.0
    return poses

# ========================================

def fromInPointToStartPoint( # 必须在findRobotPoseByInPoint之后调用
    in_point: tuple[float, float],
    start_point: tuple[float, float],
    start_corner: int,
    robot: Robot,
    start_heading: float | None = None,
    start_arm_angle: float | None = None,
    sample_rate: float = 10,
    motion_segments: list[MotionSegment] | None = None,
) -> list[dict]:
    """
    从入点驶向耕牛法起始点，并旋转摆臂到指向角落，生成完整 pose 序列。
    过程中需要不断更新robot动态参数
    """
    poses = []
    node_0 = robot.car_position
    node_2 = start_point
    if start_heading is None:
        start_heading = 0 if start_corner in (0, 1) else 180
    if start_arm_angle is None:
        start_arm_angle = robot.arm_angle
    # 确定中间拐点node_1(L形路径)
    if robot.heading in (0, 180):
        node_1 = (node_2[0], node_0[1])
    elif robot.heading in (90, 270):
        node_1 = (node_0[0], node_2[1])
    else:
        raise ValueError(f"@fromInPointToStartPoint: robot.heading {robot.heading} 不合法，无法确定node_1")
    # 然后生成pose序列
    heading_0_to_1 = computeHeading(node_0, node_1)
    heading_1_to_2 = computeHeading(node_1, node_2)
    # 生成pose序列
    poses += _generateCarTurnPoses(robot, node_0, robot.heading, heading_0_to_1, robot.arm_angle, robot.angular_velocity_limit, sample_rate, motion_segments)
    poses += _generateCarMovePoses(robot, node_0, node_1, heading_0_to_1, robot.arm_angle, robot.speed_limit, sample_rate, motion_segments, "entry")
    poses += _generateCarTurnPoses(robot, node_1, robot.heading, heading_1_to_2, robot.arm_angle, robot.angular_velocity_limit, sample_rate, motion_segments)
    poses += _generateCarMovePoses(robot, node_1, node_2, heading_1_to_2, robot.arm_angle, robot.speed_limit, sample_rate, motion_segments, "entry")
    poses += _generateCarTurnPoses(robot, node_2, robot.heading, start_heading, robot.arm_angle, robot.angular_velocity_limit, sample_rate, motion_segments)
    poses += _generateArmTurnPoses(robot, node_2, robot.heading, robot.arm_angle, start_arm_angle, robot.arm_angular_velocity_limit, sample_rate, motion_segments, "arm_setup")
    return poses

def _normalizeHeadingToCardinal(heading: float) -> float:
    """将航向角归一化到最近的 0/90/180/270"""
    cardinals = [0, 90, 180, 270]
    return min(cardinals, key=lambda c: min(abs(c - heading), 360 - abs(c - heading)))

def fromStartPointToOutPoint(
    start_point: tuple[float, float],
    start_corner: int,
    out_point: tuple[float, float],
    rect: Rect,
    robot: Robot,
    sample_rate: float = 10,
    motion_segments: list[MotionSegment] | None = None,
) -> list[dict]:
    """
    从起点到出点，记录机器人位姿
    """
    poses = []
    robot.heading = _normalizeHeadingToCardinal(robot.heading)
    segments = _horizontalSweepSegments(rect, robot, start_corner)
    for node_a, node_b, heading, arm_start, arm_end in segments:
        if math.dist(robot.car_position, node_a) > 1e-9:
            transit_heading = computeHeading(robot.car_position, node_a)
            poses += _generateCarTurnPoses(
                robot,
                robot.car_position,
                robot.heading,
                transit_heading,
                robot.arm_angle,
                robot.angular_velocity_limit,
                sample_rate,
                motion_segments,
            )
            poses += _generateCarMovePoses(
                robot,
                robot.car_position,
                node_a,
                transit_heading,
                robot.arm_angle,
                robot.speed_limit,
                sample_rate,
                motion_segments,
                "lane_change",
            )
        poses += _generateCarTurnPoses(
            robot,
            node_a,
            robot.heading,
            heading,
            robot.arm_angle,
            robot.angular_velocity_limit,
            sample_rate,
            motion_segments,
        )
        poses += _generateArmTurnPoses(
            robot,
            node_a,
            robot.heading,
            robot.arm_angle,
            arm_start,
            robot.arm_angular_velocity_limit,
            sample_rate,
            motion_segments,
            "arm_setup",
        )
        poses += _generateSweepingMovePoses(
            robot,
            node_a,
            node_b,
            heading,
            robot.arm_angle,
            arm_end,
            robot.speed_limit,
            robot.arm_angular_velocity_limit,
            sample_rate,
            motion_segments,
        )

    exit_robot = findRobotPoseByInPoint(rect, out_point, Robot(
        disc_radius=robot.disc_radius,
        arm_length=robot.arm_length,
        car_width=robot.car_width,
        car_half_length=robot.car_half_length,
        pivot_to_car_center=robot.pivot_to_car_center,
        speed_limit=robot.speed_limit,
        angular_velocity_limit=robot.angular_velocity_limit,
        arm_angle_limit=robot.arm_angle_limit,
        arm_angular_velocity_limit=robot.arm_angular_velocity_limit,
        heading=robot.heading,
        arm_angle=robot.arm_angle,
        car_position=robot.car_position,
    ))
    if math.dist(robot.car_position, exit_robot.car_position) > 1e-9:
        transit_heading = computeHeading(robot.car_position, exit_robot.car_position)
        poses += _generateCarTurnPoses(
            robot,
            robot.car_position,
            robot.heading,
            transit_heading,
            robot.arm_angle,
            robot.angular_velocity_limit,
            sample_rate,
            motion_segments,
        )
        poses += _generateCarMovePoses(
            robot,
            robot.car_position,
            exit_robot.car_position,
            transit_heading,
            robot.arm_angle,
            robot.speed_limit,
            sample_rate,
            motion_segments,
            "exit",
        )
    poses += _generateCarTurnPoses(
        robot,
        robot.car_position,
        robot.heading,
        exit_robot.heading,
        robot.arm_angle,
        robot.angular_velocity_limit,
        sample_rate,
        motion_segments,
    )
    poses += _generateArmTurnPoses(
        robot,
        robot.car_position,
        robot.heading,
        robot.arm_angle,
        exit_robot.arm_angle,
        robot.arm_angular_velocity_limit,
        sample_rate,
        motion_segments,
        "arm_retract",
    )
    return poses

def boustrophedonCoverage(
    rect: Rect,
    in_point: tuple[float, float], # 必定在矩形边界上
    out_point: tuple[float, float], # 必定在矩形边界上
    robot: Robot,
    sample_rate: float = 10,
    motion_segments: list[MotionSegment] | None = None,
) -> list[dict]:
    """给定起点和终点，返回[dict(heading, arm_angle, car_speed, car_angular_velocity, car_position)]"""
    poses = []
    # 首先从最靠近起点的矩形顶角上开始覆盖
    start_point, start_corner, start_heading, start_arm_angle = _nearestCornerStartPose(rect, in_point, robot)
    poses += fromInPointToStartPoint(
        in_point,
        start_point,
        start_corner,
        robot,
        start_heading=start_heading,
        start_arm_angle=start_arm_angle,
        sample_rate=sample_rate,
        motion_segments=motion_segments,
    )
    poses += fromStartPointToOutPoint(
        start_point,
        start_corner,
        out_point,
        rect,
        robot,
        sample_rate,
        motion_segments,
    )
    return poses

def visualizeCoverage(
    map: np.ndarray, # uint8，灰度图，黑色为0，白色为255
    rect: Rect,
    poses: list[dict], # 蕴含了in_point和out_point的信息
    robot: Robot,
    pixel_size: float,
    sample_rate: float,
    output_path: str,
) -> None:
    """
    可视化覆盖过程
    """
    frames = []
    # 拷贝map并绘制矩形
    map_copy = map.copy()
    _drawRect(map_copy, rect, pixel_size, color=216)
    # 根据poses绘制覆盖路径
    for pose in poses:
        frame = map_copy.copy()
        # 更新robot的位姿
        robot.heading = pose['heading']
        robot.arm_angle = pose['arm_angle']
        robot.car_speed = pose['car_speed']
        robot.car_angular_velocity = pose['car_angular_velocity']
        robot.car_position = pose['car_position']
        # 绘制robot
        _drawRobot(frame, robot, pixel_size)
        # 转换为PIL图像
        frame = Image.fromarray(frame)
        frames.append(frame)
    # 保存gif
    duration = 1000 / sample_rate # 每帧间隔时间，单位为毫秒
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )
    # print(f"Coverage visualization saved to {output_path}")

def _drawRect(
    map: np.ndarray,
    rect: Rect,
    pixel_size: float,
    color: int,
) -> None:
    """
    绘制矩形
    """
    x1, y1 = coordinateWorldToMap((rect.x1, rect.y1), pixel_size)
    x2, y2 = coordinateWorldToMap((rect.x2, rect.y2), pixel_size)
    map[y1:y2, x1:x2] = color
    return None

def _drawLine(
    map: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
    pixel_size: float,
    color: int,
) -> None:
    """
    绘制线段
    """
    x1, y1 = coordinateWorldToMap(start, pixel_size)
    x2, y2 = coordinateWorldToMap(end, pixel_size)
    
    # 计算线段长度（像素数）
    dx = x2 - x1
    dy = y2 - y1
    length = int(np.sqrt(dx**2 + dy**2))  # 像素长度
    
    # 在起点和终点之间均匀采样点
    for i in range(length + 1):
        t = i / length if length > 0 else 0  # 归一化参数 [0, 1]
        x = int(x1 + t * dx)
        y = int(y1 + t * dy)
        
        # 检查边界，然后设置颜色
        if 0 <= y < map.shape[0] and 0 <= x < map.shape[1]:
            map[y, x] = color
    return None

def _drawCircle(
    map: np.ndarray,
    center: tuple[float, float],
    radius: float,
    pixel_size: float,
    color: int,
) -> None:
    """
    绘制圆
    """
    cx, cy = coordinateWorldToMap(center, pixel_size)
    radius_map = int(radius / pixel_size)
    # 计算圆的边界框
    x_min = max(0, cx - radius_map)
    x_max = min(map.shape[1], cx + radius_map + 1)
    y_min = max(0, cy - radius_map)
    y_max = min(map.shape[0], cy + radius_map + 1)
    # 遍历边界框内的每个像素
    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            # 计算像素中心到圆心的距离
            dx = x - cx
            dy = y - cy
            distance = math.sqrt(dx**2 + dy**2)
            # 如果距离小于等于半径，则绘制该像素
            if distance <= radius_map:
                map[y, x] = color
    return None

def _pointInPolygon(
    px: int, py: int,
    vertices: list[tuple[int, int]],
) -> bool:
    """
    射线法判断点是否在多边形内部
    vertices: 多边形顶点列表（地图坐标），按顺序排列
    """
    n = len(vertices)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        if yi != yj and ((yi > py) != (yj > py)):
            t = (xj - xi) * (py - yi) / (yj - yi) + xi
            if px < t:
                inside = not inside
        j = i
    return inside


def _drawCar(
    map: np.ndarray,
    robot: Robot,
    pixel_size: float,
    color: int,
) -> None:
    """绘制汽车"""
    cx, cy = robot.car_position
    alpha = math.radians(robot.heading)
    cos_a = math.cos(alpha)
    sin_a = math.sin(alpha)

    # 局部坐标系四个角（中心为原点，x轴沿车长方向）
    local_corners = [
        (-robot.car_half_length, -robot.car_width / 2),
        (robot.car_half_length, -robot.car_width / 2),
        (robot.car_half_length, robot.car_width / 2),
        (-robot.car_half_length, robot.car_width / 2),
    ]

    # 旋转并平移到世界坐标，再转换到地图坐标
    vertices: list[tuple[int, int]] = []
    for lx, ly in local_corners:
        wx = cx + lx * cos_a - ly * sin_a
        wy = cy + lx * sin_a + ly * cos_a
        mx, my = coordinateWorldToMap((wx, wy), pixel_size)
        vertices.append((mx, my))

    # 计算地图坐标系下的边界框
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    x_min = max(0, min(xs))
    x_max = min(map.shape[1], max(xs) + 1)
    y_min = max(0, min(ys))
    y_max = min(map.shape[0], max(ys) + 1)

    # 对边界框内每个像素做点-in-多边形判断
    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            if _pointInPolygon(x, y, vertices):
                map[y, x] = color

def _drawRobot(
    map: np.ndarray,
    robot: Robot,
    pixel_size: float,
) -> None:
    """
    绘制机器人
    """
    # 先画car
    heading = robot.heading
    _drawCar(
        map=map,
        robot=robot,
        pixel_size=pixel_size,
        color=128,
    )
    # 再画arm（车为头、摆臂/圆盘为尾：Car --[pivot_to_car_center]--> Pivot --[arm_length]--> Disc）
    # 摆臂旋转中心，在车中心后方（与heading相反方向）pivot_to_car_center 处
    heading_rad = math.radians(robot.heading)
    pivot_point = (
        robot.car_position[0] - robot.pivot_to_car_center * math.cos(heading_rad),
        robot.car_position[1] - robot.pivot_to_car_center * math.sin(heading_rad)
    )
    # 摆臂末端（圆盘中心），从摆臂旋转中心沿 heading+180+arm_angle 方向延长 arm_length（即车后方）
    arm_heading_rad = math.radians(robot.heading + 180 + robot.arm_angle)
    disc_center = (
        pivot_point[0] + robot.arm_length * math.cos(arm_heading_rad),
        pivot_point[1] + robot.arm_length * math.sin(arm_heading_rad)
    )
    _drawLine(map, pivot_point, disc_center, pixel_size, color=128)
    # 再画disc
    _drawCircle(map, disc_center, robot.disc_radius, pixel_size, color=128)
    return None

def mainPipeline(
    in_point: tuple[float, float],
    out_point: tuple[float, float],
    robot: Robot,
    pixel_size: float,
    output_path: str,
) -> None:
    """
    主流程
    """
    rect = Rect(0, 0, 10, 10)
    in_point = findNearestEdgePoint(rect, in_point)
    out_point = findNearestEdgePoint(rect, out_point)
    # robot = Robot(
    # # 静态信息
    # disc_radius=0.25,
    # arm_length=1.2,
    # car_width=0.8,
    # car_half_length=1.0,
    # pivot_to_car_center=0.3,

    # speed_limit=1.0,
    # angular_velocity_limit=30.0,
    # arm_angle_limit=(-90, 90),
    # arm_angular_velocity_limit=90.0,

    # # 动态信息
    # heading=0.0,
    # arm_angle=0.0,
    # arm_angular_velocity=0.0,
    # car_speed=0.0,
    # car_angular_velocity=0.0,
    # car_position=(0.0, 0.0),
    # )
    robot = findRobotPoseByInPoint(rect, in_point, robot)
    poses = boustrophedonCoverage(rect, in_point, out_point, robot, sample_rate=10)
    map = np.full((200, 200), 255, dtype=np.uint8)
    visualizeCoverage(
        map=map,
        rect=rect,
        poses=poses,
        robot=robot,
        pixel_size=pixel_size,
        sample_rate=10,
        output_path=output_path,
    )

if __name__ == "__main__":
    mainPipeline(
        in_point=(2, 0),
        out_point=(10, 0),
        robot=Robot(
            disc_radius=0.25,
            arm_length=1.2,
            car_width=0.8,
            car_half_length=1.0,
            pivot_to_car_center=0.3,
            speed_limit=1.0,
            angular_velocity_limit=30.0,
            arm_angle_limit=(-90, 90),
            arm_angular_velocity_limit=90.0,
            heading=0.0,
            arm_angle=0.0,
            arm_angular_velocity=0.0,
            car_speed=0.0,
            car_angular_velocity=0.0,
            car_position=(5, 5),
        ),
        pixel_size=0.1,
        output_path='coverage.gif'
    )
    print("Coverage visualization saved to coverage.gif")
