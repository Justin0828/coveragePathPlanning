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
    arm_angle_limit: tuple[float, float] = (0, 90) # 摆臂角度限制
    arm_angular_velocity_limit: float = 0.0 # 摆臂角速度限制

    # 动态信息
    heading: float = 0.0 # 航向角
    arm_angle: float = 0.0 # 摆臂角度，角度制，正向为逆时针
    arm_angular_velocity: float = 0.0 # 摆臂角速度，角度制，正向为逆时针
    car_speed: float = 0.0 # 车速度，>= 0，用 heading 确定方向
    car_angular_velocity: float = 0.0 # 车角速度，角度制，正向为逆时针
    car_position: tuple[float, float] = (0.0, 0.0) # 车中心点位置

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

def findStartPoint( # robot靠纵向边停靠，返回(car_center起始点, 角落编号)
    rect: Rect,
    in_point: tuple[float, float],
    robot: Robot,
) -> tuple[tuple[float, float], int]:
    """
    给定矩形和世界坐标系中的入点，确定起始点信息
    返回 (car_center起始点, 角落编号)
    角落编号: 0=左上, 1=左下, 2=右上, 3=右下
    """
    x, y = in_point
    a = x - rect.x1  # 到左边的距离
    b = rect.y2 - y  # 到上边的距离
    c = rect.x2 - x  # 到右边的距离
    d = y - rect.y1  # 到下边的距离
    half_band_width = math.sqrt(robot.arm_length**2 - (robot.car_half_length - robot.pivot_to_car_center)**2)
    if a <= c and b <= d: # 左 & 上
        return (rect.x1 + robot.car_half_length, rect.y2 - half_band_width), 0
    if a <= c and d < b: # 左 & 下
        return (rect.x1 + robot.car_half_length, rect.y1 + half_band_width), 1
    if c < a and b <= d: # 右 & 上
        return (rect.x2 - robot.car_half_length, rect.y2 - half_band_width), 2
    if c < a and d < b: # 右 & 下
        return (rect.x2 - robot.car_half_length, rect.y1 + half_band_width), 3
    raise ValueError(f"@findStartPoint: in_point {in_point} 无法确定car_center起始点")

# ==================== 通用 Pose 生成器 ====================
def _generateCarTurnPoses(
    robot: Robot,
    car_position: tuple[float, float],
    current_heading: float,
    target_heading: float,
    arm_angle: float,
    angular_velocity_limit: float,
    sample_rate: float,
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

def _generateArmTurnPoses(
    robot: Robot,
    car_position: tuple[float, float],
    heading: float,
    arm_angle_start: float,
    arm_angle_end: float,
    arm_angular_velocity_limit: float,
    sample_rate: float,
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
    sample_rate: float = 10,
) -> list[dict]:
    """
    从入点驶向耕牛法起始点，并旋转摆臂到指向角落，生成完整 pose 序列。
    过程中需要不断更新robot动态参数
    """
    poses = []
    car_disc_distance = robot.pivot_to_car_center + robot.arm_length
    # 首先计算node_0(in_car_center)和node_2(start_point)
    if robot.heading == 0:
        node_0 = (in_point[0] - car_disc_distance, in_point[1])
    elif robot.heading == 90:
        node_0 = (in_point[0], in_point[1] + car_disc_distance)
    elif robot.heading == 180:
        node_0 = (in_point[0] + car_disc_distance, in_point[1])
    elif robot.heading == 270:
        node_0 = (in_point[0], in_point[1] - car_disc_distance)
    else:
        raise ValueError(f"@fromInPointToStartPoint: robot.heading {robot.heading} 不合法，无法确定node_0")
    node_2 = start_point
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
    poses += _generateCarTurnPoses(robot, node_0, robot.heading, heading_0_to_1, robot.arm_angle, robot.angular_velocity_limit, sample_rate)
    poses += _generateCarMovePoses(robot, node_0, node_1, heading_0_to_1, robot.arm_angle, robot.speed_limit, sample_rate)
    poses += _generateCarTurnPoses(robot, node_1, robot.heading, heading_1_to_2, robot.arm_angle, robot.angular_velocity_limit, sample_rate)
    poses += _generateCarMovePoses(robot, node_1, node_2, heading_1_to_2, robot.arm_angle, robot.speed_limit, sample_rate)
    heading_start = 0 if start_corner in (0, 1) else 180
    poses += _generateCarTurnPoses(robot, node_2, heading_1_to_2, heading_start, robot.arm_angle, robot.angular_velocity_limit, sample_rate)
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
) -> list[dict]:
    """
    从起点到出点，记录机器人位姿
    """
    poses = []
    robot.heading = _normalizeHeadingToCardinal(robot.heading)  # _findNodes 仅支持 0/90/180/270
    primary_heading = robot.heading  # capture before loop mutations
    lateral_headings = {(primary_heading + 90) % 360, (primary_heading + 270) % 360}
    nodes = _findNodes(rect, robot, start_point, start_corner, out_point)
    n_segments = len(nodes) - 1
    for i, (node_a, node_b) in enumerate(zip(nodes[:-1], nodes[1:])):
        is_last = (i == n_segments - 1)
        target_heading = computeHeading(node_a, node_b) % 360
        is_lateral = any(abs((target_heading - lh + 180) % 360 - 180) < 5.0 for lh in lateral_headings)
        sweep = not is_last and not is_lateral
        poses += _fromNodeToNode(node_a, node_b, robot, sample_rate, sweep_arm=sweep)
    return poses

def _findNodes(
    rect: Rect,
    robot: Robot,
    start_point: tuple[float, float],
    start_corner: int,
    out_point: tuple[float, float],
) -> list[tuple[float, float]]:
    """
    确定耕牛法路径上的节点
    """
    nodes = []
    half_band_width = math.sqrt(robot.arm_length**2 - (robot.car_half_length - robot.pivot_to_car_center)**2)
    if robot.heading == 0:
        layers_num = math.ceil((rect.y2 - rect.y1) / (2 * half_band_width))
        nodes_num = layers_num * 2
        nodes.append(start_point)  # 起点
        work_length = rect.x2 - rect.x1 - robot.car_half_length - half_band_width
        for i in range(1, nodes_num):
            if i % 2 == 1:
                if i % 4 == 1:
                    nodes.append((nodes[i-1][0] + work_length, nodes[i-1][1]))
                else:
                    nodes.append((nodes[i-1][0] - work_length, nodes[i-1][1]))
            else:
                if start_corner == 0: # 左上角
                    nodes.append((nodes[i-1][0], nodes[i-1][1] - 2 * half_band_width))
                elif start_corner == 1: # 左下角
                    nodes.append((nodes[i-1][0], nodes[i-1][1] + 2 * half_band_width))
                else:
                    raise ValueError("无法确定耕牛节点")

    elif robot.heading == 90:
        layers_num = math.ceil((rect.x2 - rect.x1) / (2 * half_band_width))
        nodes_num = layers_num * 2
        nodes.append(start_point)  # 起点
        work_length = rect.y2 - rect.y1 - robot.car_half_length - half_band_width
        for i in range(1, nodes_num):
            if i % 2 == 1:
                if i % 4 == 1:
                    nodes.append((nodes[i-1][0], nodes[i-1][1] + work_length))
                else:
                    nodes.append((nodes[i-1][0], nodes[i-1][1] - work_length))
            else:
                if start_corner == 1: # 左下角  
                    nodes.append((nodes[i-1][0] + 2 * half_band_width, nodes[i-1][1]))
                elif start_corner == 3: # 右下角
                    nodes.append((nodes[i-1][0] - 2 * half_band_width, nodes[i-1][1]))
                else:
                    raise ValueError("无法确定耕牛节点")

    elif robot.heading == 180:
        layers_num = math.ceil((rect.y2 - rect.y1) / (2 * half_band_width))
        nodes_num = layers_num * 2
        nodes.append(start_point)  # 起点
        work_length = rect.x2 - rect.x1 - robot.car_half_length - half_band_width
        for i in range(1, nodes_num):
            if i % 2 == 1:
                if i % 4 == 1:
                    nodes.append((nodes[i-1][0] - work_length, nodes[i-1][1]))
                else:
                    nodes.append((nodes[i-1][0] + work_length, nodes[i-1][1]))
            else:
                if start_corner == 2: # 右上角
                    nodes.append((nodes[i-1][0], nodes[i-1][1] - 2 * half_band_width))
                elif start_corner == 3: # 右下角
                    nodes.append((nodes[i-1][0], nodes[i-1][1] + 2 * half_band_width))
                else:
                    raise ValueError("无法确定耕牛节点")

    elif robot.heading == 270:
        layers_num = math.ceil((rect.x2 - rect.x1) / (2 * half_band_width))
        nodes_num = layers_num * 2
        nodes.append(start_point)  # 起点
        work_length = rect.y2 - rect.y1 - robot.car_half_length - half_band_width
        for i in range(1, nodes_num):
            if i % 2 == 1:
                if i % 4 == 1:
                    nodes.append((nodes[i-1][0] - work_length, nodes[i-1][1]))
                else:
                    nodes.append((nodes[i-1][0] + work_length, nodes[i-1][1]))
            else:
                if start_corner == 0: # 左上角
                    nodes.append((nodes[i-1][0] + 2 * half_band_width, nodes[i-1][1]))
                elif start_corner == 2: # 右上角
                    nodes.append((nodes[i-1][0] - 2 * half_band_width, nodes[i-1][1]))
                else:
                    raise ValueError("无法确定耕牛节点")
    # 最后添加终点
    nodes.append(out_point)
    return nodes

def _fromNodeToNode(
    node_a: tuple[float, float],
    node_b: tuple[float, float],
    robot: Robot,
    sample_rate: float = 10,
    sweep_arm: bool = False,
) -> list[dict]:
    """
    从节点到节点，记录机器人位姿
    sweep_arm=True: arm rotates to arm_angle_limit[1] before the move (working lane pass)
    sweep_arm=False: arm returns to 0° (lateral step or final transition)
    """
    target_heading = computeHeading(node_a, node_b)
    target_arm_angle = robot.arm_angle_limit[1] if sweep_arm else 0.0
    poses = []
    poses += _generateArmTurnPoses(robot, node_a, robot.heading, robot.arm_angle, target_arm_angle, robot.arm_angular_velocity_limit, sample_rate)
    poses += _generateCarTurnPoses(robot, node_a, robot.heading, target_heading, robot.arm_angle, robot.angular_velocity_limit, sample_rate)
    poses += _generateCarMovePoses(robot, node_a, node_b, target_heading, robot.arm_angle, robot.speed_limit, sample_rate)
    return poses

def boustrophedonCoverage(
    rect: Rect,
    in_point: tuple[float, float], # 必定在矩形边界上
    out_point: tuple[float, float], # 必定在矩形边界上
    robot: Robot,
    sample_rate: float = 10,
) -> list[dict]:
    """给定起点和终点，返回[dict(heading, arm_angle, car_speed, car_angular_velocity, car_position)]"""
    poses = []
    # 首先从最靠近起点的矩形顶角上开始覆盖
    start_point, start_corner = findStartPoint(rect, in_point, robot)
    poses += fromInPointToStartPoint(in_point, start_point, start_corner, robot, sample_rate)
    poses += fromStartPointToOutPoint(start_point, start_corner, out_point, rect, robot, sample_rate)
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
    # arm_angle_limit=(0, 90),
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
            arm_angle_limit=(0, 90),
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
