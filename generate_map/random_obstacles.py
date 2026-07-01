from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class Rect:
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

def sampleRect( # 后续可以替换这个函数的逻辑，使用BIM获得障碍物地图
    workspace: 'Rect',
    width_range: tuple[float, float],
    height_range: tuple[float, float],
) -> 'Rect' | None:
    """在workspace中随机采样一个矩形障碍物"""
    x_min = workspace.x1
    x_max = workspace.x2
    y_min = workspace.y1
    y_max = workspace.y2
    x1 = np.random.uniform(x_min, x_max)
    y1 = np.random.uniform(y_min, y_max)
    x2 = x1 + np.random.uniform(width_range[0], width_range[1])
    y2 = y1 + np.random.uniform(height_range[0], height_range[1])
    if x2 > x_max or y2 > y_max:
        return None
    return Rect(x1, y1, x2, y2)

def drawRect(
    map: np.ndarray, # uint8，灰度图，黑色为0，白色为255
    rect: 'Rect',
    pixel_size: float,
    color: int = 0, # 黑色
) -> None:
    """在map中绘制一个矩形障碍物"""
    x1, y1, x2, y2 = rect.x1, rect.y1, rect.x2, rect.y2
    x1_map, y1_map = coordinateWorldToMap((x1, y1), pixel_size)
    x2_map, y2_map = coordinateWorldToMap((x2, y2), pixel_size)
    map[y1_map:y2_map, x1_map:x2_map] = color

def generateMap(
    world_width: float,
    world_height: float,
    pixel_size: float,
    width_range: tuple[float, float],
    height_range: tuple[float, float],
    n_obstacles: int,
    max_trials: int = 100,
) -> np.ndarray:
    """生成障碍物地图"""
    workspace = Rect(0, 0, world_width, world_height)
    # 初始化map
    workspace_map_x2, workspace_map_y2 = coordinateWorldToMap((workspace.x2, workspace.y2), pixel_size)
    map = np.full((workspace_map_y2, workspace_map_x2), 255, dtype=np.uint8)
    # 绘制障碍物
    for _ in range(n_obstacles):
        for _ in range(max_trials):
            rect = sampleRect(workspace, width_range, height_range)
            if rect is not None:
                drawRect(map, rect, pixel_size)
                break
        else:
            raise ValueError(f"Failed to sample obstacle after {max_trials} trials")
    return map

def saveMapPNG(map: np.ndarray, file_path: str) -> None: # origin='lower'表示地图的左下角为原点，点（x, y）用map[y, x]表示
    """保存地图到PNG文件"""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(file_path, map, cmap='gray', origin='lower')

def main():
    parser = argparse.ArgumentParser(description="生成随机矩形障碍物占据地图。")
    parser.add_argument(
        "--output",
        default="experiments/fields2cover_comparison/maps/random_map.png",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    np.random.seed(args.seed)
    world_width = 20.0
    world_height = 40.0
    pixel_size = 0.2
    width_range = (1, 3)
    height_range = (1, 5)
    n_obstacles = 10
    max_trials = 100
    map = generateMap(world_width, world_height, pixel_size, width_range, height_range, n_obstacles, max_trials)
    saveMapPNG(map, args.output)

if __name__ == "__main__":
    main()
