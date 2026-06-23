from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

@dataclass
class Rect:
    x1: float # 左下角x
    y1: float # 左下角y
    x2: float # 右上角x
    y2: float # 右上角y

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

def loadMap(file_path: str) -> np.ndarray:
    """读取PNG灰度地图为 ndarray"""
    img = Image.open(file_path).convert('L')
    # 上下翻转，使坐标系原点在左下角，与保存时的origin='lower'保持一致
    return np.flipud(np.array(img))

def showMap(map: np.ndarray) -> None:
    """显示map（为了显示覆盖后的map）"""
    plt.imshow(map, cmap='gray', origin='lower')
    plt.show()

def saveMap(map: np.ndarray, file_path: str) -> None:
    """保存map为PNG灰度图"""
    plt.imsave(file_path, map, cmap='gray', origin='lower')
    return None

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

def coordinateRowcolToGrid(
    n_rows: int, # 行数
    row: int,
    col: int,
) -> tuple[int, int]:
    """将行、列坐标(row, col)转换为栅格坐标(x, y)"""
    return col, n_rows - row - 1

def coordinateGridToWorld(
    x: int,
    y: int,
    grid_size: int, # 一个单元栅格边长为多少像素
    pixel_size: float, # 一个像素对应多少米
) -> tuple[float, float]:
    """将栅格坐标(x, y)转换为世界坐标系中的坐标"""
    return x * grid_size * pixel_size, y * grid_size * pixel_size

def resterizeMap(
    map: np.ndarray,
    grid_size: int, # 一个单元栅格边长为多少像素
) -> np.ndarray:
    """将map栅格化，每个单元栅格边长为grid_size"""
    # map的行数，列数
    n_rows, n_cols = map.shape
    # 为了后续栅格化，需要对map进行扩展，补全使用0（障碍物）
    new_n_rows, new_n_cols = int(np.ceil(n_rows / grid_size)) * grid_size, int(np.ceil(n_cols / grid_size)) * grid_size
    extended_map = np.zeros((new_n_rows, new_n_cols), dtype=np.bool_)
    extended_map[0:n_rows, 0:n_cols] = map == 0 # map是灰度图，0为障碍物，1为空闲；map==0代表障碍物，因此在bool图中赋值为1，表示is_obstacle
    # 补全后，计算栅格化后的地图尺寸
    grid_n_rows = int(new_n_rows / grid_size)
    grid_n_cols = int(new_n_cols / grid_size)
    # 栅格化后的地图
    grid_map = np.zeros((grid_n_rows, grid_n_cols), dtype=np.bool_)
    for i in range(grid_n_rows): # 遍历y
        for j in range(grid_n_cols): # 遍历x
            # 获取当前栅格对应的extended_map的Rect区域
            x1 = j * grid_size
            x2 = x1 + grid_size
            y1 = i * grid_size
            y2 = y1 + grid_size
            # 判断当前区域是否存在障碍物
            if np.any(extended_map[y1:y2, x1:x2] == 1):
                grid_map[i, j] = 1 # 整栅格为障碍物
    return grid_map

def generateCandidates(
    grid_map: np.ndarray,
    min_edge_length: float,
    pixel_size: float,
    grid_size: int,
    k: int,
) -> list[Rect]:
    """行扫描候选矩形：对每一行，向上扩展高度，向左右扩展宽度"""
    candidates = []
    for row in range(grid_map.shape[0]):
        candidates.extend(_computeRowCandidates(grid_map, row, min_edge_length, pixel_size, grid_size))
    return candidates


def generateColCandidates(
    grid_map: np.ndarray,
    min_edge_length: float,
    pixel_size: float,
    grid_size: int,
    k: int,
) -> list[Rect]:
    """列扫描候选矩形：对每一列，向右扩展宽度，向上下扩展高度"""
    candidates = []
    for col in range(grid_map.shape[1]):
        candidates.extend(_computeColCandidates(grid_map, col, min_edge_length, pixel_size, grid_size))
    return candidates


def maskRects(
    grid_map: np.ndarray,
    rects: list[Rect],
    pixel_size: float,
    grid_size: int,
) -> np.ndarray:
    """返回 grid_map 的副本，将 rects 覆盖的区域标记为障碍物（值=1）"""
    masked = grid_map.copy()
    n_rows = grid_map.shape[0]
    cell_size = grid_size * pixel_size
    for rect in rects:
        col1 = int(round(rect.x1 / cell_size))
        col2 = int(round(rect.x2 / cell_size))
        gy1 = int(round(rect.y1 / cell_size))  # world y bottom → low grid_y
        gy2 = int(round(rect.y2 / cell_size))  # world y top   → high grid_y
        # grid_map row = n_rows - grid_y - 1, so higher grid_y → smaller row
        row1 = max(0, n_rows - gy2)
        row2 = min(n_rows, n_rows - gy1)
        col1 = max(0, col1)
        col2 = min(grid_map.shape[1], col2)
        if row1 < row2 and col1 < col2:
            masked[row1:row2, col1:col2] = 1
    return masked


def buildCoverCount(
    rects: list[Rect],
    map_shape: tuple[int, int],
    pixel_size: float,
) -> np.ndarray:
    """根据已选矩形列表，构建 cover_count 初始值（用于后续阶段的贪婪选择）"""
    cover_count = np.zeros(map_shape, dtype=np.int32)
    for rect in rects:
        _updateCoverCount(rect, cover_count, pixel_size)
    return cover_count

def _computeRowCandidates(
    grid_map: np.ndarray,
    row: int,
    min_edge_length: float,
    pixel_size: float, # 一个像素对应多少米
    grid_size: int, # 一个栅格边长为多少像素
) -> list[Rect]:
    """计算当前行的矩形候选列表"""
    # 获取高度list：从当前行开始，向上遍历，遇到障碍物或边界为止，记录高度
    heights = np.zeros(grid_map.shape[1], dtype=np.int32)
    candidates = []
    for col in range(grid_map.shape[1]):
        n_rows = grid_map.shape[0]
        x, y = coordinateRowcolToGrid(n_rows, row, col)
        if grid_map[y, x] == 1:
            heights[col] = 0
            continue
        h = 0
        r = row
        while grid_map[y, x] == 0 and y < grid_map.shape[0] - 1:
            h += 1
            y += 1
        heights[col] = h
        # 左右扩展
        left_idx = col
        while left_idx > 0 and heights[left_idx - 1] >= h:
            left_idx -= 1
        right_idx = col
        while right_idx < grid_map.shape[1] - 1 and heights[right_idx + 1] >= h:
            right_idx += 1
        # 添加候选矩形：当前拟添加(left_idx, row)到(right_idx, row + h)的矩形，需要检查边长
        # 先计算Rect
        x1, y1 = coordinateGridToWorld(left_idx, n_rows - row - 1, grid_size, pixel_size)
        x2, y2 = coordinateGridToWorld(right_idx + 1, n_rows - row + h, grid_size, pixel_size)
        if x2 - x1 >= min_edge_length and y2 - y1 >= min_edge_length:
            candidates.append(Rect(x1, y1, x2, y2))
    return candidates

def _computeColCandidates(
    grid_map: np.ndarray,
    col: int,
    min_edge_length: float,
    pixel_size: float,
    grid_size: int,
) -> list[Rect]:
    """计算当前列的矩形候选列表：向右扩展宽度，上下扩展高度"""
    n_rows = grid_map.shape[0]
    n_cols = grid_map.shape[1]
    widths = np.zeros(n_rows, dtype=np.int32)
    candidates = []
    for row in range(n_rows):
        x, y = coordinateRowcolToGrid(n_rows, row, col)
        if grid_map[y, x] == 1:
            widths[row] = 0
            continue
        # 向右扩展宽度（+x 方向）
        w = 0
        cx = x
        while grid_map[y, cx] == 0 and cx < n_cols - 1:
            w += 1
            cx += 1
        widths[row] = w
        # 上下扩展（行索引方向：row 减小 = y 增大 = 世界坐标向上）
        top_idx = row
        while top_idx > 0 and widths[top_idx - 1] >= w:
            top_idx -= 1
        bottom_idx = row
        while bottom_idx < n_rows - 1 and widths[bottom_idx + 1] >= w:
            bottom_idx += 1
        # 转换为世界坐标
        x1, _ = coordinateGridToWorld(col, 0, grid_size, pixel_size)
        x2, _ = coordinateGridToWorld(col + w, 0, grid_size, pixel_size)
        _, y1 = coordinateGridToWorld(0, n_rows - bottom_idx - 1, grid_size, pixel_size)
        _, y2 = coordinateGridToWorld(0, n_rows - top_idx, grid_size, pixel_size)
        if x2 - x1 >= min_edge_length and y2 - y1 >= min_edge_length:
            candidates.append(Rect(x1, y1, x2, y2))
    return candidates


def greedySelection(
    candidates: list[Rect],
    map: np.ndarray, # unit8，为了计算覆盖率
    pixel_size: float,
    alpha: float = 0.0, # 矩形数量惩罚
    beta: float = 0.0, # 重复覆盖惩罚
    initial_cover_count: np.ndarray | None = None, # 前一阶段已覆盖区域
) -> list[Rect]:
    """贪婪选择，每次选择后计算优化函数的变化值，直到当前选择变化值<=threshold"""
    selected_rectangles = []
    cover_count = initial_cover_count.copy() if initial_cover_count is not None else np.zeros_like(map, dtype=np.int32)
    remaining_candidates = candidates.copy()
    
    while remaining_candidates:
        best_gain = -float('inf')
        best_rectangle = None
        best_idx = -1
        
        # 遍历所有候选矩形，找到增益最大的
        for idx, rect in enumerate(remaining_candidates):
            gain = _computeGain(rect, cover_count, pixel_size, alpha, beta)
            if gain > best_gain:
                best_gain = gain
                best_rectangle = rect
                best_idx = idx
        
        # 如果最佳增益<=0，停止选择
        if best_gain <= 0:
            break
            
        # 选择最佳矩形
        selected_rectangles.append(best_rectangle)
        remaining_candidates.pop(best_idx)
        
        # 更新覆盖计数
        _updateCoverCount(best_rectangle, cover_count, pixel_size)
        
    return selected_rectangles

def _computeGain(
    rect: Rect,
    cover_count: np.ndarray,
    pixel_size: float,
    alpha: float, # 矩形数量惩罚
    beta: float, # 重复覆盖惩罚
) -> float:
    """计算矩形的优化函数值，每次的函数变化量用newly - alpha - beta * overlap来计算"""
    x1, y1 = coordinateWorldToMap((rect.x1, rect.y1), pixel_size)
    x2, y2 = coordinateWorldToMap((rect.x2, rect.y2), pixel_size)
    sub_cover_count = cover_count[y1:y2, x1:x2]
    newly = np.sum(sub_cover_count == 0) * pixel_size ** 2  # 新覆盖的面积
    overlap = np.sum(sub_cover_count >= 1) * pixel_size ** 2  # 重复覆盖的面积
    return newly - alpha - beta * overlap

def _updateCoverCount(
    rect: Rect,
    cover_count: np.ndarray,
    pixel_size: float,
) -> None:
    """更新覆盖计数"""
    x1, y1 = coordinateWorldToMap((rect.x1, rect.y1), pixel_size)
    x2, y2 = coordinateWorldToMap((rect.x2, rect.y2), pixel_size)
    cover_count[y1:y2, x1:x2] += 1

def _expandPoint(
    grid_map: np.ndarray,
    row: int,
    col: int,
    pixel_size: float,
    grid_size: int,
) -> Rect:
    """从 (row, col) 出发，向四方扩展到最大无障碍矩形（使用原始 grid_map，不受已选矩形限制）"""
    n_rows, n_cols = grid_map.shape
    # 水平扩展（当前行）
    left_col = col
    while left_col > 0 and grid_map[row, left_col - 1] == 0:
        left_col -= 1
    right_col = col
    while right_col < n_cols - 1 and grid_map[row, right_col + 1] == 0:
        right_col += 1
    # 在 [left_col, right_col] 范围内垂直扩展
    top_row = row
    while top_row > 0 and np.all(grid_map[top_row - 1, left_col:right_col + 1] == 0):
        top_row -= 1
    bot_row = row
    while bot_row < n_rows - 1 and np.all(grid_map[bot_row + 1, left_col:right_col + 1] == 0):
        bot_row += 1
    # 转换为世界坐标（top_row=最小行索引=最高世界y，bot_row=最大行索引=最低世界y）
    x1, _ = coordinateGridToWorld(left_col,       0,               grid_size, pixel_size)
    x2, _ = coordinateGridToWorld(right_col + 1,  0,               grid_size, pixel_size)
    _, y1  = coordinateGridToWorld(0, n_rows - bot_row - 1,        grid_size, pixel_size)
    _, y2  = coordinateGridToWorld(0, n_rows - top_row,            grid_size, pixel_size)
    return Rect(x1, y1, x2, y2)


def randomExpansionSession(
    grid_map: np.ndarray,
    selected: list[Rect],
    pixel_size: float,
    grid_size: int,
    min_edge_length: float,
    max_consecutive_failures: int = 5,
    seed: int | None = None,
) -> list[Rect]:
    """在已选矩形之间的空白处随机采样起始点并扩展，补充覆盖空缺。
    连续失败 max_consecutive_failures 次后停止。"""
    import random
    rng = random.Random(seed)
    result = list(selected)
    consecutive_failures = 0

    while consecutive_failures < max_consecutive_failures:
        # 计算未被任何已选矩形覆盖的空闲栅格
        masked = maskRects(grid_map, result, pixel_size, grid_size)
        uncovered_rows, uncovered_cols = np.where(masked == 0)
        if len(uncovered_rows) == 0:
            break

        idx = rng.randint(0, len(uncovered_rows) - 1)
        row, col = int(uncovered_rows[idx]), int(uncovered_cols[idx])

        rect = _expandPoint(grid_map, row, col, pixel_size, grid_size)

        if rect.x2 - rect.x1 >= min_edge_length and rect.y2 - rect.y1 >= min_edge_length:
            result.append(rect)
            consecutive_failures = 0
        else:
            consecutive_failures += 1

    return result


def drawSegmentation(
    map: np.ndarray,
    selected_rectangles: list[Rect],
    pixel_size: float,
) -> np.ndarray:
    """在map中绘制选中的矩形"""
    for rect in selected_rectangles:
        _drawRect(map, rect, pixel_size)
        _drawBoundary(map, rect, pixel_size)
    return map

def _drawRect(
    map: np.ndarray,
    rect: Rect,
    pixel_size: float,
) -> None:
    """在map中绘制一个矩形"""
    x1, y1, x2, y2 = rect.x1, rect.y1, rect.x2, rect.y2
    x1_map, y1_map = coordinateWorldToMap((x1, y1), pixel_size)
    x2_map, y2_map = coordinateWorldToMap((x2, y2), pixel_size)
    map[y1_map:y2_map, x1_map:x2_map] = 128 # 128为灰色，表示覆盖
    return None

def _drawBoundary(
    map: np.ndarray,
    rect: Rect,
    pixel_size: float,
) -> None:
    """在map中绘制一个矩形的边界"""
    x1, y1, x2, y2 = rect.x1, rect.y1, rect.x2, rect.y2
    x1_map, y1_map = coordinateWorldToMap((x1, y1), pixel_size)
    x2_map, y2_map = coordinateWorldToMap((x2, y2), pixel_size)
    # numpy数组索引顺序是[行, 列]，即[y, x]
    map[y1_map:y2_map, x1_map] = 64  # 左边界
    map[y1_map:y2_map, x2_map - 1] = 64  # 右边界
    map[y1_map, x1_map:x2_map] = 64  # 下边界
    map[y2_map - 1, x1_map:x2_map] = 64  # 上边界
    return None

def main():
    pixel_size = 0.5
    grid_size = 1
    min_edge_length = 2.5
    alpha = 0.5
    beta = 2.0

    map = loadMap("map.png")
    grid_map = resterizeMap(map, grid_size)

    # Phase 1: row candidates
    candidates_row = generateCandidates(grid_map, min_edge_length, pixel_size, grid_size, 10)
    selected_row = greedySelection(candidates_row, map, pixel_size, alpha, beta)
    print(f"Phase 1 (行扫描): {len(candidates_row)} 个候选, 选出 {len(selected_row)} 个矩形")

    # Phase 2: col candidates on remaining free space
    masked_grid = maskRects(grid_map, selected_row, pixel_size, grid_size)
    candidates_col = generateColCandidates(masked_grid, min_edge_length, pixel_size, grid_size, 10)
    init_cover = buildCoverCount(selected_row, map.shape, pixel_size)
    selected_col = greedySelection(candidates_col, map, pixel_size, alpha, beta, initial_cover_count=init_cover)
    print(f"Phase 2 (列扫描): {len(candidates_col)} 个候选, 选出 {len(selected_col)} 个矩形")

    selected_rectangles = selected_row + selected_col
    print(f"合计选出 {len(selected_rectangles)} 个矩形")

    # Random expansion session: fill uncovered gaps
    selected_rectangles = randomExpansionSession(grid_map, selected_rectangles, pixel_size, grid_size, min_edge_length)
    print(f"随机扩展后: {len(selected_rectangles)} 个矩形")

    # Connectivity repair — local import avoids circular dependency at module level
    import build_graph as bg  # noqa: PLC0415
    # Bridge pool: use UNMASKED col candidates so bridges can span across row-selected regions
    candidates_col_unmasked = generateColCandidates(grid_map, min_edge_length, pixel_size, grid_size, 10)
    all_cands_bg = [bg.Rect(r.x1, r.y1, r.x2, r.y2) for r in candidates_row + candidates_col_unmasked]
    selected_bg  = [bg.Rect(r.x1, r.y1, r.x2, r.y2) for r in selected_rectangles]
    repaired_bg  = bg.repairConnectivity(selected_bg, all_cands_bg, min_edge_length)
    if len(repaired_bg) > len(selected_bg):
        print(f"连通性修复: 插入了 {len(repaired_bg) - len(selected_bg)} 个桥接矩形")
    repaired = [Rect(r.x1, r.y1, r.x2, r.y2) for r in repaired_bg]

    final_map = drawSegmentation(map.copy(), repaired, pixel_size)
    saveMap(final_map, "selected_rectangles.png")
    return None

if __name__ == "__main__":
    main()