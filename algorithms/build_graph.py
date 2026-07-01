from __future__ import annotations

import segmentation as seg
import rectangle_coverage as rc
from typing import List, Protocol, Tuple
import math
import random
from dataclasses import dataclass

@dataclass
class Rect:
    x1: float
    y1: float
    x2: float
    y2: float

class RobotCostConfig(Protocol):
    """图排序代价所需的机器人配置子集。"""

    disc_radius: float
    arm_length: float
    car_half_length: float
    pivot_to_car_center: float
    speed_limit: float
    angular_velocity_limit: float
    arm_angle_limit: tuple[float, float]

def isAdjacent(
    rect1: Rect, 
    rect2: Rect, 
    min_edge_length: float
) -> Tuple[bool, Tuple[float, float] | None]:
    # 没有重合部分时
    if rect1.x2 < rect2.x1 or rect1.x1 > rect2.x2 or rect1.y2 < rect2.y1 or rect1.y1 > rect2.y2:
        # 看相邻边长
        if rect1.x2 == rect2.x1 or rect1.x1 == rect2.x2:
            if min(abs(rect2.y2 - rect1.y1), abs(rect2.y1 - rect1.y2)) > min_edge_length:
                return True, (rect1.x2 if rect1.x2 == rect2.x1 else rect1.x1, (rect2.y2 + rect1.y1) / 2 if abs(rect2.y2 - rect1.y1) < abs(rect1.y2 - rect2.y1) else (rect2.y1 + rect1.y2) / 2)
            else:
                return False, None
        elif rect1.y2 == rect2.y1 or rect1.y1 == rect2.y2:
            if min(abs(rect2.x2 - rect1.x1), abs(rect2.x1 - rect1.x2)) > min_edge_length:
                return True, (rect1.y2 if rect1.y2 == rect2.y1 else rect1.y1, (rect2.x2 + rect1.x1) / 2 if abs(rect2.x2 - rect1.x1) < abs(rect1.x2 - rect2.x1) else (rect2.x1 + rect1.x2) / 2)
            else:
                return False, None
        else:
            return False, None
    # 有重合部分时
    else:
        intersect_rect = Rect(max(rect1.x1, rect2.x1), max(rect1.y1, rect2.y1), min(rect1.x2, rect2.x2), min(rect1.y2, rect2.y2))
        diagonal_length = math.sqrt((intersect_rect.x2 - intersect_rect.x1) ** 2 + (intersect_rect.y2 - intersect_rect.y1) ** 2)
        if diagonal_length > min_edge_length:
            return True, ((intersect_rect.x1 + intersect_rect.x2) / 2, (intersect_rect.y1 + intersect_rect.y2) / 2)
        else:
            return False, None

def getAdjacencyGraph(
    rect_list: List[Rect],
    min_edge_length: float,
) -> List[List[int]]:
    adjacency_graph = [[0 for _ in range(len(rect_list))] for _ in range(len(rect_list))]
    for i in range(len(rect_list)):
        for j in range(i + 1, len(rect_list)):
            is_adjacent, _ = isAdjacent(rect_list[i], rect_list[j], min_edge_length)
            if is_adjacent:
                adjacency_graph[i][j] = 1
                adjacency_graph[j][i] = 1
    return adjacency_graph

def _maxVariableArmLaneWidth(robot: RobotCostConfig) -> float:
    """Maximum cross-track width covered by one pass with a +/-90 degree arm swing."""
    lower, upper = robot.arm_angle_limit
    if lower > upper:
        raise ValueError("arm_angle_limit 下限不能大于上限。")
    if lower > 0 or upper < 0:
        raise ValueError("arm_angle_limit 必须包含 0 度以支持回收姿态。")
    swing = min(abs(lower), abs(upper), 90.0)
    return 2.0 * (robot.arm_length * math.sin(math.radians(swing)) + robot.disc_radius)


def _validateCostConfig(robot: RobotCostConfig) -> None:
    if robot.disc_radius <= 0:
        raise ValueError("robot.disc_radius 必须为正数。")
    if robot.arm_length <= 0:
        raise ValueError("robot.arm_length 必须为正数。")
    if robot.speed_limit <= 0:
        raise ValueError("robot.speed_limit 必须为正数。")
    if robot.angular_velocity_limit <= 0:
        raise ValueError("robot.angular_velocity_limit 必须为正数。")


def computeCoverageCost(rect: Rect, robot: RobotCostConfig) -> float:
    """Estimate boustrophedon coverage time from configured robot kinematics."""
    _validateCostConfig(robot)
    lane_width = _maxVariableArmLaneWidth(robot)
    width = rect.x2 - rect.x1
    height = rect.y2 - rect.y1
    if width >= height:
        num_lanes = max(1, math.ceil(height / lane_width))
        lane_length = width
    else:
        num_lanes = max(1, math.ceil(width / lane_width))
        lane_length = height
    travel_time = num_lanes * lane_length / robot.speed_limit
    turn_time = max(0, num_lanes - 1) * 180.0 / robot.angular_velocity_limit
    return travel_time + turn_time


def computeTransitionCost(
    rect_a: Rect,
    rect_b: Rect,
    robot: RobotCostConfig,
) -> float:
    """Estimate center-to-center transition time from configured kinematics."""
    _validateCostConfig(robot)
    ca = ((rect_a.x1 + rect_a.x2) / 2, (rect_a.y1 + rect_a.y2) / 2)
    cb = ((rect_b.x1 + rect_b.x2) / 2, (rect_b.y1 + rect_b.y2) / 2)
    dist = math.sqrt((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2)
    return (
        dist / robot.speed_limit
        + 180.0 / robot.angular_velocity_limit
    )  # travel + one re-orientation turn


def _traversal_order_cost(
    rect_list: List[Rect],
    order: List[int],
    robot: RobotCostConfig,
) -> float:
    """与旧版 DP 一致的总代价：首矩形覆盖 + 每条边转移 + 新矩形覆盖。"""
    if not order:
        return 0.0
    visited = {order[0]}
    total = computeCoverageCost(rect_list[order[0]], robot)
    for i in range(len(order) - 1):
        next_rect = order[i + 1]
        total += computeTransitionCost(rect_list[order[i]], rect_list[next_rect], robot)
        if next_rect not in visited:
            total += computeCoverageCost(rect_list[next_rect], robot)
            visited.add(next_rect)
    return total


def _minimum_spanning_tree_walk(
    rect_list: List[Rect],
    adjacency_graph: List[List[int]],
    start_rect: int,
    reachable_rects: List[int],
    robot: RobotCostConfig,
) -> List[int]:
    """构造一个允许回访节点、且一定覆盖整个连通分量的开放式游走。

    先在可达邻接图上用 Prim 算法构造转场时间最小生成树，再深度优先遍历。
    从起点到树上最远节点的路径放在最后访问，因此该路径无需折返；其余树边
    最多经过两次。复杂度为 O(V^2)，不会出现旧版 bitmask DP 的指数爆炸。
    """
    reachable_set = set(reachable_rects)
    if len(reachable_set) <= 1:
        return [start_rect]

    # Prim minimum spanning tree.  parent[v] is also the rooted-tree parent
    # used below to identify the final non-backtracking branch.
    in_tree = {start_rect}
    parent: dict[int, int] = {}
    tree: dict[int, list[int]] = {node: [] for node in reachable_rects}
    # For every node outside the tree, cache its cheapest edge into the tree.
    # Updating this cache after each insertion makes Prim O(V^2) on the
    # adjacency-matrix representation used by this project.
    best_connection: dict[int, tuple[float, int]] = {}

    def update_connections(node: int) -> None:
        for neighbor, is_adjacent in enumerate(adjacency_graph[node]):
            if not is_adjacent or neighbor not in reachable_set or neighbor in in_tree:
                continue
            edge = (
                computeTransitionCost(rect_list[node], rect_list[neighbor], robot),
                node,
            )
            if neighbor not in best_connection or edge < best_connection[neighbor]:
                best_connection[neighbor] = edge

    update_connections(start_rect)
    while len(in_tree) < len(reachable_set):
        candidates = [node for node in best_connection if node not in in_tree]
        if not candidates:
            # reachable_rects comes from a graph search, so this indicates an
            # inconsistent adjacency matrix rather than a normal route failure.
            raise ValueError("可达矩形的邻接图不一致，无法构造生成树。")

        neighbor = min(
            candidates,
            key=lambda candidate: (*best_connection[candidate], candidate),
        )
        _, node = best_connection[neighbor]
        in_tree.add(neighbor)
        parent[neighbor] = node
        tree[node].append(neighbor)
        tree[neighbor].append(node)
        update_connections(neighbor)

    # In a tree, an open traversal starting at start_rect is shortest when it
    # ends at the farthest node: every off-path edge is used twice, while the
    # start-to-end path is used only once.
    distances = {start_rect: 0.0}
    stack = [start_rect]
    while stack:
        node = stack.pop()
        for neighbor in tree[node]:
            if parent.get(node) == neighbor:
                continue
            distances[neighbor] = distances[node] + computeTransitionCost(
                rect_list[node], rect_list[neighbor], robot
            )
            stack.append(neighbor)
    end_rect = max(distances, key=lambda node: (distances[node], -node))

    # Mark the unique start-to-end branch.  At every node, visit side branches
    # first (with backtracking), then enter this branch without coming back.
    final_child: dict[int, int] = {}
    node = end_rect
    while node != start_rect:
        parent_node = parent[node]
        final_child[parent_node] = node
        node = parent_node

    order = [start_rect]

    def append_subtree(node: int, parent_node: int | None) -> None:
        children = sorted(neighbor for neighbor in tree[node] if neighbor != parent_node)
        last_child = final_child.get(node)
        for child in children:
            if child == last_child:
                continue
            order.append(child)
            append_subtree(child, node)
            order.append(node)
        if last_child is not None:
            order.append(last_child)
            append_subtree(last_child, node)

    append_subtree(start_rect, None)
    return order


def _getConnectedComponents(
    rect_list: List[Rect],
    min_edge_length: float,
) -> List[List[int]]:
    """Return list of connected components; each is a list of rect indices."""
    n = len(rect_list)
    adj = getAdjacencyGraph(rect_list, min_edge_length)
    visited = [False] * n
    components = []
    for start in range(n):
        if visited[start]:
            continue
        comp: List[int] = []
        stack = [start]
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            comp.append(node)
            for nb, connected in enumerate(adj[node]):
                if connected and not visited[nb]:
                    stack.append(nb)
        components.append(comp)
    return components


def repairConnectivity(
    selected: List[Rect],
    all_candidates: List[Rect],
    min_edge_length: float,
    grid_map=None,
    pixel_size: float | None = None,
    grid_size: int | None = None,
) -> List[Rect]:
    """Post-selection connectivity repair.

    Finds disconnected components in the adjacency graph of `selected` and
    iteratively inserts the smallest bridge rectangle from `all_candidates`
    that connects two different components, until the graph is fully connected
    or no bridge can be found.
    """
    if grid_map is not None and (pixel_size is None or grid_size is None):
        raise ValueError("检查桥接矩形障碍物时必须同时提供 pixel_size 和 grid_size。")

    def is_valid(rect: Rect) -> bool:
        if grid_map is None:
            return True
        return seg.isRectObstacleFree(
            seg.Rect(rect.x1, rect.y1, rect.x2, rect.y2),
            grid_map,
            pixel_size,
            grid_size,
        )

    invalid_selected = [rect for rect in selected if not is_valid(rect)]
    if invalid_selected:
        raise ValueError("已选矩形与障碍物相交，拒绝执行连通性修复。")

    result = list(selected)
    selected_keys = set((r.x1, r.y1, r.x2, r.y2) for r in result)
    bridge_pool = [r for r in all_candidates
                   if (r.x1, r.y1, r.x2, r.y2) not in selected_keys and is_valid(r)]

    while True:
        components = _getConnectedComponents(result, min_edge_length)
        if len(components) <= 1:
            break

        best_bridge: Rect | None = None
        best_area = float('inf')

        for cand in bridge_pool:
            # Count how many distinct components this candidate touches
            touched = set()
            for comp_idx, comp in enumerate(components):
                for rect_idx in comp:
                    adj, _ = isAdjacent(cand, result[rect_idx], min_edge_length)
                    if adj:
                        touched.add(comp_idx)
                        break  # one match per component is enough
            if len(touched) >= 2:
                area = (cand.x2 - cand.x1) * (cand.y2 - cand.y1)
                if area < best_area:
                    best_area = area
                    best_bridge = cand

        if best_bridge is None:
            break  # no bridge available — cannot connect further

        result.append(best_bridge)
        key = (best_bridge.x1, best_bridge.y1, best_bridge.x2, best_bridge.y2)
        selected_keys.add(key)
        bridge_pool = [r for r in bridge_pool
                       if (r.x1, r.y1, r.x2, r.y2) not in selected_keys]

    return result


def findReachableFromStart(
    start_rect: int,
    adjacency_graph: List[List[int]],
) -> List[int]:
    """从start_rect出发，找到所有可达的矩形"""
    visited = [False] * len(adjacency_graph) # 创建访问标记数组，初始都为false
    visited[start_rect] = True # 将起点标记为已访问
    stack = [start_rect] # 创建栈，将起点压入栈
    while stack:
        current_rect = stack.pop() # 从栈中弹出当前矩形
        for v_rect, is_related in enumerate(adjacency_graph[current_rect]): # v_rect是当前矩形可能的相邻矩形，is_related表示是否相邻
            if is_related and not visited[v_rect]: # 如果有边且未访问
                visited[v_rect] = True # 标记为已访问
                stack.append(v_rect) # 将相邻矩形压入栈
    return [i for i, v in enumerate(visited) if v] # 返回所有可达矩形的索引


# --- 以下为 findMinTimeTraversalOrder 的旧实现（bitmask + 优先队列），时间复杂度随 n 指数级；保留备查 ---
# import heapq
#
# def findMinTimeTraversalOrder(
#     rect_list: List[Rect],
#     adjacency_graph: List[List[int]],
#     start_rect: int,
# ) -> List[int]:
#     """找到从start_rect出发，遍历所有矩形的最短时间顺序"""
#     n = len(rect_list)
#     reachable_rects = findReachableFromStart(start_rect, adjacency_graph)
#     FULL_REACHABLE = 0
#     for rect in reachable_rects:
#         FULL_REACHABLE |= (1 << rect)
#
#     start_mask = 1 << start_rect
#     INF = float('inf')
#     dist = [[INF] * (1 << n) for _ in range(n)]
#     parent = [[None] * (1 << n) for _ in range(n)]
#
#     init_cost = computeCoverageCost(rect_list[start_rect])
#     dist[start_rect][start_mask] = init_cost
#     pq = [(init_cost, start_rect, start_mask)]
#
#     best_end = None
#     best_cost = None
#     while pq:
#         cost, curr, mask = heapq.heappop(pq)
#         if cost != dist[curr][mask]:
#             continue
#         if mask == FULL_REACHABLE:
#             if best_cost is None or cost < best_cost:
#                 best_cost = cost
#                 best_end = curr
#                 break
#         for rect_v, is_related in enumerate(adjacency_graph[curr]):
#             if not is_related:
#                 continue
#             is_visited = ((mask >> rect_v) & 1) == 1
#             step_cost = computeTransitionCost(rect_list[curr], rect_list[rect_v])
#             if not is_visited:
#                 step_cost += computeCoverageCost(rect_list[rect_v])
#             new_mask = mask | (1 << rect_v)
#             new_cost = cost + step_cost
#             if new_cost < dist[rect_v][new_mask]:
#                 dist[rect_v][new_mask] = new_cost
#                 parent[rect_v][new_mask] = (curr, mask)
#                 heapq.heappush(pq, (new_cost, rect_v, new_mask))
#     seq = []
#     curr, mask = best_end, FULL_REACHABLE
#     while True:
#         seq.append(curr)
#         p = parent[curr][mask]
#         if p is None:
#             break
#         curr, mask = p
#     seq.reverse()
#     return seq


def findMinTimeTraversalOrder(
    rect_list: List[Rect],
    adjacency_graph: List[List[int]],
    start_rect: int,
    robot: RobotCostConfig,
) -> List[int]:
    """从 start_rect 出发，沿邻接边访问所有可达矩形的近似最优顺序。

    同时生成两类候选：
    1. 加权贪心 + Warnsdorff 次序寻找无需回访的路径；
    2. 最小生成树开放式遍历，允许回访中转矩形并保证覆盖整个连通分量。

    最后按与旧版 bitmask DP 相同的代价模型选择较优路线。整体为多项式复杂度，
    避免 n 较大时 O(n * 2^n) 的状态空间爆炸。
    """
    n = len(rect_list)
    reachable_rects = findReachableFromStart(start_rect, adjacency_graph)
    if not reachable_rects:
        return []
    reachable_set = set(reachable_rects)
    target = len(reachable_set)
    if target == 1:
        return [start_rect]

    def _warnsdorff_degree(node: int, visited: set[int]) -> int:
        """未访问邻接点数越少，越该优先走（类似骑士巡游的 Warnsdorff 规则）。"""
        return sum(
            1
            for k in range(n)
            if adjacency_graph[node][k] and k in reachable_set and k not in visited
        )

    def _one_greedy_run(rng: random.Random, pick_top: int) -> List[int] | None:
        """pick_top=1 为纯贪心；>1 时在排序后的前 pick_top 名中随机取，以探索不同哈密顿路径。"""
        visited: set[int] = {start_rect}
        order: List[int] = [start_rect]
        current = start_rect
        while len(visited) < target:
            cands: list[tuple[float, int, int]] = []
            for j in range(n):
                if not adjacency_graph[current][j] or j not in reachable_set or j in visited:
                    continue
                # 所有可达矩形最终都要覆盖一次，覆盖成本之和与顺序无关；
                # 此处只比较会随顺序变化的转场成本。
                step = computeTransitionCost(rect_list[current], rect_list[j], robot)
                wd = _warnsdorff_degree(j, visited)
                cands.append((step, wd, j))
            if not cands:
                return None
            cands.sort(key=lambda t: (t[0], t[1]))
            k = min(pick_top, len(cands))
            chosen = rng.choice(cands[:k])[2]
            visited.add(chosen)
            order.append(chosen)
            current = chosen
        return order

    # 该候选允许重复节点，并对任意连通邻接图都能给出有效路线。即使图中
    # 不存在哈密顿路径，后面的贪心搜索失败也不会中断 pipeline。
    best_order = _minimum_spanning_tree_walk(
        rect_list, adjacency_graph, start_rect, reachable_rects, robot
    )
    best_cost = _traversal_order_cost(rect_list, best_order, robot)
    # 先纯贪心一次，再对前 2～4 名随机抽样多轮，在可行完整路径里取总代价最小
    attempts: list[tuple[int, int]] = [(1, 1)] + [(2, 128)] + [(3, 160)] + [(4, 160)]
    for pick_top, reps in attempts:
        for t in range(reps):
            rng = random.Random(42 + pick_top * 10_000 + t)
            tour = _one_greedy_run(rng, pick_top)
            if tour is None:
                continue
            c = _traversal_order_cost(rect_list, tour, robot)
            if c < best_cost:
                best_cost = c
                best_order = tour

    return best_order

def main():
    """与 main_pipeline.segment_map_into_rectangles 一致的分割流程，再跑邻接图与 TSP 顺序。"""
    from src.configuration import load_experiment_config

    robot = load_experiment_config().robot
    map_path = "map.png"
    grid_size = 1
    min_edge_length = 2.5
    pixel_size = 0.5
    alpha = 0.5
    beta = 2.0
    candidate_top_k = 10

    raw_map = seg.loadMap(map_path)
    grid_map = seg.resterizeMap(raw_map, grid_size)

    candidates_row = seg.generateCandidates(
        grid_map, min_edge_length, pixel_size, grid_size, candidate_top_k
    )
    selected_row = seg.greedySelection(
        candidates_row, raw_map, pixel_size, alpha, beta
    )

    masked_grid = seg.maskRects(grid_map, selected_row, pixel_size, grid_size)
    candidates_col = seg.generateColCandidates(
        masked_grid, min_edge_length, pixel_size, grid_size, candidate_top_k
    )
    init_cover = seg.buildCoverCount(selected_row, raw_map.shape, pixel_size)
    selected_col = seg.greedySelection(
        candidates_col,
        raw_map,
        pixel_size,
        alpha,
        beta,
        initial_cover_count=init_cover,
    )

    selected_rects = selected_row + selected_col
    print(
        f"[segmentation] row={len(selected_row)} col={len(selected_col)} "
        f"total={len(selected_rects)}"
    )

    selected_rects = seg.randomExpansionSession(
        grid_map, selected_rects, pixel_size, grid_size, min_edge_length
    )
    print(f"[random expansion] total={len(selected_rects)} rects")

    candidates_col_unmasked = seg.generateColCandidates(
        grid_map, min_edge_length, pixel_size, grid_size, candidate_top_k
    )
    all_candidates_bg = [
        Rect(r.x1, r.y1, r.x2, r.y2) for r in candidates_row + candidates_col_unmasked
    ]
    selected_bg = [Rect(r.x1, r.y1, r.x2, r.y2) for r in selected_rects]
    rect_list = repairConnectivity(
        selected_bg,
        all_candidates_bg,
        min_edge_length,
        grid_map=grid_map,
        pixel_size=pixel_size,
        grid_size=grid_size,
    )
    if len(rect_list) > len(selected_bg):
        print(f"[connectivity] inserted {len(rect_list) - len(selected_bg)} bridge rect(s)")

    if not rect_list:
        raise ValueError("分割结果为空，无法建图。")

    adjacency_graph = getAdjacencyGraph(rect_list, min_edge_length)
    start_rect = max(
        range(len(rect_list)),
        key=lambda i: (rect_list[i].x2 - rect_list[i].x1)
        * (rect_list[i].y2 - rect_list[i].y1),
    )
    reachable_rects = findReachableFromStart(start_rect, adjacency_graph)
    unreachable_rects = [i for i in range(len(rect_list)) if i not in reachable_rects]
    print(f"reachable_rects: {reachable_rects}")
    print(f"unreachable_rects: {unreachable_rects}")
    seq = findMinTimeTraversalOrder(rect_list, adjacency_graph, start_rect, robot)
    print(f"start_rect (largest area): {start_rect}")
    print(f"seq: {seq}")
    return None

if __name__ == "__main__":
    main()
