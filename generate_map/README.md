# Map generation

地图生成与覆盖算法刻意解耦，以便后续接入随机障碍物、BIM、Fields2Cover 或其他地图来源。

- `random_obstacles.py`：随机矩形障碍物地图。
- `fields2cover.py`：Fields2Cover GeoJSON 到占据栅格图。

生成目标必须显式属于某个实验的 `maps/`。例如：

```bash
python generate_map/fields2cover.py \
  ../Fields2Cover/experiments/maps/parkingGraph.geojson \
  experiments/fields2cover_comparison/maps/parkingGraph.png \
  --pixel-size 0.5
```

segmentation 对比实验维护自己的地图副本，位于
`experiments/segmentation_order_comparison/maps/`。
