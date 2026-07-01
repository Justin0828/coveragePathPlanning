# Segmentation comparison maps

这里保存 segmentation 扫描策略对比专用的地图副本，不从其他实验目录读取输入。
黑色（0）表示障碍物，白色（255）表示空闲区域，世界坐标原点位于左下角。

- `map_test1.png`：基础测试地图；
- `parkingGraph.png`：由 Fields2Cover 的 `parkingGraph.geojson` 转换得到；
- `parkingGraph.png.json`：`parkingGraph.png` 的坐标转换元数据。

每增加一张 `<map_name>.png`，应在三个策略的 `configs/<strategy>/` 下分别增加
`experiment_<map_name>.json`。三份配置仅允许
`segmentation.candidate_scan_strategy` 和输出目录中的策略名不同。
