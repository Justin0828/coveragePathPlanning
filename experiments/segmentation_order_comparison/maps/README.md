# Segmentation comparison maps

这里保存 segmentation 扫描策略对比专用的地图副本，不从其他实验目录读取输入。
黑色（0）表示障碍物，白色（255）表示空闲区域，世界坐标原点位于左下角。

- `map_test1.png`：基础测试地图；
- `parkingGraph.png`：由 Fields2Cover 的 `parkingGraph.geojson` 转换得到；
- `parkingGraph.png.json`：`parkingGraph.png` 的坐标转换元数据。
- `scan_comb_corridors.png`：中央横向作业厅连接多条错位竖井，形成梳状混合通道；
- `scan_mixed_weave.png`：不对称横厅、竖井和短支廊交织，形成紧凑的混合方向结构；
- `scan_offset_rooms.png`：主横厅配合偏置竖井和双侧短廊，形成非对称房间—通道组合。

这三张图均由 `search_directional_maps.py` 在固定的机器人、分割、连通性修复和轨迹
参数下筛选得到。筛选要求自由空间为单一四连通分量，并以“row-then-col 相对两个
单方向基线的全轨迹覆盖率优势”为主指标、以总任务时长优势为次指标；最终三张图上
row-then-col 都同时取得最高全轨迹覆盖率和最短总时长。

| 地图 | row-only（全轨迹 / s） | col-only（全轨迹 / s） | row-then-col（全轨迹 / s） |
|---|---:|---:|---:|
| `scan_comb_corridors` | 93.82% / 2065.09 | 93.94% / 2032.55 | **94.01% / 1986.42** |
| `scan_mixed_weave` | 95.26% / 1532.65 | 95.49% / 1575.63 | **95.70% / 1522.77** |
| `scan_offset_rooms` | 91.97% / 1496.73 | 91.51% / 1443.31 | **92.06% / 1374.38** |

每增加一张 `<map_name>.png`，应在三个策略的 `configs/<strategy>/` 下分别增加
`experiment_<map_name>.json`。三份配置仅允许
`segmentation.candidate_scan_strategy` 和输出目录中的策略名不同。
