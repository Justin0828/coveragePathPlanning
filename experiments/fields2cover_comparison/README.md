# Fields2Cover comparison

这里保存原有的 Fields2Cover 方法对比实验：

- `configs/`：每张地图对应的实验配置；
- `maps/`：实验输入地图；
- `results/<map_name>/`：按地图名保存图像、视频和指标，重复运行直接覆盖。

统一机器人基线位于 `experiments/configs/robot_config_default.json`。

从仓库根目录运行：

```bash
python -m src.main_pipeline --config experiments/fields2cover_comparison/configs/experiment_map_test1.json
python -m src.main_pipeline --config experiments/fields2cover_comparison/configs/experiment_parkingGraph.json
```
