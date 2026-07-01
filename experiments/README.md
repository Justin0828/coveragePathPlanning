# Experiments

- `connectivity_repair_ablation/`：候选桥接连通性补全的配对消融实验；
- `fields2cover_comparison/`：与 Fields2Cover 的方法对比；
- `segmentation_order_comparison/`：row/col 候选扫描顺序对比。
- `vehicle_speed_comparison/`：固定摆臂角速度、改变车速的覆盖率—覆盖时长实验。

四类对比实验保持相同的一级目录结构：

```text
experiments/
├── configs/
│   └── robot_config_default.json
├── fields2cover_comparison/
│   ├── configs/
│   ├── maps/
│   └── results/
├── segmentation_order_comparison/
│   ├── configs/
│   ├── maps/
│   ├── run_comparison.py
│   └── results/
├── connectivity_repair_ablation/
│   ├── configs/
│   ├── maps/
│   ├── run_ablation.py
│   └── results/
└── vehicle_speed_comparison/
    ├── configs/
    ├── maps/
    ├── run_comparison.py
    ├── plot_results.py
    └── results/
```

前三类实验共享 `configs/robot_config_default.json`；车速实验的多档机器人配置由该
基线派生，且除 `speed_limit` 外保持一致。各实验分别维护自己的 `maps/`，避免实验
输入跨目录依赖。实验名严格等于地图文件名（不含 `.png`）。重复运行相同地图和配置
时直接覆盖原结果。
