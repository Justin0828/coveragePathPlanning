# Segmentation candidate-scan comparison

这个实验与其他方法对比实验平行，单独研究候选矩形扫描策略：

- `row_only`：只在原始栅格地图上生成 row 候选；
- `col_only`：只在原始栅格地图上生成 col 候选；
- `row_then_col`：先选择 row 候选，再遮蔽其覆盖区域并从剩余区域生成 col 候选。

本实验拥有独立的 `maps/`，不读取 `fields2cover_comparison/maps/`。每张地图在三个
策略目录中各有一份 `experiment_<map_name>.json`；实验名等于地图文件名（不含扩展名）。
同一地图的三份配置除 `segmentation.candidate_scan_strategy` 和输出目录中的策略名外
保持一致。机器人统一使用 `experiments/configs/robot_config_default.json`。

从仓库根目录运行：

```bash
# 运行配置齐全的全部地图和三种策略
python experiments/segmentation_order_comparison/run_comparison.py

# 只运行指定地图；--map 可重复
python experiments/segmentation_order_comparison/run_comparison.py --map parkingGraph
```

结果写入：

```text
experiments/segmentation_order_comparison/results/
├── map_test1/
│   ├── row_only/
│   ├── col_only/
│   ├── row_then_col/
│   └── comparison_summary.json
└── parkingGraph/
    ├── row_only/
    ├── col_only/
    ├── row_then_col/
    └── comparison_summary.json
```

每个策略目录都包含 `config_snapshot.json`、`segmentation.png`、
`traversal_order.png`、`coverage.mp4`、`coverage_final.png` 和 `metrics.json`。
`metrics.json` 会分别记录贪婪选择、随机补洞后、连通修复后的矩形分割覆盖率；
`comparison_summary.json` 同时汇总分割覆盖率、作业圆盘覆盖率和全轨迹圆盘覆盖率。
正式圆盘指标由连续运动段的参数曲线计算，不依赖视频的 pose 采样率；
`coverage_final.png` 使用作业圆盘掩膜。

重复运行同一地图时覆盖上述同名目录和汇总文件，不创建时间戳或数字后缀。
