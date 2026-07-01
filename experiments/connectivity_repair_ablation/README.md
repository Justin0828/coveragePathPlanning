# Connectivity-repair ablation

这个实验只改变矩形分割后的连通性策略：

- `without_repair`：`connectivity.strategy = "none"`，直接使用随机补洞后的矩形；
- `with_repair`：`connectivity.strategy = "candidate_bridge"`，从完整候选池中插入桥接矩形。

候选生成、贪婪选择、随机补洞、起始矩形、机器人参数、遍历算法、局部覆盖和覆盖评估均保持一致。起始矩形在连通性步骤之前确定，两组运行还会校验补全前矩形集合的 SHA-256。若无补全组包含多个分量，后续遍历只覆盖起点所在分量，结果会显式记录不可达矩形和可达分区面积，不能仅凭更短的运行时长判断其更优。

当前仅使用仓库已有的 `map_test1` 和 `parkingGraph`。若某张地图没有插入桥接矩形，汇总中的 `ablation_triggered` 为 `false`，该地图按自然负对照解释。

运行全部配对实验：

```bash
python experiments/connectivity_repair_ablation/run_ablation.py
```

只运行一张地图：

```bash
python experiments/connectivity_repair_ablation/run_ablation.py --map parkingGraph
```

结果结构：

```text
results/
├── map_test1/
│   ├── without_repair/
│   ├── with_repair/
│   └── comparison_summary.json
├── parkingGraph/
│   ├── without_repair/
│   ├── with_repair/
│   └── comparison_summary.json
└── comparison_summary.csv
```

每个变体保留完整的 segmentation、遍历顺序、覆盖图片、视频和 `metrics.json`。配对汇总报告连通分量、桥接数量和面积、起点可达比例、作业抹盘覆盖率、连续运动段时长、单位覆盖面积时长，以及 `with_repair - without_repair` 的指标差值。
