# pathCoverage

一个面向可重复实验的覆盖路径规划仓库。算法实现保持冻结，地图生成、实验配置、主编排和结果可视化彼此解耦。

## 目录

```text
algorithms/                 # 冻结的算法实现与哈希清单
generate_map/               # 可替换的地图生成逻辑
src/main_pipeline.py        # 全局实验编排入口
src/configuration.py        # JSON 配置读取与校验
src/visualization.py        # 遍历顺序、覆盖视频和覆盖结果图
experiments/fields2cover_comparison/       # 与 Fields2Cover 的方法对比
experiments/segmentation_order_comparison/ # row/col 扫描顺序消融实验
experiments/connectivity_repair_ablation/  # 连通性补全配对消融实验
tests/                      # 结构、配置与可视化测试
```

## 快速开始

要求 Python 3.10 或更新版本。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.main_pipeline --config experiments/fields2cover_comparison/configs/experiment_map_test1.json
```

成功运行时，标准输出是完整 JSON；同一份内容保存到
`experiments/fields2cover_comparison/results/map_test1/metrics.json`。结果目录还包含：

- `segmentation.png`：矩形分割图；
- `traversal_order.png`：矩形中心间的有向遍历顺序，箭头标签表示转移步数；
- `coverage.mp4`：低采样位姿上的蓝色累计圆盘显示、黄色车体轨迹与红色机器人；
- `coverage_final.png`：与 JSON `work_disc` 指标使用同一参数化圆盘扫掠掩膜的最终覆盖图；
- `config_snapshot.json`：本次运行的完整配置快照。

统一机器人基线位于 `experiments/configs/robot_config_default.json`。新地图实验使用
`experiment_<map_name>.json`，且 `experiment_name` 必须等于地图文件名（不含扩展名）。
所有相对路径均相对于仓库根目录解析，因此从任意工作目录启动结果一致。

运行 row-only、col-only、row-then-col 三组候选扫描对比：

```bash
python experiments/segmentation_order_comparison/run_comparison.py
```

产物按 `experiments/segmentation_order_comparison/results/<map_name>/<strategy>/`
组织，每张地图的汇总指标位于其 `comparison_summary.json`。重复运行直接覆盖同名结果。

运行带/不带候选桥接的连通性补全消融：

```bash
python experiments/connectivity_repair_ablation/run_ablation.py
```

两组配置会校验补全前矩形哈希和起始矩形一致，并汇总连通分量、可达区域、
圆盘覆盖率及连续运动段时长。当前两张地图若未触发桥接，会被明确标记为自然负对照。

## 验证

```bash
python -m unittest discover -v
```

测试会验证冻结算法源码的 SHA-256，并核对连通性策略、受控配置、覆盖统计和最终蓝色像素，同时确认规划条带指标与连续运动时长不受低采样位姿密度影响。
