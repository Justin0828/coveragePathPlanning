# Vehicle-speed comparison

本实验固定地图、分割、连通修复、遍历顺序、机器人几何和摆臂角速度，只改变汽车
线速度，测量作业圆盘覆盖率与连续轨迹覆盖时长之间的权衡。

## 实验变量

`configs/robots/` 包含 `0.5、1.0、1.5、2.0、2.5、3.0 m/s` 六档机器人配置。
这些 JSON 从 `experiments/configs/robot_config_default.json` 派生，除
`speed_limit` 外完全一致，摆臂角速度固定为 `90 deg/s`。

实验配置使用：

```json
"motion": {"work_speed_policy": "commanded"}
```

该模式让覆盖作业段严格执行配置车速，以便高速组可以真实表现漏覆盖。其他三个实验
未配置该字段时仍默认使用 `coverage_safe`，即按照摆臂扫描能力自动限制作业车速。

每张地图首先使用最低车速配置计算一次基准矩形遍历顺序，全部速度组随后强制复用该
顺序。runner 还会校验机器人受控变量、矩形集合和遍历顺序，防止路线差异污染车速
实验。

## 运行

从仓库根目录执行：

```bash
# 全部地图、全部速度
python experiments/vehicle_speed_comparison/run_comparison.py

# 指定地图
python experiments/vehicle_speed_comparison/run_comparison.py --map parkingGraph

# 指定一个或多个速度
python experiments/vehicle_speed_comparison/run_comparison.py \
  --map map_test1 --speed 1.0 --speed 2.0
```

## 结果

```text
results/<map>/
├── speed_0p5/
│   ├── config_snapshot.json
│   ├── segmentation.png
│   ├── traversal_order.png
│   ├── coverage.mp4
│   ├── coverage_final.png
│   └── metrics.json
├── ...
├── comparison_summary.json
├── comparison_summary.csv
├── coverage_duration_curve.png
└── speed_effect_curve.png
```

主图 `coverage_duration_curve.png` 以总连续轨迹时长为横轴、最终作业圆盘覆盖率为
纵轴，并为每个点标注车速。`speed_effect_curve.png` 以车速为横轴，同时展示覆盖率
和总时长。

正式覆盖率使用 `coverage.work_disc.coverage_ratio`，时长使用
`trajectory.duration_seconds`；两者均来自连续运动段，不使用 Python 运行耗时。
