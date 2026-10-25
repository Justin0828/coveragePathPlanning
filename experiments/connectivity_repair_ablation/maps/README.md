# Maps

本目录冻结连通性补全消融实验使用的地图：

- `map_test1.png`
- `parkingGraph.png`
- `repair_zigzag_chain.png`：三个错位房间通过窄桥相连，预修复矩形图稳定分裂为多个分量；
- `repair_four_hubs.png`：四个房间围绕中心十字连接区，候选池中存在小面积中心桥接矩形；
- `repair_ladder_bridge.png`：上下双廊和左右房间形成梯形结构，预修复结果保留两个主要分量。

两组变体读取同一文件，runner 同时校验配置与补全前矩形集合，确保连通性策略是唯一实验变量。
