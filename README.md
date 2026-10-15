# 通用低空无人机航线规划与安全评估平台

本项目由早期多个无人机路径规划算法原型重构而来，面向建筑密集环境中的低空航线方案生成。平台将GIS空间数据、OBJ三维模型、三维障碍建模、A*与RRT系列算法、安全校验和飞控任务包导出整合为一条完整任务链路。

```mermaid
flowchart LR
    A[GIS或OBJ三维数据] --> B[坐标统一与障碍建模]
    B --> C[任务参数配置]
    C --> D[运动学约束A*规划]
    D --> E[全航段安全复核]
    E --> F[GeoJSON KMZ 图件与报告]
```

## 项目解决的问题

传统算法Demo通常只展示“在几个方块之间画一条线”。本平台处理的是一项完整任务：

1. 导入真实建筑物和禁飞区空间数据。
2. 统一坐标系统，将WGS84经纬度转换为本地米制坐标。
3. 根据建筑高度或楼层数构建三维障碍体。
4. 配置起降点、最低航路高度、最高航高、安全距离和飞行速度。
5. 使用带进入方向状态的约束A*统一规划地面起点至地面终点。
6. 在搜索阶段限制爬升角和转向角，并对全航段进行安全复核。
7. 输出WGS84航线GeoJSON、航点CSV、三维成果图和任务报告。
8. 将验证后的航线导出为通用飞控任务JSON和KML/KMZ任务包。

## 两种场景输入

| 模式 | 输入 | 建模方式 | 适用场景 |
|---|---|---|---|
| GIS任务模式 | GeoJSON / OSM | 建筑多边形拉伸与安全膨胀 | 大范围航线方案与地理坐标输出 |
| 三维模型模式 | OBJ三角网模型 | 连通体包围盒或保守体素柱 | 局部精细场景与三维模型验证 |

OBJ模式不会简单地把模型顶点当作全部障碍物。平台支持按不连通建筑体生成保守包围体，也支持把三角面投影为体素柱并向场景基底填充，以降低模型表面空洞造成的漏检风险。

## 真实案例

仓库中附带河海大学江宁校区案例。建筑轮廓来自OpenStreetMap公开数据，共74栋建筑，其中56栋具有`height`或`building:levels`信息。建筑高度按以下优先级确定：

1. 使用`height`字段；
2. 使用`building:levels × 3.3m`估算；
3. 信息缺失时采用15m默认高度。

公开数据仅用于算法与流程验证，数据版权归OpenStreetMap贡献者所有，遵循ODbL 1.0。

当前工程将建筑物多边形的外接矩形拉伸为三维包围盒，用于快速且偏保守的碰撞检测；即使建筑轮廓不规则，也不会因为简化而低估障碍范围。后续可替换为精确多边形棱柱模型。

## 快速运行

```bash
python -m pip install -r requirements.txt
python scripts/prepare_osm_case.py
python run_mission.py --config config/hhu_delivery_mission.json
python scripts/run_campus_experiments.py
python run_obj_mission.py --config config/obj_demo_mission.json
python scripts/export_flight_package.py
python -m unittest discover -s tests -v
```

运行成功后生成：

```text
outputs/hhu_delivery_mission/
├── mission_report.json       # 结构化任务报告
├── mission_report.md         # 可阅读报告
├── mission_dashboard.html    # 可直接打开的任务成果页面
├── mission_route_3d.png      # 三维航线成果图
├── mission_route_topdown.png # 建筑与航线俯视图
├── route_local.csv           # 本地米制航点
├── route_wgs84.geojson       # WGS84航线，可在QGIS中打开
└── search_tree_3d.png        # RRT系列搜索过程

outputs/obj_demo_mission/
├── obj_mission_report.json   # OBJ模型规划结果
├── obj_route.csv             # 本地三维坐标航点
├── obj_route.obj             # 可与场景共同加载的OBJ折线
├── obj_route_3d.png          # 三维模型障碍与航线
└── obj_route_topdown.png     # 模型航线俯视图

outputs/hhu_delivery_mission/flight_package/
├── flight_mission.json       # 厂商无关任务交换格式
├── flight_route.kml          # 标准KML航线
└── flight_route.kmz          # 压缩任务包
```

## 飞控适配边界

`flight_export.py`输出的是经过校验的厂商无关任务交换格式，不会连接或控制真实飞行器。接入具体机型时，可以在此基础上增加DJI Cloud API、MAVLink或厂商SDK适配器，同时保留鉴权、设备状态检查和仿真验证环节。

## 代码结构

```text
uav_planning/
├── environment.py       # 三维障碍物、边界和碰撞检测
├── scenarios.py         # 场景与GeoJSON解析
├── planners.py          # 约束A*及几何基线算法
├── analysis.py          # 路径平滑、指标和安全校验
├── mission.py           # 起飞—巡航—降落任务编排与导出
├── mesh_adapter.py      # OBJ解析、连通体和保守体素柱建模
├── obj_mission.py       # 三维模型任务编排与成果导出
├── flight_export.py     # 通用飞控任务JSON与KML/KMZ
└── visualization.py     # 三维环境、路径和搜索树

legacy/                  # 原始算法代码和演进记录
scripts/                 # 真实空间数据准备脚本
config/                  # 任务参数
data/                    # 原始公开数据与标准化案例
tests/                   # 自动化测试
```

进一步材料：

- `METHODOLOGY.md`：论文依据、参数来源和能力边界；
- `PROJECT_ACCEPTANCE.md`：功能验收表与随包案例指标；
- `DATA_SOURCES.md`：公开数据来源、字段处理和许可说明；
- `legacy/PROVENANCE.md`：原始代码与重构模块的对应关系。

## 面试介绍

> 项目最初从A*、RRT、遗传算法和DQN的独立验证开始，后续参考城市无人机航迹规划论文，将校园任务重构为带方向状态的三维约束A*：地面起点至地面终点统一搜索，在搜索阶段检查最大爬升角、最大转向角、最低航路高度、最高航高和建筑安全距离，并对建筑上空飞越设置风险代价。河海大学江宁校区案例使用74栋OSM建筑数据，同时提供分辨率敏感性实验和可复现任务报告。

## 能力边界

平台定位于航线方案生成和飞行前风险评估。它不替代飞控系统的实时定位、动态避障和监管审批。实际接入具体无人机时，还需根据机型标定续航、爬升率、转弯半径和通信链路约束。

