# 通用低空无人机航线规划与安全评估平台

本项目由早期多个无人机路径规划算法原型重构而来，面向建筑密集环境中的低空航线方案生成。平台将GIS空间数据、三维障碍建模、A*与RRT系列算法、安全校验和成果导出整合为一条完整任务链路。

```mermaid
flowchart LR
    A[公开或业务GIS数据] --> B[坐标统一与三维建模]
    B --> C[任务参数配置]
    C --> D[A* RRT RRT*规划]
    D --> E[全航段安全复核]
    E --> F[GeoJSON CSV 图件与报告]
```

## 项目解决的问题

传统算法Demo通常只展示“在几个方块之间画一条线”。本平台处理的是一项完整任务：

1. 导入真实建筑物和禁飞区空间数据。
2. 统一坐标系统，将WGS84经纬度转换为本地米制坐标。
3. 根据建筑高度或楼层数构建三维障碍体。
4. 配置起降点、巡航高度、安全距离和飞行速度。
5. 使用A*、RRT或RRT*规划无碰撞路径。
6. 对起飞、巡航、降落全航段进行安全复核。
7. 输出WGS84航线GeoJSON、航点CSV、三维成果图和任务报告。

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
```

## 代码结构

```text
uav_planning/
├── environment.py       # 三维障碍物、边界和碰撞检测
├── scenarios.py         # 场景与GeoJSON解析
├── planners.py          # A*、RRT、RRT*
├── analysis.py          # 路径平滑、指标和安全校验
├── mission.py           # 起飞—巡航—降落任务编排与导出
└── visualization.py     # 三维环境、路径和搜索树

legacy/                  # 原始算法代码和演进记录
scripts/                 # 真实空间数据准备脚本
config/                  # 任务参数
data/                    # 原始公开数据与标准化案例
tests/                   # 自动化测试
```

进一步材料：

- `PROJECT_ACCEPTANCE.md`：功能验收表与随包案例指标；
- `DATA_SOURCES.md`：公开数据来源、字段处理和许可说明；
- `legacy/PROVENANCE.md`：原始代码与重构模块的对应关系。

## 面试介绍

> 项目最初从A*、RRT、遗传算法和DQN的独立验证开始，后续我将可稳定复现的A*、RRT和RRT*重构为统一规划内核，并补充了真实空间数据接入、坐标转换、建筑三维化、起降任务编排、安全校验及航线导出。平台可以读取建筑物和禁飞区GeoJSON，配置起终点、安全距离及巡航高度，输出可在QGIS中直接查看的WGS84航线和任务评估报告。河海大学江宁校区案例使用了74栋公开建筑数据，用于验证整套处理链路。

## 能力边界

平台定位于航线方案生成和飞行前风险评估。它不替代飞控系统的实时定位、动态避障和监管审批。实际接入具体无人机时，还需根据机型标定续航、爬升率、转弯半径和通信链路约束。

