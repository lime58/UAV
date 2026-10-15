# 论文依据与建模边界

本项目不再把“升到楼顶以上后直飞”视为有效的城市低空路径规划。校园案例采用完整地面起点到地面终点搜索，并在搜索过程中检查爬升角、转向角、最低航路高度、最高航高、建筑安全距离和建筑上空飞越代价。

## 直接借鉴的方法

1. Rienecker, Hildebrand and Pfifer (2023), *Energy optimal 3D flight path planning for unmanned aerial vehicle in urban environments*, CEAS Aeronautical Journal, DOI: 10.1007/s13272-023-00666-x。
   - 三维离散A*；
   - 状态保留进入节点前的方向，用于搜索阶段约束航向变化；
   - 指出普通等距三维邻接会产生不合理的45°爬升；
   - 根据飞行器爬升能力调整水平和垂直离散尺度；
   - 将爬升能量、转弯能力和轨迹平滑纳入完整流程；
   - 使用多组任务，而不是用单条示例路线证明算法。

2. *Research on 3D layered visibility graph route network model and multi-objective path planning for UAVs in complex urban environments* (2025), Aerospace Science and Technology。
   - 城市低空三维分层航路；
   - 飞行高度限制与机动保护区；
   - 路径长度和路径置信度的多目标权衡；
   - 将安全、可靠性与公众接受度纳入航路选择。

3. Thoma et al. (2024), *Obstacle encounter probability dependent local path planner for UAV operation in urban environments*, CEAS Aeronautical Journal, DOI: 10.1007/s13272-024-00746-6。
   - “绕过建筑”与“飞越建筑”是需要比较代价的两种策略；
   - 飞行高度不能由固定偏好决定；
   - 评价同时报告任务成功、航程、时间和估算能耗。

## 本仓库的工程化选择

论文不会替具体校园任务给出唯一参数。`config/hhu_delivery_mission.json`中的最低航路高度、安全裕度、代价权重和离散尺度属于可复现实验配置，不声称是法规值或真实机型标定值。`scripts/run_campus_experiments.py`提供10、20、30和40米水平分辨率敏感性实验，避免把单一格网大小包装成普适结论。10米水平/2.5米垂直配置能将单步爬升角限制在约14.0°以内；更粗尺度仅作为效率与离散误差对照。

当前版本使用加权A*提高大范围校园场景的搜索效率，因此结果是可行的近似低代价路径，不宣称全局最优。建筑物仍采用OSM轮廓外接矩形的保守包围盒；项目属于仿真与飞行前方案验证，不代表完成了实机飞行或监管审批。
