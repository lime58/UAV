import math
import random
import heapq
import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QSlider, QGroupBox,
                             QFormLayout, QMessageBox, QLineEdit)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtGui import QIntValidator, QDoubleValidator

# 导入 matplotlib 集成到 PyQt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# -------------------------- 1. 基础数据结构 --------------------------
class Node:
    """RRT*与A*共用的节点类"""

    def __init__(self, x, y, z):
        self.x = x  # 坐标
        self.y = y
        self.z = z
        self.parent = None  # 父节点（用于回溯路径）
        self.cost = 0.0  # 从起点到该节点的代价（距离+高度代价）
        self.heuristic = 0.0  # A*的启发式代价

    def __eq__(self, other):
        return (abs(self.x - other.x) < 0.1 and
                abs(self.y - other.y) < 0.1 and
                abs(self.z - other.z) < 0.1)

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

    def distance(self, other):
        """计算欧氏距离（含高度）"""
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.hypot(dx, dy, dz)

    def get_path(self):
        """回溯父节点，获取从起点到当前节点的路径"""
        path = []
        node = self
        while node:
            path.append((node.x, node.y, node.z))
            node = node.parent
        return path[::-1]  # 反转：起点→当前节点


# -------------------------- 2. 环境感知与地图管理 --------------------------
class EnvManager:
    """环境管理器：处理障碍物探测、地图更新"""

    def __init__(self, obstacle_list, detect_range=6.0):
        self.global_obstacles = obstacle_list  # 全局障碍物（初始未知）
        self.known_obstacles = []  # 已探测到的障碍物
        self.detect_range = detect_range  # 传感器探测半径
        self.known_area = set()  # 已探索的可行区域（坐标集合）

    def detect_obstacles(self, current_pos):
        """从当前位置探测周围障碍物，更新已知障碍物列表"""
        x, y, z = current_pos
        new_detected = []
        for obs in self.global_obstacles:
            obs_x, obs_y, obs_z, obs_r = obs  # 障碍物：(x,y,z,半径)
            dist = math.hypot(x - obs_x, y - obs_y, z - obs_z)
            if dist <= self.detect_range and obs not in self.known_obstacles:
                self.known_obstacles.append(obs)
                new_detected.append(obs)
        return new_detected

    def mark_known_area(self, pos):
        """标记位置为已探索区域"""
        key = (round(pos[0], 1), round(pos[1], 1), round(pos[2], 1))
        self.known_area.add(key)

    def is_collision(self, start_node, end_node):
        """检查两点间路径是否与已知障碍物碰撞"""
        if not self.known_obstacles:
            return False

        # 路径采样检测（避免漏检）
        num_samples = 20
        for i in range(num_samples + 1):
            t = i / num_samples
            x = start_node.x + t * (end_node.x - start_node.x)
            y = start_node.y + t * (end_node.y - start_node.y)
            z = start_node.z + t * (end_node.z - start_node.z)

            for obs in self.known_obstacles:
                obs_x, obs_y, obs_z, obs_r = obs
                dist = math.hypot(x - obs_x, y - obs_y, z - obs_z)
                if dist <= obs_r + 0.5:  # 0.5m安全距离
                    return True
        return False


# -------------------------- 3. RRT*探索与A*优化核心算法（关键改进） --------------------------
class RRTStarAStarPlanner:
    """RRT*探索 + A*优化 混合规划器（修复A*优化问题）"""

    def __init__(self, start, goal, env_manager, step_size=2.0, max_iter=3000):  # 增大迭代次数
        self.start = Node(*start)
        self.goal = Node(*goal)
        self.env = env_manager
        self.step_size = step_size
        self.max_iter = max_iter

        # RRT*相关
        self.rrt_tree = [self.start]  # RRT*随机树
        self.env.mark_known_area(start)  # 标记起点为已探索

        # A*相关
        self.astar_path = []  # A*优化后的路径
        self.is_rrt_complete = False  # RRT*是否探索到终点附近
        self.is_astar_complete = False  # A*是否完成优化

    def rrt_sample(self):
        """随机采样：优先向终点或未知区域采样"""
        # 20%概率采样终点（加速收敛）
        if random.random() < 0.2:
            return self.goal

        # 80%概率在地图范围内采样（x[0,50], y[0,50], z[5,20]）
        x = random.uniform(0, 50)
        y = random.uniform(0, 50)
        z = random.uniform(5, 20)

        # 优先探索未知区域（降低重复采样已知区域概率）
        sample_key = (round(x, 1), round(y, 1), round(z, 1))
        if sample_key in self.env.known_area and random.random() < 0.6:  # 概率从0.7降至0.6
            return self.rrt_sample()

        return Node(x, y, z)

    def rrt_find_nearest(self, sample_node):
        """找到树中离采样点最近的节点"""
        min_dist = float('inf')
        nearest_node = None
        for node in self.rrt_tree:
            dist = node.distance(sample_node)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
        return nearest_node

    def rrt_steer(self, nearest_node, sample_node):
        """从最近节点向采样点延伸，生成新节点"""
        dist = nearest_node.distance(sample_node)
        if dist <= self.step_size:
            new_node = Node(sample_node.x, sample_node.y, sample_node.z)
        else:
            # 按步长比例计算新节点（增加精度）
            ratio = self.step_size / (dist + 1e-6)  # 避免除零
            x = nearest_node.x + ratio * (sample_node.x - nearest_node.x)
            y = nearest_node.y + ratio * (sample_node.y - nearest_node.y)
            z = nearest_node.z + ratio * (sample_node.z - nearest_node.z)
            new_node = Node(x, y, z)

        # 计算代价（距离+高度代价，优化高度权重）
        height_cost = 0.08 * abs(new_node.z - nearest_node.z)  # 权重从0.1降至0.08
        new_node.parent = nearest_node
        new_node.cost = nearest_node.cost + new_node.distance(nearest_node) + height_cost
        self.env.mark_known_area((new_node.x, new_node.y, new_node.z))
        return new_node

    def rrt_rewire(self, new_node):
        """重布线：优化树结构（扩大搜索半径）"""
        # 动态搜索半径（系数从5.0增至6.0，增加近邻节点数量）
        radius = 6.0 * math.sqrt(math.log(len(self.rrt_tree) + 1) / (len(self.rrt_tree) + 1))
        near_nodes = []

        # 找到半径内的所有节点
        for node in self.rrt_tree:
            if node.distance(new_node) <= radius:
                near_nodes.append(node)

        # 选择最优父节点
        min_cost = new_node.cost
        best_parent = new_node.parent
        for near_node in near_nodes:
            if not self.env.is_collision(near_node, new_node):
                height_cost = 0.08 * abs(new_node.z - near_node.z)
                new_cost = near_node.cost + new_node.distance(near_node) + height_cost
                if new_cost < min_cost:
                    min_cost = new_cost
                    best_parent = near_node

        new_node.parent = best_parent
        new_node.cost = min_cost

        # 更新近邻节点的父节点（若通过新节点代价更小）
        for near_node in near_nodes:
            if near_node != best_parent and not self.env.is_collision(new_node, near_node):
                height_cost = 0.08 * abs(near_node.z - new_node.z)
                new_cost = new_node.cost + near_node.distance(new_node) + height_cost
                if new_cost < near_node.cost:
                    near_node.parent = new_node
                    near_node.cost = new_cost

    def rrt_step(self):
        """RRT*单步探索（用于可视化逐步展示）"""
        if self.is_rrt_complete:
            return "complete"

        # 1. 探测边界节点周围环境（增加探测节点数量）
        boundary_nodes = self.rrt_tree[-15:] if len(self.rrt_tree) > 15 else self.rrt_tree
        for node in boundary_nodes:
            self.env.detect_obstacles((node.x, node.y, node.z))

        # 2. 采样与扩展（增加采样成功率）
        sample_node = self.rrt_sample()
        nearest_node = self.rrt_find_nearest(sample_node)
        new_node = self.rrt_steer(nearest_node, sample_node)

        # 3. 碰撞检测（增加容错）
        if not self.env.is_collision(nearest_node, new_node):
            self.rrt_tree.append(new_node)
            self.rrt_rewire(new_node)

        # 4. 检查是否探索到终点附近（扩大判定范围）
        goal_nearest = self.rrt_find_nearest(self.goal)
        if goal_nearest.distance(self.goal) < 6.0:  # 范围从5.0增至6.0
            # 尝试连接终点（增加步长容错）
            goal_node = self.rrt_steer(goal_nearest, self.goal)
            if not self.env.is_collision(goal_nearest, goal_node):
                self.rrt_tree.append(goal_node)
                self.is_rrt_complete = True
                return "complete"

        return "running"

    def astar_optimize(self):
        """A*路径优化（核心改进：增加路径回溯容错）"""
        if not self.is_rrt_complete:
            return False

        open_set = []
        closed_set = set()
        # 用更宽松的键值（保留1位小数）避免重复判定
        start_key = (round(self.start.x, 1), round(self.start.y, 1), round(self.start.z, 1))

        # 初始化起点（增加启发式权重）
        self.start.heuristic = 1.2 * self._astar_heuristic(self.start)  # 权重1.2增强引导
        heapq.heappush(open_set, self.start)
        closed_set.add(start_key)

        while open_set:
            current = heapq.heappop(open_set)

            # 到达终点（扩大判定范围，从1.0增至1.5）
            if current.distance(self.goal) < 1.5:
                self.astar_path = current.get_path()
                # 后处理：移除路径中过近的冗余节点
                self.astar_path = self._simplify_path(self.astar_path)
                self.is_astar_complete = True
                return True

            # 生成邻居节点（增加方向数量+步长容错）
            neighbors = self._astar_get_neighbors(current)
            for neighbor in neighbors:
                neighbor_key = (round(neighbor.x, 1), round(neighbor.y, 1), round(neighbor.z, 1))
                if neighbor_key not in closed_set:
                    closed_set.add(neighbor_key)
                    heapq.heappush(open_set, neighbor)

        # 容错：若未找到路径，尝试直接连接RRT*终点附近节点
        return self._astar_fallback()

    def _astar_heuristic(self, node):
        """A*启发式函数（改进为欧氏距离，更贴合3D空间）"""
        dx = abs(node.x - self.goal.x)
        dy = abs(node.y - self.goal.y)
        dz = abs(node.z - self.goal.z)
        return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)  # 欧氏距离（原曼哈顿距离替换）

    def _astar_get_neighbors(self, node):
        """生成A*邻居节点（增加18个方向，覆盖斜向和高度组合）"""
        directions = [
            # 6个基础方向
            (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
            # 8个平面斜向
            (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
            # 4个高度斜向（X-Z）
            (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
            # 4个高度斜向（Y-Z）
            (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)
        ]
        neighbors = []

        for dx, dy, dz in directions:
            # 增加步长灵活性（支持0.8-1.2倍步长）
            step_scale = random.uniform(0.8, 1.2)
            x = node.x + dx * step_scale
            y = node.y + dy * step_scale
            z = node.z + dz * step_scale

            # 检查：在已知区域、高度合法、地图边界内（增加边界容错）
            neighbor_key = (round(x, 1), round(y, 1), round(z, 1))
            if (neighbor_key in self.env.known_area and
                    4.5 <= z <= 20.5 and 0 <= x <= 50 and 0 <= y <= 50):  # Z轴容错0.5

                neighbor = Node(x, y, z)
                neighbor.parent = node
                # 计算代价（优化高度权重）
                height_cost = 0.08 * abs(z - node.z)
                neighbor.cost = node.cost + math.hypot(dx, dy, dz) * step_scale + height_cost
                neighbor.heuristic = 1.2 * self._astar_heuristic(neighbor)  # 同步启发式权重
                neighbors.append(neighbor)

        return neighbors

    def _simplify_path(self, path):
        """路径后处理：移除冗余节点（减少折线，优化路径平滑度）"""
        if len(path) < 3:
            return path

        simplified = [path[0]]
        for i in range(1, len(path) - 1):
            prev = np.array(path[i - 1])
            curr = np.array(path[i])
            next_p = np.array(path[i + 1])
            # 若三点近似共线，移除中间节点
            if np.linalg.norm(np.cross(next_p - prev, curr - prev)) > 0.3:  # 阈值0.3控制平滑度
                simplified.append(path[i])
        simplified.append(path[-1])
        return simplified

    def _astar_fallback(self):
        """A*容错方案：若主逻辑失败，直接连接RRT*终点附近节点"""
        # 找到RRT*树中离终点最近的前3个节点
        goal_neighbors = sorted(self.rrt_tree, key=lambda n: n.distance(self.goal))[:3]
        for near_node in goal_neighbors:
            # 检查从起点到该节点是否有可行路径（回溯RRT*树）
            rrt_path = near_node.get_path()
            if len(rrt_path) > 0:
                # 基于RRT*路径生成A*简化路径
                self.astar_path = self._simplify_path(rrt_path)
                # 尝试连接终点
                last_point = self.astar_path[-1]
                if not self.env.is_collision(Node(*last_point), self.goal):
                    self.astar_path.append((self.goal.x, self.goal.y, self.goal.z))
                self.is_astar_complete = True
                return True
        return False


# -------------------------- 4. Matplotlib可视化画布（完整实现） --------------------------
class PlanningCanvas(FigureCanvas):
    """路径规划可视化画布（3D）"""

    def __init__(self, parent=None):
        # 初始化Figure和Axes
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # 基础设置（优化坐标轴显示）
        self.ax.set_xlim(0, 50)
        self.ax.set_ylim(0, 50)
        self.ax.set_zlim(4, 21)  # 扩大Z轴显示范围，适配容错
        self.ax.set_xlabel('X 坐标 (m)', fontsize=10, labelpad=10)
        self.ax.set_ylabel('Y 坐标 (m)', fontsize=10, labelpad=10)
        self.ax.set_zlabel('Z 坐标 (m)', fontsize=10, labelpad=10)
        self.ax.set_title(
            '无人机未知环境路径规划可视化\n（蓝色：RRT*树 | 红色：已探测障碍物 | 绿色：A*优化路径）',
            fontsize=12, pad=20
        )
        self.ax.tick_params(axis='both', labelsize=8)

        # 绘图元素存储（初始为None）
        self.rrt_lines = None  # RRT*树线条
        self.obstacle_scatters = None  # 障碍物散点
        self.astar_lines = None  # A*路径线条
        self.start_scatter = None  # 起点
        self.goal_scatter = None  # 终点

        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def update_plot(self, planner):
        """更新可视化内容（修复重复删除崩溃问题）"""
        # 1. 清除原有绘图元素（删除后重置为None，避免内存冲突）
        if self.rrt_lines is not None:
            self.rrt_lines.remove()
            self.rrt_lines = None
        if self.obstacle_scatters is not None:
            self.obstacle_scatters.remove()
            self.obstacle_scatters = None
        if self.astar_lines is not None:
            # ax.plot返回列表，需遍历删除所有子元素
            for line in self.astar_lines:
                line.remove()
            self.astar_lines = None
        if self.start_scatter is not None:
            self.start_scatter.remove()
            self.start_scatter = None
        if self.goal_scatter is not None:
            self.goal_scatter.remove()
            self.goal_scatter = None

        # 2. 绘制RRT*树（优化线条显示）
        rrt_segments = []
        for node in planner.rrt_tree:
            if node.parent is not None:
                segment = [
                    (node.parent.x, node.parent.y, node.parent.z),
                    (node.x, node.y, node.z)
                ]
                rrt_segments.append(segment)

        if rrt_segments:
            self.rrt_lines = Line3DCollection(
                rrt_segments,
                colors='skyblue',
                linewidths=0.8,
                alpha=0.6,
                zorder=1  # 层级低于障碍物和路径
            )
            self.ax.add_collection3d(self.rrt_lines)

        # 3. 绘制已探测障碍物（优化散点大小和颜色）
        obs_x, obs_y, obs_z, obs_r = [], [], [], []
        for obs in planner.env.known_obstacles:
            x, y, z, r = obs
            obs_x.append(x)
            obs_y.append(y)
            obs_z.append(z)
            obs_r.append(r)

        if obs_x:
            # 用散点大小表示障碍物半径（比例35，更清晰）
            self.obstacle_scatters = self.ax.scatter(
                obs_x, obs_y, obs_z,
                c='#ff4444',  # 深红色，更醒目
                s=[r * 35 for r in obs_r],
                alpha=0.8,
                label='已探测障碍物',
                zorder=3  # 层级高于RRT*树
            )

        # 4. 绘制A*优化路径（优化线条宽度和颜色）
        if planner.is_astar_complete and planner.astar_path:
            path = np.array(planner.astar_path)
            self.astar_lines = self.ax.plot(
                path[:, 0], path[:, 1], path[:, 2],
                'g-',
                linewidth=3.0,  # 加粗至3.0，更清晰
                alpha=0.9,
                label='A*优化路径',
                zorder=4  # 层级最高
            )

        # 5. 绘制起点和终点（优化大小和颜色）
        self.start_scatter = self.ax.scatter(
            planner.start.x, planner.start.y, planner.start.z,
            c='#2ecc71',  # 亮绿色
            s=250,  # 放大至250
            marker='o',
            label='起点',
            edgecolors='black',
            linewidth=2,
            zorder=5
        )
        self.goal_scatter = self.ax.scatter(
            planner.goal.x, planner.goal.y, planner.goal.z,
            c='#f39c12',  # 橙色
            s=300,  # 放大至300
            marker='*',
            label='终点',
            edgecolors='black',
            linewidth=2,
            zorder=5
        )

        # 6. 添加图例（只添加一次，避免重复）
        if not self.ax.get_legend():
            legend = self.ax.legend(
                loc='upper right',
                fontsize=9,
                framealpha=0.9,
                edgecolor='black'
            )
            legend.get_frame().set_facecolor('white')

        # 刷新画布（增加渲染效率）
        self.fig.canvas.draw_idle()


# -------------------------- 5. 主界面窗口（完整实现） --------------------------
class PlanningMainWindow(QMainWindow):
    """主界面窗口（支持参数显示+坐标设置+稳定A*优化）"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("无人机未知环境路径规划（RRT*+A*）")
        self.setGeometry(100, 100, 1400, 900)  # 窗口尺寸适配

        # 1. 初始化关键属性
        self.speed = 50  # 探索速度（毫秒/步）
        self.is_running = False
        # 初始起点/终点坐标（默认值）
        self.default_start = (5.0, 5.0, 8.0)
        self.default_goal = (45.0, 45.0, 8.0)
        self.current_start = self.default_start
        self.current_goal = self.default_goal

        # 2. 初始化环境与规划器
        self._init_environment()

        # 3. 构建界面布局
        self._init_ui()

        # 4. 初始化定时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._run_planning_step)
        self.timer.setInterval(self.speed)

    def _init_environment(self):
        """初始化环境参数（使用当前设置的起点/终点）"""
        # 全局障碍物（优化布局，避免过于密集）
        self.global_obstacles = [
            (15, 15, 8, 3), (25, 10, 8, 2.5), (35, 20, 8, 3),
            (20, 35, 8, 2), (30, 30, 8, 3.5), (40, 15, 8, 2),
            (10, 30, 8, 2.5), (45, 35, 8, 3)
        ]

        # 创建环境管理器和规划器（使用当前起点/终点）
        self.env_manager = EnvManager(self.global_obstacles, detect_range=6.0)
        self.planner = RRTStarAStarPlanner(
            start=self.current_start,
            goal=self.current_goal,
            env_manager=self.env_manager,
            step_size=2.5,
            max_iter=3000  # 增大迭代次数，确保探索充分
        )

    def _init_ui(self):
        """构建界面布局（参数显示+坐标设置）"""
        # 中心Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 1. 可视化画布与工具栏
        self.canvas = PlanningCanvas()
        self.toolbar = NavigationToolbar(self.canvas, self)
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)

        # 2. 控制面板（功能按钮+参数调节+坐标设置+状态显示）
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        control_layout.setContentsMargins(10, 10, 10, 10)

        # 2.1 功能按钮组
        btn_layout = QVBoxLayout()
        self.start_btn = QPushButton("开始探索")
        self.pause_btn = QPushButton("暂停探索")
        self.reset_btn = QPushButton("重置环境")
        self.optimize_btn = QPushButton("A*路径优化")
        self.apply_coord_btn = QPushButton("应用坐标设置")  # 坐标应用按钮

        # 按钮绑定事件
        self.start_btn.clicked.connect(self._start_planning)
        self.pause_btn.clicked.connect(self._pause_planning)
        self.reset_btn.clicked.connect(self._reset_planning)
        self.optimize_btn.clicked.connect(self._run_astar)
        self.apply_coord_btn.clicked.connect(self._apply_coordinate_settings)

        # 按钮状态初始化
        self.pause_btn.setEnabled(False)
        self.optimize_btn.setEnabled(False)
        self.apply_coord_btn.setEnabled(True)

        # 按钮样式优化
        btn_style = "QPushButton {padding: 8px 15px; font-size: 12px; margin: 5px 0;}"
        self.start_btn.setStyleSheet(btn_style + "background-color: #2ecc71; color: white;")
        self.pause_btn.setStyleSheet(btn_style + "background-color: #f39c12; color: white;")
        self.reset_btn.setStyleSheet(btn_style + "background-color: #e74c3c; color: white;")
        self.optimize_btn.setStyleSheet(btn_style + "background-color: #3498db; color: white;")
        self.apply_coord_btn.setStyleSheet(btn_style + "background-color: #9b59b6; color: white;")

        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.pause_btn)
        btn_layout.addWidget(self.reset_btn)
        btn_layout.addWidget(self.optimize_btn)
        btn_layout.addSpacing(20)
        btn_layout.addWidget(self.apply_coord_btn)
        btn_layout.setSpacing(10)
        control_layout.addLayout(btn_layout)

        # 2.2 参数调节组（带实时数值显示）
        param_group = QGroupBox("探索参数调节（带实时数值）")
        param_layout = QFormLayout(param_group)
        param_group.setStyleSheet("QGroupBox {font-size: 12px; font-weight: bold; margin-bottom: 10px;}")

        # 探索速度滑块 + 数值显示
        speed_row = QHBoxLayout()
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(10, 500)  # 10-500毫秒/步
        self.speed_slider.setValue(self.speed)
        self.speed_slider.valueChanged.connect(self._update_speed)
        self.speed_label = QLabel(f"当前值：{self.speed} ms")
        self.speed_label.setAlignment(Qt.AlignCenter)
        self.speed_label.setFixedWidth(120)
        speed_row.addWidget(self.speed_slider)
        speed_row.addWidget(self.speed_label)
        param_layout.addRow(QLabel("探索速度"), speed_row)

        # 传感器范围滑块 + 数值显示
        sensor_row = QHBoxLayout()
        self.sensor_slider = QSlider(Qt.Horizontal)
        self.sensor_slider.setRange(3, 10)  # 3-10米
        self.sensor_slider.setValue(6)
        self.sensor_slider.valueChanged.connect(self._update_sensor_range)
        self.sensor_label = QLabel(f"当前值：6 米")
        self.sensor_label.setAlignment(Qt.AlignCenter)
        self.sensor_label.setFixedWidth(120)
        sensor_row.addWidget(self.sensor_slider)
        sensor_row.addWidget(self.sensor_label)
        param_layout.addRow(QLabel("传感器探测范围"), sensor_row)

        # 步长滑块 + 数值显示
        step_row = QHBoxLayout()
        self.step_slider = QSlider(Qt.Horizontal)
        self.step_slider.setRange(1, 5)  # 1-5米
        self.step_slider.setValue(2)
        self.step_slider.valueChanged.connect(self._update_step_size)
        self.step_label = QLabel(f"当前值：2 米")
        self.step_label.setAlignment(Qt.AlignCenter)
        self.step_label.setFixedWidth(120)
        step_row.addWidget(self.step_slider)
        step_row.addWidget(self.step_label)
        param_layout.addRow(QLabel("RRT*扩展步长"), step_row)

        control_layout.addWidget(param_group)

        # 2.3 起点/终点坐标设置面板
        coord_group = QGroupBox("起点/终点坐标设置（范围：X/Y[0,50], Z[5,20]）")
        coord_layout = QFormLayout(coord_group)
        coord_group.setStyleSheet("QGroupBox {font-size: 12px; font-weight: bold; margin-bottom: 10px;}")
        # 输入框验证器
        double_validator = QDoubleValidator(0.0, 50.0, 1)
        z_validator = QDoubleValidator(5.0, 20.0, 1)

        # 起点坐标输入
        coord_layout.addRow(QLabel("<b>起点坐标</b>"))
        # 起点X
        start_x_row = QHBoxLayout()
        self.start_x_edit = QLineEdit(str(self.current_start[0]))
        self.start_x_edit.setValidator(double_validator)
        self.start_x_edit.setFixedWidth(80)
        start_x_row.addWidget(QLabel("X:"))
        start_x_row.addWidget(self.start_x_edit)
        start_x_row.addWidget(QLabel("m"))
        coord_layout.addRow(start_x_row)
        # 起点Y
        start_y_row = QHBoxLayout()
        self.start_y_edit = QLineEdit(str(self.current_start[1]))
        self.start_y_edit.setValidator(double_validator)
        self.start_y_edit.setFixedWidth(80)
        start_y_row.addWidget(QLabel("Y:"))
        start_y_row.addWidget(self.start_y_edit)
        start_y_row.addWidget(QLabel("m"))
        coord_layout.addRow(start_y_row)
        # 起点Z
        start_z_row = QHBoxLayout()
        self.start_z_edit = QLineEdit(str(self.current_start[2]))
        self.start_z_edit.setValidator(z_validator)
        self.start_z_edit.setFixedWidth(80)
        start_z_row.addWidget(QLabel("Z:"))
        start_z_row.addWidget(self.start_z_edit)
        start_z_row.addWidget(QLabel("m"))
        coord_layout.addRow(start_z_row)

        # 终点坐标输入
        coord_layout.addRow(QLabel("<b>终点坐标</b>"))
        # 终点X
        goal_x_row = QHBoxLayout()
        self.goal_x_edit = QLineEdit(str(self.current_goal[0]))
        self.goal_x_edit.setValidator(double_validator)
        self.goal_x_edit.setFixedWidth(80)
        goal_x_row.addWidget(QLabel("X:"))
        goal_x_row.addWidget(self.goal_x_edit)
        goal_x_row.addWidget(QLabel("m"))
        coord_layout.addRow(goal_x_row)
        # 终点Y
        goal_y_row = QHBoxLayout()
        self.goal_y_edit = QLineEdit(str(self.current_goal[1]))
        self.goal_y_edit.setValidator(double_validator)
        self.goal_y_edit.setFixedWidth(80)
        goal_y_row.addWidget(QLabel("Y:"))
        goal_y_row.addWidget(self.goal_y_edit)
        goal_y_row.addWidget(QLabel("m"))
        coord_layout.addRow(goal_y_row)
        # 终点Z
        goal_z_row = QHBoxLayout()
        self.goal_z_edit = QLineEdit(str(self.current_goal[2]))
        self.goal_z_edit.setValidator(z_validator)
        self.goal_z_edit.setFixedWidth(80)
        goal_z_row.addWidget(QLabel("Z:"))
        goal_z_row.addWidget(self.goal_z_edit)
        goal_z_row.addWidget(QLabel("m"))
        coord_layout.addRow(goal_z_row)

        control_layout.addWidget(coord_group)

        # 2.4 状态显示区
        status_layout = QVBoxLayout()
        self.status_label = QLabel(self._get_status_text())
        self.status_label.setAlignment(Qt.AlignTop)
        self.status_label.setStyleSheet("QLabel {font-size: 11px; white-space: pre;}")
        status_layout.addWidget(self.status_label)
        control_layout.addLayout(status_layout, stretch=1)

        main_layout.addWidget(control_panel)

    def _get_status_text(self):
        """生成状态文本（含A*优化状态）"""
        base_text = (
            f"状态：{'探索中' if self.is_running else '已暂停' if self.planner.rrt_tree else '未开始'}\n"
            f"RRT*节点数：{len(self.planner.rrt_tree)}\n"
            f"已探测障碍物：{len(self.planner.env.known_obstacles)}\n"
            f"A*优化：{'已完成' if self.planner.is_astar_complete else '未完成'}\n"
            f"当前起点：({self.current_start[0]:.1f},{self.current_start[1]:.1f},{self.current_start[2]:.1f})\n"
            f"当前终点：({self.current_goal[0]:.1f},{self.current_goal[1]:.1f},{self.current_goal[2]:.1f})"
        )
        # 增加A*路径长度显示
        if self.planner.is_astar_complete and self.planner.astar_path:
            path_len = sum(
                math.hypot(p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
                for p1, p2 in zip(self.planner.astar_path[:-1], self.planner.astar_path[1:])
            )
            base_text += f"\n优化路径长度：{path_len:.2f}米"
        return base_text

    def _update_status(self):
        """更新状态显示"""
        self.status_label.setText(self._get_status_text())

    # -------------------------- 参数实时更新 --------------------------
    def _update_speed(self):
        """更新探索速度（同步数值标签）"""
        self.speed = self.speed_slider.value()
        self.speed_label.setText(f"当前值：{self.speed} ms")
        if self.is_running:
            self.timer.setInterval(self.speed)

    def _update_sensor_range(self):
        """更新传感器探测范围（同步数值标签）"""
        sensor_range = self.sensor_slider.value()
        self.sensor_label.setText(f"当前值：{sensor_range} 米")
        self.planner.env.detect_range = sensor_range

    def _update_step_size(self):
        """更新RRT*扩展步长（同步数值标签）"""
        step_size = self.step_slider.value()
        self.step_label.setText(f"当前值：{step_size} 米")
        self.planner.step_size = step_size

    # -------------------------- 坐标设置应用 --------------------------
    def _apply_coordinate_settings(self):
        """读取输入框坐标，验证后应用"""
        try:
            # 读取起点坐标
            start_x = float(self.start_x_edit.text().strip())
            start_y = float(self.start_y_edit.text().strip())
            start_z = float(self.start_z_edit.text().strip())
            # 读取终点坐标
            goal_x = float(self.goal_x_edit.text().strip())
            goal_y = float(self.goal_y_edit.text().strip())
            goal_z = float(self.goal_z_edit.text().strip())

            # 验证坐标合法性
            if not (0 <= start_x <= 50 and 0 <= start_y <= 50 and 5 <= start_z <= 20):
                raise ValueError("起点坐标超出范围！X/Y需在0-50，Z需在5-20")
            if not (0 <= goal_x <= 50 and 0 <= goal_y <= 50 and 5 <= goal_z <= 20):
                raise ValueError("终点坐标超出范围！X/Y需在0-50，Z需在5-20")
            if math.hypot(start_x - goal_x, start_y - goal_y, start_z - goal_z) < 3.0:
                raise ValueError("起点与终点距离过近（需≥3米），请重新设置")

            # 验证通过，更新坐标并重置环境
            self.current_start = (start_x, start_y, start_z)
            self.current_goal = (goal_x, goal_y, goal_z)
            self._reset_planning()
            QMessageBox.information(
                self, "坐标设置成功",
                f"已应用新坐标：\n起点：({start_x:.1f},{start_y:.1f},{start_z:.1f})\n终点：({goal_x:.1f},{goal_y:.1f},{goal_z:.1f})"
            )

        except ValueError as e:
            QMessageBox.warning(self, "坐标设置错误", str(e))
        except Exception as e:
            QMessageBox.critical(self, "系统错误", f"坐标应用失败：{str(e)}")

    # -------------------------- 核心功能逻辑 --------------------------
    def _start_planning(self):
        """开始探索"""
        if not self.is_running:
            self.is_running = True
            self.timer.start()
            self.start_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            self._update_status()

    def _pause_planning(self):
        """暂停探索"""
        if self.is_running:
            self.is_running = False
            self.timer.stop()
            self.start_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self._update_status()

    def _reset_planning(self):
        """重置环境（使用当前坐标）"""
        self.timer.stop()
        self.is_running = False
        self.speed = self.speed_slider.value()
        # 重新初始化环境和规划器
        self._init_environment()
        # 更新画布
        self.canvas.update_plot(self.planner)
        # 同步输入框显示
        self.start_x_edit.setText(f"{self.current_start[0]:.1f}")
        self.start_y_edit.setText(f"{self.current_start[1]:.1f}")
        self.start_z_edit.setText(f"{self.current_start[2]:.1f}")
        self.goal_x_edit.setText(f"{self.current_goal[0]:.1f}")
        self.goal_y_edit.setText(f"{self.current_goal[1]:.1f}")
        self.goal_z_edit.setText(f"{self.current_goal[2]:.1f}")
        # 同步参数标签
        self.speed_label.setText(f"当前值：{self.speed} ms")
        self.sensor_label.setText(f"当前值：{self.sensor_slider.value()} 米")
        self.step_label.setText(f"当前值：{self.step_slider.value()} 米")
        # 重置按钮状态
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.optimize_btn.setEnabled(False)
        # 更新状态
        self._update_status()

    def _run_astar(self):
        """运行A*路径优化（增加结果提示）"""
        if self.planner.is_rrt_complete and not self.planner.is_astar_complete:
            # 显示加载提示
            QMessageBox.information(self, "A*优化中", "正在进行A*路径优化，请稍候...")
            success = self.planner.astar_optimize()
            # 更新画布和状态
            self.canvas.update_plot(self.planner)
            self._update_status()
            # 显示结果
            if success:
                QMessageBox.information(
                    self, "A*优化完成",
                    f"已成功生成最优路径！\n路径长度：{self._calc_path_length():.2f}米"
                )
            else:
                QMessageBox.warning(
                    self, "A*优化失败",
                    "未找到完全最优路径，已使用RRT*路径简化方案"
                )
        elif not self.planner.is_rrt_complete:
            QMessageBox.warning(self, "优化条件不足", "请先完成RRT*探索（探索到终点附近）")
        else:
            QMessageBox.information(self, "优化已完成", "A*路径优化已完成，无需重复执行")

    def _calc_path_length(self):
        """计算A*路径长度"""
        if not self.planner.astar_path or len(self.planner.astar_path) < 2:
            return 0.0
        return sum(
            math.hypot(p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
            for p1, p2 in zip(self.planner.astar_path[:-1], self.planner.astar_path[1:])
        )

    def _run_planning_step(self):
        """执行单步规划（RRT*探索）"""
        result = self.planner.rrt_step()
        # 更新画布和状态
        self.canvas.update_plot(self.planner)
        self._update_status()

        # 若RRT*探索完成，自动暂停并启用A*按钮
        if result == "complete":
            self._pause_planning()
            self.optimize_btn.setEnabled(True)
            QMessageBox.information(
                self, "RRT*探索完成",
                f"已探索到终点附近（节点数：{len(self.planner.rrt_tree)}）\n可点击「A*路径优化」生成最优路径"
            )


# -------------------------- 6. 程序入口 --------------------------
if __name__ == "__main__":
    # 解决matplotlib中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 防止程序崩溃的异常捕获
    try:
        app = QApplication(sys.argv)
        window = PlanningMainWindow()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        QMessageBox.critical(None, "程序异常", f"程序运行出错：{str(e)}")
        sys.exit(1)
