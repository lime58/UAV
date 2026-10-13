import math
import sys
import random
import numpy as np
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QSlider, QGroupBox, QFormLayout, QMessageBox,
                             QTextEdit, QScrollArea, QSplitter)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 设置matplotlib支持中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.titlesize"] = 10


class Node:
    __slots__ = ['x', 'y', 'z', 'parent', 'cost']

    def __init__(self, x, y, z, parent=None):
        self.x = x
        self.y = y
        self.z = z
        self.parent = parent
        self.cost = 0.0

    def distance_sq_to(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return dx ** 2 + dy ** 2 + dz ** 2

    def distance_to(self, other):
        return math.sqrt(self.distance_sq_to(other))

    def update_cost(self):
        if self.parent:
            self.cost = self.parent.cost + self.distance_to(self.parent)


class Building:
    __slots__ = ['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max',
                 'safe_x_min', 'safe_x_max', 'safe_y_min', 'safe_y_max',
                 'safe_z_min', 'safe_z_max', 'color']

    def __init__(self, x_center, y_center, z_base, width, length, height, color):
        self.x_min = x_center - width / 2
        self.x_max = x_center + width / 2
        self.y_min = y_center - length / 2
        self.y_max = y_center + length / 2
        self.z_min = z_base
        self.z_max = z_base + height
        self.color = color

        self.safe_x_min = 0
        self.safe_x_max = 0
        self.safe_y_min = 0
        self.safe_y_max = 0
        self.safe_z_min = 0
        self.safe_z_max = 0

    def set_safe_bounds(self, safe_distance):
        self.safe_x_min = self.x_min - safe_distance
        self.safe_x_max = self.x_max + safe_distance
        self.safe_y_min = self.y_min - safe_distance
        self.safe_y_max = self.y_max + safe_distance
        self.safe_z_min = self.z_min - safe_distance
        self.safe_z_max = self.z_max + safe_distance

    def get_faces(self):
        return [
            [(self.x_min, self.y_min, self.z_min), (self.x_max, self.y_min, self.z_min),
             (self.x_max, self.y_max, self.z_min), (self.x_min, self.y_max, self.z_min)],
            [(self.x_min, self.y_min, self.z_max), (self.x_max, self.y_min, self.z_max),
             (self.x_max, self.y_max, self.z_max), (self.x_min, self.y_max, self.z_max)],
            [(self.x_min, self.y_min, self.z_min), (self.x_max, self.y_min, self.z_min),
             (self.x_max, self.y_min, self.z_max), (self.x_min, self.y_min, self.z_max)],
            [(self.x_min, self.y_max, self.z_min), (self.x_max, self.y_max, self.z_min),
             (self.x_max, self.y_max, self.z_max), (self.x_min, self.y_max, self.z_max)],
            [(self.x_min, self.y_min, self.z_min), (self.x_min, self.y_max, self.z_min),
             (self.x_min, self.y_max, self.z_max), (self.x_min, self.y_min, self.z_max)],
            [(self.x_max, self.y_min, self.z_min), (self.x_max, self.y_max, self.z_min),
             (self.x_max, self.y_max, self.z_max), (self.x_max, self.y_min, self.z_max)]
        ]


class OptimizedRRTPlanner:
    def __init__(self, start, goal, buildings,
                 safe_distance=3.0, base_step_size=2.0, min_height=0.5,
                 max_iter=5000, goal_sample_rate=20,
                 x_range=(0, 60), y_range=(0, 60), z_range=(0, 30)):
        self.start = Node(start[0], start[1], start[2])
        self.goal = Node(goal[0], goal[1], goal[2])

        self.buildings = buildings
        self.safe_distance = safe_distance
        self.safe_distance_sq = safe_distance ** 2
        self.min_height = min_height

        self.max_iter = max_iter
        self.goal_sample_rate = goal_sample_rate
        self.base_step_size = base_step_size
        self.dynamic_step_factor = 1.5

        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

        self.goal_threshold = base_step_size * 1.5
        self.goal_threshold_sq = self.goal_threshold ** 2

        self.tree = [self.start]
        self.start.update_cost()

        self.attraction_strength = 0.3

        for b in self.buildings:
            b.set_safe_bounds(safe_distance)

    def is_collision(self, node):
        x, y, z = node.x, node.y, node.z

        if (x < self.x_range[0] or x > self.x_range[1] or
                y < self.y_range[0] or y > self.y_range[1] or
                z < self.z_range[0] or z > self.z_range[1]):
            return True

        for b in self.buildings:
            if (x < b.safe_x_min or x > b.safe_x_max or
                    y < b.safe_y_min or y > b.safe_y_max or
                    z < b.safe_z_min or z > b.safe_z_max):
                continue

            dx = max(b.x_min - x, 0, x - b.x_max)
            dy = max(b.y_min - y, 0, y - b.y_max)
            dz = max(b.z_min - z, 0, z - b.z_max)

            if dx * dx + dy * dy + dz * dz < self.safe_distance_sq:
                return True

        if node != self.start and node != self.goal and z < self.min_height:
            return True

        return False

    def random_node(self):
        if random.randint(0, 100) < self.goal_sample_rate:
            return self.goal

        x = random.uniform(self.x_range[0], self.x_range[1])
        y = random.uniform(self.y_range[0], self.y_range[1])
        z = random.uniform(max(self.min_height, self.z_range[0]), self.z_range[1])

        if random.random() < 0.7:
            goal_dir_x = self.goal.x - self.start.x
            goal_dir_y = self.goal.y - self.start.y
            goal_dir_z = self.goal.z - self.start.z
            max_dist = math.sqrt(goal_dir_x ** 2 + goal_dir_y ** 2 + goal_dir_z ** 2)

            if max_dist > 0:
                x += self.attraction_strength * (goal_dir_x / max_dist) * self.base_step_size
                y += self.attraction_strength * (goal_dir_y / max_dist) * self.base_step_size
                z += self.attraction_strength * (goal_dir_z / max_dist) * self.base_step_size

                x = max(self.x_range[0], min(x, self.x_range[1]))
                y = max(self.y_range[0], min(y, self.y_range[1]))
                z = max(max(self.min_height, self.z_range[0]), min(z, self.z_range[1]))

        return Node(x, y, z)

    def nearest_node(self, random_node):
        min_dist_sq = float('inf')
        nearest_node = None

        for node in self.tree:
            dist_sq = node.distance_sq_to(random_node)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                nearest_node = node

        return nearest_node

    def steer(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        dz = to_node.z - from_node.z

        dist = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        if dist > self.base_step_size * 2:
            step = self.base_step_size * self.dynamic_step_factor
        else:
            step = self.base_step_size

        if dist <= step:
            new_node = Node(to_node.x, to_node.y, to_node.z, from_node)
        else:
            ratio = step / dist
            new_x = from_node.x + dx * ratio
            new_y = from_node.y + dy * ratio
            new_z = from_node.z + dz * ratio
            if from_node != self.start and to_node != self.goal:
                new_z = max(new_z, self.min_height)
            new_node = Node(new_x, new_y, new_z, from_node)

        new_node.update_cost()
        return new_node

    def is_path_valid(self, from_node, to_node):
        dist = from_node.distance_to(to_node)
        num_samples = max(2, min(5, int(dist / self.base_step_size) + 1))

        for i in range(1, num_samples + 1):
            ratio = i / num_samples
            x = from_node.x + (to_node.x - from_node.x) * ratio
            y = from_node.y + (to_node.y - from_node.y) * ratio
            z = from_node.z + (to_node.z - from_node.z) * ratio

            if from_node != self.start and to_node != self.goal and z < self.min_height:
                z = self.min_height

            sample_node = Node(x, y, z)
            if self.is_collision(sample_node):
                return False
        return True

    def plan(self):
        if self.is_collision(self.start) or self.is_collision(self.goal):
            return None

        closest_node = self.start
        min_dist_to_goal = self.start.distance_to(self.goal)

        for i in range(self.max_iter):
            if i % 1000 == 0 and i > 0:
                self.goal_sample_rate = min(40, self.goal_sample_rate + 5)

            rand_node = self.random_node()
            nearest_node = self.nearest_node(rand_node)
            new_node = self.steer(nearest_node, rand_node)

            if not self.is_collision(new_node) and self.is_path_valid(nearest_node, new_node):
                self.tree.append(new_node)

                current_dist = new_node.distance_to(self.goal)
                if current_dist < min_dist_to_goal:
                    min_dist_to_goal = current_dist
                    closest_node = new_node

                    if current_dist < self.goal_threshold:
                        if self.is_path_valid(new_node, self.goal):
                            self.goal.parent = new_node
                            self.goal.update_cost()
                            self.tree.append(self.goal)

                            path = []
                            current = self.goal
                            while current:
                                path.append((current.x, current.y, current.z))
                                current = current.parent
                            return path[::-1]

        path = []
        current = closest_node
        while current:
            path.append((current.x, current.y, current.z))
            current = current.parent
        return path[::-1]


class PlanningThread(QThread):
    finished = pyqtSignal(object, object)
    error = pyqtSignal(str)

    def __init__(self, planner):
        super().__init__()
        self.planner = planner

    def run(self):
        try:
            path = self.planner.plan()
            self.finished.emit(path, self.planner.tree)
        except Exception as e:
            self.error.emit(str(e))


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = plt.figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111, projection='3d')
        self.axes.set_proj_type('persp')
        self.axes.xaxis._axinfo["grid"]['color'] = (0.9, 0.9, 0.9, 0.3)
        self.axes.yaxis._axinfo["grid"]['color'] = (0.9, 0.9, 0.9, 0.3)
        self.axes.zaxis._axinfo["grid"]['color'] = (0.9, 0.9, 0.9, 0.3)

        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.85)


class DronePathVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于RRT算法的无人机三维路径规划")
        self.resize(1200, 700)

        self.is_planning = False
        self.path = None  # 存储完整路径
        self.rrt_tree = None
        self.plan_time = 0.0
        self.min_flight_height = 0.5

        # 演示相关变量
        self.demo_step = 0  # 当前演示步骤
        self.demo_timer = QTimer(self)
        self.demo_timer.timeout.connect(self._auto_advance_demo)  # 自动前进演示
        self.is_demo_running = False

        # 障碍物定义
        self.buildings = [
            # Building(15, 15, 0, 10, 10, 20, (0.6, 0.2, 0.2, 0.8)),
            # Building(35, 35, 0, 12, 12, 18, (0.2, 0.6, 0.2, 0.8))

            # 中心区域建筑群
            Building(15, 15, 0, 10, 10, 20, (0.6, 0.2, 0.2, 0.8)),  # 红色建筑
            Building(35, 35, 0, 12, 12, 18, (0.2, 0.6, 0.2, 0.8)),  # 绿色建筑
            Building(50, 20, 0, 8, 8, 15, (0.2, 0.2, 0.6, 0.8)),  # 蓝色建筑

            # 左侧建筑群
            Building(10, 30, 0, 6, 6, 12, (0.8, 0.4, 0.4, 0.8)),
            Building(10, 50, 0, 7, 7, 10, (0.4, 0.8, 0.4, 0.8)),

            # 右侧建筑群
            Building(50, 40, 0, 9, 9, 22, (0.4, 0.4, 0.8, 0.8)),
            Building(55, 55, 0, 5, 5, 16, (0.8, 0.8, 0.4, 0.8)),

            # 分散障碍物
            Building(25, 45, 0, 6, 6, 14, (0.6, 0.6, 0.2, 0.8)),
            Building(30, 10, 0, 8, 8, 19, (0.6, 0.2, 0.6, 0.8)),
            Building(40, 25, 0, 7, 7, 13, (0.2, 0.6, 0.6, 0.8)),
            Building(20, 20, 0, 5, 5, 11, (0.8, 0.2, 0.4, 0.8))
        ]

        # 初始化UI
        self._init_ui()
        self.init_visualization()

    def _init_ui(self):
        """初始化用户界面"""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # 左侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setFixedWidth(250)

        # 起点坐标输入
        start_group = QGroupBox("起点坐标")
        start_layout = QFormLayout()
        start_layout.setRowWrapPolicy(QFormLayout.DontWrapRows)
        start_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.start_x = QLineEdit("0")
        self.start_x.setFixedWidth(80)
        self.start_y = QLineEdit("0")
        self.start_y.setFixedWidth(80)
        self.start_z = QLineEdit("0")
        self.start_z.setFixedWidth(80)
        start_layout.addRow("X:", self.start_x)
        start_layout.addRow("Y:", self.start_y)
        start_layout.addRow("Z:", self.start_z)
        start_group.setLayout(start_layout)

        # 终点坐标输入
        goal_group = QGroupBox("终点坐标")
        goal_layout = QFormLayout()
        goal_layout.setRowWrapPolicy(QFormLayout.DontWrapRows)
        goal_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.goal_x = QLineEdit("50")
        self.goal_x.setFixedWidth(80)
        self.goal_y = QLineEdit("50")
        self.goal_y.setFixedWidth(80)
        self.goal_z = QLineEdit("0")
        self.goal_z.setFixedWidth(80)
        goal_layout.addRow("X:", self.goal_x)
        goal_layout.addRow("Y:", self.goal_y)
        goal_layout.addRow("Z:", self.goal_z)
        goal_group.setLayout(goal_layout)

        # 安全距离设置
        safe_group = QGroupBox("安全距离")
        safe_layout = QVBoxLayout()
        self.safe_distance = QSlider(Qt.Horizontal)
        self.safe_distance.setRange(1, 100)
        self.safe_distance.setValue(30)
        self.safe_label = QLabel("3.0")
        self.safe_distance.valueChanged.connect(self.update_safe_label)
        safe_layout.addWidget(self.safe_distance)
        safe_layout.addWidget(self.safe_label)
        safe_group.setLayout(safe_layout)

        # 最低飞行高度设置
        height_group = QGroupBox("最低飞行高度")
        height_layout = QVBoxLayout()
        self.min_height_slider = QSlider(Qt.Horizontal)
        self.min_height_slider.setRange(0, 50)
        self.min_height_slider.setValue(10)
        self.height_label = QLabel("1.0")
        self.min_height_slider.valueChanged.connect(self.update_height_label)
        height_layout.addWidget(self.min_height_slider)
        height_layout.addWidget(self.height_label)
        height_group.setLayout(height_layout)

        # 步长大小设置
        step_group = QGroupBox("步长大小")
        step_layout = QVBoxLayout()
        self.step_size = QSlider(Qt.Horizontal)
        self.step_size.setRange(1, 20)
        self.step_size.setValue(3)
        self.step_label = QLabel("3.0")
        self.step_size.valueChanged.connect(self.update_step_label)
        step_layout.addWidget(self.step_size)
        step_layout.addWidget(self.step_label)
        step_group.setLayout(step_layout)

        # RRT算法参数设置
        rrt_group = QGroupBox("RRT算法参数")
        rrt_layout = QFormLayout()
        rrt_layout.setRowWrapPolicy(QFormLayout.DontWrapRows)
        rrt_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.max_iter = QLineEdit("5000")
        self.max_iter.setFixedWidth(80)
        self.goal_sample = QLineEdit("20")
        self.goal_sample.setFixedWidth(80)
        rrt_layout.addRow("最大迭代次数:", self.max_iter)
        rrt_layout.addRow("目标采样概率(%):", self.goal_sample)
        rrt_group.setLayout(rrt_layout)

        # 演示速度设置 (0.001s到0.01s)
        speed_group = QGroupBox("演示速度 (0.001-0.01s)")
        speed_layout = QVBoxLayout()
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 10)  # 1对应0.001s，10对应0.01s
        self.speed_slider.setValue(5)  # 默认0.005s
        self.speed_label = QLabel("0.005s")
        self.speed_slider.valueChanged.connect(self.update_speed_label)
        speed_layout.addWidget(self.speed_slider)
        speed_layout.addWidget(self.speed_label)
        speed_group.setLayout(speed_layout)

        # 按钮布局 - 左下角添加演示按钮
        button_layout = QVBoxLayout()
        self.plan_button = QPushButton("规划路径")
        self.plan_button.setMinimumHeight(30)
        self.plan_button.clicked.connect(self.plan_path)
        self.reset_button = QPushButton("重置视图")
        self.reset_button.setMinimumHeight(30)
        self.reset_button.clicked.connect(self.reset_view)

        # 演示控制按钮
        self.step_demo_button = QPushButton("单步演示")
        self.step_demo_button.setMinimumHeight(30)
        self.step_demo_button.clicked.connect(self.single_step_demo)
        self.step_demo_button.setEnabled(False)

        self.full_demo_button = QPushButton("完整演示")
        self.full_demo_button.setMinimumHeight(30)
        self.full_demo_button.clicked.connect(self.toggle_full_demo)
        self.full_demo_button.setEnabled(False)

        button_layout.addWidget(self.plan_button)
        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.step_demo_button)
        button_layout.addWidget(self.full_demo_button)
        button_layout.addStretch()

        # 性能指标
        self.time_label = QLabel("规划时间: -- 秒")
        time_group = QGroupBox("性能指标")
        time_layout = QVBoxLayout()
        time_layout.addWidget(self.time_label)
        time_group.setLayout(time_layout)

        # 添加所有控件到控制面板
        control_layout.addWidget(start_group)
        control_layout.addWidget(goal_group)
        control_layout.addWidget(safe_group)
        control_layout.addWidget(height_group)
        control_layout.addWidget(step_group)
        control_layout.addWidget(rrt_group)
        control_layout.addWidget(speed_group)
        control_layout.addWidget(time_group)
        control_layout.addLayout(button_layout)
        control_layout.addStretch()
        control_layout.setSpacing(8)

        # 右侧布局
        right_splitter = QSplitter(Qt.Vertical)

        # 可视化区域
        visual_widget = QWidget()
        visual_layout = QVBoxLayout(visual_widget)

        self.canvas = MplCanvas(self, width=8, height=6, dpi=100)
        self.toolbar = NavigationToolbar(self.canvas, self)
        visual_layout.addWidget(self.toolbar)
        visual_layout.addWidget(self.canvas)

        # 路径信息区域
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        info_title = QLabel("路径点坐标信息")
        info_title.setAlignment(Qt.AlignCenter)
        info_title.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.path_info = QTextEdit()
        self.path_info.setReadOnly(True)
        self.path_info.setMinimumHeight(150)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.path_info)
        info_layout.addWidget(info_title)
        info_layout.addWidget(scroll_area)

        right_splitter.addWidget(visual_widget)
        right_splitter.addWidget(info_widget)
        right_splitter.setSizes([500, 200])

        main_layout.addWidget(control_panel)
        main_layout.addWidget(right_splitter, 1)

    def update_safe_label(self):
        value = self.safe_distance.value() / 10.0
        self.safe_label.setText(f"{value:.1f}")

    def update_step_label(self):
        value = self.step_size.value() / 1.0
        self.step_label.setText(f"{value:.1f}")

    def update_height_label(self):
        value = self.min_height_slider.value() / 10.0
        self.height_label.setText(f"{value:.1f}")
        self.min_flight_height = value
        self.init_visualization()

    def update_speed_label(self):
        # 转换滑块值(1-10)到时间(0.001-0.01秒)
        value = self.speed_slider.value() / 1000.0
        self.speed_label.setText(f"{value:.3f}s")
        # 如果演示正在运行，实时更新计时器间隔
        if self.is_demo_running:
            self.demo_timer.setInterval(int(value * 1000))

    def create_surface_arrays(self):
        """创建表面绘制所需的数组"""
        x = np.linspace(0, 60, 2, dtype=np.float64)
        y = np.linspace(0, 60, 2, dtype=np.float64)
        return np.meshgrid(x, y)

    def init_visualization(self):
        """初始化可视化界面，不显示路径"""
        ax = self.canvas.axes
        ax.clear()
        self.path_info.clear()

        try:
            X, Y = self.create_surface_arrays()

            # 绘制地面参考平面
            Z_ground = np.zeros(X.shape, dtype=np.float64)
            ax.plot_surface(X, Y, Z_ground, color='lightgray', alpha=0.3)

            # 绘制建筑物
            for building in self.buildings:
                faces = building.get_faces()
                poly3d = Poly3DCollection(faces, facecolors=building.color,
                                          edgecolors='black', linewidths=1)
                ax.add_collection3d(poly3d)

            # 绘制最低飞行高度参考平面
            z_height = self.min_flight_height
            Z_height = np.full(X.shape, z_height, dtype=np.float64)
            ax.plot([], [], [], 'blue', alpha=0.1, label=f'最低飞行高度: {z_height:.1f}')
            ax.plot_surface(X, Y, Z_height, color='blue', alpha=0.1)

            # 绘制起点和终点
            start = (float(self.start_x.text()), float(self.start_y.text()), float(self.start_z.text()))
            goal = (float(self.goal_x.text()), float(self.goal_y.text()), float(self.goal_z.text()))
            ax.scatter(start[0], start[1], start[2], color='green', s=150, label='起点', marker='o')
            ax.scatter(goal[0], goal[1], goal[2], color='red', s=150, label='终点', marker='*')

            # 设置坐标轴
            ax.set_xlabel('X轴', fontsize=8)
            ax.set_ylabel('Y轴', fontsize=8)
            ax.set_zlabel('Z轴', fontsize=8)
            ax.set_title('无人机三维路径规划', fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.set_xlim(0, 60)
            ax.set_ylim(0, 60)
            ax.set_zlim(0, 30)
            ax.legend(fontsize=8)

            self.canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "可视化错误", f"创建3D表面时出错: {str(e)}")

    def show_demo_path(self, step):
        """根据当前步骤显示路径"""
        ax = self.canvas.axes
        ax.clear()

        try:
            X, Y = self.create_surface_arrays()

            # 绘制地面参考平面
            Z_ground = np.zeros(X.shape, dtype=np.float64)
            ax.plot_surface(X, Y, Z_ground, color='lightgray', alpha=0.3)

            # 绘制最低飞行高度参考平面
            z_height = self.min_flight_height
            Z_height = np.full(X.shape, z_height, dtype=np.float64)
            ax.plot([], [], [], 'blue', alpha=0.1, label=f'最低飞行高度: {z_height:.1f}')
            ax.plot_surface(X, Y, Z_height, color='blue', alpha=0.1)

            # 绘制建筑物
            for building in self.buildings:
                faces = building.get_faces()
                poly3d = Poly3DCollection(faces, facecolors=building.color,
                                          edgecolors='black', linewidths=1)
                ax.add_collection3d(poly3d)

            # 获取起点和终点
            start = (float(self.start_x.text()), float(self.start_y.text()), float(self.start_z.text()))
            goal = (float(self.goal_x.text()), float(self.goal_y.text()), float(self.goal_z.text()))

            # 绘制起点和终点
            ax.scatter(start[0], start[1], start[2], color='green', s=150, label='起点', marker='o')
            ax.scatter(goal[0], goal[1], goal[2], color='red', s=150, label='终点', marker='*')

            # 绘制路径（只显示到当前步骤）
            if self.path and len(self.path) > 1 and step > 0:
                # 截取到当前步骤的路径
                current_path = self.path[:step + 1]
                x = [p[0] for p in current_path]
                y = [p[1] for p in current_path]
                z = [p[2] for p in current_path]

                # 绘制路径线和路径点
                ax.plot(x, y, z, 'b-', linewidth=2, label='规划路径')
                ax.scatter(x, y, z, color='blue', s=50, alpha=0.7, marker='o')

                # 突出显示当前位置
                ax.scatter(x[-1], y[-1], z[-1], color='yellow', s=100, alpha=1.0, marker='*', label='当前位置')

                # 显示路径信息
                self.path_info.clear()
                total_length = 0.0
                info_text = "路径点坐标列表：\n\n"

                for i, (px, py, pz) in enumerate(current_path):
                    info_text += f"步骤 {i}: ({px:.1f}, {py:.1f}, {pz:.1f})\n"

                    if i > 0:
                        dx = px - current_path[i - 1][0]
                        dy = py - current_path[i - 1][1]
                        dz = pz - current_path[i - 1][2]
                        total_length += math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

                info_text += f"\n当前进度: 步骤 {step}/{len(self.path) - 1}"
                info_text += f"\n已规划长度: {total_length:.2f} 单位"
                info_text += f"\n规划总时间: {self.plan_time:.4f} 秒"
                self.path_info.setText(info_text)

            # 设置坐标轴
            ax.set_xlabel('X轴', fontsize=8)
            ax.set_ylabel('Y轴', fontsize=8)
            ax.set_zlabel('Z轴', fontsize=8)
            ax.set_title('无人机三维路径规划', fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.set_xlim(0, 60)
            ax.set_ylim(0, 60)
            ax.set_zlim(0, 30)
            ax.legend(fontsize=8)

            self.canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "可视化错误", f"显示路径时出错: {str(e)}")

    def plan_path(self):
        """开始路径规划"""
        if self.is_planning:
            return

        try:
            start = (
                float(self.start_x.text()),
                float(self.start_y.text()),
                float(self.start_z.text())
            )
            goal = (
                float(self.goal_x.text()),
                float(self.goal_y.text()),
                float(self.goal_z.text())
            )

            max_iter = int(self.max_iter.text())
            goal_sample_rate = int(self.goal_sample.text())

            if goal_sample_rate < 0 or goal_sample_rate > 100:
                QMessageBox.critical(self, "参数错误", "目标采样概率必须在0-100之间！")
                return

            # 更新UI状态
            self.plan_button.setEnabled(False)
            self.is_planning = True
            self.time_label.setText("规划中...")
            self.path_info.setText("正在规划路径，请稍候...")
            self.step_demo_button.setEnabled(False)
            self.full_demo_button.setEnabled(False)

            # 创建规划器
            planner = OptimizedRRTPlanner(
                start,
                goal,
                self.buildings,
                safe_distance=self.safe_distance.value() / 10.0,
                base_step_size=self.step_size.value(),
                min_height=self.min_flight_height,
                max_iter=max_iter,
                goal_sample_rate=goal_sample_rate
            )

            # 启动规划线程
            self.thread = PlanningThread(planner)
            self.thread.finished.connect(self.on_planning_finished)
            self.thread.error.connect(self.on_planning_error)
            self.thread.start()

            self.plan_start_time = time.time()

        except ValueError:
            QMessageBox.critical(self, "输入错误", "请输入有效的数值！")
            self.plan_button.setEnabled(True)
            self.is_planning = False

    def on_planning_finished(self, path, tree):
        """规划完成回调"""
        self.plan_time = time.time() - self.plan_start_time
        self.time_label.setText(f"规划时间: {self.plan_time:.4f} 秒")
        self.path = path
        self.rrt_tree = tree
        self.demo_step = 0  # 重置演示步骤

        if self.path and len(self.path) > 1:
            # 规划成功，不显示完整路径，只显示初始状态
            self.init_visualization()
            self.path_info.setText(f"路径规划完成！共 {len(self.path) - 1} 步。\n点击'单步演示'或'完整演示'查看路径。")
            # 启用演示按钮
            self.step_demo_button.setEnabled(True)
            self.full_demo_button.setEnabled(True)
        else:
            # 规划失败
            QMessageBox.warning(self, "规划结果", f"在{self.max_iter.text()}次迭代内未找到完整路径！")
            self.init_visualization()
            self.step_demo_button.setEnabled(False)
            self.full_demo_button.setEnabled(False)

        self.plan_button.setEnabled(True)
        self.is_planning = False

    def on_planning_error(self, error_msg):
        """规划错误回调"""
        QMessageBox.critical(self, "规划错误", f"路径规划过程中发生错误: {error_msg}")
        self.time_label.setText("规划时间: 错误")
        self.plan_button.setEnabled(True)
        self.is_planning = False
        self.step_demo_button.setEnabled(False)
        self.full_demo_button.setEnabled(False)

    def reset_view(self):
        """重置视图"""
        if self.is_planning or self.is_demo_running:
            return

        # 停止演示（如果正在运行）
        self.demo_timer.stop()
        self.is_demo_running = False
        self.full_demo_button.setText("完整演示")

        # 重置变量
        self.path = None
        self.rrt_tree = None
        self.demo_step = 0
        self.time_label.setText("规划时间: -- 秒")
        self.step_demo_button.setEnabled(False)
        self.full_demo_button.setEnabled(False)

        # 重新初始化可视化
        self.init_visualization()

    def single_step_demo(self):
        """单步演示 - 每次点击前进一个步骤"""
        if not self.path or len(self.path) <= 1 or self.is_demo_running:
            return

        # 检查是否已到达终点
        if self.demo_step >= len(self.path) - 1:
            # 重置演示
            self.demo_step = 0
            self.init_visualization()
            self.path_info.setText(f"路径演示已完成！共 {len(self.path) - 1} 步。\n再次点击'单步演示'重新开始。")
        else:
            # 前进一个步骤
            self.demo_step += 1
            self.show_demo_path(self.demo_step)

    def toggle_full_demo(self):
        """切换完整演示状态"""
        if not self.path or len(self.path) <= 1:
            return

        if self.is_demo_running:
            # 停止演示
            self.demo_timer.stop()
            self.is_demo_running = False
            self.full_demo_button.setText("完整演示")
        else:
            # 开始自动演示
            self.is_demo_running = True
            self.full_demo_button.setText("停止演示")

            # 根据滑块设置演示速度
            speed = self.speed_slider.value() / 1000.0  # 转换为秒
            self.demo_timer.setInterval(int(speed * 1000))  # 转换为毫秒

            # 如果当前在起点，先显示第一步
            if self.demo_step == 0:
                self.demo_step = 1
                self.show_demo_path(self.demo_step)

            # 启动定时器，自动播放
            self.demo_timer.start()

    def _auto_advance_demo(self):
        """自动演示时的步骤推进"""
        if not self.is_demo_running:
            return

        if self.demo_step < len(self.path) - 1:
            # 继续前进
            self.demo_step += 1
            self.show_demo_path(self.demo_step)
        else:
            # 演示完成，停止定时器
            self.demo_timer.stop()
            self.is_demo_running = False
            self.full_demo_button.setText("完整演示")
            self.path_info.append("\n演示已完成！")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    window = DronePathVisualizer()
    window.show()
    sys.exit(app.exec_())

#没加单步演示和全程演示
# import math
# import sys
# import random
# import numpy as np
# import time
# from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
#                              QHBoxLayout, QLabel, QLineEdit, QPushButton,
#                              QSlider, QGroupBox, QFormLayout, QMessageBox,
#                              QTextEdit, QScrollArea, QSplitter)
# from PyQt5.QtCore import Qt, QThread, pyqtSignal
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#
# # 设置matplotlib支持中文显示
# plt.rcParams["font.family"] = ["SimHei"]
# plt.rcParams["axes.unicode_minus"] = False
# plt.rcParams["figure.titlesize"] = 10
#
#
# class Node:
#     __slots__ = ['x', 'y', 'z', 'parent', 'cost']
#
#     def __init__(self, x, y, z, parent=None):
#         self.x = x
#         self.y = y
#         self.z = z
#         self.parent = parent
#         self.cost = 0.0
#
#     def distance_sq_to(self, other):
#         dx = self.x - other.x
#         dy = self.y - other.y
#         dz = self.z - other.z
#         return dx **2 + dy** 2 + dz **2
#
#     def distance_to(self, other):
#         return math.sqrt(self.distance_sq_to(other))
#
#     def update_cost(self):
#         if self.parent:
#             self.cost = self.parent.cost + self.distance_to(self.parent)
#
#
# class Building:
#     __slots__ = ['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max',
#                  'safe_x_min', 'safe_x_max', 'safe_y_min', 'safe_y_max',
#                  'safe_z_min', 'safe_z_max', 'color']
#
#     def __init__(self, x_center, y_center, z_base, width, length, height, color):
#         self.x_min = x_center - width / 2
#         self.x_max = x_center + width / 2
#         self.y_min = y_center - length / 2
#         self.y_max = y_center + length / 2
#         self.z_min = z_base
#         self.z_max = z_base + height
#         self.color = color
#
#         self.safe_x_min = 0
#         self.safe_x_max = 0
#         self.safe_y_min = 0
#         self.safe_y_max = 0
#         self.safe_z_min = 0
#         self.safe_z_max = 0
#
#     def set_safe_bounds(self, safe_distance):
#         self.safe_x_min = self.x_min - safe_distance
#         self.safe_x_max = self.x_max + safe_distance
#         self.safe_y_min = self.y_min - safe_distance
#         self.safe_y_max = self.y_max + safe_distance
#         self.safe_z_min = self.z_min - safe_distance
#         self.safe_z_max = self.z_max + safe_distance
#
#     def get_faces(self):
#         return [
#             [(self.x_min, self.y_min, self.z_min), (self.x_max, self.y_min, self.z_min),
#              (self.x_max, self.y_max, self.z_min), (self.x_min, self.y_max, self.z_min)],
#             [(self.x_min, self.y_min, self.z_max), (self.x_max, self.y_min, self.z_max),
#              (self.x_max, self.y_max, self.z_max), (self.x_min, self.y_max, self.z_max)],
#             [(self.x_min, self.y_min, self.z_min), (self.x_max, self.y_min, self.z_min),
#              (self.x_max, self.y_min, self.z_max), (self.x_min, self.y_min, self.z_max)],
#             [(self.x_min, self.y_max, self.z_min), (self.x_max, self.y_max, self.z_min),
#              (self.x_max, self.y_max, self.z_max), (self.x_min, self.y_max, self.z_max)],
#             [(self.x_min, self.y_min, self.z_min), (self.x_min, self.y_max, self.z_min),
#              (self.x_min, self.y_max, self.z_max), (self.x_min, self.y_min, self.z_max)],
#             [(self.x_max, self.y_min, self.z_min), (self.x_max, self.y_max, self.z_min),
#              (self.x_max, self.y_max, self.z_max), (self.x_max, self.y_min, self.z_max)]
#         ]
#
#
# class OptimizedRRTPlanner:
#     def __init__(self, start, goal, buildings,
#                  safe_distance=3.0, base_step_size=2.0, min_height=0.5,
#                  max_iter=5000, goal_sample_rate=20,
#                  x_range=(0, 60), y_range=(0, 60), z_range=(0, 30)):
#         self.start = Node(start[0], start[1], start[2])
#         self.goal = Node(goal[0], goal[1], goal[2])
#
#         self.buildings = buildings
#         self.safe_distance = safe_distance
#         self.safe_distance_sq = safe_distance **2
#         self.min_height = min_height
#
#         self.max_iter = max_iter
#         self.goal_sample_rate = goal_sample_rate
#         self.base_step_size = base_step_size
#         self.dynamic_step_factor = 1.5
#
#         self.x_range = x_range
#         self.y_range = y_range
#         self.z_range = z_range
#
#         self.goal_threshold = base_step_size * 1.5
#         self.goal_threshold_sq = self.goal_threshold** 2
#
#         self.tree = [self.start]
#         self.start.update_cost()
#
#         self.attraction_strength = 0.3
#
#         for b in self.buildings:
#             b.set_safe_bounds(safe_distance)
#
#     def is_collision(self, node):
#         x, y, z = node.x, node.y, node.z
#
#         if (x < self.x_range[0] or x > self.x_range[1] or
#                 y < self.y_range[0] or y > self.y_range[1] or
#                 z < self.z_range[0] or z > self.z_range[1]):
#             return True
#
#         for b in self.buildings:
#             if (x < b.safe_x_min or x > b.safe_x_max or
#                     y < b.safe_y_min or y > b.safe_y_max or
#                     z < b.safe_z_min or z > b.safe_z_max):
#                 continue
#
#             dx = max(b.x_min - x, 0, x - b.x_max)
#             dy = max(b.y_min - y, 0, y - b.y_max)
#             dz = max(b.z_min - z, 0, z - b.z_max)
#
#             if dx * dx + dy * dy + dz * dz < self.safe_distance_sq:
#                 return True
#
#         if node != self.start and node != self.goal and z < self.min_height:
#             return True
#
#         return False
#
#     def random_node(self):
#         if random.randint(0, 100) < self.goal_sample_rate:
#             return self.goal
#
#         x = random.uniform(self.x_range[0], self.x_range[1])
#         y = random.uniform(self.y_range[0], self.y_range[1])
#         z = random.uniform(max(self.min_height, self.z_range[0]), self.z_range[1])
#
#         if random.random() < 0.7:
#             goal_dir_x = self.goal.x - self.start.x
#             goal_dir_y = self.goal.y - self.start.y
#             goal_dir_z = self.goal.z - self.start.z
#             max_dist = math.sqrt(goal_dir_x **2 + goal_dir_y** 2 + goal_dir_z **2)
#
#             if max_dist > 0:
#                 x += self.attraction_strength * (goal_dir_x / max_dist) * self.base_step_size
#                 y += self.attraction_strength * (goal_dir_y / max_dist) * self.base_step_size
#                 z += self.attraction_strength * (goal_dir_z / max_dist) * self.base_step_size
#
#                 x = max(self.x_range[0], min(x, self.x_range[1]))
#                 y = max(self.y_range[0], min(y, self.y_range[1]))
#                 z = max(max(self.min_height, self.z_range[0]), min(z, self.z_range[1]))
#
#         return Node(x, y, z)
#
#     def nearest_node(self, random_node):
#         min_dist_sq = float('inf')
#         nearest_node = None
#
#         for node in self.tree:
#             dist_sq = node.distance_sq_to(random_node)
#             if dist_sq < min_dist_sq:
#                 min_dist_sq = dist_sq
#                 nearest_node = node
#
#         return nearest_node
#
#     def steer(self, from_node, to_node):
#         dx = to_node.x - from_node.x
#         dy = to_node.y - from_node.y
#         dz = to_node.z - from_node.z
#
#         dist = math.sqrt(dx **2 + dy** 2 + dz **2)
#
#         if dist > self.base_step_size * 2:
#             step = self.base_step_size * self.dynamic_step_factor
#         else:
#             step = self.base_step_size
#
#         if dist <= step:
#             new_node = Node(to_node.x, to_node.y, to_node.z, from_node)
#         else:
#             ratio = step / dist
#             new_x = from_node.x + dx * ratio
#             new_y = from_node.y + dy * ratio
#             new_z = from_node.z + dz * ratio
#             if from_node != self.start and to_node != self.goal:
#                 new_z = max(new_z, self.min_height)
#             new_node = Node(new_x, new_y, new_z, from_node)
#
#         new_node.update_cost()
#         return new_node
#
#     def is_path_valid(self, from_node, to_node):
#         dist = from_node.distance_to(to_node)
#         num_samples = max(2, min(5, int(dist / self.base_step_size) + 1))
#
#         for i in range(1, num_samples + 1):
#             ratio = i / num_samples
#             x = from_node.x + (to_node.x - from_node.x) * ratio
#             y = from_node.y + (to_node.y - from_node.y) * ratio
#             z = from_node.z + (to_node.z - from_node.z) * ratio
#
#             if from_node != self.start and to_node != self.goal and z < self.min_height:
#                 z = self.min_height
#
#             sample_node = Node(x, y, z)
#             if self.is_collision(sample_node):
#                 return False
#         return True
#
#     def plan(self):
#         if self.is_collision(self.start) or self.is_collision(self.goal):
#             return None
#
#         closest_node = self.start
#         min_dist_to_goal = self.start.distance_to(self.goal)
#
#         for i in range(self.max_iter):
#             if i % 1000 == 0 and i > 0:
#                 self.goal_sample_rate = min(40, self.goal_sample_rate + 5)
#
#             rand_node = self.random_node()
#             nearest_node = self.nearest_node(rand_node)
#             new_node = self.steer(nearest_node, rand_node)
#
#             if not self.is_collision(new_node) and self.is_path_valid(nearest_node, new_node):
#                 self.tree.append(new_node)
#
#                 current_dist = new_node.distance_to(self.goal)
#                 if current_dist < min_dist_to_goal:
#                     min_dist_to_goal = current_dist
#                     closest_node = new_node
#
#                     if current_dist < self.goal_threshold:
#                         if self.is_path_valid(new_node, self.goal):
#                             self.goal.parent = new_node
#                             self.goal.update_cost()
#                             self.tree.append(self.goal)
#
#                             path = []
#                             current = self.goal
#                             while current:
#                                 path.append((current.x, current.y, current.z))
#                                 current = current.parent
#                             return path[::-1]
#
#         path = []
#         current = closest_node
#         while current:
#             path.append((current.x, current.y, current.z))
#             current = current.parent
#         return path[::-1]
#
#
# class PlanningThread(QThread):
#     finished = pyqtSignal(object, object)
#     error = pyqtSignal(str)
#
#     def __init__(self, planner):
#         super().__init__()
#         self.planner = planner
#
#     def run(self):
#         try:
#             path = self.planner.plan()
#             self.finished.emit(path, self.planner.tree)
#         except Exception as e:
#             self.error.emit(str(e))
#
#
# class MplCanvas(FigureCanvas):
#     def __init__(self, parent=None, width=5, height=4, dpi=100):
#         self.fig = plt.figure(figsize=(width, height), dpi=dpi)
#         self.axes = self.fig.add_subplot(111, projection='3d')
#         self.axes.set_proj_type('persp')
#         self.axes.xaxis._axinfo["grid"]['color'] = (0.9, 0.9, 0.9, 0.3)
#         self.axes.yaxis._axinfo["grid"]['color'] = (0.9, 0.9, 0.9, 0.3)
#         self.axes.zaxis._axinfo["grid"]['color'] = (0.9, 0.9, 0.9, 0.3)
#
#         super(MplCanvas, self).__init__(self.fig)
#         self.setParent(parent)
#         self.fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.85)
#
#
# class DronePathVisualizer(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("无人机三维路径规划")
#         self.resize(1200, 700)
#
#         self.is_planning = False
#         self.path = None
#         self.rrt_tree = None
#         self.plan_time = 0.0
#         self.min_flight_height = 0.5
#
#         self.buildings = [
#             Building(15, 15, 0, 10, 10, 20, (0.6, 0.2, 0.2, 0.8)),
#             Building(35, 35, 0, 12, 12, 18, (0.2, 0.6, 0.2, 0.8))
#         ]
#
#         main_widget = QWidget()
#         self.setCentralWidget(main_widget)
#         main_layout = QHBoxLayout(main_widget)
#         main_layout.setContentsMargins(5, 5, 5, 5)
#
#         control_panel = QWidget()
#         control_layout = QVBoxLayout(control_panel)
#         control_panel.setFixedWidth(250)
#
#         # 起点坐标输入
#         start_group = QGroupBox("起点坐标")
#         start_layout = QFormLayout()
#         start_layout.setRowWrapPolicy(QFormLayout.DontWrapRows)
#         start_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
#         self.start_x = QLineEdit("0")
#         self.start_x.setFixedWidth(80)
#         self.start_y = QLineEdit("0")
#         self.start_y.setFixedWidth(80)
#         self.start_z = QLineEdit("0")
#         self.start_z.setFixedWidth(80)
#         start_layout.addRow("X:", self.start_x)
#         start_layout.addRow("Y:", self.start_y)
#         start_layout.addRow("Z:", self.start_z)
#         start_group.setLayout(start_layout)
#
#         # 终点坐标输入
#         goal_group = QGroupBox("终点坐标")
#         goal_layout = QFormLayout()
#         goal_layout.setRowWrapPolicy(QFormLayout.DontWrapRows)
#         goal_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
#         self.goal_x = QLineEdit("50")
#         self.goal_x.setFixedWidth(80)
#         self.goal_y = QLineEdit("50")
#         self.goal_y.setFixedWidth(80)
#         self.goal_z = QLineEdit("0")
#         self.goal_z.setFixedWidth(80)
#         goal_layout.addRow("X:", self.goal_x)
#         goal_layout.addRow("Y:", self.goal_y)
#         goal_layout.addRow("Z:", self.goal_z)
#         goal_group.setLayout(goal_layout)
#
#         # 其他控制面板组件...
#         safe_group = QGroupBox("安全距离")
#         safe_layout = QVBoxLayout()
#         self.safe_distance = QSlider(Qt.Horizontal)
#         self.safe_distance.setRange(1, 100)
#         self.safe_distance.setValue(30)
#         self.safe_label = QLabel("3.0")
#         self.safe_distance.valueChanged.connect(self.update_safe_label)
#         safe_layout.addWidget(self.safe_distance)
#         safe_layout.addWidget(self.safe_label)
#         safe_group.setLayout(safe_layout)
#
#         height_group = QGroupBox("最低飞行高度")
#         height_layout = QVBoxLayout()
#         self.min_height_slider = QSlider(Qt.Horizontal)
#         self.min_height_slider.setRange(0, 50)
#         self.min_height_slider.setValue(10)
#         self.height_label = QLabel("1.0")
#         self.min_height_slider.valueChanged.connect(self.update_height_label)
#         height_layout.addWidget(self.min_height_slider)
#         height_layout.addWidget(self.height_label)
#         height_group.setLayout(height_layout)
#
#         step_group = QGroupBox("步长大小")
#         step_layout = QVBoxLayout()
#         self.step_size = QSlider(Qt.Horizontal)
#         self.step_size.setRange(1, 20)
#         self.step_size.setValue(3)
#         self.step_label = QLabel("3.0")
#         self.step_size.valueChanged.connect(self.update_step_label)
#         step_layout.addWidget(self.step_size)
#         step_layout.addWidget(self.step_label)
#         step_group.setLayout(step_layout)
#
#         rrt_group = QGroupBox("RRT算法参数")
#         rrt_layout = QFormLayout()
#         rrt_layout.setRowWrapPolicy(QFormLayout.DontWrapRows)
#         rrt_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
#         self.max_iter = QLineEdit("5000")
#         self.max_iter.setFixedWidth(80)
#         self.goal_sample = QLineEdit("20")
#         self.goal_sample.setFixedWidth(80)
#         rrt_layout.addRow("最大迭代次数:", self.max_iter)
#         rrt_layout.addRow("目标采样概率(%):", self.goal_sample)
#         rrt_group.setLayout(rrt_layout)
#
#         button_layout = QVBoxLayout()
#         self.plan_button = QPushButton("规划路径")
#         self.plan_button.setMinimumHeight(30)
#         self.plan_button.clicked.connect(self.plan_path)
#         self.reset_button = QPushButton("重置视图")
#         self.reset_button.setMinimumHeight(30)
#         self.reset_button.clicked.connect(self.reset_view)
#         button_layout.addWidget(self.plan_button)
#         button_layout.addWidget(self.reset_button)
#         button_layout.addStretch()
#
#         self.time_label = QLabel("规划时间: -- 秒")
#         time_group = QGroupBox("性能指标")
#         time_layout = QVBoxLayout()
#         time_layout.addWidget(self.time_label)
#         time_group.setLayout(time_layout)
#
#         control_layout.addWidget(start_group)
#         control_layout.addWidget(goal_group)
#         control_layout.addWidget(safe_group)
#         control_layout.addWidget(height_group)
#         control_layout.addWidget(step_group)
#         control_layout.addWidget(rrt_group)
#         control_layout.addWidget(time_group)
#         control_layout.addLayout(button_layout)
#         control_layout.addStretch()
#         control_layout.setSpacing(8)
#
#         right_splitter = QSplitter(Qt.Vertical)
#
#         visual_widget = QWidget()
#         visual_layout = QVBoxLayout(visual_widget)
#
#         self.canvas = MplCanvas(self, width=8, height=6, dpi=100)
#         self.toolbar = NavigationToolbar(self.canvas, self)
#         visual_layout.addWidget(self.toolbar)
#         visual_layout.addWidget(self.canvas)
#
#         info_widget = QWidget()
#         info_layout = QVBoxLayout(info_widget)
#         info_title = QLabel("路径点坐标信息")
#         info_title.setAlignment(Qt.AlignCenter)
#         info_title.setStyleSheet("font-weight: bold; font-size: 12px;")
#         self.path_info = QTextEdit()
#         self.path_info.setReadOnly(True)
#         self.path_info.setMinimumHeight(150)
#         scroll_area = QScrollArea()
#         scroll_area.setWidgetResizable(True)
#         scroll_area.setWidget(self.path_info)
#         info_layout.addWidget(info_title)
#         info_layout.addWidget(scroll_area)
#
#         right_splitter.addWidget(visual_widget)
#         right_splitter.addWidget(info_widget)
#         right_splitter.setSizes([500, 200])
#
#         main_layout.addWidget(control_panel)
#         main_layout.addWidget(right_splitter, 1)
#
#         self.init_visualization()
#
#     def update_safe_label(self):
#         value = self.safe_distance.value() / 10.0
#         self.safe_label.setText(f"{value:.1f}")
#
#     def update_step_label(self):
#         value = self.step_size.value() / 1.0
#         self.step_label.setText(f"{value:.1f}")
#
#     def update_height_label(self):
#         value = self.min_height_slider.value() / 10.0
#         self.height_label.setText(f"{value:.1f}")
#         self.min_flight_height = value
#         self.init_visualization()
#
#     def create_surface_arrays(self):
#         """专门创建表面绘制所需的数组，确保类型正确"""
#         # 创建基础线性空间
#         x = np.linspace(0, 60, 2, dtype=np.float64)
#         y = np.linspace(0, 60, 2, dtype=np.float64)
#
#         # 创建网格（确保是numpy数组）
#         X, Y = np.meshgrid(x, y)
#
#         # 验证类型
#         if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
#             raise TypeError("网格必须是numpy数组")
#
#         return X, Y
#
#     def init_visualization(self):
#         ax = self.canvas.axes
#         ax.clear()
#         self.path_info.clear()
#
#         try:
#             # 获取正确的网格数组
#             X, Y = self.create_surface_arrays()
#
#             # 绘制地面参考平面（确保Z是正确的numpy数组）
#             Z_ground = np.zeros(X.shape, dtype=np.float64)
#             ax.plot_surface(X, Y, Z_ground, color='lightgray', alpha=0.3)
#
#             # 绘制建筑物
#             for building in self.buildings:
#                 faces = building.get_faces()
#                 poly3d = Poly3DCollection(faces, facecolors=building.color,
#                                           edgecolors='black', linewidths=1)
#                 ax.add_collection3d(poly3d)
#
#             # 绘制最低飞行高度参考平面
#             z_height = self.min_flight_height
#             Z_height = np.full(X.shape, z_height, dtype=np.float64)
#             # 创建临时线条用于图例显示
#             ax.plot([], [], [], 'blue', alpha=0.1, label=f'最低飞行高度: {z_height:.1f}')
#             ax.plot_surface(X, Y, Z_height, color='blue', alpha=0.1)
#
#             # 设置坐标轴
#             ax.set_xlabel('X轴', fontsize=8)
#             ax.set_ylabel('Y轴', fontsize=8)
#             ax.set_zlabel('Z轴', fontsize=8)
#             ax.set_title('无人机三维路径规划', fontsize=10)
#             ax.tick_params(axis='both', which='major', labelsize=7)
#             ax.set_xlim(0, 60)
#             ax.set_ylim(0, 60)
#             ax.set_zlim(0, 30)
#             ax.legend(fontsize=8)
#
#             self.canvas.draw()
#
#         except Exception as e:
#             QMessageBox.critical(self, "可视化错误", f"创建3D表面时出错: {str(e)}")
#             print(f"调试信息: 错误类型={type(e)}, 错误内容={str(e)}")
#
#     def show_final_path(self):
#         ax = self.canvas.axes
#         ax.clear()
#
#         try:
#             # 获取正确的网格数组
#             X, Y = self.create_surface_arrays()
#
#             # 绘制地面参考平面
#             Z_ground = np.zeros(X.shape, dtype=np.float64)
#             ax.plot_surface(X, Y, Z_ground, color='lightgray', alpha=0.3)
#
#             # 绘制最低飞行高度参考平面
#             z_height = self.min_flight_height
#             Z_height = np.full(X.shape, z_height, dtype=np.float64)
#             # 创建临时线条用于图例显示
#             ax.plot([], [], [], 'blue', alpha=0.1, label=f'最低飞行高度: {z_height:.1f}')
#             ax.plot_surface(X, Y, Z_height, color='blue', alpha=0.1)
#
#             # 绘制建筑物
#             for building in self.buildings:
#                 faces = building.get_faces()
#                 poly3d = Poly3DCollection(faces, facecolors=building.color,
#                                           edgecolors='black', linewidths=1)
#                 ax.add_collection3d(poly3d)
#
#             # 获取起点和终点
#             start = (float(self.start_x.text()), float(self.start_y.text()), float(self.start_z.text()))
#             goal = (float(self.goal_x.text()), float(self.goal_y.text()), float(self.goal_z.text()))
#
#             # 绘制起点和终点
#             ax.scatter(start[0], start[1], start[2], color='green', s=150, label='起点', marker='o')
#             ax.scatter(goal[0], goal[1], goal[2], color='red', s=150, label='终点', marker='*')
#
#             # 绘制RRT树
#             if self.rrt_tree and len(self.rrt_tree) > 1:
#                 draw_nodes = self.rrt_tree[-1000:] if len(self.rrt_tree) > 1000 else self.rrt_tree
#                 for node in draw_nodes:
#                     if node.parent:
#                         ax.plot(
#                             [node.x, node.parent.x],
#                             [node.y, node.parent.y],
#                             [node.z, node.parent.z],
#                             'gray', linewidth=0.5, alpha=0.3
#                         )
#
#             # 绘制路径
#             if self.path and len(self.path) > 1:
#                 x = [p[0] for p in self.path]
#                 y = [p[1] for p in self.path]
#                 z = [p[2] for p in self.path]
#
#                 ax.plot(x, y, z, 'b-', linewidth=2, label='规划路径')
#                 ax.scatter(x, y, z, color='blue', s=50, alpha=0.7, marker='o')
#
#                 # 显示路径信息
#                 self.path_info.clear()
#                 total_length = 0.0
#                 info_text = "路径点坐标列表：\n\n"
#                 info_text += f"起点: ({start[0]:.1f}, {start[1]:.1f}, {start[2]:.1f})\n\n"
#
#                 for i, (px, py, pz) in enumerate(self.path):
#                     info_text += f"步骤 {i}: ({px:.1f}, {py:.1f}, {pz:.1f})"
#                     if i > 0 and i < len(self.path) - 1 and pz < self.min_flight_height - 0.1:
#                         info_text += " (低于最低高度)\n"
#                     else:
#                         info_text += "\n"
#
#                     if i > 0:
#                         dx = px - self.path[i - 1][0]
#                         dy = py - self.path[i - 1][1]
#                         dz = pz - self.path[i - 1][2]
#                         total_length += math.sqrt(dx **2 + dy** 2 + dz **2)
#
#                 info_text += f"\n终点: ({goal[0]:.1f}, {goal[1]:.1f}, {goal[2]:.1f})\n"
#                 info_text += f"\n路径总长度: {total_length:.2f} 单位"
#                 info_text += f"\n规划时间: {self.plan_time:.4f} 秒"
#                 info_text += f"\n最低飞行高度: {self.min_flight_height:.1f} 单位"
#                 self.path_info.setText(info_text)
#
#             # 设置坐标轴
#             ax.set_xlabel('X轴', fontsize=8)
#             ax.set_ylabel('Y轴', fontsize=8)
#             ax.set_zlabel('Z轴', fontsize=8)
#             ax.set_title('无人机三维路径规划', fontsize=10)
#             ax.tick_params(axis='both', which='major', labelsize=7)
#             ax.set_xlim(0, 60)
#             ax.set_ylim(0, 60)
#             ax.set_zlim(0, 30)
#             ax.legend(fontsize=8)
#
#             self.canvas.draw()
#
#         except Exception as e:
#             QMessageBox.critical(self, "可视化错误", f"显示路径时出错: {str(e)}")
#             print(f"调试信息: 错误类型={type(e)}, 错误内容={str(e)}")
#
#     def plan_path(self):
#         if self.is_planning:
#             return
#
#         try:
#             start = (
#                 float(self.start_x.text()),
#                 float(self.start_y.text()),
#                 float(self.start_z.text())
#             )
#             goal = (
#                 float(self.goal_x.text()),
#                 float(self.goal_y.text()),
#                 float(self.goal_z.text())
#             )
#
#             max_iter = int(self.max_iter.text())
#             goal_sample_rate = int(self.goal_sample.text())
#
#             if goal_sample_rate < 0 or goal_sample_rate > 100:
#                 QMessageBox.critical(self, "参数错误", "目标采样概率必须在0-100之间！")
#                 return
#
#             self.plan_button.setEnabled(False)
#             self.is_planning = True
#             self.time_label.setText("规划中...")
#             self.path_info.setText("正在规划路径，请稍候...")
#
#             planner = OptimizedRRTPlanner(
#                 start,
#                 goal,
#                 self.buildings,
#                 safe_distance=self.safe_distance.value() / 10.0,
#                 base_step_size=self.step_size.value(),
#                 min_height=self.min_flight_height,
#                 max_iter=max_iter,
#                 goal_sample_rate=goal_sample_rate
#             )
#
#             self.thread = PlanningThread(planner)
#             self.thread.finished.connect(self.on_planning_finished)
#             self.thread.error.connect(self.on_planning_error)
#             self.thread.start()
#
#             self.plan_start_time = time.time()
#
#         except ValueError:
#             QMessageBox.critical(self, "输入错误", "请输入有效的数值！")
#             self.plan_button.setEnabled(True)
#             self.is_planning = False
#
#     def on_planning_finished(self, path, tree):
#         self.plan_time = time.time() - self.plan_start_time
#         self.time_label.setText(f"规划时间: {self.plan_time:.4f} 秒")
#         self.path = path
#         self.rrt_tree = tree
#
#         if self.path and len(self.path) > 1:
#             self.show_final_path()
#         else:
#             QMessageBox.warning(self, "规划结果", f"在{self.max_iter.text()}次迭代内未找到完整路径！")
#             self.show_final_path()
#
#         self.plan_button.setEnabled(True)
#         self.is_planning = False
#
#     def on_planning_error(self, error_msg):
#         QMessageBox.critical(self, "规划错误", f"路径规划过程中发生错误: {error_msg}")
#         self.time_label.setText("规划时间: 错误")
#         self.plan_button.setEnabled(True)
#         self.is_planning = False
#
#     def reset_view(self):
#         if self.is_planning:
#             return
#
#         self.path = None
#         self.rrt_tree = None
#         self.time_label.setText("规划时间: -- 秒")
#         self.init_visualization()
#
#
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     if hasattr(Qt, 'AA_EnableHighDpiScaling'):
#         app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
#     if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
#         app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
#
#     window = DronePathVisualizer()
#     window.show()
#     sys.exit(app.exec_())



#增加单布全程演示，但坐标位置不对
# import sys
# import math
# import random
# import time
# import numpy as np
# from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
#                              QGroupBox, QFormLayout, QLineEdit, QSlider, QLabel, QPushButton,
#                              QSplitter, QTextEdit, QScrollArea, QMessageBox)
# from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# import matplotlib.pyplot as plt
#
# plt.rcParams["font.family"] = ["SimHei"]
# plt.rcParams["axes.unicode_minus"] = False
# plt.rcParams["figure.titlesize"] = 10
#
# class Building:
#     def __init__(self, x, y, z, width, depth, height, color):
#         self.x_min = x
#         self.x_max = x + width
#         self.y_min = y
#         self.y_max = y + depth
#         self.z_min = z
#         self.z_max = z + height
#         self.color = color
#         self.safe_x_min = self.x_min
#         self.safe_x_max = self.x_max
#         self.safe_y_min = self.y_min
#         self.safe_y_max = self.y_max
#         self.safe_z_min = self.z_min
#         self.safe_z_max = self.z_max
#
#     def set_safe_bounds(self, safe_distance):
#         self.safe_x_min = self.x_min - safe_distance
#         self.safe_x_max = self.x_max + safe_distance
#         self.safe_y_min = self.y_min - safe_distance
#         self.safe_y_max = self.y_max + safe_distance
#         self.safe_z_min = self.z_min - safe_distance
#         self.safe_z_max = self.z_max + safe_distance
#
#     def get_faces(self):
#         x = [self.x_min, self.x_max, self.x_max, self.x_min, self.x_min]
#         y = [self.y_min, self.y_min, self.y_max, self.y_max, self.y_min]
#
#         faces = []
#
#         # 底面
#         z = [self.z_min] * 5
#         faces.append(list(zip(x, y, z)))
#
#         # 顶面
#         z = [self.z_max] * 5
#         faces.append(list(zip(x, y, z)))
#
#         # 前面
#         z = [self.z_min, self.z_min, self.z_max, self.z_max, self.z_min]
#         faces.append(list(zip([self.x_max] * 5, y, z)))
#
#         # 后面
#         faces.append(list(zip([self.x_min] * 5, y, z)))
#
#         # 左面
#         x_left = [self.x_min, self.x_max, self.x_max, self.x_min, self.x_min]
#         faces.append(list(zip(x_left, [self.y_min] * 5, z)))
#
#         # 右面
#         faces.append(list(zip(x_left, [self.y_max] * 5, z)))
#
#         return faces
#
#
# class Node:
#     __slots__ = ['x', 'y', 'z', 'parent', 'cost']
#
#     def __init__(self, x, y, z, parent=None):
#         self.x = x
#         self.y = y
#         self.z = z
#         self.parent = parent
#         self.cost = 0.0
#
#     def distance_sq_to(self, other):
#         dx = self.x - other.x
#         dy = self.y - other.y
#         dz = self.z - other.z
#         return dx ** 2 + dy ** 2 + dz ** 2
#
#     def distance_to(self, other):
#         return math.sqrt(self.distance_sq_to(other))
#
#     def update_cost(self):
#         if self.parent:
#             self.cost = self.parent.cost + self.distance_to(self.parent)
#
#     def __eq__(self, other):
#         if not isinstance(other, Node):
#             return False
#         return (abs(self.x - other.x) < 1e-6 and
#                 abs(self.y - other.y) < 1e-6 and
#                 abs(self.z - other.z) < 1e-6)
#
#     def __hash__(self):
#         return hash((round(self.x, 6), round(self.y, 6), round(self.z, 6)))
#
#
# class OptimizedRRTPlanner:
#     def __init__(self, start, goal, buildings,
#                  safe_distance=3.0, base_step_size=2.0, min_height=0.5,
#                  max_iter=5000, goal_sample_rate=20,
#                  x_range=(0, 60), y_range=(0, 60), z_range=(0, 30)):
#         self.start = Node(start[0], start[1], start[2])
#         self.goal = Node(goal[0], goal[1], goal[2])
#
#         self.buildings = buildings
#         self.safe_distance = safe_distance
#         self.safe_distance_sq = safe_distance ** 2
#         self.min_height = min_height
#
#         self.max_iter = max_iter
#         self.goal_sample_rate = goal_sample_rate
#         self.base_step_size = base_step_size
#         self.dynamic_step_factor = 1.5
#
#         self.x_range = x_range
#         self.y_range = y_range
#         self.z_range = z_range
#
#         self.goal_threshold = base_step_size * 1.5
#         self.goal_threshold_sq = self.goal_threshold ** 2
#
#         self.tree = [self.start]
#         self.start.update_cost()
#
#         self.attraction_strength = 0.3
#
#         for b in self.buildings:
#             b.set_safe_bounds(safe_distance)
#
#     def is_collision(self, node):
#         x, y, z = node.x, node.y, node.z
#
#         if (x < self.x_range[0] or x > self.x_range[1] or
#                 y < self.y_range[0] or y > self.y_range[1] or
#                 z < self.z_range[0] or z > self.z_range[1]):
#             return True
#
#         for b in self.buildings:
#             if (x < b.safe_x_min or x > b.safe_x_max or
#                     y < b.safe_y_min or y > b.safe_y_max or
#                     z < b.safe_z_min or z > b.safe_z_max):
#                 continue
#
#             dx = max(b.x_min - x, 0, x - b.x_max)
#             dy = max(b.y_min - y, 0, y - b.y_max)
#             dz = max(b.z_min - z, 0, z - b.z_max)
#
#             if dx * dx + dy * dy + dz * dz < self.safe_distance_sq:
#                 return True
#
#         if node != self.start and node != self.goal and z < self.min_height:
#             return True
#
#         return False
#
#     def random_node(self):
#         if random.randint(0, 100) < self.goal_sample_rate:
#             return self.goal
#
#         x = random.uniform(self.x_range[0], self.x_range[1])
#         y = random.uniform(self.y_range[0], self.y_range[1])
#         z = random.uniform(max(self.min_height, self.z_range[0]), self.z_range[1])
#
#         if random.random() < 0.7:
#             goal_dir_x = self.goal.x - self.start.x
#             goal_dir_y = self.goal.y - self.start.y
#             goal_dir_z = self.goal.z - self.start.z
#             max_dist = math.sqrt(goal_dir_x ** 2 + goal_dir_y ** 2 + goal_dir_z ** 2)
#
#             if max_dist > 0:
#                 x += self.attraction_strength * (goal_dir_x / max_dist) * self.base_step_size
#                 y += self.attraction_strength * (goal_dir_y / max_dist) * self.base_step_size
#                 z += self.attraction_strength * (goal_dir_z / max_dist) * self.base_step_size
#
#                 x = max(self.x_range[0], min(x, self.x_range[1]))
#                 y = max(self.y_range[0], min(y, self.y_range[1]))
#                 z = max(max(self.min_height, self.z_range[0]), min(z, self.z_range[1]))
#
#         return Node(x, y, z)
#
#     def nearest_node(self, random_node):
#         min_dist_sq = float('inf')
#         nearest_node = None
#
#         for node in self.tree:
#             dist_sq = node.distance_sq_to(random_node)
#             if dist_sq < min_dist_sq:
#                 min_dist_sq = dist_sq
#                 nearest_node = node
#
#         return nearest_node
#
#     def steer(self, from_node, to_node):
#         dx = to_node.x - from_node.x
#         dy = to_node.y - from_node.y
#         dz = to_node.z - from_node.z
#
#         dist = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
#
#         if dist > self.base_step_size * 2:
#             step = self.base_step_size * self.dynamic_step_factor
#         else:
#             step = self.base_step_size
#
#         if dist <= step:
#             new_node = Node(to_node.x, to_node.y, to_node.z, from_node)
#         else:
#             ratio = step / dist
#             new_x = from_node.x + dx * ratio
#             new_y = from_node.y + dy * ratio
#             new_z = from_node.z + dz * ratio
#             if from_node != self.start and to_node != self.goal:
#                 new_z = max(new_z, self.min_height)
#             new_node = Node(new_x, new_y, new_z, from_node)
#
#         new_node.update_cost()
#         return new_node
#
#     def is_path_valid(self, from_node, to_node):
#         dist = from_node.distance_to(to_node)
#         num_samples = max(2, min(5, int(dist / self.base_step_size) + 1))
#
#         for i in range(1, num_samples + 1):
#             ratio = i / num_samples
#             x = from_node.x + (to_node.x - from_node.x) * ratio
#             y = from_node.y + (to_node.y - from_node.y) * ratio
#             z = from_node.z + (to_node.z - from_node.z) * ratio
#
#             if from_node != self.start and to_node != self.goal and z < self.min_height:
#                 z = self.min_height
#
#             sample_node = Node(x, y, z)
#             if self.is_collision(sample_node):
#                 return False
#         return True
#
#     def plan(self):
#         if self.is_collision(self.start) or self.is_collision(self.goal):
#             return None
#
#         closest_node = self.start
#         min_dist_to_goal = self.start.distance_to(self.goal)
#
#         for i in range(self.max_iter):
#             if i % 1000 == 0 and i > 0:
#                 self.goal_sample_rate = min(40, self.goal_sample_rate + 5)
#
#             rand_node = self.random_node()
#             nearest_node = self.nearest_node(rand_node)
#             new_node = self.steer(nearest_node, rand_node)
#
#             if not self.is_collision(new_node) and self.is_path_valid(nearest_node, new_node):
#                 self.tree.append(new_node)
#
#                 current_dist = new_node.distance_to(self.goal)
#                 if current_dist < min_dist_to_goal:
#                     min_dist_to_goal = current_dist
#                     closest_node = new_node
#
#                     if current_dist < self.goal_threshold:
#                         if self.is_path_valid(new_node, self.goal):
#                             self.goal.parent = new_node
#                             self.goal.update_cost()
#                             self.tree.append(self.goal)
#
#                             path = []
#                             current = self.goal
#                             while current:
#                                 path.append((current.x, current.y, current.z))
#                                 current = current.parent
#                             return path[::-1]
#
#         path = []
#         current = closest_node
#         while current:
#             path.append((current.x, current.y, current.z))
#             current = current.parent
#         return path[::-1]
#
#
# class PlanningThread(QThread):
#     finished = pyqtSignal(object, object)
#     error = pyqtSignal(str)
#
#     def __init__(self, planner):
#         super().__init__()
#         self.planner = planner
#
#     def run(self):
#         try:
#             path = self.planner.plan()
#             self.finished.emit(path, self.planner.tree)
#         except Exception as e:
#             self.error.emit(str(e))
#
#
# class MplCanvas(FigureCanvas):
#     def __init__(self, parent=None, width=5, height=4, dpi=100):
#         self.fig = plt.figure(figsize=(width, height), dpi=dpi)
#         self.axes = self.fig.add_subplot(111, projection='3d')
#         self.axes.set_proj_type('persp')
#         self.axes.xaxis._axinfo["grid"]['color'] = (0.9, 0.9, 0.9, 0.3)
#         self.axes.yaxis._axinfo["grid"]['color'] = (0.9, 0.9, 0.9, 0.3)
#         self.axes.zaxis._axinfo["grid"]['color'] = (0.9, 0.9, 0.9, 0.3)
#
#         super(MplCanvas, self).__init__(self.fig)
#         self.setParent(parent)
#         self.fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.85)
#
#
# class DronePathVisualizer(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("无人机三维路径规划")
#         self.resize(1400, 800)
#
#         self.is_planning = False
#         self.path = None
#         self.rrt_tree = None
#         self.plan_time = 0.0
#         self.min_flight_height = 0.5
#         self.current_step = 0
#         self.animation_timer = QTimer(self)
#         self.animation_timer.timeout.connect(self.animate_step)
#
#         self.buildings = [
#             Building(15, 15, 0, 10, 10, 20, (0.6, 0.2, 0.2, 0.8)),
#             Building(35, 35, 0, 12, 12, 18, (0.2, 0.6, 0.2, 0.8))
#         ]
#
#         main_widget = QWidget()
#         self.setCentralWidget(main_widget)
#         main_layout = QHBoxLayout(main_widget)
#         main_layout.setContentsMargins(5, 5, 5, 5)
#
#         control_panel = QWidget()
#         control_layout = QVBoxLayout(control_panel)
#         control_panel.setFixedWidth(250)
#
#         # 起点坐标输入
#         start_group = QGroupBox("起点坐标")
#         start_layout = QFormLayout()
#         start_layout.setRowWrapPolicy(QFormLayout.DontWrapRows)
#         start_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
#         self.start_x = QLineEdit("0")
#         self.start_x.setFixedWidth(80)
#         self.start_y = QLineEdit("0")
#         self.start_y.setFixedWidth(80)
#         self.start_z = QLineEdit("0")
#         self.start_z.setFixedWidth(80)
#         start_layout.addRow("X:", self.start_x)
#         start_layout.addRow("Y:", self.start_y)
#         start_layout.addRow("Z:", self.start_z)
#         start_group.setLayout(start_layout)
#
#         # 终点坐标输入
#         goal_group = QGroupBox("终点坐标")
#         goal_layout = QFormLayout()
#         goal_layout.setRowWrapPolicy(QFormLayout.DontWrapRows)
#         goal_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
#         self.goal_x = QLineEdit("50")
#         self.goal_x.setFixedWidth(80)
#         self.goal_y = QLineEdit("50")
#         self.goal_y.setFixedWidth(80)
#         self.goal_z = QLineEdit("0")
#         self.goal_z.setFixedWidth(80)
#         goal_layout.addRow("X:", self.goal_x)
#         goal_layout.addRow("Y:", self.goal_y)
#         goal_layout.addRow("Z:", self.goal_z)
#         goal_group.setLayout(goal_layout)
#
#         # 安全距离设置
#         safe_group = QGroupBox("安全距离")
#         safe_layout = QVBoxLayout()
#         self.safe_distance = QSlider(Qt.Horizontal)
#         self.safe_distance.setRange(1, 100)
#         self.safe_distance.setValue(30)
#         self.safe_label = QLabel("3.0")
#         self.safe_distance.valueChanged.connect(self.update_safe_label)
#         safe_layout.addWidget(self.safe_distance)
#         safe_layout.addWidget(self.safe_label)
#         safe_group.setLayout(safe_layout)
#
#         # 最低飞行高度设置
#         height_group = QGroupBox("最低飞行高度")
#         height_layout = QVBoxLayout()
#         self.min_height_slider = QSlider(Qt.Horizontal)
#         self.min_height_slider.setRange(0, 50)
#         self.min_height_slider.setValue(10)
#         self.height_label = QLabel("1.0")
#         self.min_height_slider.valueChanged.connect(self.update_height_label)
#         height_layout.addWidget(self.min_height_slider)
#         height_layout.addWidget(self.height_label)
#         height_group.setLayout(height_layout)
#
#         # 步长大小设置
#         step_group = QGroupBox("步长大小")
#         step_layout = QVBoxLayout()
#         self.step_size = QSlider(Qt.Horizontal)
#         self.step_size.setRange(1, 20)
#         self.step_size.setValue(3)
#         self.step_label = QLabel("3.0")
#         self.step_size.valueChanged.connect(self.update_step_label)
#         step_layout.addWidget(self.step_size)
#         step_layout.addWidget(self.step_label)
#         step_group.setLayout(step_layout)
#
#         # RRT算法参数设置
#         rrt_group = QGroupBox("RRT算法参数")
#         rrt_layout = QFormLayout()
#         rrt_layout.setRowWrapPolicy(QFormLayout.DontWrapRows)
#         rrt_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
#         self.max_iter = QLineEdit("5000")
#         self.max_iter.setFixedWidth(80)
#         self.goal_sample = QLineEdit("20")
#         self.goal_sample.setFixedWidth(80)
#         rrt_layout.addRow("最大迭代次数:", self.max_iter)
#         rrt_layout.addRow("目标采样概率(%):", self.goal_sample)
#         rrt_group.setLayout(rrt_layout)
#
#         # 动画速度设置
#         animation_speed_group = QGroupBox("动画速度")
#         animation_speed_layout = QVBoxLayout()
#         self.animation_speed = QSlider(Qt.Horizontal)
#         self.animation_speed.setRange(1, 10)  # 对应0.001到0.01秒
#         self.animation_speed.setValue(5)
#         self.animation_speed_label = QLabel("0.005s")
#         self.animation_speed.valueChanged.connect(self.update_animation_speed_label)
#         animation_speed_layout.addWidget(self.animation_speed)
#         animation_speed_layout.addWidget(self.animation_speed_label)
#         animation_speed_group.setLayout(animation_speed_layout)
#
#         # 按钮布局
#         button_layout = QVBoxLayout()
#         self.plan_button = QPushButton("规划路径")
#         self.plan_button.setMinimumHeight(30)
#         self.plan_button.clicked.connect(self.plan_path)
#         self.step_button = QPushButton("单步演示")
#         self.step_button.setMinimumHeight(30)
#         self.step_button.clicked.connect(self.step_demo)
#         self.step_button.setEnabled(False)
#         self.full_button = QPushButton("完整演示")
#         self.full_button.setMinimumHeight(30)
#         self.full_button.clicked.connect(self.full_demo)
#         self.full_button.setEnabled(False)
#         self.reset_button = QPushButton("重置视图")
#         self.reset_button.setMinimumHeight(30)
#         self.reset_button.clicked.connect(self.reset_view)
#         button_layout.addWidget(self.plan_button)
#         button_layout.addWidget(self.step_button)
#         button_layout.addWidget(self.full_button)
#         button_layout.addWidget(self.reset_button)
#         button_layout.addStretch()
#
#         # 性能指标
#         self.time_label = QLabel("规划时间: -- 秒")
#         time_group = QGroupBox("性能指标")
#         time_layout = QVBoxLayout()
#         time_layout.addWidget(self.time_label)
#         time_group.setLayout(time_layout)
#
#         # 控制面板布局
#         control_layout.addWidget(start_group)
#         control_layout.addWidget(goal_group)
#         control_layout.addWidget(safe_group)
#         control_layout.addWidget(height_group)
#         control_layout.addWidget(step_group)
#         control_layout.addWidget(rrt_group)
#         control_layout.addWidget(animation_speed_group)  # 添加动画速度控制
#         control_layout.addWidget(time_group)
#         control_layout.addLayout(button_layout)
#         control_layout.addStretch()
#         control_layout.setSpacing(8)
#
#         # 右侧布局
#         right_splitter = QSplitter(Qt.Vertical)
#
#         visual_widget = QWidget()
#         visual_layout = QVBoxLayout(visual_widget)
#
#         self.canvas = MplCanvas(self, width=8, height=6, dpi=100)
#         self.toolbar = NavigationToolbar(self.canvas, self)
#         visual_layout.addWidget(self.toolbar)
#         visual_layout.addWidget(self.canvas)
#
#         info_widget = QWidget()
#         info_layout = QVBoxLayout(info_widget)
#         info_title = QLabel("路径点坐标信息")
#         info_title.setAlignment(Qt.AlignCenter)
#         info_title.setStyleSheet("font-weight: bold; font-size: 12px;")
#         self.path_info = QTextEdit()
#         self.path_info.setReadOnly(True)
#         self.path_info.setMinimumHeight(150)
#         scroll_area = QScrollArea()
#         scroll_area.setWidgetResizable(True)
#         scroll_area.setWidget(self.path_info)
#         info_layout.addWidget(info_title)
#         info_layout.addWidget(scroll_area)
#
#         right_splitter.addWidget(visual_widget)
#         right_splitter.addWidget(info_widget)
#         right_splitter.setSizes([500, 200])
#
#         main_layout.addWidget(control_panel)
#         main_layout.addWidget(right_splitter, 1)
#
#         self.init_visualization()
#
#     def update_safe_label(self):
#         value = self.safe_distance.value() / 10.0
#         self.safe_label.setText(f"{value:.1f}")
#
#     def update_step_label(self):
#         value = self.step_size.value() / 1.0
#         self.step_label.setText(f"{value:.1f}")
#
#     def update_height_label(self):
#         value = self.min_height_slider.value() / 10.0
#         self.height_label.setText(f"{value:.1f}")
#         self.min_flight_height = value
#         self.init_visualization()
#
#     def update_animation_speed_label(self):
#         # 转换为0.001到0.01秒之间的值
#         value = self.animation_speed.value() / 1000.0
#         self.animation_speed_label.setText(f"{value:.3f}s")
#
#     def create_surface_arrays(self):
#         """专门创建表面绘制所需的数组，确保类型正确"""
#         x = np.linspace(0, 60, 2, dtype=np.float64)
#         y = np.linspace(0, 60, 2, dtype=np.float64)
#         X, Y = np.meshgrid(x, y)
#
#         if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
#             raise TypeError("网格必须是numpy数组")
#
#         return X, Y
#
#     def init_visualization(self):
#         ax = self.canvas.axes
#         ax.clear()
#         self.path_info.clear()
#
#         try:
#             X, Y = self.create_surface_arrays()
#
#             # 绘制地面参考平面
#             Z_ground = np.zeros(X.shape, dtype=np.float64)
#             ax.plot_surface(X, Y, Z_ground, color='lightgray', alpha=0.3)
#
#             # 绘制建筑物
#             for building in self.buildings:
#                 faces = building.get_faces()
#                 poly3d = Poly3DCollection(faces, facecolors=building.color,
#                                           edgecolors='black', linewidths=1)
#                 ax.add_collection3d(poly3d)
#
#             # 绘制最低飞行高度参考平面
#             z_height = self.min_flight_height
#             Z_height = np.full(X.shape, z_height, dtype=np.float64)
#             ax.plot([], [], [], 'blue', alpha=0.1, label=f'最低飞行高度: {z_height:.1f}')
#             ax.plot_surface(X, Y, Z_height, color='blue', alpha=0.1)
#
#             # 设置坐标轴
#             ax.set_xlabel('X轴', fontsize=8)
#             ax.set_ylabel('Y轴', fontsize=8)
#             ax.set_zlabel('Z轴', fontsize=8)
#             ax.set_title('无人机三维路径规划', fontsize=10)
#             ax.tick_params(axis='both', which='major', labelsize=7)
#             ax.set_xlim(0, 60)
#             ax.set_ylim(0, 60)
#             ax.set_zlim(0, 30)
#             ax.legend(fontsize=8)
#
#             self.canvas.draw()
#
#         except Exception as e:
#             QMessageBox.critical(self, "可视化错误", f"创建3D表面时出错: {str(e)}")
#             print(f"调试信息: 错误类型={type(e)}, 错误内容={str(e)}")
#
#     def show_final_path(self, show_all=True):
#         ax = self.canvas.axes
#         ax.clear()
#
#         try:
#             X, Y = self.create_surface_arrays()
#
#             # 绘制地面参考平面
#             Z_ground = np.zeros(X.shape, dtype=np.float64)
#             ax.plot_surface(X, Y, Z_ground, color='lightgray', alpha=0.3)
#
#             # 绘制最低飞行高度参考平面
#             z_height = self.min_flight_height
#             Z_height = np.full(X.shape, z_height, dtype=np.float64)
#             ax.plot([], [], [], 'blue', alpha=0.1, label=f'最低飞行高度: {z_height:.1f}')
#             ax.plot_surface(X, Y, Z_height, color='blue', alpha=0.1)
#
#             # 绘制建筑物
#             for building in self.buildings:
#                 faces = building.get_faces()
#                 poly3d = Poly3DCollection(faces, facecolors=building.color,
#                                           edgecolors='black', linewidths=1)
#                 ax.add_collection3d(poly3d)
#
#             # 获取起点和终点
#             start = (float(self.start_x.text()), float(self.start_y.text()), float(self.start_z.text()))
#             goal = (float(self.goal_x.text()), float(self.goal_y.text()), float(self.goal_z.text()))
#
#             # 绘制起点和终点
#             ax.scatter(start[0], start[1], start[2], color='green', s=150, label='起点', marker='o')
#             ax.scatter(goal[0], goal[1], goal[2], color='red', s=150, label='终点', marker='*')
#
#             # 绘制RRT树
#             if self.rrt_tree and len(self.rrt_tree) > 1:
#                 draw_nodes = self.rrt_tree[-1000:] if len(self.rrt_tree) > 1000 else self.rrt_tree
#                 for node in draw_nodes:
#                     if node.parent:
#                         ax.plot(
#                             [node.x, node.parent.x],
#                             [node.y, node.parent.y],
#                             [node.z, node.parent.z],
#                             'gray', linewidth=0.5, alpha=0.3
#                         )
#
#             # 绘制路径
#             if self.path and len(self.path) > 1:
#                 # 确定要显示的路径部分
#                 if show_all or self.current_step >= len(self.path) - 1:
#                     display_path = self.path
#                 else:
#                     display_path = self.path[:self.current_step + 1]
#
#                 x = [p[0] for p in display_path]
#                 y = [p[1] for p in display_path]
#                 z = [p[2] for p in display_path]
#
#                 ax.plot(x, y, z, 'b-', linewidth=2, label='规划路径')
#                 ax.scatter(x, y, z, color='blue', s=50, alpha=0.7, marker='o')
#
#                 # 如果不是显示全部路径，突出显示当前位置
#                 if not show_all and self.current_step < len(self.path):
#                     current_pos = self.path[self.current_step]
#                     ax.scatter(current_pos[0], current_pos[1], current_pos[2],
#                                color='yellow', s=100, alpha=1.0, marker='*')
#
#                 # 显示路径信息
#                 self.path_info.clear()
#                 total_length = 0.0
#                 info_text = "路径点坐标列表：\n\n"
#                 info_text += f"起点: ({start[0]:.1f}, {start[1]:.1f}, {start[2]:.1f})\n\n"
#
#                 for i, (px, py, pz) in enumerate(self.path):
#                     info_text += f"步骤 {i}: ({px:.1f}, {py:.1f}, {pz:.1f})"
#                     if i > 0 and i < len(self.path) - 1 and pz < self.min_flight_height - 0.1:
#                         info_text += " (低于最低高度)\n"
#                     else:
#                         info_text += "\n"
#
#                     if i > 0:
#                         dx = px - self.path[i - 1][0]
#                         dy = py - self.path[i - 1][1]
#                         dz = pz - self.path[i - 1][2]
#                         total_length += math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
#
#                 info_text += f"\n终点: ({goal[0]:.1f}, {goal[1]:.1f}, {goal[2]:.1f})\n"
#                 info_text += f"\n路径总长度: {total_length:.2f} 单位"
#                 info_text += f"\n规划时间: {self.plan_time:.4f} 秒"
#                 info_text += f"\n最低飞行高度: {self.min_flight_height:.1f} 单位"
#                 self.path_info.setText(info_text)
#
#             # 设置坐标轴
#             ax.set_xlabel('X轴', fontsize=8)
#             ax.set_ylabel('Y轴', fontsize=8)
#             ax.set_zlabel('Z轴', fontsize=8)
#             ax.set_title('无人机三维路径规划', fontsize=10)
#             ax.tick_params(axis='both', which='major', labelsize=7)
#             ax.set_xlim(0, 60)
#             ax.set_ylim(0, 60)
#             ax.set_zlim(0, 30)
#             ax.legend(fontsize=8)
#
#             self.canvas.draw()
#
#         except Exception as e:
#             QMessageBox.critical(self, "可视化错误", f"显示路径时出错: {str(e)}")
#             print(f"调试信息: 错误类型={type(e)}, 错误内容={str(e)}")
#
#     def plan_path(self):
#         if self.is_planning:
#             return
#
#         try:
#             start = (
#                 float(self.start_x.text()),
#                 float(self.start_y.text()),
#                 float(self.start_z.text())
#             )
#             goal = (
#                 float(self.goal_x.text()),
#                 float(self.goal_y.text()),
#                 float(self.goal_z.text())
#             )
#
#             max_iter = int(self.max_iter.text())
#             goal_sample_rate = int(self.goal_sample.text())
#
#             if goal_sample_rate < 0 or goal_sample_rate > 100:
#                 QMessageBox.critical(self, "参数错误", "目标采样概率必须在0-100之间！")
#                 return
#
#             self.plan_button.setEnabled(False)
#             self.step_button.setEnabled(False)
#             self.full_button.setEnabled(False)
#             self.is_planning = True
#             self.time_label.setText("规划中...")
#             self.path_info.setText("正在规划路径，请稍候...")
#             self.current_step = 0
#
#             planner = OptimizedRRTPlanner(
#                 start,
#                 goal,
#                 self.buildings,
#                 safe_distance=self.safe_distance.value() / 10.0,
#                 base_step_size=self.step_size.value(),
#                 min_height=self.min_flight_height,
#                 max_iter=max_iter,
#                 goal_sample_rate=goal_sample_rate
#             )
#
#             self.thread = PlanningThread(planner)
#             self.thread.finished.connect(self.on_planning_finished)
#             self.thread.error.connect(self.on_planning_error)
#             self.thread.start()
#
#             self.plan_start_time = time.time()
#
#         except ValueError:
#             QMessageBox.critical(self, "输入错误", "请输入有效的数值！")
#             self.plan_button.setEnabled(True)
#             self.is_planning = False
#
#     def on_planning_finished(self, path, tree):
#         self.plan_time = time.time() - self.plan_start_time
#         self.time_label.setText(f"规划时间: {self.plan_time:.4f} 秒")
#         self.path = path
#         self.rrt_tree = tree
#         self.current_step = 0
#
#         if self.path and len(self.path) > 1:
#             self.show_final_path()
#             self.step_button.setEnabled(True)
#             self.full_button.setEnabled(True)
#         else:
#             QMessageBox.warning(self, "规划结果", f"在{self.max_iter.text()}次迭代内未找到完整路径！")
#             self.show_final_path()
#             self.step_button.setEnabled(False)
#             self.full_button.setEnabled(False)
#
#         self.plan_button.setEnabled(True)
#         self.is_planning = False
#
#     def on_planning_error(self, error_msg):
#         QMessageBox.critical(self, "规划错误", f"路径规划过程中发生错误: {error_msg}")
#         self.time_label.setText("规划时间: 错误")
#         self.plan_button.setEnabled(True)
#         self.step_button.setEnabled(False)
#         self.full_button.setEnabled(False)
#         self.is_planning = False
#
#     def reset_view(self):
#         if self.is_planning:
#             return
#
#         self.animation_timer.stop()
#         self.path = None
#         self.rrt_tree = None
#         self.current_step = 0
#         self.time_label.setText("规划时间: -- 秒")
#         self.step_button.setEnabled(False)
#         self.full_button.setEnabled(False)
#         self.init_visualization()
#
#     def step_demo(self):
#         """单步演示路径"""
#         if self.path and self.current_step < len(self.path) - 1:
#             self.current_step += 1
#             self.show_final_path(show_all=False)
#         else:
#             QMessageBox.information(self, "提示", "已到达路径终点")
#
#     def full_demo(self):
#         """完整演示路径"""
#         if self.path:
#             # 停止任何正在进行的动画
#             self.animation_timer.stop()
#             # 重置到起点
#             self.current_step = 0
#             self.show_final_path(show_all=False)
#             # 设置动画间隔时间，转换为毫秒
#             interval = int(self.animation_speed.value() * 1)  # 1-10对应1-10毫秒
#             self.animation_timer.setInterval(interval)
#             self.animation_timer.start()
#
#     def animate_step(self):
#         """动画演示的每一步"""
#         if self.current_step < len(self.path) - 1:
#             self.current_step += 1
#             self.show_final_path(show_all=False)
#         else:
#             self.animation_timer.stop()
#
#
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = DronePathVisualizer()
#     window.show()
#     sys.exit(app.exec_())

# import sys
# import time
# import random
# import math
# import numpy as np
# from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
#                              QGroupBox, QFormLayout, QLineEdit, QSlider, QLabel, QPushButton,
#                              QSplitter, QScrollArea, QTextEdit, QMessageBox)
# from PyQt5.QtCore import Qt, QThread, pyqtSignal
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#
# # 设置matplotlib支持中文显示
# plt.rcParams["font.family"] = ["SimHei"]
# plt.rcParams["axes.unicode_minus"] = False
# plt.rcParams["figure.titlesize"] = 10
#
#
# # 节点类 - 与多障碍物版本保持一致
# class Node:
#     __slots__ = ['x', 'y', 'z', 'parent', 'cost']
#
#     def __init__(self, x, y, z, parent=None):
#         self.x = x  # x坐标
#         self.y = y  # y坐标
#         self.z = z  # z坐标
#         self.parent = parent  # 父节点
#         self.cost = 0.0  # 路径成本
#
#     def distance_sq_to(self, other):
#         """计算与另一个节点的平方距离（避免开方，提高效率）"""
#         dx = self.x - other.x
#         dy = self.y - other.y
#         dz = self.z - other.z
#         return dx ** 2 + dy ** 2 + dz ** 2
#
#     def distance_to(self, other):
#         """计算与另一个节点的直线距离"""
#         return math.sqrt(self.distance_sq_to(other))
#
#     def update_cost(self):
#         """根据父节点更新路径成本"""
#         if self.parent:
#             self.cost = self.parent.cost + self.distance_to(self.parent)
#
#
# # 建筑物类 - 与多障碍物版本坐标逻辑一致
# class Building:
#     __slots__ = ['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max',
#                  'safe_x_min', 'safe_x_max', 'safe_y_min', 'safe_y_max',
#                  'safe_z_min', 'safe_z_max', 'color']
#
#     def __init__(self, x_center, y_center, z_base, width, length, height, color):
#         # 障碍物本身的坐标范围（基于中心点计算）
#         self.x_min = x_center - width / 2
#         self.x_max = x_center + width / 2
#         self.y_min = y_center - length / 2
#         self.y_max = y_center + length / 2
#         self.z_min = z_base  # 底部z坐标
#         self.z_max = z_base + height  # 顶部z坐标
#         self.color = color
#
#         # 安全边界（后续通过set_safe_bounds设置）
#         self.safe_x_min = 0
#         self.safe_x_max = 0
#         self.safe_y_min = 0
#         self.safe_y_max = 0
#         self.safe_z_min = 0
#         self.safe_z_max = 0
#
#     def set_safe_bounds(self, safe_distance):
#         """设置障碍物的安全边界（扩大障碍物范围，避免碰撞）"""
#         self.safe_x_min = self.x_min - safe_distance
#         self.safe_x_max = self.x_max + safe_distance
#         self.safe_y_min = self.y_min - safe_distance
#         self.safe_y_max = self.y_max + safe_distance
#         self.safe_z_min = self.z_min - safe_distance
#         self.safe_z_max = self.z_max + safe_distance
#
#     def get_faces(self):
#         """返回建筑物的6个面，用于3D绘制"""
#         # 底部
#         bottom = [(self.x_min, self.y_min, self.z_min),
#                   (self.x_max, self.y_min, self.z_min),
#                   (self.x_max, self.y_max, self.z_min),
#                   (self.x_min, self.y_max, self.z_min)]
#
#         # 顶部
#         top = [(self.x_min, self.y_min, self.z_max),
#                (self.x_max, self.y_min, self.z_max),
#                (self.x_max, self.y_max, self.z_max),
#                (self.x_min, self.y_max, self.z_max)]
#
#         # 前面
#         front = [(self.x_min, self.y_min, self.z_min),
#                  (self.x_max, self.y_min, self.z_min),
#                  (self.x_max, self.y_min, self.z_max),
#                  (self.x_min, self.y_min, self.z_max)]
#
#         # 后面
#         back = [(self.x_min, self.y_max, self.z_min),
#                 (self.x_max, self.y_max, self.z_min),
#                 (self.x_max, self.y_max, self.z_max),
#                 (self.x_min, self.y_max, self.z_max)]
#
#         # 左面
#         left = [(self.x_min, self.y_min, self.z_min),
#                 (self.x_min, self.y_max, self.z_min),
#                 (self.x_min, self.y_max, self.z_max),
#                 (self.x_min, self.y_min, self.z_max)]
#
#         # 右面
#         right = [(self.x_max, self.y_min, self.z_min),
#                  (self.x_max, self.y_max, self.z_min),
#                  (self.x_max, self.y_max, self.z_max),
#                  (self.x_max, self.y_min, self.z_max)]
#
#         return [bottom, top, front, back, left, right]
#
#
# # RRT规划器类 - 坐标逻辑与多障碍物版本统一
# class RRTPlanner:
#     def __init__(self, start, goal, buildings,
#                  safe_distance=3.0, step_size=2.0, min_height=0.5,
#                  max_iter=5000, goal_sample_rate=20,
#                  x_range=(0, 60), y_range=(0, 60), z_range=(0, 30)):
#         # 起点和终点（三维坐标）
#         self.start = Node(start[0], start[1], start[2])
#         self.goal = Node(goal[0], goal[1], goal[2])
#
#         self.buildings = buildings  # 障碍物列表
#         self.safe_distance = safe_distance  # 安全距离
#         self.safe_distance_sq = safe_distance ** 2  # 安全距离平方
#         self.min_height = min_height  # 最低飞行高度
#
#         # 算法参数
#         self.max_iter = max_iter  # 最大迭代次数
#         self.goal_sample_rate = goal_sample_rate  # 目标点采样概率（%）
#         self.step_size = step_size  # 步长
#
#         # 坐标范围（三维）
#         self.x_range = x_range
#         self.y_range = y_range
#         self.z_range = z_range
#
#         # 目标阈值（接近目标的判定距离）
#         self.goal_threshold = step_size * 1.5
#         self.goal_threshold_sq = self.goal_threshold ** 2
#
#         # 初始化树
#         self.tree = [self.start]
#         self.start.update_cost()
#
#         # 设置障碍物安全边界
#         for b in self.buildings:
#             b.set_safe_bounds(safe_distance)
#
#         # 记录规划过程
#         self.planning_history = []
#         self.final_path = None
#
#     def is_collision(self, node):
#         """检查节点是否碰撞（与多障碍物版本逻辑一致）"""
#         x, y, z = node.x, node.y, node.z
#
#         # 检查是否超出坐标范围
#         if (x < self.x_range[0] or x > self.x_range[1] or
#                 y < self.y_range[0] or y > self.y_range[1] or
#                 z < self.z_range[0] or z > self.z_range[1]):
#             return True  # 超出范围，视为碰撞
#
#         # 检查是否与障碍物或其安全边界碰撞
#         for b in self.buildings:
#             # 快速排除：不在障碍物安全边界范围内，直接跳过
#             if (x < b.safe_x_min or x > b.safe_x_max or
#                     y < b.safe_y_min or y > b.safe_y_max or
#                     z < b.safe_z_min or z > b.safe_z_max):
#                 continue
#
#             # 计算节点到障碍物的最短距离（三维）
#             dx = max(b.x_min - x, 0, x - b.x_max)  # x方向距离（若在范围内则为0）
#             dy = max(b.y_min - y, 0, y - b.y_max)  # y方向距离
#             dz = max(b.z_min - z, 0, z - b.z_max)  # z方向距离
#
#             # 距离平方小于安全距离平方，视为碰撞
#             if dx * dx + dy * dy + dz * dz < self.safe_distance_sq:
#                 return True
#
#         # 检查是否低于最低飞行高度（起点和终点除外）
#         if node != self.start and node != self.goal and z < self.min_height:
#             return True
#
#         return False  # 无碰撞
#
#     def is_segment_collision(self, node1, node2):
#         """检查两个节点之间的线段是否碰撞"""
#         # 线段采样检测
#         distance = node1.distance_to(node2)
#         if distance == 0:
#             return False
#
#         # 采样点数（根据距离动态调整）
#         num_samples = max(5, int(distance / (self.step_size / 2)))
#
#         for i in range(1, num_samples):
#             ratio = i / num_samples
#             x = node1.x + (node2.x - node1.x) * ratio
#             y = node1.y + (node2.y - node1.y) * ratio
#             z = node1.z + (node2.z - node1.z) * ratio
#
#             if self.is_collision(Node(x, y, z)):
#                 return True
#
#         return False
#
#     def get_random_node(self):
#         """生成随机节点（三维坐标）"""
#         # 以一定概率直接采样目标点
#         if random.randint(0, 100) < self.goal_sample_rate:
#             return self.goal
#
#         # 随机生成三维坐标点
#         x = random.uniform(self.x_range[0], self.x_range[1])
#         y = random.uniform(self.y_range[0], self.y_range[1])
#         z = random.uniform(self.z_range[0], self.z_range[1])
#
#         return Node(x, y, z)
#
#     def find_nearest_node(self, random_node):
#         """在树中找到离随机节点最近的节点"""
#         min_dist_sq = float('inf')
#         nearest_node = None
#
#         for node in self.tree:
#             dist_sq = node.distance_sq_to(random_node)
#             if dist_sq < min_dist_sq:
#                 min_dist_sq = dist_sq
#                 nearest_node = node
#
#         return nearest_node
#
#     def steer(self, from_node, to_node):
#         """从from_node向to_node移动固定步长，生成新节点"""
#         distance = from_node.distance_to(to_node)
#
#         # 如果距离小于步长，直接返回目标节点
#         if distance <= self.step_size:
#             return to_node
#
#         # 计算单位向量
#         ratio = self.step_size / distance
#         x = from_node.x + (to_node.x - from_node.x) * ratio
#         y = from_node.y + (to_node.y - from_node.y) * ratio
#         z = from_node.z + (to_node.z - from_node.z) * ratio
#
#         return Node(x, y, z)
#
#     def plan(self):
#         """RRT路径规划主函数"""
#         # 检查起点和终点是否合法
#         if self.is_collision(self.start):
#             return None, "起点在障碍物或边界外"
#         if self.is_collision(self.goal):
#             return None, "终点在障碍物或边界外"
#
#         # 检查起点到终点是否可以直接到达
#         if not self.is_segment_collision(self.start, self.goal):
#             self.final_path = [self.start, self.goal]
#             self.planning_history.append((self.tree.copy(), None))
#             return self._extract_path(self.goal), "找到直接路径"
#
#         # 开始RRT搜索
#         for i in range(self.max_iter):
#             # 生成随机节点
#             random_node = self.get_random_node()
#
#             # 找到最近节点
#             nearest_node = self.find_nearest_node(random_node)
#
#             # 生成新节点
#             new_node = self.steer(nearest_node, random_node)
#             new_node.parent = nearest_node
#             new_node.update_cost()
#
#             # 检查新节点是否碰撞
#             if not self.is_collision(new_node) and not self.is_segment_collision(nearest_node, new_node):
#                 # 添加到树中
#                 self.tree.append(new_node)
#                 self.planning_history.append((self.tree.copy(), new_node))
#
#                 # 检查是否到达目标附近
#                 if new_node.distance_sq_to(self.goal) < self.goal_threshold_sq:
#                     # 连接到目标点
#                     self.goal.parent = new_node
#                     self.goal.update_cost()
#                     if not self.is_segment_collision(new_node, self.goal):
#                         self.final_path = self._extract_path(self.goal)
#                         self.planning_history.append((self.tree.copy(), self.goal))
#                         return self.final_path, f"在迭代 {i + 1} 次后找到路径"
#
#         # 未找到路径
#         return None, f"超过最大迭代次数({self.max_iter})，未找到路径"
#
#     def _extract_path(self, node):
#         """从目标节点回溯到起点，提取路径"""
#         path = []
#         current = node
#
#         while current is not None:
#             path.append((current.x, current.y, current.z))
#             current = current.parent
#
#         # 反转路径，从起点到终点
#         return path[::-1]
#
#
# # matplotlib画布类
# class MplCanvas(FigureCanvas):
#     def __init__(self, parent=None, width=5, height=4, dpi=100):
#         self.fig = plt.figure(figsize=(width, height), dpi=dpi)
#         self.axes = self.fig.add_subplot(111, projection='3d')
#         super(MplCanvas, self).__init__(self.fig)
#         self.setParent(parent)
#         self.fig.tight_layout()
#
#
# # 规划线程类
# class PlanningThread(QThread):
#     finished = pyqtSignal(list, str)
#     error = pyqtSignal(str)
#
#     def __init__(self, planner):
#         super().__init__()
#         self.planner = planner
#
#     def run(self):
#         try:
#             path, message = self.planner.plan()
#             self.finished.emit(path, message)
#         except Exception as e:
#             self.error.emit(str(e))
#
#
# # 主窗口类
# class RRTPathVisualizer(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("RRT路径规划演示 (坐标逻辑统一版)")
#         self.resize(1200, 700)
#
#         self.is_planning = False
#         self.path = None
#         self.plan_time = 0.0
#         self.planner = None
#         self.current_step = 0  # 用于演示的步骤
#         self.demo_path = None  # 插值后的演示路径
#
#         # 障碍物设置（与多障碍物版本坐标逻辑一致）
#         self.buildings = [
#             Building(15, 15, 0, 10, 10, 20, (0.6, 0.2, 0.2, 0.8)),  # 中心(15,15)，宽10，长10，高20
#             Building(35, 35, 0, 12, 12, 18, (0.2, 0.6, 0.2, 0.8)),  # 中心(35,35)，宽12，长12，高18
#         ]
#
#         # 初始化UI
#         self.init_ui()
#         self.init_visualization()
#
#     def init_ui(self):
#         """初始化UI界面"""
#         main_widget = QWidget()
#         self.setCentralWidget(main_widget)
#         main_layout = QHBoxLayout(main_widget)
#         main_layout.setContentsMargins(5, 5, 5, 5)
#
#         # 左侧控制面板
#         control_panel = QWidget()
#         control_layout = QVBoxLayout(control_panel)
#         control_panel.setFixedWidth(250)
#
#         # 起点坐标输入
#         start_group = QGroupBox("起点坐标")
#         start_layout = QFormLayout()
#         start_layout.setRowWrapPolicy(QFormLayout.DontWrapRows)
#         start_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
#         self.start_x = QLineEdit("0")
#         self.start_x.setFixedWidth(80)
#         self.start_y = QLineEdit("0")
#         self.start_y.setFixedWidth(80)
#         self.start_z = QLineEdit("0")
#         self.start_z.setFixedWidth(80)
#         start_layout.addRow("X:", self.start_x)
#         start_layout.addRow("Y:", self.start_y)
#         start_layout.addRow("Z:", self.start_z)
#         start_group.setLayout(start_layout)
#
#         # 终点坐标输入
#         goal_group = QGroupBox("终点坐标")
#         goal_layout = QFormLayout()
#         goal_layout.setRowWrapPolicy(QFormLayout.DontWrapRows)
#         goal_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
#         self.goal_x = QLineEdit("50")
#         self.goal_x.setFixedWidth(80)
#         self.goal_y = QLineEdit("50")
#         self.goal_y.setFixedWidth(80)
#         self.goal_z = QLineEdit("0")
#         self.goal_z.setFixedWidth(80)
#         goal_layout.addRow("X:", self.goal_x)
#         goal_layout.addRow("Y:", self.goal_y)
#         goal_layout.addRow("Z:", self.goal_z)
#         goal_group.setLayout(goal_layout)
#
#         # 安全距离设置
#         safe_group = QGroupBox("安全距离")
#         safe_layout = QVBoxLayout()
#         self.safe_distance = QSlider(Qt.Horizontal)
#         self.safe_distance.setRange(1, 100)
#         self.safe_distance.setValue(30)  # 3.0
#         self.safe_label = QLabel("3.0")
#         self.safe_distance.valueChanged.connect(self.update_safe_label)
#         safe_layout.addWidget(self.safe_distance)
#         safe_layout.addWidget(self.safe_label)
#         safe_group.setLayout(safe_layout)
#
#         # 最低飞行高度设置
#         height_group = QGroupBox("最低飞行高度")
#         height_layout = QVBoxLayout()
#         self.min_height_slider = QSlider(Qt.Horizontal)
#         self.min_height_slider.setRange(0, 50)
#         self.min_height_slider.setValue(10)  # 1.0
#         self.height_label = QLabel("1.0")
#         self.min_height_slider.valueChanged.connect(self.update_height_label)
#         height_layout.addWidget(self.min_height_slider)
#         height_layout.addWidget(self.height_label)
#         height_group.setLayout(height_layout)
#
#         # RRT参数设置
#         rrt_group = QGroupBox("RRT算法参数")
#         rrt_layout = QFormLayout()
#         rrt_layout.setRowWrapPolicy(QFormLayout.DontWrapRows)
#         rrt_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
#         self.max_iter = QLineEdit("5000")
#         self.max_iter.setFixedWidth(80)
#         self.step_size = QLineEdit("2.0")
#         self.step_size.setFixedWidth(80)
#         self.goal_sample = QLineEdit("20")
#         self.goal_sample.setFixedWidth(80)
#         rrt_layout.addRow("最大迭代次数:", self.max_iter)
#         rrt_layout.addRow("步长:", self.step_size)
#         rrt_layout.addRow("目标采样率(%):", self.goal_sample)
#         rrt_group.setLayout(rrt_layout)
#
#         # 按钮布局
#         button_layout = QVBoxLayout()
#         self.plan_button = QPushButton("规划路径")
#         self.plan_button.setMinimumHeight(30)
#         self.plan_button.clicked.connect(self.plan_path)
#         self.reset_button = QPushButton("重置视图")
#         self.reset_button.setMinimumHeight(30)
#         self.reset_button.clicked.connect(self.reset_view)
#
#         # 演示按钮
#         self.step_demo_button = QPushButton("单步演示")
#         self.step_demo_button.setMinimumHeight(30)
#         self.step_demo_button.clicked.connect(self.step_demo)
#         self.step_demo_button.setEnabled(False)
#
#         self.full_demo_button = QPushButton("完整演示")
#         self.full_demo_button.setMinimumHeight(30)
#         self.full_demo_button.clicked.connect(self.full_demo)
#         self.full_demo_button.setEnabled(False)
#
#         button_layout.addWidget(self.plan_button)
#         button_layout.addWidget(self.reset_button)
#         button_layout.addWidget(self.step_demo_button)
#         button_layout.addWidget(self.full_demo_button)
#         button_layout.addStretch()
#
#         # 状态显示
#         self.time_label = QLabel("规划时间: -- 秒")
#         self.status_label = QLabel("状态: 就绪")
#         status_group = QGroupBox("状态信息")
#         status_layout = QVBoxLayout()
#         status_layout.addWidget(self.time_label)
#         status_layout.addWidget(self.status_label)
#         status_group.setLayout(status_layout)
#
#         # 组装控制面板
#         control_layout.addWidget(start_group)
#         control_layout.addWidget(goal_group)
#         control_layout.addWidget(safe_group)
#         control_layout.addWidget(height_group)
#         control_layout.addWidget(rrt_group)
#         control_layout.addWidget(status_group)
#         control_layout.addLayout(button_layout)
#         control_layout.addStretch()
#         control_layout.setSpacing(8)
#
#         # 右侧分割器
#         right_splitter = QSplitter(Qt.Vertical)
#
#         # 可视化区域
#         visual_widget = QWidget()
#         visual_layout = QVBoxLayout(visual_widget)
#
#         self.canvas = MplCanvas(self, width=8, height=6, dpi=100)
#         self.toolbar = NavigationToolbar(self.canvas, self)
#         visual_layout.addWidget(self.toolbar)
#         visual_layout.addWidget(self.canvas)
#
#         # 路径信息区域
#         info_widget = QWidget()
#         info_layout = QVBoxLayout(info_widget)
#         info_title = QLabel("路径点坐标信息")
#         info_title.setAlignment(Qt.AlignCenter)
#         info_title.setStyleSheet("font-weight: bold; font-size: 12px;")
#         self.path_info = QTextEdit()
#         self.path_info.setReadOnly(True)
#         self.path_info.setMinimumHeight(150)
#         scroll_area = QScrollArea()
#         scroll_area.setWidgetResizable(True)
#         scroll_area.setWidget(self.path_info)
#         info_layout.addWidget(info_title)
#         info_layout.addWidget(scroll_area)
#
#         right_splitter.addWidget(visual_widget)
#         right_splitter.addWidget(info_widget)
#         right_splitter.setSizes([500, 200])
#
#         main_layout.addWidget(control_panel)
#         main_layout.addWidget(right_splitter, 1)
#
#     def update_safe_label(self):
#         """更新安全距离显示"""
#         value = self.safe_distance.value() / 10.0
#         self.safe_label.setText(f"{value:.1f}")
#
#     def update_height_label(self):
#         """更新最低高度显示"""
#         value = self.min_height_slider.value() / 10.0
#         self.height_label.setText(f"{value:.1f}")
#
#     def create_surface_arrays(self):
#         """创建地面和高度参考面的数组"""
#         x = np.linspace(0, 60, 2, dtype=np.float64)
#         y = np.linspace(0, 60, 2, dtype=np.float64)
#         return np.meshgrid(x, y)
#
#     def init_visualization(self):
#         """初始化可视化界面"""
#         ax = self.canvas.axes
#         ax.clear()
#         self.path_info.clear()
#
#         try:
#             X, Y = self.create_surface_arrays()
#
#             # 绘制地面参考平面
#             Z_ground = np.zeros(X.shape, dtype=np.float64)
#             ax.plot_surface(X, Y, Z_ground, color='lightgray', alpha=0.3)
#
#             # 绘制建筑物
#             for building in self.buildings:
#                 faces = building.get_faces()
#                 poly3d = Poly3DCollection(faces, facecolors=building.color,
#                                           edgecolors='black', linewidths=1)
#                 ax.add_collection3d(poly3d)
#
#             # 绘制最低飞行高度参考平面
#             min_height = self.min_height_slider.value() / 10.0
#             Z_height = np.full(X.shape, min_height, dtype=np.float64)
#             ax.plot([], [], [], 'blue', alpha=0.1, label=f'最低飞行高度: {min_height:.1f}')
#             ax.plot_surface(X, Y, Z_height, color='blue', alpha=0.1)
#
#             # 绘制起点和终点
#             start = (float(self.start_x.text()), float(self.start_y.text()), float(self.start_z.text()))
#             goal = (float(self.goal_x.text()), float(self.goal_y.text()), float(self.goal_z.text()))
#             ax.scatter(start[0], start[1], start[2], color='green', s=150, label='起点', marker='o')
#             ax.scatter(goal[0], goal[1], goal[2], color='red', s=150, label='终点', marker='*')
#
#             # 设置坐标轴
#             ax.set_xlabel('X轴', fontsize=8)
#             ax.set_ylabel('Y轴', fontsize=8)
#             ax.set_zlabel('Z轴', fontsize=8)
#             ax.set_title('RRT路径规划演示', fontsize=10)
#             ax.tick_params(axis='both', which='major', labelsize=7)
#             ax.set_xlim(0, 60)
#             ax.set_ylim(0, 60)
#             ax.set_zlim(0, 30)
#             ax.legend(fontsize=8)
#
#             self.canvas.draw()
#
#         except Exception as e:
#             QMessageBox.critical(self, "可视化错误", f"初始化可视化时出错: {str(e)}")
#
#     def interpolate_path(self, path):
#         """对路径进行插值，使演示更平滑"""
#         if not path or len(path) < 2:
#             return path
#
#         interpolated = []
#         step_size = float(self.step_size.text()) / 2  # 演示步长为规划步长的一半
#
#         for i in range(len(path) - 1):
#             p1 = np.array(path[i])
#             p2 = np.array(path[i + 1])
#             segment_length = np.linalg.norm(p2 - p1)
#
#             # 计算需要的插值点数
#             num_points = max(2, int(segment_length / step_size) + 1)
#
#             # 生成插值点
#             for j in range(num_points):
#                 ratio = j / (num_points - 1)
#                 x = p1[0] + (p2[0] - p1[0]) * ratio
#                 y = p1[1] + (p2[1] - p1[1]) * ratio
#                 z = p1[2] + (p2[2] - p1[2]) * ratio
#                 interpolated.append((x, y, z))
#
#         return interpolated
#
#     def show_path(self, step=None):
#         """显示路径（支持分步演示）"""
#         ax = self.canvas.axes
#         ax.clear()
#
#         try:
#             X, Y = self.create_surface_arrays()
#
#             # 绘制地面参考平面
#             Z_ground = np.zeros(X.shape, dtype=np.float64)
#             ax.plot_surface(X, Y, Z_ground, color='lightgray', alpha=0.3)
#
#             # 绘制最低飞行高度参考平面
#             min_height = self.min_height_slider.value() / 10.0
#             Z_height = np.full(X.shape, min_height, dtype=np.float64)
#             ax.plot([], [], [], 'blue', alpha=0.1, label=f'最低飞行高度: {min_height:.1f}')
#             ax.plot_surface(X, Y, Z_height, color='blue', alpha=0.1)
#
#             # 绘制建筑物
#             for building in self.buildings:
#                 faces = building.get_faces()
#                 poly3d = Poly3DCollection(faces, facecolors=building.color,
#                                           edgecolors='black', linewidths=1)
#                 ax.add_collection3d(poly3d)
#
#             # 获取起点和终点
#             start = (float(self.start_x.text()), float(self.start_y.text()), float(self.start_z.text()))
#             goal = (float(self.goal_x.text()), float(self.goal_y.text()), float(self.goal_z.text()))
#
#             # 绘制起点和终点
#             ax.scatter(start[0], start[1], start[2], color='green', s=150, label='起点', marker='o')
#             ax.scatter(goal[0], goal[1], goal[2], color='red', s=150, label='终点', marker='*')
#
#             # 绘制路径
#             if self.path and len(self.path) > 1:
#                 # 准备演示路径
#                 if self.demo_path is None:
#                     self.demo_path = self.interpolate_path(self.path)
#
#                 # 确定要显示的路径部分
#                 display_path = self.demo_path
#                 if step is not None and 0 <= step < len(self.demo_path):
#                     display_path = self.demo_path[:step + 1]
#                     current_point = self.demo_path[step]
#                     ax.scatter(current_point[0], current_point[1], current_point[2],
#                                color='yellow', s=100, label='当前位置', marker='d')
#
#                 x = [p[0] for p in display_path]
#                 y = [p[1] for p in display_path]
#                 z = [p[2] for p in display_path]
#
#                 ax.plot(x, y, z, 'b-', linewidth=2, label='规划路径')
#
#                 # 显示路径信息
#                 self.path_info.clear()
#                 total_length = 0.0
#                 info_text = "路径点坐标列表：\n\n"
#
#                 for i, (px, py, pz) in enumerate(self.path):
#                     info_text += f"点 {i}: ({px:.1f}, {py:.1f}, {pz:.1f})\n"
#                     if i > 0:
#                         dx = px - self.path[i - 1][0]
#                         dy = py - self.path[i - 1][1]
#                         dz = pz - self.path[i - 1][2]
#                         total_length += math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
#
#                 info_text += f"\n路径总长度: {total_length:.2f} 单位"
#                 info_text += f"\n规划时间: {self.plan_time:.4f} 秒"
#                 self.path_info.setText(info_text)
#
#             # 设置坐标轴
#             ax.set_xlabel('X轴', fontsize=8)
#             ax.set_ylabel('Y轴', fontsize=8)
#             ax.set_zlabel('Z轴', fontsize=8)
#             ax.set_title('RRT路径规划演示', fontsize=10)
#             ax.tick_params(axis='both', which='major', labelsize=7)
#             ax.set_xlim(0, 60)
#             ax.set_ylim(0, 60)
#             ax.set_zlim(0, 30)
#             ax.legend(fontsize=8)
#
#             self.canvas.draw()
#
#         except Exception as e:
#             QMessageBox.critical(self, "可视化错误", f"显示路径时出错: {str(e)}")
#
#     def plan_path(self):
#         """开始路径规划"""
#         if self.is_planning:
#             return
#
#         try:
#             # 获取输入参数
#             start = (
#                 float(self.start_x.text()),
#                 float(self.start_y.text()),
#                 float(self.start_z.text())
#             )
#             goal = (
#                 float(self.goal_x.text()),
#                 float(self.goal_y.text()),
#                 float(self.goal_z.text())
#             )
#
#             max_iter = int(self.max_iter.text())
#             step_size = float(self.step_size.text())
#             goal_sample = int(self.goal_sample.text())
#             safe_distance = self.safe_distance.value() / 10.0
#             min_height = self.min_height_slider.value() / 10.0
#
#             # 参数验证
#             if max_iter <= 0:
#                 QMessageBox.critical(self, "参数错误", "最大迭代次数必须为正数！")
#                 return
#             if step_size <= 0:
#                 QMessageBox.critical(self, "参数错误", "步长必须为正数！")
#                 return
#             if goal_sample < 0 or goal_sample > 100:
#                 QMessageBox.critical(self, "参数错误", "目标采样率必须在0-100之间！")
#                 return
#
#             # 初始化规划器
#             self.planner = RRTPlanner(
#                 start, goal, self.buildings,
#                 safe_distance=safe_distance,
#                 step_size=step_size,
#                 min_height=min_height,
#                 max_iter=max_iter,
#                 goal_sample_rate=goal_sample
#             )
#
#             # 更新UI状态
#             self.plan_button.setEnabled(False)
#             self.is_planning = True
#             self.time_label.setText("规划中...")
#             self.status_label.setText("状态: 正在规划路径")
#             self.path_info.setText("正在规划路径，请稍候...")
#             self.step_demo_button.setEnabled(False)
#             self.full_demo_button.setEnabled(False)
#
#             # 启动规划线程
#             self.thread = PlanningThread(self.planner)
#             self.thread.finished.connect(self.on_planning_finished)
#             self.thread.error.connect(self.on_planning_error)
#             self.thread.start()
#
#             self.plan_start_time = time.time()
#
#         except ValueError:
#             QMessageBox.critical(self, "输入错误", "请输入有效的数值！")
#             self.status_label.setText("状态: 输入错误")
#
#     def on_planning_finished(self, path, message):
#         """规划完成回调"""
#         self.plan_time = time.time() - self.plan_start_time
#         self.time_label.setText(f"规划时间: {self.plan_time:.4f} 秒")
#         self.status_label.setText(f"状态: {message}")
#         self.path = path
#         self.current_step = 0  # 重置演示步骤
#         self.demo_path = None  # 重置演示路径
#
#         # 更新UI
#         if self.path and len(self.path) > 1:
#             self.step_demo_button.setEnabled(True)
#             self.full_demo_button.setEnabled(True)
#             self.show_path(0)  # 显示第一步
#         else:
#             QMessageBox.warning(self, "规划结果", message)
#             self.show_path()
#
#         self.plan_button.setEnabled(True)
#         self.is_planning = False
#
#     def on_planning_error(self, error_msg):
#         """规划错误回调"""
#         QMessageBox.critical(self, "规划错误", f"路径规划出错: {error_msg}")
#         self.time_label.setText("规划时间: 错误")
#         self.status_label.setText("状态: 规划出错")
#         self.plan_button.setEnabled(True)
#         self.is_planning = False
#
#     def reset_view(self):
#         """重置视图"""
#         if self.is_planning:
#             return
#
#         self.path = None
#         self.current_step = 0
#         self.demo_path = None
#         self.time_label.setText("规划时间: -- 秒")
#         self.status_label.setText("状态: 就绪")
#         self.step_demo_button.setEnabled(False)
#         self.full_demo_button.setEnabled(False)
#         self.init_visualization()
#
#     def step_demo(self):
#         """单步演示路径"""
#         if not self.demo_path or self.is_planning:
#             return
#
#         # 显示下一步
#         self.current_step += 1
#         if self.current_step >= len(self.demo_path):
#             self.current_step = len(self.demo_path) - 1  # 保持在最后一步
#             self.status_label.setText("状态: 演示完成")
#         else:
#             self.status_label.setText(f"状态: 演示中 (步骤 {self.current_step}/{len(self.demo_path) - 1})")
#
#         self.show_path(self.current_step)
#
#     def full_demo(self):
#         """完整演示路径"""
#         if not self.demo_path or self.is_planning:
#             return
#
#         # 禁用按钮防止重复操作
#         self.step_demo_button.setEnabled(False)
#         self.full_demo_button.setEnabled(False)
#         self.plan_button.setEnabled(False)
#
#         # 重置到第一步
#         self.current_step = 0
#         self.show_path(self.current_step)
#         self.status_label.setText(f"状态: 完整演示中 (步骤 {self.current_step}/{len(self.demo_path) - 1})")
#
#         # 逐步显示路径
#         for step in range(1, len(self.demo_path)):
#             self.current_step = step
#             self.show_path(step)
#             self.status_label.setText(f"状态: 完整演示中 (步骤 {step}/{len(self.demo_path) - 1})")
#             QApplication.processEvents()  # 刷新界面
#             time.sleep(0.05)  # 控制演示速度
#
#         self.status_label.setText("状态: 演示完成")
#         # 重新启用按钮
#         self.step_demo_button.setEnabled(True)
#         self.full_demo_button.setEnabled(True)
#         self.plan_button.setEnabled(True)
#
#
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = RRTPathVisualizer()
#     window.show()
#     sys.exit(app.exec_())
