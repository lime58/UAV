import math
import heapq
import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSlider, QGroupBox, QFormLayout, QMessageBox,
    QSplitter, QTextEdit
)
from PyQt5.QtCore import Qt, QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time

# 设置matplotlib支持中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.titlesize"] = 10


class Node:
    """路径规划中的节点类，代表三维空间中的一个点"""

    def __init__(self, x, y, z, cost=0, parent=None):
        self.x = x
        self.y = y
        self.z = z
        self.cost = cost
        self.parent = parent
        self.heuristic = 0

    def total_cost(self):
        return self.cost + self.heuristic

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return (round(self.x, 1) == round(other.x, 1) and
                round(self.y, 1) == round(other.y, 1) and
                round(self.z, 1) == round(other.z, 1))

    def __lt__(self, other):
        return self.total_cost() < other.total_cost()

    def __hash__(self):
        return hash((round(self.x, 1), round(self.y, 1), round(self.z, 1)))

    def __repr__(self):
        return f"Node({self.x:.1f}, {self.y:.1f}, {self.z:.1f}, cost={self.cost:.2f})"


class Building:
    """建筑物类，代表障碍物"""

    def __init__(self, x, y, z, width, length, height, color=(0.5, 0.5, 0.5, 0.7)):
        self.x = x
        self.y = y
        self.z = z
        self.width = width
        self.length = length
        self.height = height
        self.color = color

        self.x_min = x - width / 2
        self.x_max = x + width / 2
        self.y_min = y - length / 2
        self.y_max = y + length / 2
        self.z_min = z
        self.z_max = z + height


class AStarPlanner:
    """优化后的A*算法路径规划器"""

    def __init__(self, start, goal, buildings, safe_distance=5, step_size=2, min_height=5):
        self.start = Node(start[0], start[1], start[2])
        self.goal = Node(goal[0], goal[1], goal[2])
        self.buildings = buildings
        self.safe_distance = safe_distance
        self.step_size = step_size
        self.min_height = min_height

        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        dz = goal[2] - start[2]
        dist = math.sqrt(dx **2 + dy** 2 + dz **2)
        if dist > 0:
            self.main_dir = (dx / dist, dy / dist, dz / dist)
        else:
            self.main_dir = (0, 0, 0)

        self.directions = [
            (step_size, 0, 0),
            (-step_size, 0, 0),
            (0, step_size, 0),
            (0, -step_size, 0),
            (0, 0, step_size),
            (0, 0, -step_size),
            (self.main_dir[0] * step_size,
             self.main_dir[1] * step_size,
             self.main_dir[2] * step_size)
        ]

    def heuristic(self, node):
        dx = node.x - self.goal.x
        dy = node.y - self.goal.y
        dz = node.z - self.goal.z
        return dx * dx + dy * dy + dz * dz

    def is_collision(self, node):
        x, y, z = node.x, node.y, node.z

        # 最小飞行高度检查
        if (node != self.start and node != self.goal) and z < self.min_height:
            return True

        if x < -5 or x > 65 or y < -5 or y > 65 or z < 0 or z > 35:
            return True

        for building in self.buildings:
            if x < building.x_min:
                dx = building.x_min - x
            elif x > building.x_max:
                dx = x - building.x_max
            else:
                dx = 0

            if y < building.y_min:
                dy = building.y_min - y
            elif y > building.y_max:
                dy = y - building.y_max
            else:
                dy = 0

            if z < building.z_min:
                dz = building.z_min - z
            elif z > building.z_max:
                dz = z - building.z_max
            else:
                dz = 0

            dist_sq = dx * dx + dy * dy + dz * dz
            safe_sq = self.safe_distance * self.safe_distance

            if dist_sq < safe_sq:
                return True

        return False

    def plan(self):
        open_heap = []
        open_set = {}
        closed_set = set()

        self.start.heuristic = self.heuristic(self.start)
        heapq.heappush(open_heap, self.start)
        open_set[self.start] = self.start

        max_iterations = 5000
        iterations = 0

        while open_heap and iterations < max_iterations:
            iterations += 1

            current_node = heapq.heappop(open_heap)

            if current_node not in open_set:
                continue

            del open_set[current_node]

            dx = current_node.x - self.goal.x
            dy = current_node.y - self.goal.y
            dz = current_node.z - self.goal.z
            dist_sq = dx * dx + dy * dy + dz * dz

            if dist_sq < (self.step_size * 1.5) **2:
                goal_node = Node(self.goal.x, self.goal.y, self.goal.z)
                goal_node.parent = current_node
                current_node = goal_node

                path = []
                while current_node:
                    path.append((round(current_node.x, 1),
                                 round(current_node.y, 1),
                                 round(current_node.z, 1)))
                    current_node = current_node.parent
                return path[::-1]

            closed_set.add(current_node)

            for dx_dir, dy_dir, dz_dir in self.directions:
                new_x = current_node.x + dx_dir
                new_y = current_node.y + dy_dir
                new_z = current_node.z + dz_dir

                new_node = Node(new_x, new_y, new_z)
                new_node.parent = current_node

                move_cost = math.sqrt(dx_dir** 2 + dy_dir **2 + dz_dir** 2)
                new_node.cost = current_node.cost + move_cost

                new_node.heuristic = self.heuristic(new_node)

                if self.is_collision(new_node):
                    continue

                if new_node in closed_set:
                    continue

                if new_node in open_set:
                    existing_node = open_set[new_node]
                    if new_node.cost < existing_node.cost:
                        existing_node.cost = new_node.cost
                        existing_node.parent = new_node.parent
                        heapq.heappush(open_heap, existing_node)
                else:
                    heapq.heappush(open_heap, new_node)
                    open_set[new_node] = new_node

        return None


class MplCanvas(FigureCanvas):
    """Matplotlib画布集成到PyQt5"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = plt.figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111, projection='3d')
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.85)


class DronePathVisualizer(QMainWindow):
    """无人机三维路径规划可视化界面"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("A星算法路径规划演示")
        self.resize(1400, 800)

        self.is_planning = False
        self.path = None
        self.current_step = 0
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.animate_step)

        self.buildings = [
            Building(15, 15, 0, 10, 10, 20, (0.6, 0.2, 0.2, 0.8)),
            Building(35, 35, 0, 12, 12, 18, (0.2, 0.6, 0.2, 0.8))
        ]

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
        self.start_z = QLineEdit("10")
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
        self.goal_z = QLineEdit("10")
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

        # 最小飞行高度设置
        min_height_group = QGroupBox("最小飞行高度")
        min_height_layout = QVBoxLayout()
        self.min_height = QSlider(Qt.Horizontal)
        self.min_height.setRange(0, 20)
        self.min_height.setValue(5)
        self.min_height_label = QLabel("5.0")
        self.min_height.valueChanged.connect(self.update_min_height_label)
        min_height_layout.addWidget(self.min_height)
        min_height_layout.addWidget(self.min_height_label)
        min_height_group.setLayout(min_height_layout)

        # 步长设置
        step_group = QGroupBox("步长大小")
        step_layout = QVBoxLayout()
        self.step_size = QSlider(Qt.Horizontal)
        self.step_size.setRange(1, 10)
        self.step_size.setValue(2)
        self.step_label = QLabel("2.0")
        self.step_size.valueChanged.connect(self.update_step_label)
        step_layout.addWidget(self.step_size)
        step_layout.addWidget(self.step_label)
        step_group.setLayout(step_layout)

        # 动画速度设置 - 修改为0.001s到0.01s范围
        speed_group = QGroupBox("动画速度")
        speed_layout = QVBoxLayout()
        self.animation_speed = QSlider(Qt.Horizontal)
        self.animation_speed.setRange(1, 10)  # 1-10对应0.001-0.01秒
        self.animation_speed.setValue(5)
        self.speed_label = QLabel("0.005s")
        self.animation_speed.valueChanged.connect(self.update_speed_label)
        speed_layout.addWidget(self.animation_speed)
        speed_layout.addWidget(self.speed_label)
        speed_group.setLayout(speed_layout)

        # 控制按钮
        button_layout = QVBoxLayout()
        self.plan_button = QPushButton("规划路径")
        self.plan_button.setMinimumHeight(30)
        self.plan_button.clicked.connect(self.plan_path)

        self.step_button = QPushButton("单步演示")
        self.step_button.setMinimumHeight(30)
        self.step_button.clicked.connect(self.step_demo)
        self.step_button.setEnabled(False)

        self.full_button = QPushButton("完整演示")
        self.full_button.setMinimumHeight(30)
        self.full_button.clicked.connect(self.full_demo)
        self.full_button.setEnabled(False)

        self.reset_button = QPushButton("重置视图")
        self.reset_button.setMinimumHeight(30)
        self.reset_button.clicked.connect(self.reset_view)

        button_layout.addWidget(self.plan_button)
        button_layout.addWidget(self.step_button)
        button_layout.addWidget(self.full_button)
        button_layout.addWidget(self.reset_button)
        button_layout.addStretch()

        # 添加到控制面板
        control_layout.addWidget(start_group)
        control_layout.addWidget(goal_group)
        control_layout.addWidget(safe_group)
        control_layout.addWidget(min_height_group)
        control_layout.addWidget(step_group)
        control_layout.addWidget(speed_group)
        control_layout.addLayout(button_layout)
        control_layout.setSpacing(8)

        # 右侧使用垂直拆分器，上方显示可视化，下方显示信息
        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.setHandleWidth(5)

        # 上方可视化区域
        visual_widget = QWidget()
        visual_layout = QVBoxLayout(visual_widget)
        self.canvas = MplCanvas(self, width=8, height=6, dpi=100)
        self.toolbar = NavigationToolbar(self.canvas, self)
        visual_layout.addWidget(self.toolbar)
        visual_layout.addWidget(self.canvas)

        # 下方信息区域
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)

        # 统一信息显示框
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setPlainText("请点击\"规划路径\"按钮开始路径规划")

        info_layout.addWidget(self.info_text)

        # 将可视化区域和信息区域添加到拆分器
        right_splitter.addWidget(visual_widget)
        right_splitter.addWidget(info_widget)
        # 设置初始大小比例（上70%，下30%）
        right_splitter.setSizes([560, 240])

        # 添加到主布局
        main_layout.addWidget(control_panel)
        main_layout.addWidget(right_splitter, 1)

        # 初始化可视化元素
        self.init_visualization()

        # 路径动画相关变量
        self.path_line = None
        self.drone_marker = None

    def update_safe_label(self):
        value = self.safe_distance.value() / 10.0
        self.safe_label.setText(f"{value:.1f}")

    def update_min_height_label(self):
        value = self.min_height.value()
        self.min_height_label.setText(f"{value:.1f}")

    def update_step_label(self):
        value = self.step_size.value()
        self.step_label.setText(f"{value:.1f}")

    # 更新动画速度标签显示
    def update_speed_label(self):
        value = self.animation_speed.value() / 1000.0  # 1-10转换为0.001-0.01秒
        self.speed_label.setText(f"{value:.3f}s")

    def init_visualization(self):
        """初始化可视化界面，绘制障碍物"""
        ax = self.canvas.axes
        ax.clear()

        # 绘制建筑物
        for building in self.buildings:
            faces = [
                [(building.x_min, building.y_min, building.z_min),
                 (building.x_max, building.y_min, building.z_min),
                 (building.x_max, building.y_max, building.z_min),
                 (building.x_min, building.y_max, building.z_min)],
                [(building.x_min, building.y_min, building.z_max),
                 (building.x_max, building.y_min, building.z_max),
                 (building.x_max, building.y_max, building.z_max),
                 (building.x_min, building.y_max, building.z_max)],
                [(building.x_min, building.y_min, building.z_min),
                 (building.x_max, building.y_min, building.z_min),
                 (building.x_max, building.y_min, building.z_max),
                 (building.x_min, building.y_min, building.z_max)],
                [(building.x_min, building.y_max, building.z_min),
                 (building.x_max, building.y_max, building.z_min),
                 (building.x_max, building.y_max, building.z_max),
                 (building.x_min, building.y_max, building.z_max)],
                [(building.x_min, building.y_min, building.z_min),
                 (building.x_min, building.y_max, building.z_min),
                 (building.x_min, building.y_max, building.z_max),
                 (building.x_min, building.y_min, building.z_max)],
                [(building.x_max, building.y_min, building.z_min),
                 (building.x_max, building.y_max, building.z_min),
                 (building.x_max, building.y_max, building.z_max),
                 (building.x_max, building.y_min, building.z_max)]
            ]

            poly3d = Poly3DCollection(faces, facecolors=building.color,
                                      edgecolors='black', linewidths=1)
            ax.add_collection3d(poly3d)

        # 绘制最小飞行高度平面（透明）
        min_h = self.min_height.value()
        x_range = [0, 60]
        y_range = [0, 60]
        xx, yy = np.meshgrid(x_range, y_range)
        z = np.full(xx.shape, min_h, dtype=np.float64)
        ax.plot_surface(xx, yy, z, alpha=0.2, color='blue')
        ax.text(50, 50, min_h + 1, f'最小高度: {min_h}', color='blue')

        # 设置坐标轴
        ax.set_xlabel('X轴', fontsize=8)
        ax.set_ylabel('Y轴', fontsize=8)
        ax.set_zlabel('Z轴', fontsize=8)
        ax.set_title('无人机三维路径规划', fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 60)
        ax.set_zlim(0, 30)

        self.canvas.draw()

    def draw_path_visualization(self, show_all=False):
        ax = self.canvas.axes
        self.init_visualization()

        try:
            start = (float(self.start_x.text()), float(self.start_y.text()), float(self.start_z.text()))
            goal = (float(self.goal_x.text()), float(self.goal_y.text()), float(self.goal_z.text()))

            ax.scatter(start[0], start[1], start[2], color='green', s=150, label='起点', marker='o')
            ax.scatter(goal[0], goal[1], goal[2], color='red', s=150, label='终点', marker='*')

            if self.path and len(self.path) > 1:
                end_idx = len(self.path) if show_all else self.current_step + 1
                x = [p[0] for p in self.path[:end_idx]]
                y = [p[1] for p in self.path[:end_idx]]
                z = [p[2] for p in self.path[:end_idx]]

                if len(x) > 1:
                    self.path_line = ax.plot(x, y, z, 'b-', linewidth=2, label='已规划路径')[0]

                # 移除未规划路径的虚线显示
                # if not show_all and end_idx < len(self.path):
                #     x_rest = [p[0] for p in self.path[end_idx - 1:]]
                #     y_rest = [p[1] for p in self.path[end_idx - 1:]]
                #     z_rest = [p[2] for p in self.path[end_idx - 1:]]
                #     ax.plot(x_rest, y_rest, z_rest, 'b--', linewidth=1, label='未规划路径')

                if self.current_step < len(self.path):
                    drone_x, drone_y, drone_z = self.path[self.current_step]
                    self.drone_marker = ax.scatter(
                        [drone_x], [drone_y], [drone_z],
                        color='blue', s=200, marker='^', label='无人机'
                    )

            ax.legend(fontsize=8)
            self.canvas.draw()

        except ValueError:
            QMessageBox.critical(self, "输入错误", "请输入有效的坐标值！")

    def calculate_path_length(self):
        """计算路径总长度"""
        if not self.path or len(self.path) < 2:
            return 0.0

        total_length = 0.0
        for i in range(1, len(self.path)):
            x1, y1, z1 = self.path[i - 1]
            x2, y2, z2 = self.path[i]
            dx = x2 - x1
            dy = y2 - y1
            dz = z2 - z1
            total_length += math.sqrt(dx **2 + dy** 2 + dz **2)
        return total_length

    def animate_step(self):
        if self.current_step < len(self.path) - 1:
            self.current_step += 1
            self.draw_path_visualization(show_all=False)
        else:
            self.animation_timer.stop()
            self.plan_button.setEnabled(True)
            self.is_planning = False

    def step_demo(self):
        if self.path and self.current_step < len(self.path) - 1:
            self.current_step += 1
            self.draw_path_visualization(show_all=False)
        else:
            QMessageBox.information(self, "提示", "已到达路径终点")

    def full_demo(self):
        if self.path:
            # 设置动画间隔时间，转换为毫秒
            interval = int(self.animation_speed.value() * 1)  # 1-10对应1-10毫秒
            self.animation_timer.setInterval(interval)
            self.animation_timer.start()

    def plan_path(self):
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

            self.plan_button.setEnabled(False)
            self.step_button.setEnabled(False)
            self.full_button.setEnabled(False)
            self.is_planning = True

            # 规划开始时显示状态信息
            self.info_text.setPlainText("正在规划路径，请稍候...\n\n")
            self.info_text.repaint()  # 立即刷新显示

            start_time = time.time()

            # 显示正在初始化规划器
            self.info_text.append("正在初始化A*路径规划器...")
            self.info_text.repaint()

            planner = AStarPlanner(
                start,
                goal,
                self.buildings,
                safe_distance=self.safe_distance.value() / 10.0,
                step_size=self.step_size.value(),
                min_height=self.min_height.value()
            )

            # 显示正在搜索路径
            self.info_text.append("正在搜索最优路径...")
            self.info_text.repaint()

            self.path = planner.plan()

            end_time = time.time()
            plan_time = end_time - start_time

            if self.path:
                # 拼接路径点信息
                path_info = "\n路径点坐标列表：\n"
                path_info += f"起点: ({start[0]}, {start[1]}, {start[2]})\n"
                for idx, point in enumerate(self.path[1:], start=1):
                    path_info += f"步骤 {idx}: ({point[0]}, {point[1]}, {point[2]})\n"
                path_info += "\n"

                # 拼接统计信息
                total_length = self.calculate_path_length()
                stats_info = "路径统计信息:\n"
                stats_info += f"终点坐标: ({goal[0]}, {goal[1]}, {goal[2]})\n"
                stats_info += f"路径总长度: {total_length:.2f} 单位\n"
                stats_info += f"路径点数量: {len(self.path)}\n"
                stats_info += f"规划时间: {plan_time:.4f} 秒\n"
                stats_info += f"最小飞行高度: {self.min_height.value()} 单位\n"
                stats_info += f"安全距离: {self.safe_distance.value() / 10.0:.1f} 单位\n"
                stats_info += f"步长大小: {self.step_size.value()} 单位\n"

                # 整合显示
                self.info_text.setPlainText(path_info + stats_info)

                # 重置当前步骤，不自动显示路径
                self.current_step = 0
                self.init_visualization()  # 只显示障碍物和起点终点，不显示路径
                self.step_button.setEnabled(True)
                self.full_button.setEnabled(True)
            else:
                self.info_text.setPlainText("规划失败，未找到有效路径！\n\n"
                                            "可能原因：\n"
                                            "- 起点或终点被障碍物阻挡\n"
                                            "- 最小飞行高度设置过高\n"
                                            "- 安全距离设置过大\n"
                                            "- 起点与终点之间无法形成有效路径")

            self.plan_button.setEnabled(True)
            self.is_planning = False

        except ValueError:
            self.info_text.setPlainText("输入错误：请输入有效的坐标值！")
            self.plan_button.setEnabled(True)
            self.is_planning = False
        except Exception as e:
            self.info_text.setPlainText(f"规划过程中发生错误: {str(e)}")
            self.plan_button.setEnabled(True)
            self.is_planning = False

    def reset_view(self):
        self.is_planning = False
        self.animation_timer.stop()
        self.path = None
        self.path_line = None
        self.drone_marker = None
        self.current_step = 0

        # 重置信息显示区域
        self.info_text.setPlainText("请点击\"规划路径\"按钮开始路径规划")

        self.init_visualization()
        self.plan_button.setEnabled(True)
        self.step_button.setEnabled(False)
        self.full_button.setEnabled(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DronePathVisualizer()
    window.show()
    sys.exit(app.exec_())

# import math
# import heapq
# import sys
# import numpy as np
# from PyQt5.QtWidgets import (
#     QApplication, QMainWindow, QWidget, QVBoxLayout,
#     QHBoxLayout, QLabel, QLineEdit, QPushButton,
#     QSlider, QGroupBox, QFormLayout, QMessageBox,
#     QSplitter, QTextEdit
# )
# from PyQt5.QtCore import Qt, QTimer
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# import time
#
# # 设置matplotlib支持中文显示
# plt.rcParams["font.family"] = ["SimHei"]
# plt.rcParams["axes.unicode_minus"] = False
# plt.rcParams["figure.titlesize"] = 10
#
#
# class Node:
#     """路径规划中的节点类，代表三维空间中的一个点"""
#
#     def __init__(self, x, y, z, cost=0, parent=None):
#         self.x = x
#         self.y = y
#         self.z = z
#         self.cost = cost
#         self.parent = parent
#         self.heuristic = 0
#
#     def total_cost(self):
#         return self.cost + self.heuristic
#
#     def __eq__(self, other):
#         if not isinstance(other, Node):
#             return False
#         return (round(self.x, 1) == round(other.x, 1) and
#                 round(self.y, 1) == round(other.y, 1) and
#                 round(self.z, 1) == round(other.z, 1))
#
#     def __lt__(self, other):
#         return self.total_cost() < other.total_cost()
#
#     def __hash__(self):
#         return hash((round(self.x, 1), round(self.y, 1), round(self.z, 1)))
#
#     def __repr__(self):
#         return f"Node({self.x:.1f}, {self.y:.1f}, {self.z:.1f}, cost={self.cost:.2f})"
#
#
# class Building:
#     """建筑物类，代表障碍物"""
#
#     def __init__(self, x, y, z, width, length, height, color=(0.5, 0.5, 0.5, 0.7)):
#         self.x = x
#         self.y = y
#         self.z = z
#         self.width = width
#         self.length = length
#         self.height = height
#         self.color = color
#
#         self.x_min = x - width / 2
#         self.x_max = x + width / 2
#         self.y_min = y - length / 2
#         self.y_max = y + length / 2
#         self.z_min = z
#         self.z_max = z + height
#
#
# class AStarPlanner:
#     """优化后的A*算法路径规划器"""
#
#     def __init__(self, start, goal, buildings, safe_distance=5, step_size=2, min_height=5):
#         self.start = Node(start[0], start[1], start[2])
#         self.goal = Node(goal[0], goal[1], goal[2])
#         self.buildings = buildings
#         self.safe_distance = safe_distance
#         self.step_size = step_size
#         self.min_height = min_height
#
#         dx = goal[0] - start[0]
#         dy = goal[1] - start[1]
#         dz = goal[2] - start[2]
#         dist = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
#         if dist > 0:
#             self.main_dir = (dx / dist, dy / dist, dz / dist)
#         else:
#             self.main_dir = (0, 0, 0)
#
#         self.directions = [
#             (step_size, 0, 0),
#             (-step_size, 0, 0),
#             (0, step_size, 0),
#             (0, -step_size, 0),
#             (0, 0, step_size),
#             (0, 0, -step_size),
#             (self.main_dir[0] * step_size,
#              self.main_dir[1] * step_size,
#              self.main_dir[2] * step_size)
#         ]
#
#     def heuristic(self, node):
#         dx = node.x - self.goal.x
#         dy = node.y - self.goal.y
#         dz = node.z - self.goal.z
#         return dx * dx + dy * dy + dz * dz
#
#     def is_collision(self, node):
#         x, y, z = node.x, node.y, node.z
#
#         # 最小飞行高度检查
#         if (node != self.start and node != self.goal) and z < self.min_height:
#             return True
#
#         if x < -5 or x > 65 or y < -5 or y > 65 or z < 0 or z > 35:
#             return True
#
#         for building in self.buildings:
#             if x < building.x_min:
#                 dx = building.x_min - x
#             elif x > building.x_max:
#                 dx = x - building.x_max
#             else:
#                 dx = 0
#
#             if y < building.y_min:
#                 dy = building.y_min - y
#             elif y > building.y_max:
#                 dy = y - building.y_max
#             else:
#                 dy = 0
#
#             if z < building.z_min:
#                 dz = building.z_min - z
#             elif z > building.z_max:
#                 dz = z - building.z_max
#             else:
#                 dz = 0
#
#             dist_sq = dx * dx + dy * dy + dz * dz
#             safe_sq = self.safe_distance * self.safe_distance
#
#             if dist_sq < safe_sq:
#                 return True
#
#         return False
#
#     def plan(self):
#         open_heap = []
#         open_set = {}
#         closed_set = set()
#
#         self.start.heuristic = self.heuristic(self.start)
#         heapq.heappush(open_heap, self.start)
#         open_set[self.start] = self.start
#
#         max_iterations = 5000
#         iterations = 0
#
#         while open_heap and iterations < max_iterations:
#             iterations += 1
#
#             current_node = heapq.heappop(open_heap)
#
#             if current_node not in open_set:
#                 continue
#
#             del open_set[current_node]
#
#             dx = current_node.x - self.goal.x
#             dy = current_node.y - self.goal.y
#             dz = current_node.z - self.goal.z
#             dist_sq = dx * dx + dy * dy + dz * dz
#
#             if dist_sq < (self.step_size * 1.5) ** 2:
#                 goal_node = Node(self.goal.x, self.goal.y, self.goal.z)
#                 goal_node.parent = current_node
#                 current_node = goal_node
#
#                 path = []
#                 while current_node:
#                     path.append((round(current_node.x, 1),
#                                  round(current_node.y, 1),
#                                  round(current_node.z, 1)))
#                     current_node = current_node.parent
#                 return path[::-1]
#
#             closed_set.add(current_node)
#
#             for dx_dir, dy_dir, dz_dir in self.directions:
#                 new_x = current_node.x + dx_dir
#                 new_y = current_node.y + dy_dir
#                 new_z = current_node.z + dz_dir
#
#                 new_node = Node(new_x, new_y, new_z)
#                 new_node.parent = current_node
#
#                 move_cost = math.sqrt(dx_dir ** 2 + dy_dir ** 2 + dz_dir ** 2)
#                 new_node.cost = current_node.cost + move_cost
#
#                 new_node.heuristic = self.heuristic(new_node)
#
#                 if self.is_collision(new_node):
#                     continue
#
#                 if new_node in closed_set:
#                     continue
#
#                 if new_node in open_set:
#                     existing_node = open_set[new_node]
#                     if new_node.cost < existing_node.cost:
#                         existing_node.cost = new_node.cost
#                         existing_node.parent = new_node.parent
#                         heapq.heappush(open_heap, existing_node)
#                 else:
#                     heapq.heappush(open_heap, new_node)
#                     open_set[new_node] = new_node
#
#         return None
#
#
# class MplCanvas(FigureCanvas):
#     """Matplotlib画布集成到PyQt5"""
#
#     def __init__(self, parent=None, width=5, height=4, dpi=100):
#         self.fig = plt.figure(figsize=(width, height), dpi=dpi)
#         self.axes = self.fig.add_subplot(111, projection='3d')
#         super(MplCanvas, self).__init__(self.fig)
#         self.setParent(parent)
#         self.fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.85)
#
#
# class DronePathVisualizer(QMainWindow):
#     """无人机三维路径规划可视化界面"""
#
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("A星算法路径规划演示")
#         self.resize(1400, 800)
#
#         self.is_planning = False
#         self.path = None
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
#         self.start_z = QLineEdit("10")
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
#         self.goal_z = QLineEdit("10")
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
#         # 最小飞行高度设置
#         min_height_group = QGroupBox("最小飞行高度")
#         min_height_layout = QVBoxLayout()
#         self.min_height = QSlider(Qt.Horizontal)
#         self.min_height.setRange(0, 20)
#         self.min_height.setValue(5)
#         self.min_height_label = QLabel("5.0")
#         self.min_height.valueChanged.connect(self.update_min_height_label)
#         min_height_layout.addWidget(self.min_height)
#         min_height_layout.addWidget(self.min_height_label)
#         min_height_group.setLayout(min_height_layout)
#
#         # 步长设置
#         step_group = QGroupBox("步长大小")
#         step_layout = QVBoxLayout()
#         self.step_size = QSlider(Qt.Horizontal)
#         self.step_size.setRange(1, 10)
#         self.step_size.setValue(2)
#         self.step_label = QLabel("2.0")
#         self.step_size.valueChanged.connect(self.update_step_label)
#         step_layout.addWidget(self.step_size)
#         step_layout.addWidget(self.step_label)
#         step_group.setLayout(step_layout)
#
#         # 动画速度设置
#         speed_group = QGroupBox("动画速度")
#         speed_layout = QVBoxLayout()
#         self.animation_speed = QSlider(Qt.Horizontal)
#         self.animation_speed.setRange(1, 100)
#         self.animation_speed.setValue(10)
#         self.speed_label = QLabel("0.10s")
#         self.animation_speed.valueChanged.connect(self.update_speed_label)
#         speed_layout.addWidget(self.animation_speed)
#         speed_layout.addWidget(self.speed_label)
#         speed_group.setLayout(speed_layout)
#
#         # 控制按钮
#         button_layout = QVBoxLayout()
#         self.plan_button = QPushButton("规划路径")
#         self.plan_button.setMinimumHeight(30)
#         self.plan_button.clicked.connect(self.plan_path)
#
#         self.step_button = QPushButton("单步演示")
#         self.step_button.setMinimumHeight(30)
#         self.step_button.clicked.connect(self.step_demo)
#         self.step_button.setEnabled(False)
#
#         self.full_button = QPushButton("完整展示")
#         self.full_button.setMinimumHeight(30)
#         self.full_button.clicked.connect(self.full_demo)
#         self.full_button.setEnabled(False)
#
#         self.reset_button = QPushButton("重置视图")
#         self.reset_button.setMinimumHeight(30)
#         self.reset_button.clicked.connect(self.reset_view)
#
#         button_layout.addWidget(self.plan_button)
#         button_layout.addWidget(self.step_button)
#         button_layout.addWidget(self.full_button)
#         button_layout.addWidget(self.reset_button)
#         button_layout.addStretch()
#
#         # 添加到控制面板
#         control_layout.addWidget(start_group)
#         control_layout.addWidget(goal_group)
#         control_layout.addWidget(safe_group)
#         control_layout.addWidget(min_height_group)
#         control_layout.addWidget(step_group)
#         control_layout.addWidget(speed_group)
#         control_layout.addLayout(button_layout)
#         control_layout.setSpacing(8)
#
#         # 右侧使用垂直拆分器，上方显示可视化，下方显示信息
#         right_splitter = QSplitter(Qt.Vertical)
#         right_splitter.setHandleWidth(5)
#
#         # 上方可视化区域
#         visual_widget = QWidget()
#         visual_layout = QVBoxLayout(visual_widget)
#         self.canvas = MplCanvas(self, width=8, height=6, dpi=100)
#         self.toolbar = NavigationToolbar(self.canvas, self)
#         visual_layout.addWidget(self.toolbar)
#         visual_layout.addWidget(self.canvas)
#
#         # 下方信息区域
#         info_widget = QWidget()
#         info_layout = QVBoxLayout(info_widget)
#
#         # 统一信息显示框
#         self.info_text = QTextEdit()
#         self.info_text.setReadOnly(True)
#         self.info_text.setPlainText("请点击\"规划路径\"按钮开始路径规划")
#
#         info_layout.addWidget(self.info_text)
#
#         # 将可视化区域和信息区域添加到拆分器
#         right_splitter.addWidget(visual_widget)
#         right_splitter.addWidget(info_widget)
#         # 设置初始大小比例（上70%，下30%）
#         right_splitter.setSizes([560, 240])
#
#         # 添加到主布局
#         main_layout.addWidget(control_panel)
#         main_layout.addWidget(right_splitter, 1)
#
#         # 初始化可视化元素
#         self.init_visualization()
#
#         # 路径动画相关变量
#         self.path_line = None
#         self.drone_marker = None
#
#     def update_safe_label(self):
#         value = self.safe_distance.value() / 10.0
#         self.safe_label.setText(f"{value:.1f}")
#
#     def update_min_height_label(self):
#         value = self.min_height.value()
#         self.min_height_label.setText(f"{value:.1f}")
#
#     def update_step_label(self):
#         value = self.step_size.value()
#         self.step_label.setText(f"{value:.1f}")
#
#     def update_speed_label(self):
#         value = self.animation_speed.value() / 100.0
#         self.speed_label.setText(f"{value:.2f}s")
#
#     def init_visualization(self):
#         """初始化可视化界面，绘制障碍物"""
#         ax = self.canvas.axes
#         ax.clear()
#
#         # 绘制建筑物
#         for building in self.buildings:
#             faces = [
#                 [(building.x_min, building.y_min, building.z_min),
#                  (building.x_max, building.y_min, building.z_min),
#                  (building.x_max, building.y_max, building.z_min),
#                  (building.x_min, building.y_max, building.z_min)],
#                 [(building.x_min, building.y_min, building.z_max),
#                  (building.x_max, building.y_min, building.z_max),
#                  (building.x_max, building.y_max, building.z_max),
#                  (building.x_min, building.y_max, building.z_max)],
#                 [(building.x_min, building.y_min, building.z_min),
#                  (building.x_max, building.y_min, building.z_min),
#                  (building.x_max, building.y_min, building.z_max),
#                  (building.x_min, building.y_min, building.z_max)],
#                 [(building.x_min, building.y_max, building.z_min),
#                  (building.x_max, building.y_max, building.z_min),
#                  (building.x_max, building.y_max, building.z_max),
#                  (building.x_min, building.y_max, building.z_max)],
#                 [(building.x_min, building.y_min, building.z_min),
#                  (building.x_min, building.y_max, building.z_min),
#                  (building.x_min, building.y_max, building.z_max),
#                  (building.x_min, building.y_min, building.z_max)],
#                 [(building.x_max, building.y_min, building.z_min),
#                  (building.x_max, building.y_max, building.z_min),
#                  (building.x_max, building.y_max, building.z_max),
#                  (building.x_max, building.y_min, building.z_max)]
#             ]
#
#             poly3d = Poly3DCollection(faces, facecolors=building.color,
#                                       edgecolors='black', linewidths=1)
#             ax.add_collection3d(poly3d)
#
#         # 绘制最小飞行高度平面（透明）
#         min_h = self.min_height.value()
#         x_range = [0, 60]
#         y_range = [0, 60]
#         xx, yy = np.meshgrid(x_range, y_range)
#         z = np.full(xx.shape, min_h, dtype=np.float64)
#         ax.plot_surface(xx, yy, z, alpha=0.2, color='blue')
#         ax.text(50, 50, min_h + 1, f'最小高度: {min_h}', color='blue')
#
#         # 设置坐标轴
#         ax.set_xlabel('X轴', fontsize=8)
#         ax.set_ylabel('Y轴', fontsize=8)
#         ax.set_zlabel('Z轴', fontsize=8)
#         ax.set_title('无人机三维路径规划', fontsize=10)
#         ax.tick_params(axis='both', which='major', labelsize=7)
#         ax.set_xlim(0, 60)
#         ax.set_ylim(0, 60)
#         ax.set_zlim(0, 30)
#
#         self.canvas.draw()
#
#     def draw_path_visualization(self, show_all=False):
#         ax = self.canvas.axes
#         self.init_visualization()
#
#         try:
#             start = (float(self.start_x.text()), float(self.start_y.text()), float(self.start_z.text()))
#             goal = (float(self.goal_x.text()), float(self.goal_y.text()), float(self.goal_z.text()))
#
#             ax.scatter(start[0], start[1], start[2], color='green', s=150, label='起点', marker='o')
#             ax.scatter(goal[0], goal[1], goal[2], color='red', s=150, label='终点', marker='*')
#
#             if self.path and len(self.path) > 1:
#                 end_idx = len(self.path) if show_all else self.current_step + 1
#                 x = [p[0] for p in self.path[:end_idx]]
#                 y = [p[1] for p in self.path[:end_idx]]
#                 z = [p[2] for p in self.path[:end_idx]]
#
#                 if len(x) > 1:
#                     self.path_line = ax.plot(x, y, z, 'b-', linewidth=2, label='已规划路径')[0]
#
#                 if not show_all and end_idx < len(self.path):
#                     x_rest = [p[0] for p in self.path[end_idx - 1:]]
#                     y_rest = [p[1] for p in self.path[end_idx - 1:]]
#                     z_rest = [p[2] for p in self.path[end_idx - 1:]]
#                     ax.plot(x_rest, y_rest, z_rest, 'b--', linewidth=1, label='未规划路径')
#
#                 if self.current_step < len(self.path):
#                     drone_x, drone_y, drone_z = self.path[self.current_step]
#                     self.drone_marker = ax.scatter(
#                         [drone_x], [drone_y], [drone_z],
#                         color='blue', s=200, marker='^', label='无人机'
#                     )
#
#             ax.legend(fontsize=8)
#             self.canvas.draw()
#
#         except ValueError:
#             QMessageBox.critical(self, "输入错误", "请输入有效的坐标值！")
#
#     def calculate_path_length(self):
#         """计算路径总长度"""
#         if not self.path or len(self.path) < 2:
#             return 0.0
#
#         total_length = 0.0
#         for i in range(1, len(self.path)):
#             x1, y1, z1 = self.path[i - 1]
#             x2, y2, z2 = self.path[i]
#             dx = x2 - x1
#             dy = y2 - y1
#             dz = z2 - z1
#             total_length += math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
#         return total_length
#
#     def animate_step(self):
#         if self.current_step < len(self.path) - 1:
#             self.current_step += 1
#             self.draw_path_visualization(show_all=False)
#         else:
#             self.animation_timer.stop()
#             self.plan_button.setEnabled(True)
#             self.is_planning = False
#
#     def step_demo(self):
#         if self.path and self.current_step < len(self.path) - 1:
#             self.current_step += 1
#             self.draw_path_visualization(show_all=False)
#         else:
#             QMessageBox.information(self, "提示", "已到达路径终点")
#
#     def full_demo(self):
#         if self.path:
#             self.current_step = len(self.path) - 1
#             self.draw_path_visualization(show_all=True)
#
#             total_length = self.calculate_path_length()
#             QMessageBox.information(self, "路径信息",
#                                     f"完整路径已展示！\n总长度: {total_length:.2f} 单位\n路径点数量: {len(self.path)}")
#
#     def plan_path(self):
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
#             self.plan_button.setEnabled(False)
#             self.step_button.setEnabled(False)
#             self.full_button.setEnabled(False)
#             self.is_planning = True
#
#             # 规划开始时显示状态信息
#             self.info_text.setPlainText("正在规划路径，请稍候...\n\n")
#             self.info_text.repaint()  # 立即刷新显示
#
#             start_time = time.time()
#
#             # 显示正在初始化规划器
#             self.info_text.append("正在初始化A*路径规划器...")
#             self.info_text.repaint()
#
#             planner = AStarPlanner(
#                 start,
#                 goal,
#                 self.buildings,
#                 safe_distance=self.safe_distance.value() / 10.0,
#                 step_size=self.step_size.value(),
#                 min_height=self.min_height.value()
#             )
#
#             # 显示正在搜索路径
#             self.info_text.append("正在搜索最优路径...")
#             self.info_text.repaint()
#
#             self.path = planner.plan()
#
#             end_time = time.time()
#             plan_time = end_time - start_time
#
#             if self.path:
#                 # 拼接路径点信息
#                 path_info = "\n路径点坐标列表：\n"
#                 path_info += f"起点: ({start[0]}, {start[1]}, {start[2]})\n"
#                 for idx, point in enumerate(self.path[1:], start=1):
#                     path_info += f"步骤 {idx}: ({point[0]}, {point[1]}, {point[2]})\n"
#                 path_info += "\n"
#
#                 # 拼接统计信息
#                 total_length = self.calculate_path_length()
#                 stats_info = "路径统计信息:\n"
#                 stats_info += f"终点坐标: ({goal[0]}, {goal[1]}, {goal[2]})\n"
#                 stats_info += f"路径总长度: {total_length:.2f} 单位\n"
#                 stats_info += f"路径点数量: {len(self.path)}\n"
#                 stats_info += f"规划时间: {plan_time:.4f} 秒\n"
#                 stats_info += f"最小飞行高度: {self.min_height.value()} 单位\n"
#                 stats_info += f"安全距离: {self.safe_distance.value() / 10.0:.1f} 单位\n"
#                 stats_info += f"步长大小: {self.step_size.value()} 单位\n"
#
#                 # 整合显示
#                 self.info_text.setPlainText(path_info + stats_info)
#
#                 self.current_step = 0
#                 self.draw_path_visualization(show_all=False)
#                 self.step_button.setEnabled(True)
#                 self.full_button.setEnabled(True)
#             else:
#                 self.info_text.setPlainText("规划失败，未找到有效路径！\n\n"
#                                             "可能原因：\n"
#                                             "- 起点或终点被障碍物阻挡\n"
#                                             "- 最小飞行高度设置过高\n"
#                                             "- 安全距离设置过大\n"
#                                             "- 起点与终点之间无法形成有效路径")
#
#             self.plan_button.setEnabled(True)
#             self.is_planning = False
#
#         except ValueError:
#             self.info_text.setPlainText("输入错误：请输入有效的坐标值！")
#             self.plan_button.setEnabled(True)
#             self.is_planning = False
#         except Exception as e:
#             self.info_text.setPlainText(f"规划过程中发生错误: {str(e)}")
#             self.plan_button.setEnabled(True)
#             self.is_planning = False
#
#     def reset_view(self):
#         self.is_planning = False
#         self.animation_timer.stop()
#         self.path = None
#         self.path_line = None
#         self.drone_marker = None
#         self.current_step = 0
#
#         # 重置信息显示区域
#         self.info_text.setPlainText("请点击\"规划路径\"按钮开始路径规划")
#
#         self.init_visualization()
#         self.plan_button.setEnabled(True)
#         self.step_button.setEnabled(False)
#         self.full_button.setEnabled(False)
#
#
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = DronePathVisualizer()
#     window.show()
#     sys.exit(app.exec_())

# import math
# import heapq
# import sys
# from PyQt5.QtWidgets import (
#     QApplication, QMainWindow, QWidget, QVBoxLayout,
#     QHBoxLayout, QLabel, QLineEdit, QPushButton,
#     QSlider, QGroupBox, QFormLayout, QMessageBox,
#     QTableWidget, QTableWidgetItem, QHeaderView
# )
# from PyQt5.QtCore import Qt, QTimer
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# import time  # 用于计算规划时间
#
# # 设置matplotlib支持中文显示
# plt.rcParams["font.family"] = ["SimHei"]
# plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号
# plt.rcParams["figure.titlesize"] = 10  # 减小图表标题字体大小
#
#
# class Node:
#     """路径规划中的节点类，代表三维空间中的一个点"""
#
#     def __init__(self, x, y, z, cost=0, parent=None):
#         self.x = x  # x坐标
#         self.y = y  # y坐标
#         self.z = z  # z坐标
#         self.cost = cost  # 从起点到当前节点的代价
#         self.parent = parent  # 父节点
#         self.heuristic = 0  # 启发式估计代价
#
#     def total_cost(self):
#         """计算总代价 f = g + h"""
#         return self.cost + self.heuristic
#
#     def __eq__(self, other):
#         """判断两个节点是否为同一个点"""
#         if not isinstance(other, Node):
#             return False
#         return (round(self.x, 1) == round(other.x, 1) and
#                 round(self.y, 1) == round(other.y, 1) and
#                 round(self.z, 1) == round(other.z, 1))
#
#     def __lt__(self, other):
#         """用于优先队列的比较"""
#         return self.total_cost() < other.total_cost()
#
#     def __hash__(self):
#         """用于集合操作的哈希函数，使用坐标值计算哈希"""
#         return hash((round(self.x, 1), round(self.y, 1), round(self.z, 1)))
#
#     def __repr__(self):
#         """节点的字符串表示，便于调试"""
#         return f"Node({self.x:.1f}, {self.y:.1f}, {self.z:.1f}, cost={self.cost:.2f})"
#
#
# class Building:
#     """建筑物类，代表障碍物"""
#
#     def __init__(self, x, y, z, width, length, height, color=(0.5, 0.5, 0.5, 0.7)):
#         self.x = x  # 中心点x坐标
#         self.y = y  # 中心点y坐标
#         self.z = z  # 底部z坐标
#         self.width = width  # x方向宽度
#         self.length = length  # y方向长度
#         self.height = height  # 高度（z方向）
#         self.color = color  # 建筑物颜色及透明度 (r, g, b, alpha)
#
#         # 预计算边界值，加速碰撞检测
#         self.x_min = x - width / 2
#         self.x_max = x + width / 2
#         self.y_min = y - length / 2
#         self.y_max = y + length / 2
#         self.z_min = z
#         self.z_max = z + height
#
#
# class AStarPlanner:
#     """优化后的A*算法路径规划器"""
#
#     def __init__(self, start, goal, buildings, safe_distance=5, step_size=2):
#         # 增加步长以减少节点数量
#         self.start = Node(start[0], start[1], start[2])
#         self.goal = Node(goal[0], goal[1], goal[2])
#         self.buildings = buildings  # 障碍物列表
#         self.safe_distance = safe_distance  # 安全距离
#         self.step_size = step_size  # 增大步长以减少节点数量
#
#         # 计算起点到终点的大致方向，用于优化搜索方向
#         dx = goal[0] - start[0]
#         dy = goal[1] - start[1]
#         dz = goal[2] - start[2]
#         dist = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
#         if dist > 0:
#             self.main_dir = (dx / dist, dy / dist, dz / dist)
#         else:
#             self.main_dir = (0, 0, 0)
#
#         # 减少方向数量，只保留关键方向（6个主要方向+1个朝向目标的方向）
#         self.directions = [
#             (step_size, 0, 0),  # 右
#             (-step_size, 0, 0),  # 左
#             (0, step_size, 0),  # 前
#             (0, -step_size, 0),  # 后
#             (0, 0, step_size),  # 上
#             (0, 0, -step_size),  # 下
#             (self.main_dir[0] * step_size,
#              self.main_dir[1] * step_size,
#              self.main_dir[2] * step_size)  # 朝向目标的方向
#         ]
#
#     def heuristic(self, node):
#         """启发式函数，使用欧氏距离的平方（不计算平方根以提高速度）"""
#         dx = node.x - self.goal.x
#         dy = node.y - self.goal.y
#         dz = node.z - self.goal.z
#         return dx * dx + dy * dy + dz * dz  # 不计算平方根，加快计算
#
#     def is_collision(self, node):
#         """优化的碰撞检测，使用预计算的边界值"""
#         x, y, z = node.x, node.y, node.z
#
#         # 先检查是否在全局边界内（快速排除）
#         if x < -5 or x > 65 or y < -5 or y > 65 or z < 0 or z > 35:
#             return True
#
#         for building in self.buildings:
#             # 计算节点到建筑物的最短距离（使用预计算的边界值）
#             # x方向距离
#             if x < building.x_min:
#                 dx = building.x_min - x
#             elif x > building.x_max:
#                 dx = x - building.x_max
#             else:
#                 dx = 0
#
#             # y方向距离
#             if y < building.y_min:
#                 dy = building.y_min - y
#             elif y > building.y_max:
#                 dy = y - building.y_max
#             else:
#                 dy = 0
#
#             # z方向距离
#             if z < building.z_min:
#                 dz = building.z_min - z
#             elif z > building.z_max:
#                 dz = z - building.z_max
#             else:
#                 dz = 0
#
#             # 三维空间中的最短距离平方（避免计算平方根）
#             dist_sq = dx * dx + dy * dy + dz * dz
#             safe_sq = self.safe_distance * self.safe_distance
#
#             # 如果距离小于安全距离，视为碰撞
#             if dist_sq < safe_sq:
#                 return True
#
#         return False
#
#     def plan(self):
#         """执行优化后的A*算法进行路径规划"""
#         # 初始化开放列表和关闭列表
#         open_heap = []
#         open_set = {}  # 使用字典存储开放列表中的节点，便于查找
#         closed_set = set()
#
#         # 将起点加入开放列表
#         self.start.heuristic = self.heuristic(self.start)
#         heapq.heappush(open_heap, self.start)
#         open_set[self.start] = self.start
#
#         # 限制最大迭代次数，防止无限循环
#         max_iterations = 5000
#         iterations = 0
#
#         while open_heap and iterations < max_iterations:
#             iterations += 1
#
#             # 从开放列表中获取总代价最小的节点
#             current_node = heapq.heappop(open_heap)
#
#             # 如果当前节点已经被处理，跳过
#             if current_node not in open_set:
#                 continue
#
#             # 从开放列表中移除当前节点
#             del open_set[current_node]
#
#             # 如果到达目标点附近，回溯路径
#             dx = current_node.x - self.goal.x
#             dy = current_node.y - self.goal.y
#             dz = current_node.z - self.goal.z
#             dist_sq = dx * dx + dy * dy + dz * dz  # 使用平方距离加速计算
#
#             if dist_sq < (self.step_size * 1.5) ** 2:  # 允许一定误差范围
#                 # 添加目标节点到路径
#                 goal_node = Node(self.goal.x, self.goal.y, self.goal.z)
#                 goal_node.parent = current_node
#                 current_node = goal_node
#
#                 path = []
#                 while current_node:
#                     path.append((round(current_node.x, 1),
#                                  round(current_node.y, 1),
#                                  round(current_node.z, 1)))
#                     current_node = current_node.parent
#                 return path[::-1]  # 反转路径，从起点到终点
#
#             # 将当前节点加入关闭列表
#             closed_set.add(current_node)
#
#             # 探索所有可能的移动方向
#             for dx_dir, dy_dir, dz_dir in self.directions:
#                 # 计算新节点的坐标
#                 new_x = current_node.x + dx_dir
#                 new_y = current_node.y + dy_dir
#                 new_z = current_node.z + dz_dir
#
#                 # 创建新节点
#                 new_node = Node(new_x, new_y, new_z)
#                 new_node.parent = current_node
#
#                 # 计算移动代价（使用平方距离避免开方）
#                 move_cost = math.sqrt(dx_dir ** 2 + dy_dir ** 2 + dz_dir ** 2)
#                 new_node.cost = current_node.cost + move_cost
#
#                 # 计算启发式代价
#                 new_node.heuristic = self.heuristic(new_node)
#
#                 # 检查是否与障碍物碰撞
#                 if self.is_collision(new_node):
#                     continue
#
#                 # 检查是否已在关闭列表中
#                 if new_node in closed_set:
#                     continue
#
#                 # 检查是否在开放列表中
#                 if new_node in open_set:
#                     # 如果新路径代价更小，则更新
#                     existing_node = open_set[new_node]
#                     if new_node.cost < existing_node.cost:
#                         existing_node.cost = new_node.cost
#                         existing_node.parent = new_node.parent
#                         heapq.heappush(open_heap, existing_node)
#                 else:
#                     # 将新节点加入开放列表
#                     heapq.heappush(open_heap, new_node)
#                     open_set[new_node] = new_node
#
#         # 如果没有找到路径或达到最大迭代次数
#         return None
#
#
# class MplCanvas(FigureCanvas):
#     """Matplotlib画布集成到PyQt5"""
#
#     def __init__(self, parent=None, width=5, height=4, dpi=100):
#         self.fig = plt.figure(figsize=(width, height), dpi=dpi)
#         self.axes = self.fig.add_subplot(111, projection='3d')
#         super(MplCanvas, self).__init__(self.fig)
#         self.setParent(parent)
#         # 调整布局，防止标题超出
#         self.fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.85)
#
#
# class DronePathVisualizer(QMainWindow):
#     """无人机三维路径规划可视化界面"""
#
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("快速无人机三维路径规划演示")
#         self.resize(1400, 800)  # 调整窗口初始大小
#
#         # 标志变量
#         self.is_planning = False
#         self.path = None
#         self.current_step = 0
#         self.animation_timer = QTimer(self)
#         self.animation_timer.timeout.connect(self.animate_step)
#
#         # 创建建筑物（障碍物）- 预计算边界值以加速碰撞检测
#         self.buildings = [
#             Building(15, 15, 0, 10, 10, 20, (0.6, 0.2, 0.2, 0.8)),  # 红色系建筑物
#             Building(35, 35, 0, 12, 12, 18, (0.2, 0.6, 0.2, 0.8))  # 绿色系建筑物
#         ]
#
#         # 创建主布局
#         main_widget = QWidget()
#         self.setCentralWidget(main_widget)
#         main_layout = QHBoxLayout(main_widget)
#         main_layout.setContentsMargins(5, 5, 5, 5)  # 减小边距
#
#         # 创建左侧控制面板
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
#         self.start_z = QLineEdit("10")
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
#         self.goal_z = QLineEdit("10")
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
#         self.safe_distance.setRange(1, 100)  # 0.1-10.0
#         self.safe_distance.setValue(30)  # 3.0
#         self.safe_label = QLabel("3.0")
#         self.safe_distance.valueChanged.connect(self.update_safe_label)
#         safe_layout.addWidget(self.safe_distance)
#         safe_layout.addWidget(self.safe_label)
#         safe_group.setLayout(safe_layout)
#
#         # 步长设置（新增）
#         step_group = QGroupBox("步长大小")
#         step_layout = QVBoxLayout()
#         self.step_size = QSlider(Qt.Horizontal)
#         self.step_size.setRange(1, 10)  # 1-10单位
#         self.step_size.setValue(2)  # 默认2单位
#         self.step_label = QLabel("2.0")
#         self.step_size.valueChanged.connect(self.update_step_label)
#         step_layout.addWidget(self.step_size)
#         step_layout.addWidget(self.step_label)
#         step_group.setLayout(step_layout)
#
#         # 动画速度设置
#         speed_group = QGroupBox("动画速度")
#         speed_layout = QVBoxLayout()
#         self.animation_speed = QSlider(Qt.Horizontal)
#         self.animation_speed.setRange(1, 100)  # 0.01-1.00秒
#         self.animation_speed.setValue(10)  # 0.10秒
#         self.speed_label = QLabel("0.10s")
#         self.animation_speed.valueChanged.connect(self.update_speed_label)
#         speed_layout.addWidget(self.animation_speed)
#         speed_layout.addWidget(self.speed_label)
#         speed_group.setLayout(speed_layout)
#
#         # 规划时间显示
#         self.plan_time_label = QLabel("规划时间: -- 秒")
#         time_group = QGroupBox("性能指标")
#         time_layout = QVBoxLayout()
#         time_layout.addWidget(self.plan_time_label)
#         time_group.setLayout(time_layout)
#
#         # 控制按钮
#         button_layout = QVBoxLayout()
#         self.plan_button = QPushButton("规划路径")
#         self.plan_button.setMinimumHeight(30)
#         self.plan_button.clicked.connect(self.plan_path)
#
#         self.step_button = QPushButton("单步演示")
#         self.step_button.setMinimumHeight(30)
#         self.step_button.clicked.connect(self.step_demo)
#         self.step_button.setEnabled(False)
#
#         self.full_button = QPushButton("完整展示")
#         self.full_button.setMinimumHeight(30)
#         self.full_button.clicked.connect(self.full_demo)
#         self.full_button.setEnabled(False)
#
#         self.reset_button = QPushButton("重置视图")
#         self.reset_button.setMinimumHeight(30)
#         self.reset_button.clicked.connect(self.reset_view)
#
#         button_layout.addWidget(self.plan_button)
#         button_layout.addWidget(self.step_button)
#         button_layout.addWidget(self.full_button)
#         button_layout.addWidget(self.reset_button)
#         button_layout.addStretch()
#
#         # 路径点表格
#         self.path_table = QTableWidget()
#         self.path_table.setColumnCount(4)
#         self.path_table.setHorizontalHeaderLabels(["序号", "X坐标", "Y坐标", "Z坐标"])
#         header = self.path_table.horizontalHeader()
#         header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
#         header.setSectionResizeMode(1, QHeaderView.Stretch)
#         header.setSectionResizeMode(2, QHeaderView.Stretch)
#         header.setSectionResizeMode(3, QHeaderView.Stretch)
#         table_group = QGroupBox("路径点坐标")
#         table_layout = QVBoxLayout()
#         table_layout.addWidget(self.path_table)
#         table_group.setLayout(table_layout)
#
#         # 添加到控制面板
#         control_layout.addWidget(start_group)
#         control_layout.addWidget(goal_group)
#         control_layout.addWidget(safe_group)
#         control_layout.addWidget(step_group)  # 新增步长设置
#         control_layout.addWidget(speed_group)
#         control_layout.addWidget(time_group)  # 新增时间显示
#         control_layout.addLayout(button_layout)
#         control_layout.addWidget(table_group)
#         control_layout.setSpacing(8)
#
#         # 创建右侧可视化区域
#         visual_widget = QWidget()
#         visual_layout = QVBoxLayout(visual_widget)
#
#         # 创建Matplotlib画布
#         self.canvas = MplCanvas(self, width=8, height=6, dpi=100)
#         self.toolbar = NavigationToolbar(self.canvas, self)
#
#         visual_layout.addWidget(self.toolbar)
#         visual_layout.addWidget(self.canvas)
#
#         # 添加到主布局
#         main_layout.addWidget(control_panel)
#         main_layout.addWidget(visual_widget, 1)
#
#         # 初始化可视化元素
#         self.init_visualization()
#
#         # 路径动画相关变量
#         self.path_line = None
#         self.drone_marker = None
#
#     def update_safe_label(self):
#         """更新安全距离标签"""
#         value = self.safe_distance.value() / 10.0
#         self.safe_label.setText(f"{value:.1f}")
#
#     def update_step_label(self):
#         """更新步长标签"""
#         value = self.step_size.value()
#         self.step_label.setText(f"{value:.1f}")
#
#     def update_speed_label(self):
#         """更新动画速度标签"""
#         value = self.animation_speed.value() / 100.0
#         self.speed_label.setText(f"{value:.2f}s")
#
#     def init_visualization(self):
#         """初始化可视化界面，绘制障碍物"""
#         ax = self.canvas.axes
#         ax.clear()
#
#         # 绘制建筑物（立体彩色带透明度）
#         for building in self.buildings:
#             # 计算建筑物的6个面
#             faces = [
#                 # 底面 (z=z_min)
#                 [(building.x_min, building.y_min, building.z_min),
#                  (building.x_max, building.y_min, building.z_min),
#                  (building.x_max, building.y_max, building.z_min),
#                  (building.x_min, building.y_max, building.z_min)],
#                 # 顶面 (z=z_max)
#                 [(building.x_min, building.y_min, building.z_max),
#                  (building.x_max, building.y_min, building.z_max),
#                  (building.x_max, building.y_max, building.z_max),
#                  (building.x_min, building.y_max, building.z_max)],
#                 # 前面 (y=y_min)
#                 [(building.x_min, building.y_min, building.z_min),
#                  (building.x_max, building.y_min, building.z_min),
#                  (building.x_max, building.y_min, building.z_max),
#                  (building.x_min, building.y_min, building.z_max)],
#                 # 后面 (y=y_max)
#                 [(building.x_min, building.y_max, building.z_min),
#                  (building.x_max, building.y_max, building.z_min),
#                  (building.x_max, building.y_max, building.z_max),
#                  (building.x_min, building.y_max, building.z_max)],
#                 # 左面 (x=x_min)
#                 [(building.x_min, building.y_min, building.z_min),
#                  (building.x_min, building.y_max, building.z_min),
#                  (building.x_min, building.y_max, building.z_max),
#                  (building.x_min, building.y_min, building.z_max)],
#                 # 右面 (x=x_max)
#                 [(building.x_max, building.y_min, building.z_min),
#                  (building.x_max, building.y_max, building.z_min),
#                  (building.x_max, building.y_max, building.z_max),
#                  (building.x_max, building.y_min, building.z_max)]
#             ]
#
#             # 创建3D多边形集合
#             poly3d = Poly3DCollection(faces, facecolors=building.color,
#                                       edgecolors='black', linewidths=1)
#             ax.add_collection3d(poly3d)
#
#         # 设置坐标轴标签，减小字体
#         ax.set_xlabel('X轴', fontsize=8)
#         ax.set_ylabel('Y轴', fontsize=8)
#         ax.set_zlabel('Z轴', fontsize=8)
#         ax.set_title('无人机三维路径规划', fontsize=10)
#
#         # 调整坐标轴刻度字体大小
#         ax.tick_params(axis='both', which='major', labelsize=7)
#
#         # 设置坐标轴范围
#         ax.set_xlim(0, 60)
#         ax.set_ylim(0, 60)
#         ax.set_zlim(0, 30)
#
#         self.canvas.draw()
#
#     def draw_path_visualization(self, show_all=False):
#         """绘制路径的可视化，show_all参数控制是否显示全部路径"""
#         ax = self.canvas.axes
#         self.init_visualization()  # 重新初始化，确保建筑物显示
#
#         try:
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
#                 # 显示到当前步骤的路径
#                 end_idx = len(self.path) if show_all else self.current_step + 1
#                 x = [p[0] for p in self.path[:end_idx]]
#                 y = [p[1] for p in self.path[:end_idx]]
#                 z = [p[2] for p in self.path[:end_idx]]
#
#                 # 已访问的路径点
#                 if len(x) > 1:
#                     self.path_line = ax.plot(x, y, z, 'b-', linewidth=2, label='已规划路径')[0]
#
#                 # 剩余路径（如果不显示全部）
#                 if not show_all and end_idx < len(self.path):
#                     x_rest = [p[0] for p in self.path[end_idx - 1:]]
#                     y_rest = [p[1] for p in self.path[end_idx - 1:]]
#                     z_rest = [p[2] for p in self.path[end_idx - 1:]]
#                     ax.plot(x_rest, y_rest, z_rest, 'b--', linewidth=1, label='未规划路径')
#
#                 # 无人机位置标记
#                 if self.current_step < len(self.path):
#                     drone_x, drone_y, drone_z = self.path[self.current_step]
#                     self.drone_marker = ax.scatter(
#                         [drone_x], [drone_y], [drone_z],
#                         color='blue', s=200, marker='^', label='无人机'
#                     )
#
#             ax.legend(fontsize=8)  # 减小图例字体
#             self.canvas.draw()
#
#         except ValueError:
#             QMessageBox.critical(self, "输入错误", "请输入有效的坐标值！")
#
#     def update_path_table(self):
#         """更新路径点表格"""
#         if not self.path:
#             self.path_table.setRowCount(0)
#             return
#
#         self.path_table.setRowCount(len(self.path))
#
#         for i, (x, y, z) in enumerate(self.path):
#             # 设置表格内容
#             self.path_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
#             self.path_table.setItem(i, 1, QTableWidgetItem(f"{x:.1f}"))
#             self.path_table.setItem(i, 2, QTableWidgetItem(f"{y:.1f}"))
#             self.path_table.setItem(i, 3, QTableWidgetItem(f"{z:.1f}"))
#
#             # 高亮显示当前步骤
#             for j in range(4):
#                 if i == self.current_step:
#                     self.path_table.item(i, j).setBackground(Qt.yellow)
#                 else:
#                     self.path_table.item(i, j).setBackground(Qt.white)
#
#     def animate_step(self):
#         """路径动画的每一步"""
#         if self.current_step < len(self.path) - 1:
#             self.current_step += 1
#             self.draw_path_visualization(show_all=False)
#             self.update_path_table()
#         else:
#             # 动画结束，停止计时器
#             self.animation_timer.stop()
#             self.plan_button.setEnabled(True)
#             self.is_planning = False
#
#     def step_demo(self):
#         """单步演示功能"""
#         if self.path and self.current_step < len(self.path) - 1:
#             self.current_step += 1
#             self.draw_path_visualization(show_all=False)
#             self.update_path_table()
#         else:
#             QMessageBox.information(self, "提示", "已到达路径终点")
#
#     def full_demo(self):
#         """完整展示功能"""
#         if self.path:
#             self.current_step = len(self.path) - 1
#             self.draw_path_visualization(show_all=True)
#             self.update_path_table()
#
#             # 计算总长度
#             total_length = sum(
#                 math.sqrt(
#                     (self.path[i + 1][0] - self.path[i][0]) ** 2 +
#                     (self.path[i + 1][1] - self.path[i][1]) ** 2 +
#                     (self.path[i + 1][2] - self.path[i][2]) ** 2
#                 ) for i in range(len(self.path) - 1)
#             )
#             QMessageBox.information(self, "路径信息",
#                                     f"完整路径已展示！\n总长度: {total_length:.2f} 单位\n路径点数量: {len(self.path)}")
#
#     def plan_path(self):
#         """规划路径并展示结果，添加时间计算"""
#         try:
#             # 获取起点和终点坐标
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
#             # 更新按钮状态
#             self.plan_button.setEnabled(False)
#             self.is_planning = True
#             self.plan_time_label.setText("规划中...")
#
#             # 记录开始时间
#             start_time = time.time()
#
#             # 创建A*规划器（使用可配置的步长）
#             planner = AStarPlanner(
#                 start,
#                 goal,
#                 self.buildings,
#                 safe_distance=self.safe_distance.value() / 10.0,
#                 step_size=self.step_size.value()
#             )
#
#             # 执行路径规划
#             self.path = planner.plan()
#
#             # 计算并显示规划时间
#             end_time = time.time()
#             plan_time = end_time - start_time
#             self.plan_time_label.setText(f"规划时间: {plan_time:.4f} 秒")
#
#             if self.path:
#                 # 初始化步骤并显示
#                 self.current_step = 0
#                 self.draw_path_visualization(show_all=False)
#                 self.update_path_table()
#
#                 # 启用控制按钮
#                 self.step_button.setEnabled(True)
#                 self.full_button.setEnabled(True)
#                 self.plan_button.setEnabled(True)
#                 self.is_planning = False
#             else:
#                 QMessageBox.critical(self, "规划失败", "无法找到从起点到终点的路径！")
#                 self.plan_button.setEnabled(True)
#                 self.is_planning = False
#
#         except ValueError:
#             QMessageBox.critical(self, "输入错误", "请输入有效的坐标值！")
#             self.plan_button.setEnabled(True)
#             self.is_planning = False
#         except Exception as e:
#             QMessageBox.critical(self, "错误", f"规划过程中发生错误: {str(e)}")
#             self.plan_button.setEnabled(True)
#             self.is_planning = False
#
#     def reset_view(self):
#         """重置视图"""
#         self.is_planning = False
#         self.animation_timer.stop()
#         self.path = None
#         self.path_line = None
#         self.drone_marker = None
#         self.current_step = 0
#         self.plan_time_label.setText("规划时间: -- 秒")
#         self.init_visualization()
#         self.update_path_table()
#         self.plan_button.setEnabled(True)
#         self.step_button.setEnabled(False)
#         self.full_button.setEnabled(False)
#
#
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = DronePathVisualizer()
#     window.show()
#     sys.exit(app.exec_())
