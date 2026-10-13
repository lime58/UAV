import os

# 解决OpenMP冲突问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import sys
import random
import numpy as np
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QSlider, QGroupBox, QFormLayout, QMessageBox,
                             QTextEdit, QScrollArea, QSplitter, QFileDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import copy

# 设置matplotlib支持中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.titlesize"] = 10


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


class DQN(nn.Module):
    """深度Q网络模型"""

    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        #输入层，接收状态向量，输出 256 维特征
        self.fc1 = nn.Linear(state_size, 256)  # 增加网络容量
        #隐藏层 1，将 256 维特征映射到 128 维
        self.fc2 = nn.Linear(256, 128)
        #隐藏层 2，将 128 维特征映射到 64 维
        self.fc3 = nn.Linear(128, 64)
        #输出层，输出每个动作的 Q 值（共 action_size 个）
        self.fc4 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


class DQNPlanner:
    def __init__(self, start, goal, buildings,
                 safe_distance=3.0, min_height=0.5,
                 num_episodes=3000, max_steps=300,
                 num_waypoints=15,
                 x_range=(0, 60), y_range=(0, 60), z_range=(0, 30)):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.buildings = buildings
        self.safe_distance = safe_distance
        self.safe_distance_sq = safe_distance ** 2
        self.min_height = min_height

        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.num_waypoints = num_waypoints

        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

        for b in self.buildings:
            b.set_safe_bounds(safe_distance)

        # 计算起点到终点的大致方向向量
        self.direction = self.goal - self.start
        self.total_distance = np.linalg.norm(self.direction)
        if self.total_distance > 0:
            self.unit_direction = self.direction / self.total_distance
        else:
            self.unit_direction = np.array([0, 0, 0])

        # 改进：增加状态空间维度，包含6个方向的障碍物距离
        self.state_size = 3 + 3 + 6  # 当前位置(x,y,z) + 目标相对位置(dx,dy,dz) + 6个方向障碍物距离
        # 改进：减小动作步长，增加动作数量提高灵活性
        self.action_size = 27  # 3个轴，每个轴有-1,0,1三种移动，共3^3=27种动作
        self.action_step = 1.0  # 步长从2减小到1，提高控制精度
        self.action_space = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).T.reshape(-1,
                                                                                                3) * self.action_step

        # 初始化DQN网络
        self.qnetwork_local = DQN(self.state_size, self.action_size)
        self.qnetwork_target = DQN(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=3e-4)  # 调整学习率

        # 经验回放
        self.memory = ReplayBuffer(buffer_size=20000, batch_size=64)  # 增大经验池

        # 超参数
        self.gamma = 0.995  # 增大折扣因子，更关注长远奖励
        self.tau = 1e-3  # 目标网络软更新参数
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.02  # 保留更高的最小探索率
        self.epsilon_decay = 0.998  # 减缓衰减速度，保持更长探索时间

        # 记录最佳路径
        self.best_path = None
        self.best_reward = -np.inf
        self.best_path_length = float('inf')  # 记录最佳路径长度

    def save_model(self, path="dqn_planner_model.pth"):
        """保存模型参数"""
        torch.save({
            'qnetwork_local': self.qnetwork_local.state_dict(),
            'qnetwork_target': self.qnetwork_target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        print(f"模型已保存至 {path}")
        return path

    def load_model(self, path="dqn_planner_model.pth"):
        """加载模型参数"""
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.qnetwork_local.load_state_dict(checkpoint['qnetwork_local'])
            self.qnetwork_target.load_state_dict(checkpoint['qnetwork_target'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            print(f"已从 {path} 加载模型")
            return True
        print(f"未找到模型文件 {path}")
        return False

    def get_obstacle_distances(self, point):
        """计算6个方向(前、后、左、右、上、下)到最近障碍物的距离"""
        x, y, z = point
        directions = [
            (1, 0, 0),  # 前 (X+)
            (-1, 0, 0),  # 后 (X-)
            (0, 1, 0),  # 右 (Y+)
            (0, -1, 0),  # 左 (Y-)
            (0, 0, 1),  # 上 (Z+)
            (0, 0, -1)  # 下 (Z-)
        ]

        max_dist = max(self.x_range[1], self.y_range[1], self.z_range[1])
        distances = []

        for dx, dy, dz in directions:
            dist = 0
            current_x, current_y, current_z = x, y, z

            # 沿方向逐步移动，直到碰到障碍物或边界
            while True:
                current_x += dx * 0.5
                current_y += dy * 0.5
                current_z += dz * 0.5
                dist += 0.5

                # 检查是否超出边界
                if (current_x < self.x_range[0] or current_x > self.x_range[1] or
                        current_y < self.y_range[0] or current_y > self.y_range[1] or
                        current_z < self.z_range[0] or current_z > self.z_range[1]):
                    distances.append(max_dist)
                    break

                # 检查是否碰撞障碍物
                if self.is_collision((current_x, current_y, current_z)):
                    distances.append(dist)
                    break

                # 防止无限循环
                if dist > max_dist:
                    distances.append(max_dist)
                    break

        # 归一化距离
        return np.array(distances) / max_dist

    def distance_to_obstacle(self, point):
        """计算点到最近障碍物的距离"""
        min_dist = float('inf')
        x, y, z = point

        for b in self.buildings:
            dx = max(b.x_min - x, 0, x - b.x_max)
            dy = max(b.y_min - y, 0, y - b.y_max)
            dz = max(b.z_min - z, 0, z - b.z_max)

            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            min_dist = min(min_dist, dist)

        return min_dist

    def is_collision(self, point, is_start_end=False):
        x, y, z = point

        # 检查边界
        if (x < self.x_range[0] or x > self.x_range[1] or
                y < self.y_range[0] or y > self.y_range[1] or
                z < self.z_range[0] or z > self.z_range[1]):
            return True

        # 起点和终点特殊处理，z 低于最低飞行高度不视为碰撞
        if is_start_end:
            return False

        # 检查障碍物安全距离
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

        # 检查最低高度（起点终点除外）
        if z < self.min_height:
            return True

        return False

    def calculate_reward(self, state, next_state, action, done, collided, path_length):
        """改进的奖励函数"""
        current_pos = state[:3]
        next_pos = next_state[:3]
        goal_rel = state[3:6]

        # 到达目标奖励 - 大幅提高
        goal_dist = np.linalg.norm(goal_rel)
        next_goal_dist = np.linalg.norm(next_state[3:6])

        # 接近目标的奖励 - 适度调整
        distance_reward = (goal_dist - next_goal_dist) * 3

        # 碰撞惩罚 - 加大惩罚力度
        collision_penalty = -200 if collided else 0

        # 边界惩罚
        boundary_penalty = 0
        x, y, z = next_pos
        if x <= self.x_range[0] + 1 or x >= self.x_range[1] - 1:
            boundary_penalty -= 30
        if y <= self.y_range[0] + 1 or y >= self.y_range[1] - 1:
            boundary_penalty -= 30
        if z <= self.z_range[0] + 1 or z >= self.z_range[1] - 1:
            boundary_penalty -= 30

        # 最低高度惩罚
        height_penalty = 0
        if z < self.min_height:
            height_penalty -= 20

        # 到达目标奖励 - 大幅提高
        goal_reward = 1000 if done else 0

        # 路径长度惩罚 - 鼓励较短路径
        path_length_penalty = -path_length * 0.01

        # 安全距离奖励 - 鼓励远离障碍物
        obstacle_dist = self.distance_to_obstacle(next_pos)
        safe_reward = min(obstacle_dist * 5, 20)  # 最高20分

        # 方向奖励 - 鼓励朝着目标方向移动
        move_dir = next_pos - current_pos
        move_dist = np.linalg.norm(move_dir)
        direction_reward = 0
        if move_dist > 0:
            cos_sim = np.dot(move_dir / move_dist, self.unit_direction)
            direction_reward = cos_sim * 5  # 方向一致奖励

        # 未到达终点的惩罚 - 回合结束但未到达终点
        episode_end_penalty = -150 if (not done and not collided and path_length > self.max_steps * 0.7) else 0

        # 总奖励
        total_reward = (distance_reward + collision_penalty + boundary_penalty +
                        height_penalty + goal_reward + path_length_penalty +
                        safe_reward + direction_reward + episode_end_penalty)

        return total_reward

    def get_state(self, position):
        """获取改进的状态：位置 + 到目标的相对位置 + 6个方向障碍物距离"""
        rel_pos = self.goal - position
        obstacle_dists = self.get_obstacle_distances(position)
        return np.concatenate([position, rel_pos, obstacle_dists])

    def act(self, state, eps=0.0):
        """根据状态选择动作"""
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # epsilon-greedy选择动作
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done):
        """经验回放和学习"""
        self.memory.add(state, action, reward, next_state, done)

        # 有足够经验后进行学习
        if len(self.memory) > self.memory.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma)

    def learn(self, experiences, gamma):
        """从经验中学习"""
        states, actions, rewards, next_states, dones = experiences

        # 从目标网络获取下一状态的Q值
        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # 计算目标Q值
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        # 从本地网络获取当前Q值
        q_expected = self.qnetwork_local(states).gather(1, actions)

        # 计算损失
        loss = F.mse_loss(q_expected, q_targets)

        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 软更新目标网络
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """软更新目标网络参数"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def is_path_valid(self, path):
        """检查路径是否有效（无碰撞）"""
        # 检查路径上所有点（除起点终点外）的碰撞
        for i, point in enumerate(path):
            is_start_end = (i == 0 or i == len(path) - 1)
            if self.is_collision(point, is_start_end):
                return False

        # 检查路径段是否碰撞
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            dist = np.linalg.norm(p2 - p1)
            num_samples = max(5, int(dist))  # 加密采样

            for j in range(1, num_samples):
                ratio = j / num_samples
                x = p1[0] + (p2[0] - p1[0]) * ratio
                y = p1[1] + (p2[1] - p1[1]) * ratio
                z = p1[2] + (p2[2] - p1[2]) * ratio

                if self.is_collision((x, y, z)):
                    return False

        return True

    def calculate_path_length(self, path):
        length = 0.0
        for i in range(len(path) - 1):
            length += np.linalg.norm(path[i + 1] - path[i])
        return length

    def generate_path_with_trained_model(self):
        """使用训练好的模型生成路径"""
        current_pos = np.copy(self.start)
        path = [current_pos.copy()]

        for _ in range(self.max_steps):
            state = self.get_state(current_pos)
            action_idx = self.act(state, eps=0.05)  # 少量探索确保灵活性
            action = self.action_space[action_idx]
            current_pos = current_pos + action

            # 检查是否到达目标
            if np.linalg.norm(current_pos - self.goal) < self.safe_distance:
                path.append(current_pos.copy())
                path.append(self.goal.copy())
                return path

            # 检查是否碰撞
            if self.is_collision(current_pos):
                return None

            path.append(current_pos.copy())

        return None

    def plan(self, use_pretrained=False):
        """主规划函数：使用DQN进行路径规划"""
        # 检查起点和终点是否合法
        if self.is_collision(self.start, is_start_end=True):
            return None, "起点在障碍物或边界外"
        if self.is_collision(self.goal, is_start_end=True):
            return None, "终点在障碍物或边界外"

        # 检查起点到终点是否可以直接到达
        direct_path = [self.start, self.goal]
        if self.is_path_valid(direct_path):
            return [tuple(self.start), tuple(self.goal)], "找到直接路径"

        # 如果使用预训练模型，直接生成路径
        if use_pretrained:
            path = self.generate_path_with_trained_model()
            if path and self.is_path_valid(path):
                return [tuple(p) for p in path], "使用预训练模型找到有效路径"
            else:
                return None, "预训练模型未能找到有效路径"

        scores = []  # 记录每轮的得分
        self.best_reward = -np.inf
        self.best_path = None
        self.best_path_length = float('inf')

        # 增加成功计数器，用于提前停止训练
        success_counter = 0
        required_successes = 10  # 连续成功次数

        for i_episode in range(1, self.num_episodes + 1):
            # 初始化状态
            current_pos = np.copy(self.start)
            state = self.get_state(current_pos)
            score = 0
            path = [current_pos.copy()]
            step_count = 0
            path_length = 0

            for step in range(self.max_steps):
                step_count = step
                # 选择动作
                action_idx = self.act(state, self.epsilon)
                action = self.action_space[action_idx]

                # 执行动作
                prev_pos = current_pos.copy()
                current_pos = current_pos + action
                path_length += np.linalg.norm(action)  # 累积路径长度

                next_state = self.get_state(current_pos)

                # 检查是否到达目标
                done = np.linalg.norm(current_pos - self.goal) < self.safe_distance

                # 检查是否碰撞
                collided = self.is_collision(current_pos)

                # 计算奖励
                reward = self.calculate_reward(state, next_state, action, done, collided, path_length)

                # 如果碰撞或到达目标，结束本轮
                if collided or done:
                    if collided:
                        # 碰撞时不将位置加入路径
                        pass
                    else:
                        path.append(current_pos.copy())
                        success_counter += 1  # 成功到达终点，增加计数器
                    self.step(state, action_idx, reward, next_state, done)
                    score += reward
                    break
                else:
                    success_counter = 0  # 重置连续成功计数器

                # 正常步骤
                path.append(current_pos.copy())
                self.step(state, action_idx, reward, next_state, done)
                score += reward

                # 更新状态
                state = next_state

            # 保存最佳路径 - 综合考虑奖励和路径长度
            if len(path) > 1:
                full_path = [self.start] + path[1:] + [self.goal]
                if self.is_path_valid(full_path):
                    current_path_length = self.calculate_path_length(full_path)
                    # 优先考虑奖励，同时考虑路径长度
                    if (score > self.best_reward or
                            (score == self.best_reward and current_path_length < self.best_path_length)):
                        self.best_reward = score
                        self.best_path = full_path
                        self.best_path_length = current_path_length

            scores.append(score)
            self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)  # 衰减探索率

            # 每100轮打印进度
            if i_episode % 100 == 0:
                avg_score = np.mean(scores[-100:])
                print(f"\rEpisode {i_episode}\t平均得分: {avg_score:.2f}\t探索率: {self.epsilon:.3f}")

            # 如果连续成功达到要求次数，提前结束训练
            if success_counter >= required_successes:
                print(f"\n在第 {i_episode} 轮达到连续成功要求，提前结束训练")
                break

        # 处理结果
        if self.best_path is not None:
            valid_path = [tuple(point) for point in self.best_path]
            if self.is_path_valid(self.best_path):
                path_length = self.calculate_path_length(self.best_path)
                return valid_path, f"找到有效路径，最佳奖励: {self.best_reward:.2f}, 路径长度: {path_length:.2f}"
            else:
                return None, f"最佳路径无效，最佳奖励: {self.best_reward:.2f}"
        else:
            return None, f"未找到有效路径，最佳奖励: {self.best_reward:.2f}"


class PlanningThread(QThread):
    finished = pyqtSignal(object, str, float)
    error = pyqtSignal(str)

    def __init__(self, planner, use_pretrained=False):
        super().__init__()
        self.planner = planner
        self.use_pretrained = use_pretrained

    def run(self):
        try:
            path, message = self.planner.plan(self.use_pretrained)
            # 发送路径、消息和最佳奖励值
            self.finished.emit(path, message, self.planner.best_reward)
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
        self.setWindowTitle("无人机三维路径规划 (带模型保存加载功能)")
        self.resize(1200, 700)

        self.is_planning = False
        self.path = None
        self.plan_time = 0.0
        self.min_flight_height = 0.5
        self.path_step_size = 0.5  # 路径插值步长
        self.model_path = "dqn_planner_model.pth"  # 默认模型路径

        # 障碍物设置
        self.buildings = [
            Building(15, 15, 0, 10, 10, 20, (0.6, 0.2, 0.2, 0.8)),
            Building(35, 35, 0, 12, 12, 18, (0.2, 0.6, 0.2, 0.8))
            # 可以添加更多障碍物
        ]

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)

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

        # 路径步长设置
        step_group = QGroupBox("路径步长")
        step_layout = QVBoxLayout()
        self.step_size_slider = QSlider(Qt.Horizontal)
        self.step_size_slider.setRange(1, 20)  # 1到2.0
        self.step_size_slider.setValue(3)  # 默认3
        self.step_label = QLabel("3.0")
        self.step_size_slider.valueChanged.connect(self.update_step_label)
        step_layout.addWidget(self.step_size_slider)
        step_layout.addWidget(self.step_label)
        step_group.setLayout(step_layout)

        # DQN算法参数设置
        dqn_group = QGroupBox("DQN算法参数")
        dqn_layout = QFormLayout()
        dqn_layout.setRowWrapPolicy(QFormLayout.DontWrapRows)
        dqn_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.episodes = QLineEdit("3000")  # 增加默认训练回合数
        self.episodes.setFixedWidth(80)
        self.max_steps = QLineEdit("300")  # 增加每回合最大步数
        self.max_steps.setFixedWidth(80)
        self.waypoints = QLineEdit("15")
        self.waypoints.setFixedWidth(80)
        dqn_layout.addRow("训练回合数:", self.episodes)
        dqn_layout.addRow("每回合最大步数:", self.max_steps)
        dqn_layout.addRow("路径点数:", self.waypoints)
        dqn_group.setLayout(dqn_layout)

        # 模型文件路径设置
        model_group = QGroupBox("模型文件设置")
        model_layout = QFormLayout()
        model_layout.setRowWrapPolicy(QFormLayout.DontWrapRows)
        model_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.model_path_edit = QLineEdit(self.model_path)
        self.model_path_edit.setFixedWidth(150)
        self.browse_button = QPushButton("浏览...")
        self.browse_button.setFixedWidth(60)
        self.browse_button.clicked.connect(self.browse_model_path)

        model_layout.addRow("模型路径:", self.model_path_edit)
        model_layout.addRow("", self.browse_button)
        model_group.setLayout(model_layout)

        # 按钮布局
        button_layout = QVBoxLayout()
        self.plan_button = QPushButton("训练并规划路径")
        self.plan_button.setMinimumHeight(30)
        self.plan_button.clicked.connect(self.plan_path)
        self.load_button = QPushButton("加载模型并规划")
        self.load_button.setMinimumHeight(30)
        self.load_button.clicked.connect(self.load_and_plan)
        self.save_button = QPushButton("保存当前模型")
        self.save_button.setMinimumHeight(30)
        self.save_button.clicked.connect(self.save_current_model)
        self.reset_button = QPushButton("重置视图")
        self.reset_button.setMinimumHeight(30)
        self.reset_button.clicked.connect(self.reset_view)

        button_layout.addWidget(self.plan_button)
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.reset_button)
        button_layout.addStretch()

        # 时间显示
        self.time_label = QLabel("规划时间: -- 秒")
        self.status_label = QLabel("状态: 就绪")
        time_group = QGroupBox("性能指标")
        time_layout = QVBoxLayout()
        time_layout.addWidget(self.time_label)
        time_layout.addWidget(self.status_label)
        time_group.setLayout(time_layout)

        # 组装控制面板
        control_layout.addWidget(start_group)
        control_layout.addWidget(goal_group)
        control_layout.addWidget(safe_group)
        control_layout.addWidget(height_group)
        control_layout.addWidget(step_group)
        control_layout.addWidget(dqn_group)
        control_layout.addWidget(model_group)
        control_layout.addWidget(time_group)
        control_layout.addLayout(button_layout)
        control_layout.addStretch()
        control_layout.setSpacing(8)

        # 右侧分割器
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

        self.init_visualization()

    def browse_model_path(self):
        """浏览模型文件路径"""
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(
            self, "选择模型保存路径", self.model_path, "PyTorch模型文件 (*.pth);;所有文件 (*)", options=options)
        if fileName:
            self.model_path = fileName
            self.model_path_edit.setText(fileName)

    def update_safe_label(self):
        value = self.safe_distance.value() / 10.0
        self.safe_label.setText(f"{value:.1f}")

    def update_height_label(self):
        value = self.min_height_slider.value() / 10.0
        self.height_label.setText(f"{value:.1f}")
        self.min_flight_height = value
        self.init_visualization()

    def update_step_label(self):
        value = self.step_size_slider.value()
        self.step_label.setText(f"{value:.1f}")
        self.path_step_size = value
        if self.path and len(self.path) > 1:
            self.show_final_path()

    def create_surface_arrays(self):
        x = np.linspace(0, 60, 2, dtype=np.float64)
        y = np.linspace(0, 60, 2, dtype=np.float64)
        X, Y = np.meshgrid(x, y)
        return X, Y

    def init_visualization(self):
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
            ax.scatter(start[0], start[1], start[2], c='green', s=50, marker='o', label='起点')
            ax.scatter(goal[0], goal[1], goal[2], c='red', s=50, marker='*', label='终点')

            # 设置坐标轴
            ax.set_xlabel('X轴', fontsize=8)
            ax.set_ylabel('Y轴', fontsize=8)
            ax.set_zlabel('Z轴', fontsize=8)
            ax.set_title('无人机三维路径规划 (带模型保存加载功能)', fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.set_xlim(0, 60)
            ax.set_ylim(0, 60)
            ax.set_zlim(0, 30)
            ax.legend(fontsize=8)

            self.canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "可视化错误", f"创建3D表面时出错: {str(e)}")
            print(f"调试信息: 错误类型={type(e)}, 错误内容={str(e)}")

    def interpolate_path(self, path):
        if len(path) < 2:
            return path

        interpolated = []
        for i in range(len(path) - 1):
            p1 = np.array(path[i])
            p2 = np.array(path[i + 1])
            segment_length = np.linalg.norm(p2 - p1)

            # 计算需要的插值点数
            num_points = max(2, int(segment_length / self.path_step_size) + 1)

            # 生成插值点
            for j in range(num_points):
                ratio = j / (num_points - 1)
                x = p1[0] + (p2[0] - p1[0]) * ratio
                y = p1[1] + (p2[1] - p1[1]) * ratio
                z = p1[2] + (p2[2] - p1[2]) * ratio
                interpolated.append((x, y, z))

        return interpolated

    def show_final_path(self):
        ax = self.canvas.axes
        ax.clear()

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

            # 绘制路径
            if self.path and len(self.path) > 1:
                # 插值使路径更平滑
                interpolated_path = self.interpolate_path(self.path)

                # 提取x, y, z坐标
                x_vals = [p[0] for p in interpolated_path]
                y_vals = [p[1] for p in interpolated_path]
                z_vals = [p[2] for p in interpolated_path]

                # 绘制路径线
                ax.plot(x_vals, y_vals, z_vals, 'b-', linewidth=2, label='规划路径')

                # 绘制路径点
                path_points_x = [p[0] for p in self.path]
                path_points_y = [p[1] for p in self.path]
                path_points_z = [p[2] for p in self.path]
                ax.scatter(path_points_x, path_points_y, path_points_z, c='blue', s=30, marker='o')

                # 显示路径点信息
                self.path_info.clear()
                for i, (x, y, z) in enumerate(self.path):
                    self.path_info.append(f"路径点 {i}: X={x:.2f}, Y={y:.2f}, Z={z:.2f}")

            # 绘制起点和终点
            start = (float(self.start_x.text()), float(self.start_y.text()), float(self.start_z.text()))
            goal = (float(self.goal_x.text()), float(self.goal_y.text()), float(self.goal_z.text()))
            ax.scatter(start[0], start[1], start[2], c='green', s=50, marker='o', label='起点')
            ax.scatter(goal[0], goal[1], goal[2], c='red', s=50, marker='*', label='终点')

            # 设置坐标轴
            ax.set_xlabel('X轴', fontsize=8)
            ax.set_ylabel('Y轴', fontsize=8)
            ax.set_zlabel('Z轴', fontsize=8)
            ax.set_title('无人机三维路径规划 (带模型保存加载功能)', fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.set_xlim(0, 60)
            ax.set_ylim(0, 60)
            ax.set_zlim(0, 30)
            ax.legend(fontsize=8)

            self.canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "可视化错误", f"绘制路径时出错: {str(e)}")

    def plan_path(self):
        """训练并规划路径"""
        if self.is_planning:
            return

        try:
            # 获取参数
            start = (float(self.start_x.text()), float(self.start_y.text()), float(self.start_z.text()))
            goal = (float(self.goal_x.text()), float(self.goal_y.text()), float(self.goal_z.text()))
            safe_distance = self.safe_distance.value() / 10.0
            min_height = self.min_height_slider.value() / 10.0
            num_episodes = int(self.episodes.text())
            max_steps = int(self.max_steps.text())
            num_waypoints = int(self.waypoints.text())
            self.model_path = self.model_path_edit.text()

            # 创建规划器
            self.planner = DQNPlanner(
                start=start,
                goal=goal,
                buildings=self.buildings,
                safe_distance=safe_distance,
                min_height=min_height,
                num_episodes=num_episodes,
                max_steps=max_steps,
                num_waypoints=num_waypoints
            )

            # 更新状态
            self.is_planning = True
            self.plan_button.setText("训练中...")
            self.status_label.setText("状态: 正在训练并规划路径...")
            self.time_label.setText("规划时间: 计算中...")

            # 启动规划线程
            self.thread = PlanningThread(self.planner, use_pretrained=False)
            self.thread.finished.connect(self.on_planning_finished)
            self.thread.error.connect(self.on_planning_error)
            self.start_time = time.time()
            self.thread.start()

        except ValueError as e:
            QMessageBox.warning(self, "输入错误", f"参数输入错误: {str(e)}")
            self.status_label.setText("状态: 就绪")

    def load_and_plan(self):
        """加载模型并规划路径"""
        if self.is_planning:
            return

        try:
            # 获取参数
            start = (float(self.start_x.text()), float(self.start_y.text()), float(self.start_z.text()))
            goal = (float(self.goal_x.text()), float(self.goal_y.text()), float(self.goal_z.text()))
            safe_distance = self.safe_distance.value() / 10.0
            min_height = self.min_height_slider.value() / 10.0
            num_episodes = int(self.episodes.text())
            max_steps = int(self.max_steps.text())
            num_waypoints = int(self.waypoints.text())
            self.model_path = self.model_path_edit.text()

            # 创建规划器
            self.planner = DQNPlanner(
                start=start,
                goal=goal,
                buildings=self.buildings,
                safe_distance=safe_distance,
                min_height=min_height,
                num_episodes=num_episodes,
                max_steps=max_steps,
                num_waypoints=num_waypoints
            )

            # 尝试加载模型
            if not self.planner.load_model(self.model_path):
                QMessageBox.warning(self, "加载失败", "未找到模型文件，将进行新训练")
                self.plan_path()
                return

            # 更新状态
            self.is_planning = True
            self.load_button.setText("规划中...")
            self.status_label.setText("状态: 使用预训练模型规划路径...")
            self.time_label.setText("规划时间: 计算中...")

            # 启动规划线程（使用预训练模型）
            self.thread = PlanningThread(self.planner, use_pretrained=True)
            self.thread.finished.connect(self.on_planning_finished)
            self.thread.error.connect(self.on_planning_error)
            self.start_time = time.time()
            self.thread.start()

        except ValueError as e:
            QMessageBox.warning(self, "输入错误", f"参数输入错误: {str(e)}")
            self.status_label.setText("状态: 就绪")

    def save_current_model(self):
        """保存当前模型"""
        if hasattr(self, 'planner'):
            try:
                self.model_path = self.model_path_edit.text()
                saved_path = self.planner.save_model(self.model_path)
                QMessageBox.information(self, "保存成功", f"模型已成功保存至:\n{saved_path}")
            except Exception as e:
                QMessageBox.critical(self, "保存失败", f"保存模型时出错: {str(e)}")
        else:
            QMessageBox.warning(self, "警告", "请先进行至少一次路径规划")

    def on_planning_finished(self, path, message, best_value):
        self.plan_time = time.time() - self.start_time
        self.is_planning = False
        self.plan_button.setText("训练并规划路径")
        self.load_button.setText("加载模型并规划")
        self.path = path
        self.time_label.setText(f"规划时间: {self.plan_time:.2f} 秒")
        self.status_label.setText(f"状态: {message}")

        # 训练完成后自动保存模型
        if hasattr(self, 'planner') and not self.thread.use_pretrained:
            try:
                self.planner.save_model(self.model_path)
            except:
                pass  # 保存失败不影响主流程

        # 显示路径
        if path:
            self.show_final_path()
        else:
            self.init_visualization()

    def on_planning_error(self, error_msg):
        self.is_planning = False
        self.plan_button.setText("训练并规划路径")
        self.load_button.setText("加载模型并规划")
        self.status_label.setText(f"状态: 规划出错")
        QMessageBox.critical(self, "规划错误", f"路径规划过程中发生错误: {error_msg}")

    def reset_view(self):
        self.init_visualization()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DronePathVisualizer()
    window.show()
    sys.exit(app.exec_())
