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
# class GeneticAlgorithmPlanner:
#     def __init__(self, start, goal, buildings,
#                  safe_distance=3.0, min_height=0.5,
#                  population_size=100, generations=500,
#                  mutation_rate=0.1, num_waypoints=10,
#                  x_range=(0, 60), y_range=(0, 60), z_range=(0, 30)):
#         self.start = np.array(start)
#         self.goal = np.array(goal)
#         self.buildings = buildings
#         self.safe_distance = safe_distance
#         self.safe_distance_sq = safe_distance ** 2
#         self.min_height = min_height
#
#         self.population_size = population_size
#         self.generations = generations
#         self.mutation_rate = mutation_rate
#         self.num_waypoints = num_waypoints
#
#         self.x_range = x_range
#         self.y_range = y_range
#         self.z_range = z_range
#
#         for b in self.buildings:
#             b.set_safe_bounds(safe_distance)
#
#         # 计算起点到终点的大致方向向量
#         self.direction = self.goal - self.start
#         self.total_distance = np.linalg.norm(self.direction)
#         if self.total_distance > 0:
#             self.unit_direction = self.direction / self.total_distance
#         else:
#             self.unit_direction = np.array([0, 0, 0])
#
#         # 记录找到的最佳路径及其适应度
#         self.best_path = None
#         self.best_fitness = 0
#
#     def distance_to_obstacle(self, point):
#         """计算点到最近障碍物的距离"""
#         min_dist = float('inf')
#         x, y, z = point
#
#         for b in self.buildings:
#             # 计算到障碍物的距离
#             dx = max(b.x_min - x, 0, x - b.x_max)
#             dy = max(b.y_min - y, 0, y - b.y_max)
#             dz = max(b.z_min - z, 0, z - b.z_max)
#
#             dist = np.sqrt(dx * dx + dy * dy + dz * dz)
#             min_dist = min(min_dist, dist)
#
#         return min_dist
#
#     def is_collision(self, point, is_start_end=False):
#         x, y, z = point
#
#         # 检查边界
#         if (x < self.x_range[0] or x > self.x_range[1] or
#                 y < self.y_range[0] or y > self.y_range[1] or
#                 z < self.z_range[0] or z > self.z_range[1]):
#             return True
#
#         # 起点和终点特殊处理，z 低于最低飞行高度不视为碰撞
#         if is_start_end:
#             return False
#
#         # 检查障碍物安全距离
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
#         # 检查最低高度（起点终点除外）
#         if z < self.min_height:
#             return True
#
#         return False
#
#     def calculate_collision_penalty(self, path):
#         """计算路径的碰撞惩罚，距离障碍物越近惩罚越大"""
#         total_penalty = 0.0
#         segment_penalty = 0.0
#         boundary_penalty = 0.0
#
#         # 对路径点的惩罚
#         for i, point in enumerate(path):
#             # 起点和终点不惩罚 z 轴高度
#             is_start_end = (i == 0 or i == len(path) - 1)
#
#             # 检查边界越界惩罚
#             if point[0] < self.x_range[0]:
#                 boundary_penalty += (self.x_range[0] - point[0]) * 10
#             elif point[0] > self.x_range[1]:
#                 boundary_penalty += (point[0] - self.x_range[1]) * 10
#
#             if point[1] < self.y_range[0]:
#                 boundary_penalty += (self.y_range[0] - point[1]) * 10
#             elif point[1] > self.y_range[1]:
#                 boundary_penalty += (point[1] - self.y_range[1]) * 10
#
#             if point[2] < self.z_range[0]:
#                 boundary_penalty += (self.z_range[0] - point[2]) * 10
#             elif point[2] > self.z_range[1]:
#                 boundary_penalty += (point[2] - self.z_range[1]) * 10
#
#             # 高度惩罚（起点终点除外）
#             if not is_start_end and point[2] < self.min_height:
#                 boundary_penalty += (self.min_height - point[2]) * 20
#
#             # 障碍物距离惩罚
#             dist = self.distance_to_obstacle(point)
#             if dist < self.safe_distance:
#                 # 距离越近惩罚越大，非线性增长
#                 penalty = (self.safe_distance - dist) / (dist + 1e-6) * 10
#                 total_penalty += penalty
#
#         # 对路径段的惩罚
#         for i in range(len(path) - 1):
#             p1 = path[i]
#             p2 = path[i + 1]
#             dist = np.linalg.norm(p2 - p1)
#             num_samples = max(5, int(dist))  # 加密采样
#
#             for j in range(1, num_samples):
#                 ratio = j / num_samples
#                 x = p1[0] + (p2[0] - p1[0]) * ratio
#                 y = p1[1] + (p2[1] - p1[1]) * ratio
#                 z = p1[2] + (p2[2] - p1[2]) * ratio
#
#                 point = (x, y, z)
#                 dist_to_obs = self.distance_to_obstacle(point)
#                 if dist_to_obs < self.safe_distance:
#                     penalty = (self.safe_distance - dist_to_obs) / (dist_to_obs + 1e-6) * 5
#                     segment_penalty += penalty
#
#         return total_penalty + segment_penalty + boundary_penalty
#
#     def is_path_valid(self, path):
#         """检查路径是否有效（无碰撞），处理起点终点 z 轴情况"""
#         # 检查路径上所有点（除起点终点外）的碰撞
#         for i, point in enumerate(path):
#             is_start_end = (i == 0 or i == len(path) - 1)
#             if self.is_collision(point, is_start_end):
#                 return False
#
#         # 检查路径段是否碰撞
#         for i in range(len(path) - 1):
#             p1 = path[i]
#             p2 = path[i + 1]
#             dist = np.linalg.norm(p2 - p1)
#             num_samples = max(5, int(dist))  # 加密采样
#
#             for j in range(1, num_samples):
#                 ratio = j / num_samples
#                 x = p1[0] + (p2[0] - p1[0]) * ratio
#                 y = p1[1] + (p2[1] - p1[1]) * ratio
#                 z = p1[2] + (p2[2] - p1[2]) * ratio
#
#                 if self.is_collision((x, y, z)):
#                     return False
#
#         return True
#
#     def calculate_path_length(self, path):
#         length = 0.0
#         for i in range(len(path) - 1):
#             length += np.linalg.norm(path[i + 1] - path[i])
#         return length
#
#     def calculate_smoothness(self, path):
#         """计算路径平滑度（相邻线段夹角的平方和）"""
#         smoothness = 0.0
#         for i in range(1, len(path) - 1):
#             v1 = path[i] - path[i - 1]
#             v2 = path[i + 1] - path[i]
#             if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
#                 cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
#                 cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 防止数值误差
#                 angle = np.arccos(cos_theta)
#                 smoothness += angle ** 2
#         return smoothness
#
#     def calculate_goal_proximity(self, path):
#         """计算路径终点与目标点的接近程度"""
#         last_point = path[-1]
#         return np.linalg.norm(last_point - self.goal)
#
#     def calculate_fitness(self, individual):
#         """改进的适应度函数：对接近可行的路径给予部分奖励"""
#         # 构建完整路径（包含起点和终点）
#         path = np.vstack((self.start, individual, self.goal))
#
#         # 计算基本指标
#         path_length = self.calculate_path_length(path)
#         smoothness = self.calculate_smoothness(path)
#         collision_penalty = self.calculate_collision_penalty(path)
#         goal_proximity = self.calculate_goal_proximity(path)
#
#         # 对可行路径给予额外奖励
#         is_valid = self.is_path_valid(path)
#         validity_bonus = 100.0 if is_valid else 0.0
#
#         # 适应度计算：综合考虑各项因素
#         # 路径越短、越平滑、越接近目标、碰撞越少，适应度越高
#         fitness = validity_bonus + \
#                   (1.0 / (path_length + 1e-6)) * 20 + \
#                   (1.0 / (smoothness + 1e-6)) * 5 + \
#                   (1.0 / (goal_proximity + 1e-6)) * 10 - \
#                   collision_penalty
#
#         # 确保适应度不为负
#         return max(fitness, 0.001)
#
#     def create_individual(self):
#         """改进的初始化方法：基于障碍物信息的启发式初始化"""
#         individual = []
#
#         # 检查是否有直接路径
#         direct_path = [self.start, self.goal]
#         if self.is_path_valid(direct_path):
#             # 如果有直接路径，生成接近直线的路径
#             for i in range(1, self.num_waypoints + 1):
#                 ratio = i / (self.num_waypoints + 1)
#                 point = self.start + ratio * (self.goal - self.start)
#                 individual.append(point)
#             return np.array(individual)
#
#         # 否则，生成避开已知障碍物的初始点
#         for i in range(1, self.num_waypoints + 1):
#             ratio = i / (self.num_waypoints + 1)
#             base_point = self.start + ratio * (self.goal - self.start)
#
#             # 检查该点是否在障碍物附近
#             for b in self.buildings:
#                 if (base_point[0] > b.safe_x_min and base_point[0] < b.safe_x_max and
#                         base_point[1] > b.safe_y_min and base_point[1] < b.safe_y_max and
#                         base_point[2] > b.safe_z_min and base_point[2] < b.safe_z_max):
#                     # 如果在障碍物附近，向上或向两侧偏移
#                     if base_point[2] < b.z_max + self.safe_distance:
#                         # 向上偏移以避开障碍物
#                         base_point[2] = b.z_max + self.safe_distance + 2
#                     else:
#                         # 向侧面偏移
#                         offset = 5 if i % 2 == 0 else -5
#                         base_point[0] += offset
#                         base_point[1] += offset if i % 3 == 0 else -offset
#
#             # 确保在范围内
#             base_point[0] = np.clip(base_point[0], self.x_range[0], self.x_range[1])
#             base_point[1] = np.clip(base_point[1], self.y_range[0], self.y_range[1])
#             base_point[2] = np.clip(base_point[2], self.z_range[0], self.z_range[1])
#
#             individual.append(base_point)
#
#         return np.array(individual)
#
#     def create_population(self):
#         """创建初始种群，确保多样性"""
#         population = []
#         # 创建基础个体
#         for _ in range(int(self.population_size * 0.7)):
#             population.append(self.create_individual())
#
#         # 添加一些随机个体以增加多样性
#         for _ in range(int(self.population_size * 0.3)):
#             individual = []
#             for _ in range(self.num_waypoints):
#                 point = np.array([
#                     random.uniform(self.x_range[0], self.x_range[1]),
#                     random.uniform(self.y_range[0], self.y_range[1]),
#                     random.uniform(self.z_range[0], self.z_range[1])
#                 ])
#                 individual.append(point)
#             population.append(np.array(individual))
#
#         return population[:self.population_size]
#
#     def select_parents(self, population, fitnesses):
#         """锦标赛选择，提高优质个体被选中的概率"""
#         # 锦标赛规模
#         tournament_size = min(5, len(population))
#
#         # 选择第一个父代
#         candidates = random.sample(list(zip(population, fitnesses)), tournament_size)
#         parent1 = max(candidates, key=lambda x: x[1])[0]
#
#         # 选择第二个父代，确保与第一个不同
#         while True:
#             candidates = random.sample(list(zip(population, fitnesses)), tournament_size)
#             parent2 = max(candidates, key=lambda x: x[1])[0]
#             if not np.array_equal(parent1, parent2):
#                 break
#
#         return parent1, parent2
#
#     def crossover(self, parent1, parent2):
#         """改进的交叉操作：混合均匀交叉和两点交叉"""
#         if np.random.random() < 0.8:  # 80%的概率进行交叉
#             # 50%概率使用两点交叉，50%使用均匀交叉
#             if np.random.random() < 0.5:
#                 # 两点交叉
#                 point1 = random.randint(1, self.num_waypoints - 2)
#                 point2 = random.randint(point1 + 1, self.num_waypoints - 1)
#
#                 child = np.copy(parent1)
#                 child[point1:point2] = parent2[point1:point2]
#                 return child
#             else:
#                 # 均匀交叉
#                 child = np.copy(parent1)
#                 for i in range(self.num_waypoints):
#                     if np.random.random() < 0.5:
#                         child[i] = parent2[i]
#                 return child
#         else:
#             # 不交叉，随机返回一个父代
#             return random.choice([parent1, parent2])
#
#     def mutate(self, individual, current_mutation_rate):
#         """改进的变异操作：自适应步长和方向"""
#         mutated = np.copy(individual)
#         mutation_strength = 2.0  # 基础变异强度
#
#         # 根据当前个体质量调整变异强度
#         # 适应度低的个体变异强度更大
#         individual_fitness = self.calculate_fitness(individual)
#         if individual_fitness < self.best_fitness * 0.5:
#             mutation_strength *= 1.5
#
#         for i in range(self.num_waypoints):
#             if np.random.random() < current_mutation_rate:
#                 # 智能变异方向：如果点太靠近障碍物，向远离方向变异
#                 dist = self.distance_to_obstacle(mutated[i])
#                 if dist < self.safe_distance:
#                     # 计算远离障碍物的方向
#                     best_dir = np.array([0.0, 0.0, 1.0])  # 默认向上
#                     max_dist = 0
#
#                     # 尝试几个方向，选择能远离障碍物的方向
#                     for _ in range(5):
#                         dir_candidate = np.random.normal(0, 1, 3)
#                         dir_candidate = dir_candidate / (np.linalg.norm(dir_candidate) + 1e-6)
#                         test_point = mutated[i] + dir_candidate * mutation_strength
#                         test_dist = self.distance_to_obstacle(test_point)
#                         if test_dist > max_dist:
#                             max_dist = test_dist
#                             best_dir = dir_candidate
#
#                     mutated[i] += best_dir * mutation_strength * 1.2
#                 else:
#                     # 随机变异
#                     mutated[i] += np.random.normal(0, mutation_strength, 3)
#
#                 # 确保在范围内
#                 mutated[i, 0] = np.clip(mutated[i, 0], self.x_range[0], self.x_range[1])
#                 mutated[i, 1] = np.clip(mutated[i, 1], self.y_range[0], self.y_range[1])
#                 mutated[i, 2] = np.clip(mutated[i, 2], self.z_range[0], self.z_range[1])
#
#         # 整体偏移变异，保持路径形状
#         if np.random.random() < 0.15:
#             shift = np.random.normal(0, mutation_strength * 0.5, 3)
#             mutated += shift
#             # 再次确保在范围内
#             for i in range(self.num_waypoints):
#                 mutated[i, 0] = np.clip(mutated[i, 0], self.x_range[0], self.x_range[1])
#                 mutated[i, 1] = np.clip(mutated[i, 1], self.y_range[0], self.y_range[1])
#                 mutated[i, 2] = np.clip(mutated[i, 2], self.z_range[0], self.z_range[1])
#
#         return mutated
#
#     def evolve_population(self, population, generation):
#         """改进的进化策略：动态调整参数"""
#         fitnesses = [self.calculate_fitness(ind) for ind in population]
#
#         # 记录全局最佳
#         current_best_idx = np.argmax(fitnesses)
#         current_best_fitness = fitnesses[current_best_idx]
#         current_best_individual = population[current_best_idx]
#
#         if current_best_fitness > self.best_fitness:
#             self.best_fitness = current_best_fitness
#             self.best_path = np.vstack((self.start, current_best_individual, self.goal))
#
#         # 动态调整变异率：早期高探索，后期高利用
#         if generation < self.generations * 0.3:
#             current_mutation_rate = min(0.3, self.mutation_rate + 0.02)
#         elif generation < self.generations * 0.7:
#             current_mutation_rate = self.mutation_rate
#         else:
#             # 后期降低变异率，但如果找不到好解则提高
#             if self.best_fitness < 50:  # 阈值表示还没有找到较好的解
#                 current_mutation_rate = min(0.25, self.mutation_rate + 0.05)
#             else:
#                 current_mutation_rate = max(0.05, self.mutation_rate - 0.01)
#
#         # 精英选择：保留10%的最佳个体
#         elite_size = max(1, int(self.population_size * 0.1))
#         elites = [population[i] for i in np.argsort(fitnesses)[-elite_size:]]
#
#         # 创建新种群
#         new_population = elites.copy()
#
#         # 生成后代
#         while len(new_population) < self.population_size:
#             parent1, parent2 = self.select_parents(population, fitnesses)
#             child = self.crossover(parent1, parent2)
#             child = self.mutate(child, current_mutation_rate)
#             new_population.append(child)
#
#         # 定期引入新的随机个体，防止早熟收敛
#         if generation % 20 == 0 and generation > 0:
#             for i in range(int(self.population_size * 0.05)):
#                 if len(new_population) < self.population_size:
#                     new_population.append(self.create_individual())
#
#         return new_population[:self.population_size], fitnesses
#
#     def plan(self):
#         """主规划函数"""
#         # 检查起点和终点是否合法（仅检查边界，不检查高度）
#         if self.is_collision(self.start, is_start_end=True):
#             return None, "起点在障碍物或边界外"
#         if self.is_collision(self.goal, is_start_end=True):
#             return None, "终点在障碍物或边界外"
#
#         # 检查起点到终点是否可以直接到达
#         direct_path = [self.start, self.goal]
#         if self.is_path_valid(direct_path):
#             return [tuple(self.start), tuple(self.goal)], "找到直接路径"
#
#         # 初始化种群
#         population = self.create_population()
#         self.best_fitness = 0
#         self.best_path = None
#
#         # 进化
#         no_improve_generations = 0
#         for generation in range(self.generations):
#             # 进化一代
#             population, fitnesses = self.evolve_population(population, generation)
#
#             # 检查是否有改进
#             current_best = max(fitnesses)
#             if current_best > self.best_fitness * 1.05:  # 5%以上的改进
#                 no_improve_generations = 0
#             else:
#                 no_improve_generations += 1
#
#             # 如果长时间没有改进，引入新的随机个体
#             if no_improve_generations > 50:
#                 # 替换20%的种群
#                 for i in range(int(self.population_size * 0.2)):
#                     idx = random.randint(0, len(population) - 1)
#                     population[idx] = self.create_individual()
#                 no_improve_generations = 0
#
#         # 检查是否找到有效路径
#         if self.best_path is not None:
#             valid_path = [tuple(point) for point in self.best_path]
#             if self.is_path_valid(self.best_path):
#                 return valid_path, f"找到有效路径，最佳适应度: {self.best_fitness:.2f}"
#             else:
#                 # 如果最佳路径仍无效，尝试优化它
#                 optimized_path = self.optimize_invalid_path(self.best_path)
#                 if optimized_path and self.is_path_valid(optimized_path):
#                     return [tuple(point) for point in optimized_path], "优化后找到有效路径"
#
#         return None, f"未找到有效路径，最佳适应度: {self.best_fitness:.2f}"
#
#     def optimize_invalid_path(self, path):
#         """尝试优化接近有效的路径"""
#         for _ in range(20):  # 尝试20次优化
#             new_path = np.copy(path)
#             # 只调整碰撞的点
#             for i in range(1, len(path) - 1):  # 跳过起点和终点
#                 if self.is_collision(path[i]):
#                     # 向远离障碍物的方向微调
#                     dir_vec = np.random.normal(0, 1, 3)
#                     dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-6)
#                     new_path[i] = path[i] + dir_vec * 1.5
#
#             if self.is_path_valid(new_path):
#                 return new_path
#         return None
#
#
# class PlanningThread(QThread):
#     finished = pyqtSignal(object, str, float)  # 添加适应度值信号
#     error = pyqtSignal(str)
#
#     def __init__(self, planner):
#         super().__init__()
#         self.planner = planner
#
#     def run(self):
#         try:
#             path, message = self.planner.plan()
#             # 发送路径、消息和最佳适应度
#             self.finished.emit(path, message, self.planner.best_fitness)
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
#         self.setWindowTitle("无人机三维路径规划 (优化遗传算法)")
#         self.resize(1200, 700)
#
#         self.is_planning = False
#         self.path = None
#         self.plan_time = 0.0
#         self.min_flight_height = 0.5
#         self.path_step_size = 0.5  # 路径插值步长
#
#         # 障碍物设置
#         self.buildings = [
#             Building(15, 15, 0, 10, 10, 20, (0.6, 0.2, 0.2, 0.8)),
#             Building(35, 35, 0, 12, 12, 18, (0.2, 0.6, 0.2, 0.8))
#             # 可以添加更多障碍物
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
#         # 路径步长设置（新增）
#         step_group = QGroupBox("路径步长")
#         step_layout = QVBoxLayout()
#         self.step_size_slider = QSlider(Qt.Horizontal)
#         self.step_size_slider.setRange(1, 20)  # 1到2.0
#         self.step_size_slider.setValue(3)  # 默认3
#         self.step_label = QLabel("3.0")
#         self.step_size_slider.valueChanged.connect(self.update_step_label)
#         step_layout.addWidget(self.step_size_slider)
#         step_layout.addWidget(self.step_label)
#         step_group.setLayout(step_layout)
#
#         # GA算法参数设置
#         ga_group = QGroupBox("遗传算法参数")
#         ga_layout = QFormLayout()
#         ga_layout.setRowWrapPolicy(QFormLayout.DontWrapRows)
#         ga_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
#         self.population_size = QLineEdit("100")
#         self.population_size.setFixedWidth(80)
#         self.generations = QLineEdit("500")
#         self.generations.setFixedWidth(80)
#         self.mutation_rate = QLineEdit("0.1")
#         self.mutation_rate.setFixedWidth(80)
#         self.waypoints = QLineEdit("10")
#         self.waypoints.setFixedWidth(80)
#         ga_layout.addRow("种群大小:", self.population_size)
#         ga_layout.addRow("进化代数:", self.generations)
#         ga_layout.addRow("变异率:", self.mutation_rate)
#         ga_layout.addRow("路径点数:", self.waypoints)
#         ga_group.setLayout(ga_layout)
#
#         # 按钮布局
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
#         # 时间显示
#         self.time_label = QLabel("规划时间: -- 秒")
#         self.status_label = QLabel("状态: 就绪")
#         time_group = QGroupBox("性能指标")
#         time_layout = QVBoxLayout()
#         time_layout.addWidget(self.time_label)
#         time_layout.addWidget(self.status_label)
#         time_group.setLayout(time_layout)
#
#         # 组装控制面板（添加步长控制）
#         control_layout.addWidget(start_group)
#         control_layout.addWidget(goal_group)
#         control_layout.addWidget(safe_group)
#         control_layout.addWidget(height_group)
#         control_layout.addWidget(step_group)  # 新增步长控制
#         control_layout.addWidget(ga_group)
#         control_layout.addWidget(time_group)
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
#         self.init_visualization()
#
#     def update_safe_label(self):
#         value = self.safe_distance.value() / 10.0
#         self.safe_label.setText(f"{value:.1f}")
#
#     def update_height_label(self):
#         value = self.min_height_slider.value() / 10.0
#         self.height_label.setText(f"{value:.1f}")
#         self.min_flight_height = value
#         self.init_visualization()
#
#     def update_step_label(self):
#         """更新步长显示"""
#         value = self.step_size_slider.value()
#         self.step_label.setText(f"{value:.1f}")
#         self.path_step_size = value
#         # 如果已有路径，重新绘制
#         if self.path and len(self.path) > 1:
#             self.show_final_path()
#
#     def create_surface_arrays(self):
#         """创建表面绘制所需的数组，确保类型正确"""
#         x = np.linspace(0, 60, 2, dtype=np.float64)
#         y = np.linspace(0, 60, 2, dtype=np.float64)
#         X, Y = np.meshgrid(x, y)
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
#             ax.set_title('无人机三维路径规划 (优化遗传算法)', fontsize=10)
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
#     def interpolate_path(self, path):
#         """根据步长对路径进行插值，使显示更平滑"""
#         if len(path) < 2:
#             return path
#
#         interpolated = []
#         for i in range(len(path) - 1):
#             p1 = np.array(path[i])
#             p2 = np.array(path[i + 1])
#             segment_length = np.linalg.norm(p2 - p1)
#
#             # 计算需要的插值点数
#             num_points = max(2, int(segment_length / self.path_step_size) + 1)
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
#     def show_final_path(self):
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
#             # 绘制路径
#             if self.path and len(self.path) > 1:
#                 # 根据步长插值路径点
#                 interpolated_path = self.interpolate_path(self.path)
#
#                 x = [p[0] for p in interpolated_path]
#                 y = [p[1] for p in interpolated_path]
#                 z = [p[2] for p in interpolated_path]
#
#                 ax.plot(x, y, z, 'b-', linewidth=2, label='规划路径')
#                 # 只绘制原始路径点，不绘制插值点
#                 original_x = [p[0] for p in self.path]
#                 original_y = [p[1] for p in self.path]
#                 original_z = [p[2] for p in self.path]
#                 ax.scatter(original_x, original_y, original_z, color='blue', s=50, alpha=0.7, marker='o')
#
#                 # 显示路径信息（包含适应度）
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
#                 info_text += f"\n最佳适应度值: {self.best_fitness:.2f}"  # 显示适应度
#                 self.path_info.setText(info_text)
#
#             # 设置坐标轴
#             ax.set_xlabel('X轴', fontsize=8)
#             ax.set_ylabel('Y轴', fontsize=8)
#             ax.set_zlabel('Z轴', fontsize=8)
#             ax.set_title('无人机三维路径规划 (优化遗传算法)', fontsize=10)
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
#             population_size = int(self.population_size.text())
#             generations = int(self.generations.text())
#             mutation_rate = float(self.mutation_rate.text())
#             num_waypoints = int(self.waypoints.text())
#
#             if mutation_rate < 0 or mutation_rate > 1:
#                 QMessageBox.critical(self, "参数错误", "变异率必须在0-1之间！")
#                 return
#
#             self.plan_button.setEnabled(False)
#             self.is_planning = True
#             self.time_label.setText("规划中...")
#             self.status_label.setText("状态: 正在规划路径")
#             self.path_info.setText("正在规划路径，请稍候...")
#
#             planner = GeneticAlgorithmPlanner(
#                 start,
#                 goal,
#                 self.buildings,
#                 safe_distance=self.safe_distance.value() / 10.0,
#                 min_height=self.min_flight_height,
#                 population_size=population_size,
#                 generations=generations,
#                 mutation_rate=mutation_rate,
#                 num_waypoints=num_waypoints
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
#             self.status_label.setText("状态: 就绪")
#
#     def on_planning_finished(self, path, message, best_fitness):
#         """接收路径、消息和最佳适应度值"""
#         self.plan_time = time.time() - self.plan_start_time
#         self.time_label.setText(f"规划时间: {self.plan_time:.4f} 秒")
#         self.status_label.setText(f"状态: {message}")
#         self.path = path
#         self.best_fitness = best_fitness  # 保存最佳适应度值
#
#         if self.path and len(self.path) > 1:
#             self.show_final_path()
#         else:
#             QMessageBox.warning(self, "规划结果", message)
#             self.show_final_path()
#
#         self.plan_button.setEnabled(True)
#         self.is_planning = False
#
#     def on_planning_error(self, error_msg):
#         QMessageBox.critical(self, "规划错误", f"路径规划过程中发生错误: {error_msg}")
#         self.time_label.setText("规划时间: 错误")
#         self.status_label.setText("状态: 规划出错")
#         self.plan_button.setEnabled(True)
#         self.is_planning = False
#
#     def reset_view(self):
#         if self.is_planning:
#             return
#
#         self.path = None
#         self.best_fitness = 0  # 重置适应度
#         self.time_label.setText("规划时间: -- 秒")
#         self.status_label.setText("状态: 就绪")
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
import sys
import time
import random
import math
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                             QGroupBox, QFormLayout, QLineEdit, QSlider, QLabel, QPushButton,
                             QSplitter, QScrollArea, QTextEdit, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 设置matplotlib支持中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.titlesize"] = 10

# 建筑物类
class Building:
    def __init__(self, x, y, z, width, depth, height, color):
        self.x = x  # 中心点x坐标
        self.y = y  # 中心点y坐标
        self.z = z  # 底部z坐标
        self.width = width  # x方向宽度
        self.depth = depth  # y方向深度
        self.height = height  # 高度
        self.color = color  # 颜色

        # 计算边界
        self.x_min = x - width / 2
        self.x_max = x + width / 2
        self.y_min = y - depth / 2
        self.y_max = y + depth / 2
        self.z_min = z
        self.z_max = z + height

        # 安全边界（初始化为与建筑边界相同）
        self.safe_x_min = self.x_min
        self.safe_x_max = self.x_max
        self.safe_y_min = self.y_min
        self.safe_y_max = self.y_max
        self.safe_z_min = self.z_min
        self.safe_z_max = self.z_max

    def set_safe_bounds(self, safe_distance):
        """设置考虑安全距离的边界"""
        self.safe_x_min = self.x_min - safe_distance
        self.safe_x_max = self.x_max + safe_distance
        self.safe_y_min = self.y_min - safe_distance
        self.safe_y_max = self.y_max + safe_distance
        self.safe_z_min = self.z_min - safe_distance
        self.safe_z_max = self.z_max + safe_distance

    def get_faces(self):
        """返回建筑物的6个面，用于3D绘制"""
        # 底部
        bottom = [(self.x_min, self.y_min, self.z_min),
                  (self.x_max, self.y_min, self.z_min),
                  (self.x_max, self.y_max, self.z_min),
                  (self.x_min, self.y_max, self.z_min)]

        # 顶部
        top = [(self.x_min, self.y_min, self.z_max),
               (self.x_max, self.y_min, self.z_max),
               (self.x_max, self.y_max, self.z_max),
               (self.x_min, self.y_max, self.z_max)]

        # 前面
        front = [(self.x_min, self.y_min, self.z_min),
                 (self.x_max, self.y_min, self.z_min),
                 (self.x_max, self.y_min, self.z_max),
                 (self.x_min, self.y_min, self.z_max)]

        # 后面
        back = [(self.x_min, self.y_max, self.z_min),
                (self.x_max, self.y_max, self.z_min),
                (self.x_max, self.y_max, self.z_max),
                (self.x_min, self.y_max, self.z_max)]

        # 左面
        left = [(self.x_min, self.y_min, self.z_min),
                (self.x_min, self.y_max, self.z_min),
                (self.x_min, self.y_max, self.z_max),
                (self.x_min, self.y_min, self.z_max)]

        # 右面
        right = [(self.x_max, self.y_min, self.z_min),
                 (self.x_max, self.y_max, self.z_min),
                 (self.x_max, self.y_max, self.z_max),
                 (self.x_max, self.y_min, self.z_max)]

        return [bottom, top, front, back, left, right]


#  matplotlib画布类
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = plt.figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111, projection='3d')
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()


# 规划线程类
class PlanningThread(QThread):
    finished = pyqtSignal(list, str, float)
    error = pyqtSignal(str)

    def __init__(self, planner):
        super().__init__()
        self.planner = planner

    def run(self):
        try:
            path, message = self.planner.plan()
            self.finished.emit(path, message, self.planner.best_fitness)
        except Exception as e:
            self.error.emit(str(e))


# 遗传算法规划器类
class GeneticAlgorithmPlanner:
    def __init__(self, start, goal, buildings,
                 safe_distance=3.0, min_height=0.5,
                 population_size=100, generations=500,
                 mutation_rate=0.1, num_waypoints=10,
                 x_range=(0, 60), y_range=(0, 60), z_range=(0, 30)):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.buildings = buildings
        self.safe_distance = safe_distance
        self.safe_distance_sq = safe_distance ** 2
        self.min_height = min_height

        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
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

        # 记录找到的最佳路径及其适应度
        self.best_path = None
        self.best_fitness = 0

    def distance_to_obstacle(self, point):
        """计算点到最近障碍物的距离"""
        min_dist = float('inf')
        x, y, z = point

        for b in self.buildings:
            # 计算到障碍物的距离
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

    def calculate_collision_penalty(self, path):
        """计算路径的碰撞惩罚，距离障碍物越近惩罚越大"""
        total_penalty = 0.0
        segment_penalty = 0.0
        boundary_penalty = 0.0

        # 对路径点的惩罚
        for i, point in enumerate(path):
            # 起点和终点不惩罚 z 轴高度
            is_start_end = (i == 0 or i == len(path) - 1)

            # 检查边界越界惩罚
            if point[0] < self.x_range[0]:
                boundary_penalty += (self.x_range[0] - point[0]) * 10
            elif point[0] > self.x_range[1]:
                boundary_penalty += (point[0] - self.x_range[1]) * 10

            if point[1] < self.y_range[0]:
                boundary_penalty += (self.y_range[0] - point[1]) * 10
            elif point[1] > self.y_range[1]:
                boundary_penalty += (point[1] - self.y_range[1]) * 10

            if point[2] < self.z_range[0]:
                boundary_penalty += (self.z_range[0] - point[2]) * 10
            elif point[2] > self.z_range[1]:
                boundary_penalty += (point[2] - self.z_range[1]) * 10

            # 高度惩罚（起点终点除外）
            if not is_start_end and point[2] < self.min_height:
                boundary_penalty += (self.min_height - point[2]) * 20

            # 障碍物距离惩罚
            dist = self.distance_to_obstacle(point)
            if dist < self.safe_distance:
                # 距离越近惩罚越大，非线性增长
                penalty = (self.safe_distance - dist) / (dist + 1e-6) * 10
                total_penalty += penalty

        # 对路径段的惩罚
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

                point = (x, y, z)
                dist_to_obs = self.distance_to_obstacle(point)
                if dist_to_obs < self.safe_distance:
                    penalty = (self.safe_distance - dist_to_obs) / (dist_to_obs + 1e-6) * 5
                    segment_penalty += penalty

        return total_penalty + segment_penalty + boundary_penalty

    def is_path_valid(self, path):
        """检查路径是否有效（无碰撞），处理起点终点 z 轴情况"""
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

    def calculate_smoothness(self, path):
        """计算路径平滑度（相邻线段夹角的平方和）"""
        smoothness = 0.0
        for i in range(1, len(path) - 1):
            v1 = path[i] - path[i - 1]
            v2 = path[i + 1] - path[i]
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 防止数值误差
                angle = np.arccos(cos_theta)
                smoothness += angle ** 2
        return smoothness

    def calculate_goal_proximity(self, path):
        """计算路径终点与目标点的接近程度"""
        last_point = path[-1]
        return np.linalg.norm(last_point - self.goal)

    def calculate_fitness(self, individual):
        """改进的适应度函数：对接近可行的路径给予部分奖励"""
        # 构建完整路径（包含起点和终点）
        path = np.vstack((self.start, individual, self.goal))

        # 计算基本指标
        path_length = self.calculate_path_length(path)
        smoothness = self.calculate_smoothness(path)
        collision_penalty = self.calculate_collision_penalty(path)
        goal_proximity = self.calculate_goal_proximity(path)

        # 对可行路径给予额外奖励
        is_valid = self.is_path_valid(path)
        validity_bonus = 100.0 if is_valid else 0.0

        # 适应度计算：综合考虑各项因素
        # 路径越短、越平滑、越接近目标、碰撞越少，适应度越高
        fitness = validity_bonus + \
                  (1.0 / (path_length + 1e-6)) * 20 + \
                  (1.0 / (smoothness + 1e-6)) * 5 + \
                  (1.0 / (goal_proximity + 1e-6)) * 10 - \
                  collision_penalty

        # 确保适应度不为负
        return max(fitness, 0.001)

    def create_individual(self):
        """改进的初始化方法：基于障碍物信息的启发式初始化"""
        individual = []

        # 检查是否有直接路径
        direct_path = [self.start, self.goal]
        if self.is_path_valid(direct_path):
            # 如果有直接路径，生成接近直线的路径
            for i in range(1, self.num_waypoints + 1):
                ratio = i / (self.num_waypoints + 1)
                point = self.start + ratio * (self.goal - self.start)
                individual.append(point)
            return np.array(individual)

        # 否则，生成避开已知障碍物的初始点
        for i in range(1, self.num_waypoints + 1):
            ratio = i / (self.num_waypoints + 1)
            base_point = self.start + ratio * (self.goal - self.start)

            # 检查该点是否在障碍物附近
            for b in self.buildings:
                if (base_point[0] > b.safe_x_min and base_point[0] < b.safe_x_max and
                        base_point[1] > b.safe_y_min and base_point[1] < b.safe_y_max and
                        base_point[2] > b.safe_z_min and base_point[2] < b.safe_z_max):
                    # 如果在障碍物附近，向上或向两侧偏移
                    if base_point[2] < b.z_max + self.safe_distance:
                        # 向上偏移以避开障碍物
                        base_point[2] = b.z_max + self.safe_distance + 2
                    else:
                        # 向侧面偏移
                        offset = 5 if i % 2 == 0 else -5
                        base_point[0] += offset
                        base_point[1] += offset if i % 3 == 0 else -offset

            # 确保在范围内
            base_point[0] = np.clip(base_point[0], self.x_range[0], self.x_range[1])
            base_point[1] = np.clip(base_point[1], self.y_range[0], self.y_range[1])
            base_point[2] = np.clip(base_point[2], self.z_range[0], self.z_range[1])

            individual.append(base_point)

        return np.array(individual)

    def create_population(self):
        """创建初始种群，确保多样性"""
        population = []
        # 创建基础个体
        for _ in range(int(self.population_size * 0.7)):
            population.append(self.create_individual())

        # 添加一些随机个体以增加多样性
        for _ in range(int(self.population_size * 0.3)):
            individual = []
            for _ in range(self.num_waypoints):
                point = np.array([
                    random.uniform(self.x_range[0], self.x_range[1]),
                    random.uniform(self.y_range[0], self.y_range[1]),
                    random.uniform(self.z_range[0], self.z_range[1])
                ])
                individual.append(point)
            population.append(np.array(individual))

        return population[:self.population_size]

    def select_parents(self, population, fitnesses):
        """锦标赛选择，提高优质个体被选中的概率"""
        # 锦标赛规模
        tournament_size = min(5, len(population))

        # 选择第一个父代
        candidates = random.sample(list(zip(population, fitnesses)), tournament_size)
        parent1 = max(candidates, key=lambda x: x[1])[0]

        # 选择第二个父代，确保与第一个不同
        while True:
            candidates = random.sample(list(zip(population, fitnesses)), tournament_size)
            parent2 = max(candidates, key=lambda x: x[1])[0]
            if not np.array_equal(parent1, parent2):
                break

        return parent1, parent2

    def crossover(self, parent1, parent2):
        """改进的交叉操作：混合均匀交叉和两点交叉"""
        if np.random.random() < 0.8:  # 80%的概率进行交叉
            # 50%概率使用两点交叉，50%使用均匀交叉
            if np.random.random() < 0.5:
                # 两点交叉
                point1 = random.randint(1, self.num_waypoints - 2)
                point2 = random.randint(point1 + 1, self.num_waypoints - 1)

                child = np.copy(parent1)
                child[point1:point2] = parent2[point1:point2]
                return child
            else:
                # 均匀交叉
                child = np.copy(parent1)
                for i in range(self.num_waypoints):
                    if np.random.random() < 0.5:
                        child[i] = parent2[i]
                return child
        else:
            # 不交叉，随机返回一个父代
            return random.choice([parent1, parent2])

    def mutate(self, individual, current_mutation_rate):
        """改进的变异操作：自适应步长和方向"""
        mutated = np.copy(individual)
        mutation_strength = 2.0  # 基础变异强度

        # 根据当前个体质量调整变异强度
        # 适应度低的个体变异强度更大
        individual_fitness = self.calculate_fitness(individual)
        if individual_fitness < self.best_fitness * 0.5:
            mutation_strength *= 1.5

        for i in range(self.num_waypoints):
            if np.random.random() < current_mutation_rate:
                # 智能变异方向：如果点太靠近障碍物，向远离方向变异
                dist = self.distance_to_obstacle(mutated[i])
                if dist < self.safe_distance:
                    # 计算远离障碍物的方向
                    best_dir = np.array([0.0, 0.0, 1.0])  # 默认向上
                    max_dist = 0

                    # 尝试几个方向，选择能远离障碍物的方向
                    for _ in range(5):
                        dir_candidate = np.random.normal(0, 1, 3)
                        dir_candidate = dir_candidate / (np.linalg.norm(dir_candidate) + 1e-6)
                        test_point = mutated[i] + dir_candidate * mutation_strength
                        test_dist = self.distance_to_obstacle(test_point)
                        if test_dist > max_dist:
                            max_dist = test_dist
                            best_dir = dir_candidate

                    mutated[i] += best_dir * mutation_strength * 1.2
                else:
                    # 随机变异
                    mutated[i] += np.random.normal(0, mutation_strength, 3)

                # 确保在范围内
                mutated[i, 0] = np.clip(mutated[i, 0], self.x_range[0], self.x_range[1])
                mutated[i, 1] = np.clip(mutated[i, 1], self.y_range[0], self.y_range[1])
                mutated[i, 2] = np.clip(mutated[i, 2], self.z_range[0], self.z_range[1])

        # 整体偏移变异，保持路径形状
        if np.random.random() < 0.15:
            shift = np.random.normal(0, mutation_strength * 0.5, 3)
            mutated += shift
            # 再次确保在范围内
            for i in range(self.num_waypoints):
                mutated[i, 0] = np.clip(mutated[i, 0], self.x_range[0], self.x_range[1])
                mutated[i, 1] = np.clip(mutated[i, 1], self.y_range[0], self.y_range[1])
                mutated[i, 2] = np.clip(mutated[i, 2], self.z_range[0], self.z_range[1])

        return mutated

    def evolve_population(self, population, generation):
        """改进的进化策略：动态调整参数"""
        fitnesses = [self.calculate_fitness(ind) for ind in population]

        # 记录全局最佳
        current_best_idx = np.argmax(fitnesses)
        current_best_fitness = fitnesses[current_best_idx]
        current_best_individual = population[current_best_idx]

        if current_best_fitness > self.best_fitness:
            self.best_fitness = current_best_fitness
            self.best_path = np.vstack((self.start, current_best_individual, self.goal))

        # 动态调整变异率：早期高探索，后期高利用
        if generation < self.generations * 0.3:
            current_mutation_rate = min(0.3, self.mutation_rate + 0.02)
        elif generation < self.generations * 0.7:
            current_mutation_rate = self.mutation_rate
        else:
            # 后期降低变异率，但如果找不到好解则提高
            if self.best_fitness < 50:  # 阈值表示还没有找到较好的解
                current_mutation_rate = min(0.25, self.mutation_rate + 0.05)
            else:
                current_mutation_rate = max(0.05, self.mutation_rate - 0.01)

        # 精英选择：保留10%的最佳个体
        elite_size = max(1, int(self.population_size * 0.1))
        elites = [population[i] for i in np.argsort(fitnesses)[-elite_size:]]

        # 创建新种群
        new_population = elites.copy()

        # 生成后代
        while len(new_population) < self.population_size:
            parent1, parent2 = self.select_parents(population, fitnesses)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child, current_mutation_rate)
            new_population.append(child)

        # 定期引入新的随机个体，防止早熟收敛
        if generation % 20 == 0 and generation > 0:
            for i in range(int(self.population_size * 0.05)):
                if len(new_population) < self.population_size:
                    new_population.append(self.create_individual())

        return new_population[:self.population_size], fitnesses

    def plan(self):
        """主规划函数"""
        # 检查起点和终点是否合法（仅检查边界，不检查高度）
        if self.is_collision(self.start, is_start_end=True):
            return None, "起点在障碍物或边界外"
        if self.is_collision(self.goal, is_start_end=True):
            return None, "终点在障碍物或边界外"

        # 检查起点到终点是否可以直接到达
        direct_path = [self.start, self.goal]
        if self.is_path_valid(direct_path):
            return [tuple(self.start), tuple(self.goal)], "找到直接路径"

        # 初始化种群
        population = self.create_population()
        self.best_fitness = 0
        self.best_path = None

        # 进化
        no_improve_generations = 0
        for generation in range(self.generations):
            # 进化一代
            population, fitnesses = self.evolve_population(population, generation)

            # 检查是否有改进
            current_best = max(fitnesses)
            if current_best > self.best_fitness * 1.05:  # 5%以上的改进
                no_improve_generations = 0
            else:
                no_improve_generations += 1

            # 如果长时间没有改进，引入新的随机个体
            if no_improve_generations > 50:
                # 替换20%的种群
                for i in range(int(self.population_size * 0.2)):
                    idx = random.randint(0, len(population) - 1)
                    population[idx] = self.create_individual()
                no_improve_generations = 0

        # 检查是否找到有效路径
        if self.best_path is not None:
            valid_path = [tuple(point) for point in self.best_path]
            if self.is_path_valid(self.best_path):
                return valid_path, f"找到有效路径，最佳适应度: {self.best_fitness:.2f}"
            else:
                # 如果最佳路径仍无效，尝试优化它
                optimized_path = self.optimize_invalid_path(self.best_path)
                if optimized_path and self.is_path_valid(optimized_path):
                    return [tuple(point) for point in optimized_path], "优化后找到有效路径"

        return None, f"未找到有效路径，最佳适应度: {self.best_fitness:.2f}"

    def optimize_invalid_path(self, path):
        """尝试优化接近有效的路径"""
        for _ in range(20):  # 尝试20次优化
            new_path = np.copy(path)
            # 只调整碰撞的点
            for i in range(1, len(path) - 1):  # 跳过起点和终点
                if self.is_collision(path[i]):
                    # 向远离障碍物的方向微调
                    dir_vec = np.random.normal(0, 1, 3)
                    dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-6)
                    new_path[i] = path[i] + dir_vec * 1.5

            if self.is_path_valid(new_path):
                return new_path
        return None


# 主窗口类
class DronePathVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("无人机三维路径规划 (优化遗传算法)")
        self.resize(1200, 700)

        self.is_planning = False
        self.path = None
        self.plan_time = 0.0
        self.min_flight_height = 0.5
        self.path_step_size = 0.5  # 路径插值步长
        self.current_step = 0  # 当前演示步骤
        self.demo_path = None  # 用于演示的插值路径

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

        # GA算法参数设置
        ga_group = QGroupBox("遗传算法参数")
        ga_layout = QFormLayout()
        ga_layout.setRowWrapPolicy(QFormLayout.DontWrapRows)
        ga_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.population_size = QLineEdit("100")
        self.population_size.setFixedWidth(80)
        self.generations = QLineEdit("500")
        self.generations.setFixedWidth(80)
        self.mutation_rate = QLineEdit("0.1")
        self.mutation_rate.setFixedWidth(80)
        self.waypoints = QLineEdit("10")
        self.waypoints.setFixedWidth(80)
        ga_layout.addRow("种群大小:", self.population_size)
        ga_layout.addRow("进化代数:", self.generations)
        ga_layout.addRow("变异率:", self.mutation_rate)
        ga_layout.addRow("路径点数:", self.waypoints)
        ga_group.setLayout(ga_layout)

        # 按钮布局
        button_layout = QVBoxLayout()
        self.plan_button = QPushButton("规划路径")
        self.plan_button.setMinimumHeight(30)
        self.plan_button.clicked.connect(self.plan_path)
        self.reset_button = QPushButton("重置视图")
        self.reset_button.setMinimumHeight(30)
        self.reset_button.clicked.connect(self.reset_view)

        # 添加演示按钮
        self.step_demo_button = QPushButton("单步演示")
        self.step_demo_button.setMinimumHeight(30)
        self.step_demo_button.clicked.connect(self.step_demo)
        self.step_demo_button.setEnabled(False)

        self.full_demo_button = QPushButton("完整演示")
        self.full_demo_button.setMinimumHeight(30)
        self.full_demo_button.clicked.connect(self.full_demo)
        self.full_demo_button.setEnabled(False)

        button_layout.addWidget(self.plan_button)
        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.step_demo_button)
        button_layout.addWidget(self.full_demo_button)
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
        control_layout.addWidget(ga_group)
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

    def update_safe_label(self):
        value = self.safe_distance.value() / 10.0
        self.safe_label.setText(f"{value:.1f}")

    def update_height_label(self):
        value = self.min_height_slider.value() / 10.0
        self.height_label.setText(f"{value:.1f}")
        self.min_flight_height = value
        self.init_visualization()

    def update_step_label(self):
        """更新步长显示"""
        value = self.step_size_slider.value()
        self.step_label.setText(f"{value:.1f}")
        self.path_step_size = value
        # 如果已有路径，重新绘制
        if self.path and len(self.path) > 1:
            self.show_final_path()

    def create_surface_arrays(self):
        """创建表面绘制所需的数组，确保类型正确"""
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

            # 设置坐标轴
            ax.set_xlabel('X轴', fontsize=8)
            ax.set_ylabel('Y轴', fontsize=8)
            ax.set_zlabel('Z轴', fontsize=8)
            ax.set_title('无人机三维路径规划 (优化遗传算法)', fontsize=10)
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
        """根据步长对路径进行插值，使显示更平滑"""
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

    def show_final_path(self, step=None):
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

            # 绘制路径
            if self.path and len(self.path) > 1:
                # 准备演示路径
                if self.demo_path is None:
                    self.demo_path = self.interpolate_path(self.path)

                # 确定要显示的路径部分
                display_path = self.demo_path
                if step is not None and 0 <= step < len(self.demo_path):
                    display_path = self.demo_path[:step + 1]
                    current_point = self.demo_path[step]
                    ax.scatter(current_point[0], current_point[1], current_point[2],
                               color='yellow', s=100, label='当前位置', marker='d')

                x = [p[0] for p in display_path]
                y = [p[1] for p in display_path]
                z = [p[2] for p in display_path]

                ax.plot(x, y, z, 'b-', linewidth=2, label='规划路径')
                # 只绘制原始路径点，不绘制插值点
                original_x = [p[0] for p in self.path]
                original_y = [p[1] for p in self.path]
                original_z = [p[2] for p in self.path]
                ax.scatter(original_x, original_y, original_z, color='blue', s=50, alpha=0.7, marker='o')

                # 显示路径信息（包含适应度）
                self.path_info.clear()
                total_length = 0.0
                info_text = "路径点坐标列表：\n\n"
                info_text += f"起点: ({start[0]:.1f}, {start[1]:.1f}, {start[2]:.1f})\n\n"

                for i, (px, py, pz) in enumerate(self.path):
                    info_text += f"步骤 {i}: ({px:.1f}, {py:.1f}, {pz:.1f})"
                    if i > 0 and i < len(self.path) - 1 and pz < self.min_flight_height - 0.1:
                        info_text += " (低于最低高度)\n"
                    else:
                        info_text += "\n"

                    if i > 0:
                        dx = px - self.path[i - 1][0]
                        dy = py - self.path[i - 1][1]
                        dz = pz - self.path[i - 1][2]
                        total_length += math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

                info_text += f"\n终点: ({goal[0]:.1f}, {goal[1]:.1f}, {goal[2]:.1f})\n"
                info_text += f"\n路径总长度: {total_length:.2f} 单位"
                info_text += f"\n规划时间: {self.plan_time:.4f} 秒"
                info_text += f"\n最低飞行高度: {self.min_flight_height:.1f} 单位"
                info_text += f"\n最佳适应度值: {self.best_fitness:.2f}"  # 显示适应度
                self.path_info.setText(info_text)

            # 设置坐标轴
            ax.set_xlabel('X轴', fontsize=8)
            ax.set_ylabel('Y轴', fontsize=8)
            ax.set_zlabel('Z轴', fontsize=8)
            ax.set_title('无人机三维路径规划 (优化遗传算法)', fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.set_xlim(0, 60)
            ax.set_ylim(0, 60)
            ax.set_zlim(0, 30)
            ax.legend(fontsize=8)

            self.canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "可视化错误", f"显示路径时出错: {str(e)}")
            print(f"调试信息: 错误类型={type(e)}, 错误内容={str(e)}")

    def plan_path(self):
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

            population_size = int(self.population_size.text())
            generations = int(self.generations.text())
            mutation_rate = float(self.mutation_rate.text())
            num_waypoints = int(self.waypoints.text())

            if mutation_rate < 0 or mutation_rate > 1:
                QMessageBox.critical(self, "参数错误", "变异率必须在0-1之间！")
                return

            self.plan_button.setEnabled(False)
            self.is_planning = True
            self.time_label.setText("规划中...")
            self.status_label.setText("状态: 正在规划路径")
            self.path_info.setText("正在规划路径，请稍候...")
            # 禁用演示按钮
            self.step_demo_button.setEnabled(False)
            self.full_demo_button.setEnabled(False)

            planner = GeneticAlgorithmPlanner(
                start,
                goal,
                self.buildings,
                safe_distance=self.safe_distance.value() / 10.0,
                min_height=self.min_flight_height,
                population_size=population_size,
                generations=generations,
                mutation_rate=mutation_rate,
                num_waypoints=num_waypoints
            )

            self.thread = PlanningThread(planner)
            self.thread.finished.connect(self.on_planning_finished)
            self.thread.error.connect(self.on_planning_error)
            self.thread.start()

            self.plan_start_time = time.time()

        except ValueError:
            QMessageBox.critical(self, "输入错误", "请输入有效的数值！")
            self.plan_button.setEnabled(True)
            self.is_planning = False
            self.status_label.setText("状态: 就绪")

    def on_planning_finished(self, path, message, best_fitness):
        """接收路径、消息和最佳适应度值"""
        self.plan_time = time.time() - self.plan_start_time
        self.time_label.setText(f"规划时间: {self.plan_time:.4f} 秒")
        self.status_label.setText(f"状态: {message}")
        self.path = path
        self.best_fitness = best_fitness  # 保存最佳适应度值
        self.current_step = 0  # 重置演示步骤
        self.demo_path = None  # 重置演示路径

        # 如果有有效路径，启用演示按钮
        if self.path and len(self.path) > 1:
            self.step_demo_button.setEnabled(True)
            self.full_demo_button.setEnabled(True)
            self.show_final_path(0)  # 显示第一步
        else:
            QMessageBox.warning(self, "规划结果", message)
            self.show_final_path()
            # 禁用演示按钮
            self.step_demo_button.setEnabled(False)
            self.full_demo_button.setEnabled(False)

        self.plan_button.setEnabled(True)
        self.is_planning = False

    def on_planning_error(self, error_msg):
        QMessageBox.critical(self, "规划错误", f"路径规划过程中发生错误: {error_msg}")
        self.time_label.setText("规划时间: 错误")
        self.status_label.setText("状态: 规划出错")
        self.plan_button.setEnabled(True)
        self.is_planning = False
        # 禁用演示按钮
        self.step_demo_button.setEnabled(False)
        self.full_demo_button.setEnabled(False)

    def reset_view(self):
        if self.is_planning:
            return

        self.path = None
        self.best_fitness = 0  # 重置适应度
        self.current_step = 0  # 重置演示步骤
        self.demo_path = None  # 重置演示路径
        self.time_label.setText("规划时间: -- 秒")
        self.status_label.setText("状态: 就绪")
        # 禁用演示按钮
        self.step_demo_button.setEnabled(False)
        self.full_demo_button.setEnabled(False)
        self.init_visualization()

    def step_demo(self):
        """单步演示：每次点击显示下一步"""
        if not self.demo_path or self.is_planning:
            return

        # 显示下一步
        self.current_step += 1
        if self.current_step >= len(self.demo_path):
            self.current_step = len(self.demo_path) - 1  # 保持在最后一步
            self.status_label.setText("状态: 演示完成")
        else:
            self.status_label.setText(f"状态: 演示中 (步骤 {self.current_step}/{len(self.demo_path) - 1})")

        self.show_final_path(self.current_step)

    def full_demo(self):
        """完整演示：自动播放整个路径"""
        if not self.demo_path or self.is_planning:
            return

        # 禁用按钮防止重复操作
        self.step_demo_button.setEnabled(False)
        self.full_demo_button.setEnabled(False)
        self.plan_button.setEnabled(False)

        # 重置到第一步
        self.current_step = 0
        self.show_final_path(self.current_step)
        self.status_label.setText(f"状态: 完整演示中 (步骤 {self.current_step}/{len(self.demo_path) - 1})")

        # 使用QApplication.processEvents()实现逐步显示
        for step in range(1, len(self.demo_path)):
            self.current_step = step
            self.show_final_path(step)
            self.status_label.setText(f"状态: 完整演示中 (步骤 {step}/{len(self.demo_path) - 1})")
            QApplication.processEvents()  # 刷新界面
            time.sleep(0.05)  # 控制演示速度

        self.status_label.setText("状态: 演示完成")
        # 重新启用按钮
        self.step_demo_button.setEnabled(True)
        self.full_demo_button.setEnabled(True)
        self.plan_button.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DronePathVisualizer()
    window.show()
    sys.exit(app.exec_())