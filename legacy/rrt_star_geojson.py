import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
from shapely.errors import TopologicalError

# ------------------------------
# 节点类定义
# ------------------------------
class Node:
    def __init__(self, position, parent=None):
        self.position = np.array(position, dtype=np.float64)  # [x, y]坐标
        self.parent = parent
        self.cost = 0.0  # 从起点到该节点的累计成本

    def update_cost(self):
        if self.parent:
            self.cost = self.parent.cost + np.linalg.norm(self.position - self.parent.position)


# ------------------------------
# RRT*算法类（读取GeoJSON数据）
# ------------------------------
class RRTStarFromGeoJSON:
    def __init__(self, geojson_path, step_size=50, goal_tolerance=100, search_radius=200):
        # 加载GeoJSON数据
        self.gdf = gpd.read_file(geojson_path)

        # 提取关键要素
        self.obstacles = self._extract_obstacles()
        self.start_point = self._extract_start_point()
        self.goal_polygon = self._extract_goal_polygon()

        # 算法参数
        self.step_size = step_size
        self.goal_tolerance = goal_tolerance
        self.search_radius = search_radius
        self.tree = [Node(self.start_point)]  # 初始化树

        # 计算环境边界
        self.bounds = self._calculate_bounds()

    def _extract_obstacles(self):
        """从GeoJSON提取障碍物（禁飞区、建筑物等）"""
        obstacle_types = ['high_rise', 'river', 'no_fly']
        obstacles = []
        for idx, row in self.gdf.iterrows():
            if row['type'] in obstacle_types:
                # 转换为Shapely多边形
                if row.geometry.geom_type == 'Polygon':
                    obstacles.append({
                        'geometry': row.geometry,
                        'weight': row['weight']
                    })
        return obstacles

    def _extract_start_point(self):
        """从GeoJSON提取起点坐标"""
        start = self.gdf[self.gdf['type'] == 'start']
        if len(start) == 0:
            raise ValueError("GeoJSON中未找到起点（type='start'）")
        return np.array(start.geometry.iloc[0].coords[0], dtype=np.float64)

    def _extract_goal_polygon(self):
        """从GeoJSON提取终点区域"""
        goal = self.gdf[self.gdf['type'] == 'goal']
        if len(goal) == 0:
            raise ValueError("GeoJSON中未找到终点（type='goal'）")
        return goal.geometry.iloc[0]

    def _calculate_bounds(self):
        """计算环境边界（用于随机点生成）"""
        minx, miny, maxx, maxy = self.gdf.total_bounds
        # 扩展边界10%，确保能搜索到边缘区域
        buffer_x = (maxx - minx) * 0.1
        buffer_y = (maxy - miny) * 0.1
        return [minx - buffer_x, miny - buffer_y, maxx + buffer_x, maxy + buffer_y]

    def get_random_point(self):
        """生成随机点（10%概率直接采样目标区域内的点）"""
        if np.random.random() < 0.1:
            # 在终点区域内随机采样
            minx, miny, maxx, maxy = self.goal_polygon.bounds
            return np.array([
                np.random.uniform(minx, maxx),
                np.random.uniform(miny, maxy)
            ])
        # 否则在全局范围内采样
        return np.array([
            np.random.uniform(self.bounds[0], self.bounds[2]),
            np.random.uniform(self.bounds[1], self.bounds[3])
        ])

    def find_nearest_node(self, random_point):
        """找到树中距离随机点最近的节点"""
        min_distance = float('inf')
        nearest_node = None
        for node in self.tree:
            distance = np.linalg.norm(node.position - random_point)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node
        return nearest_node

    def find_near_nodes(self, new_node):
        """找到新节点周围搜索半径内的所有节点"""
        near_nodes = []
        for node in self.tree:
            if np.linalg.norm(node.position - new_node.position) < self.search_radius:
                near_nodes.append(node)
        return near_nodes

    def steer(self, from_node, to_point):
        """从当前节点向目标点移动固定步长"""
        direction = to_point - from_node.position
        distance = np.linalg.norm(direction)

        if distance < self.step_size:
            new_node = Node(to_point, from_node)
        else:
            new_position = from_node.position + (direction / distance) * self.step_size
            new_node = Node(new_position, from_node)

        new_node.update_cost()
        return new_node

    def is_collision_free(self, point1, point2):
        """检查两点连线是否与障碍物碰撞"""
        # 创建线段
        line = LineString([tuple(point1), tuple(point2)])

        # 检查与所有障碍物的交集
        for obs in self.obstacles:
            try:
                if line.intersects(obs['geometry']):
                    return False
            except TopologicalError:
                # 处理几何拓扑错误
                return False
        return True

    def is_goal_reached(self, point):
        """检查点是否到达目标区域"""
        return self.goal_polygon.contains(Point(point))

    def choose_best_parent(self, new_node, near_nodes):
        """选择最优父节点"""
        if not near_nodes:
            return new_node

        min_cost = new_node.cost
        best_parent = new_node.parent

        for node in near_nodes:
            if self.is_collision_free(node.position, new_node.position):
                candidate_cost = node.cost + np.linalg.norm(new_node.position - node.position)
                if candidate_cost < min_cost:
                    min_cost = candidate_cost
                    best_parent = node

        new_node.parent = best_parent
        new_node.update_cost()
        return new_node

    def rewire(self, new_node, near_nodes):
        """重连附近节点"""
        for node in near_nodes:
            if (self.is_collision_free(new_node.position, node.position) and
                    new_node.cost + np.linalg.norm(node.position - new_node.position) < node.cost):
                node.parent = new_node
                node.update_cost()

    def plan_path(self, max_iterations=2000):
        """规划路径主函数"""
        for _ in range(max_iterations):
            random_point = self.get_random_point()
            nearest_node = self.find_nearest_node(random_point)
            new_node = self.steer(nearest_node, random_point)

            # 碰撞检测
            if not self.is_collision_free(nearest_node.position, new_node.position):
                continue

            # 优化父节点
            near_nodes = self.find_near_nodes(new_node)
            new_node = self.choose_best_parent(new_node, near_nodes)

            # 添加到树并优化
            self.tree.append(new_node)
            self.rewire(new_node, near_nodes)

            # 检查是否到达目标
            if self.is_goal_reached(new_node.position):
                return self.reconstruct_path(new_node)

        # 如果未到达目标，返回最近路径
        print(f"达到最大迭代次数({max_iterations})，返回当前最优路径")
        return self.find_closest_path_to_goal()

    def find_closest_path_to_goal(self):
        """找到离目标最近的节点并重建路径"""
        min_dist = float('inf')
        closest_node = None
        for node in self.tree:
            dist = self.goal_polygon.distance(Point(node.position))
            if dist < min_dist:
                min_dist = dist
                closest_node = node
        return self.reconstruct_path(closest_node)

    def reconstruct_path(self, node):
        """重建从起点到目标的路径"""
        path = []
        current_node = node
        while current_node is not None:
            path.append(current_node.position)
            current_node = current_node.parent
        return path[::-1]  # 反转路径，从起点到目标


# ------------------------------
# 可视化函数
# ------------------------------
def visualize_result(planner, path):
    """可视化路径规划结果"""
    fig, ax = plt.subplots(figsize=(12, 10))

    # 绘制障碍物
    for obs in planner.obstacles:
        if obs['geometry'].geom_type == 'Polygon':
            x, y = obs['geometry'].exterior.xy
            ax.fill(x, y, alpha=0.5, color='gray',
                    label='障碍物' if '障碍物' not in ax.get_legend_handles_labels()[1] else "")

    # 绘制起点和目标区域
    ax.scatter(planner.start_point[0], planner.start_point[1],
               c='green', s=100, marker='o', label='起点')

    goal_x, goal_y = planner.goal_polygon.exterior.xy
    ax.fill(goal_x, goal_y, alpha=0.3, color='red', label='终点区域')

    # 绘制树
    for node in planner.tree:
        if node.parent:
            ax.plot(
                [node.position[0], node.parent.position[0]],
                [node.position[1], node.parent.position[1]],
                'b-', alpha=0.1
            )

    # 绘制路径
    if path:
        path_array = np.array(path)
        ax.plot(path_array[:, 0], path_array[:, 1], 'r-', linewidth=2, label='规划路径')

    # 设置图表
    ax.set_aspect('equal')
    ax.set_xlabel('X坐标 (米)')
    ax.set_ylabel('Y坐标 (米)')
    ax.legend()
    ax.set_title('基于GeoJSON数据的RRT*路径规划')
    plt.grid(True, alpha=0.3)
    plt.show()


# ------------------------------
# 主程序
# ------------------------------
if __name__ == "__main__":
    # GeoJSON文件路径（请替换为实际文件路径）
    geojson_file = "ceshi.geo.json"

    try:
        # 创建规划器实例
        planner = RRTStarFromGeoJSON(
            geojson_path=geojson_file,
            step_size=50,  # 步长（米）
            goal_tolerance=100,  # 目标容差（米）
            search_radius=200  # 搜索半径（米）
        )

        # 规划路径
        path = planner.plan_path(max_iterations=2000)

        # 输出路径信息
        if path:
            total_length = sum(np.linalg.norm(path[i + 1] - path[i]) for i in range(len(path) - 1))
            print(f"路径规划完成，总长度: {total_length:.2f} 米")
            print(f"路径点数量: {len(path)}")

            # 可视化结果
            visualize_result(planner, path)
        else:
            print("路径规划失败")

    except Exception as e:
        print(f"执行出错: {str(e)}")

# # ------------------------------
# # 基础节点类（必须最先定义）
# # ------------------------------
# class Node:
#     """节点类，存储位置、父节点和路径成本"""
#
#     def __init__(self, position, parent=None):
#         self.position = np.array(position, dtype=np.float64)
#         self.parent = parent
#         self.cost = 0.0  # 从起点到该节点的累计距离成本
#
#     def update_cost(self):
#         """更新节点成本（父节点成本 + 到父节点的距离）"""
#         if self.parent:
#             self.cost = self.parent.cost + np.linalg.norm(self.position - self.parent.position)
#
#
# # ------------------------------
# # 二维RRT类（供巡航阶段使用）
# # ------------------------------
# class Simple2DRRT:
#     def __init__(self, start, goal, bounds, obstacles, step_size=0.8):
#         self.start = Node(start[:2])  # 只取X,Y坐标
#         self.goal = Node(goal[:2])
#         self.bounds = bounds[:2]  # 只考虑X,Y范围
#         self.obstacles = obstacles
#         self.step_size = step_size
#         self.tree = [self.start]
#
#     def is_collision_free(self, p1, p2):
#         """检查二维平面上两点连线是否碰撞"""
#         num_samples = 10
#         for i in range(num_samples + 1):
#             t = i / num_samples
#             x = p1[0] + t * (p2[0] - p1[0])
#             y = p1[1] + t * (p2[1] - p1[1])
#             for obs in self.obstacles:
#                 pos = np.array(obs["position"])
#                 size = np.array(obs["size"])
#                 if (pos[0] <= x <= pos[0] + size[0] and
#                         pos[1] <= y <= pos[1] + size[1]):
#                     return False
#         return True
#
#     def plan(self):
#         """规划二维路径"""
#         for _ in range(800):
#             # 生成随机点（20%概率直接采样目标点）
#             if np.random.random() < 0.2:
#                 rand_point = self.goal.position
#             else:
#                 rand_point = np.random.uniform(0, self.bounds)
#
#             # 找到最近节点
#             nearest = min(self.tree, key=lambda n: np.linalg.norm(n.position - rand_point))
#             dir_vec = rand_point - nearest.position
#             dist = np.linalg.norm(dir_vec)
#
#             # 计算新节点位置
#             if dist < self.step_size:
#                 new_pos = rand_point
#             else:
#                 new_pos = nearest.position + (dir_vec / dist) * self.step_size
#
#             new_node = Node(new_pos, nearest)
#             new_node.update_cost()
#
#             # 碰撞检测
#             if self.is_collision_free(nearest.position, new_pos):
#                 self.tree.append(new_node)
#                 # 检查是否到达目标
#                 if np.linalg.norm(new_pos - self.goal.position) < 0.8:
#                     self.goal.parent = new_node
#                     return self.reconstruct_path(self.goal)
#         return None
#
#     def reconstruct_path(self, node):
#         """重建路径"""
#         path = []
#         while node:
#             path.append(node.position)
#             node = node.parent
#         return path[::-1]
#
#
# # ------------------------------
# # 三维RRT类（供升空和降落阶段使用）
# # ------------------------------
# class Simple3DRRT:
#     def __init__(self, start, goal, space_bounds, obstacles, step_size=0.5):
#         self.start = Node(start)
#         self.goal = Node(goal)
#         self.bounds = space_bounds
#         self.obstacles = obstacles
#         self.step_size = step_size
#         self.tree = [self.start]
#
#     def is_collision_free(self, p1, p2):
#         """检查三维空间中两点连线是否碰撞"""
#         num_samples = 10
#         for i in range(num_samples + 1):
#             t = i / num_samples
#             x = p1[0] + t * (p2[0] - p1[0])
#             y = p1[1] + t * (p2[1] - p1[1])
#             z = p1[2] + t * (p2[2] - p1[2])
#             for obs in self.obstacles:
#                 pos = np.array(obs["position"])
#                 size = np.array(obs["size"])
#                 if (pos[0] <= x <= pos[0] + size[0] and
#                         pos[1] <= y <= pos[1] + size[1] and
#                         pos[2] <= z <= pos[2] + size[2]):
#                     return False
#         return True
#
#     def plan(self):
#         """规划三维路径"""
#         for _ in range(500):
#             # 生成随机点（20%概率直接采样目标点）
#             if np.random.random() < 0.2:
#                 rand_point = self.goal.position
#             else:
#                 rand_point = np.random.uniform(0, self.bounds)
#
#             # 找到最近节点
#             nearest = min(self.tree, key=lambda n: np.linalg.norm(n.position - rand_point))
#             dir_vec = rand_point - nearest.position
#             dist = np.linalg.norm(dir_vec)
#
#             # 计算新节点位置
#             if dist < self.step_size:
#                 new_pos = rand_point
#             else:
#                 new_pos = nearest.position + (dir_vec / dist) * self.step_size
#
#             new_node = Node(new_pos, nearest)
#             new_node.update_cost()
#
#             # 碰撞检测
#             if self.is_collision_free(nearest.position, new_pos):
#                 self.tree.append(new_node)
#                 # 检查是否到达目标
#                 if np.linalg.norm(new_pos - self.goal.position) < 0.5:
#                     self.goal.parent = new_node
#                     return self.goal
#         return None
#
#     def reconstruct_path(self, node):
#         """重建路径"""
#         path = []
#         while node:
#             path.append(node.position)
#             node = node.parent
#         return path[::-1]
#
#
# # ------------------------------
# # 分层路径规划主类（最后定义，确保依赖类已加载）
# # ------------------------------
# class LayeredRRTStar:
#     def __init__(self, start, goal, cruise_alt, obstacles,
#                  step_size=0.8, goal_tolerance=1.0):
#         self.start = np.array(start)  # 起点（地面）
#         self.goal = np.array(goal)  # 终点（地面）
#         self.cruise_alt = cruise_alt  # 巡航高度
#         self.obstacles = obstacles
#         self.step_size = step_size
#         self.goal_tolerance = goal_tolerance
#
#         # 中间节点
#         self.cruise_start = None  # (x, y, cruise_alt)
#         self.cruise_end = None  # (x, y, cruise_alt)
#
#         # 三段路径存储
#         self.climb_path = []  # 升空路径
#         self.cruise_path = []  # 巡航路径
#         self.descent_path = []  # 降落路径
#
#     def get_max_obstacle_height(self, area_bounds=None):
#         """计算指定区域内的最大障碍物高度"""
#         max_height = 0.0
#         for obs in self.obstacles:
#             if obs["type"] == "cuboid":
#                 height = obs["position"][2] + obs["size"][2]
#                 if height > max_height:
#                     if area_bounds is None:
#                         max_height = height
#                     else:
#                         min_x, min_y, max_x, max_y = area_bounds
#                         obs_x, obs_y = obs["position"][0], obs["position"][1]
#                         if (min_x <= obs_x <= max_x) and (min_y <= obs_y <= max_y):
#                             max_height = height
#         return max_height
#
#     def adjust_cruise_altitude(self):
#         """根据障碍物自动调整巡航高度"""
#         buffer = 2.0
#         min_x = min(self.start[0], self.goal[0]) - buffer
#         max_x = max(self.start[0], self.goal[0]) + buffer
#         min_y = min(self.start[1], self.goal[1]) - buffer
#         max_y = max(self.start[1], self.goal[1]) + buffer
#         area_bounds = (min_x, min_y, max_x, max_y)
#
#         max_obs_height = self.get_max_obstacle_height(area_bounds)
#         required_alt = max_obs_height + 1.5  # 安全余量
#         if required_alt > self.cruise_alt:
#             print(f"自动调整巡航高度: {self.cruise_alt}m → {required_alt}m")
#             self.cruise_alt = required_alt
#         return self.cruise_alt
#
#     def plan_climb_path(self):
#         """规划从起点垂直升空到巡航高度的路径"""
#         # 优先尝试垂直升空（最节能）
#         climb_nodes = [Node([self.start[0], self.start[1], z])
#                        for z in np.arange(self.start[2], self.cruise_alt + 0.5, 0.5)]
#
#         # 检查垂直路径是否碰撞
#         for i in range(1, len(climb_nodes)):
#             if not self.is_collision_free(
#                     climb_nodes[i - 1].position, climb_nodes[i].position
#             ):
#                 print("垂直升空路径有障碍，规划绕行升空路径...")
#                 # 使用三维RRT规划绕行路径
#                 climb_rrt = Simple3DRRT(
#                     start=self.start,
#                     goal=[self.start[0], self.start[1], self.cruise_alt],
#                     space_bounds=space_bounds,
#                     obstacles=self.obstacles
#                 )
#                 goal_node = climb_rrt.plan()
#                 if goal_node:
#                     path = climb_rrt.reconstruct_path(goal_node)
#                     self.cruise_start = path[-1]
#                     return path
#                 else:
#                     return None
#
#         self.cruise_start = climb_nodes[-1].position
#         return [node.position for node in climb_nodes]
#
#     def plan_cruise_path(self):
#         """在巡航高度进行二维路径规划"""
#         # 构造二维障碍物（仅考虑巡航高度处的障碍物）
#         obs_2d = []
#         for obs in self.obstacles:
#             pos = obs["position"]
#             size = obs["size"]
#             # 检查障碍物是否影响巡航高度
#             if (pos[2] <= self.cruise_alt <= pos[2] + size[2]):
#                 obs_2d.append({
#                     "type": "rect",
#                     "position": [pos[0], pos[1]],
#                     "size": [size[0], size[1]]
#                 })
#
#         # 使用二维RRT规划路径
#         planner_2d = Simple2DRRT(
#             start=self.cruise_start,
#             goal=[self.goal[0], self.goal[1], self.cruise_alt],
#             bounds=space_bounds,
#             obstacles=obs_2d
#         )
#         path_2d = planner_2d.plan()
#         if not path_2d:
#             raise ValueError("巡航路径规划失败")
#
#         # 将二维路径恢复为三维（添加巡航高度）
#         self.cruise_end = [path_2d[-1][0], path_2d[-1][1], self.cruise_alt]
#         return [[p[0], p[1], self.cruise_alt] for p in path_2d]
#
#     def plan_descent_path(self):
#         """规划从巡航高度垂直降落到终点的路径"""
#         # 优先尝试垂直降落
#         descent_nodes = [Node([self.cruise_end[0], self.cruise_end[1], z])
#                          for z in np.arange(self.cruise_alt, self.goal[2] - 0.5, -0.5)]
#         descent_nodes.append(Node(self.goal))  # 确保到达终点
#
#         # 检查垂直降落路径是否碰撞
#         for i in range(1, len(descent_nodes)):
#             if not self.is_collision_free(
#                     descent_nodes[i - 1].position, descent_nodes[i].position
#             ):
#                 print("垂直降落路径有障碍，规划绕行降落路径...")
#                 # 使用三维RRT规划绕行路径
#                 descent_rrt = Simple3DRRT(
#                     start=self.cruise_end,
#                     goal=self.goal,
#                     space_bounds=space_bounds,
#                     obstacles=self.obstacles
#                 )
#                 goal_node = descent_rrt.plan()
#                 if goal_node:
#                     return descent_rrt.reconstruct_path(goal_node)
#                 else:
#                     return None
#
#         return [node.position for node in descent_nodes]
#
#     def is_collision_free(self, point1, point2):
#         """检查两点连线是否碰撞（三维）"""
#         num_samples = 10
#         for i in range(num_samples + 1):
#             t = i / num_samples
#             x = point1[0] + t * (point2[0] - point1[0])
#             y = point1[1] + t * (point2[1] - point1[1])
#             z = point1[2] + t * (point2[2] - point1[2])
#             if self.is_in_obstacle([x, y, z]):
#                 return False
#         return True
#
#     def is_in_obstacle(self, point):
#         """检查点是否在障碍物内"""
#         for obs in self.obstacles:
#             if obs["type"] == "cuboid":
#                 pos = np.array(obs["position"])
#                 size = np.array(obs["size"])
#                 if (pos[0] <= point[0] <= pos[0] + size[0] and
#                         pos[1] <= point[1] <= pos[1] + size[1] and
#                         pos[2] <= point[2] <= pos[2] + size[2]):
#                     return True
#         return False
#
#     def plan_complete_path(self):
#         """规划完整路径：升空→巡航→降落"""
#         # 1. 调整巡航高度
#         self.adjust_cruise_altitude()
#
#         # 2. 规划升空路径
#         self.climb_path = self.plan_climb_path()
#         if not self.climb_path:
#             raise ValueError("升空路径规划失败")
#
#         # 3. 规划巡航路径
#         self.cruise_path = self.plan_cruise_path()
#         if not self.cruise_path:
#             raise ValueError("巡航路径规划失败")
#
#         # 4. 规划降落路径
#         self.descent_path = self.plan_descent_path()
#         if not self.descent_path:
#             raise ValueError("降落路径规划失败")
#
#         # 合并路径（去重相邻点）
#         full_path = self.climb_path[:-1] + self.cruise_path[:-1] + self.descent_path
#         return full_path
#
#
# # ------------------------------
# # 可视化函数
# # ------------------------------
# def plot_full_path(planner, full_path):
#     """可视化完整路径"""
#     fig = plt.figure(figsize=(12, 10))
#     ax = fig.add_subplot(111, projection='3d')
#
#     # 绘制障碍物
#     for obs in planner.obstacles:
#         pos = np.array(obs["position"])
#         size = np.array(obs["size"])
#         x, y, z = pos
#         dx, dy, dz = size
#         ax.bar3d(x, y, z, dx, dy, dz, color='gray', alpha=0.5)
#
#     # 标记巡航高度平面
#     x_range = np.linspace(0, space_bounds[0], 2)
#     y_range = np.linspace(0, space_bounds[1], 2)
#     X, Y = np.meshgrid(x_range, y_range)
#     Z = np.full_like(X, planner.cruise_alt)
#     ax.plot_surface(X, Y, Z, alpha=0.1, color='blue')
#     ax.text(0.5, 0.5, planner.cruise_alt + 0.5,
#             f'巡航高度: {planner.cruise_alt}m', color='blue')
#
#     # 绘制三段路径
#     climb = np.array(planner.climb_path)
#     cruise = np.array(planner.cruise_path)
#     descent = np.array(planner.descent_path)
#
#     ax.plot(climb[:, 0], climb[:, 1], climb[:, 2], 'g-', linewidth=2, label='升空路径')
#     ax.plot(cruise[:, 0], cruise[:, 1], cruise[:, 2], 'b-', linewidth=2, label='巡航路径')
#     ax.plot(descent[:, 0], descent[:, 1], descent[:, 2], 'orange', linewidth=2, label='降落路径')
#
#     # 标记起点和终点
#     ax.scatter(planner.start[0], planner.start[1], planner.start[2],
#                c='green', s=150, marker='o', label='起点')
#     ax.scatter(planner.goal[0], planner.goal[1], planner.goal[2],
#                c='red', s=150, marker='x', label='终点')
#
#     # 设置坐标轴
#     ax.set_xlabel('X 坐标')
#     ax.set_ylabel('Y 坐标')
#     ax.set_zlabel('Z 坐标')
#     ax.set_xlim(0, space_bounds[0])
#     ax.set_ylim(0, space_bounds[1])
#     ax.set_zlim(0, space_bounds[2])
#     ax.legend()
#     plt.title('无人机分层路径规划（升空-巡航-降落）')
#     plt.show()
#
#
# # ------------------------------
# # 主程序（放在最后执行）
# # ------------------------------
# if __name__ == "__main__":
#     # 环境参数
#     space_bounds = np.array([20, 20, 10])  # X, Y, Z范围
#     cruise_altitude = 6.0  # 初始巡航高度
#
#     # 障碍物定义
#     obstacles = [
#         # 地面障碍物
#         {"type": "cuboid", "position": [3, 3, 0], "size": [2, 2, 4]},
#         {"type": "cuboid", "position": [15, 15, 0], "size": [3, 3, 3]},
#
#         # 空中障碍物
#         {"type": "cuboid", "position": [8, 5, 0], "size": [2, 2, 5]},
#         {"type": "cuboid", "position": [10, 10, 0], "size": [2, 2, 7]},
#         {"type": "cuboid", "position": [5, 12, 4], "size": [2, 2, 3]},
#         {"type": "cuboid", "position": [12, 8, 0], "size": [1, 1, 5.5]},
#     ]
#
#     # 起点和终点（地面）
#     start_point = [1, 1, 0]
#     goal_point = [18, 18, 0]
#
#     # 创建规划器实例（此时LayeredRRTStar类已完全定义）
#     planner = LayeredRRTStar(
#         start=start_point,
#         goal=goal_point,
#         cruise_alt=cruise_altitude,
#         obstacles=obstacles
#     )
#
#     # 规划并显示路径
#     try:
#         full_path = planner.plan_complete_path()
#         total_length = np.sum(np.linalg.norm(np.diff(full_path, axis=0), axis=1))
#         print(f"路径规划完成，总长度: {total_length:.2f}m")
#         plot_full_path(planner, full_path)
#     except Exception as e:
#         print(f"路径规划失败: {e}")

# space_bounds = np.array([15, 15, 10])
#
# # 增加多种障碍物（立方体），模拟更复杂的环境
# obstacles = [
#     # 中心区域障碍物
#     {"type": "cuboid", "position": [4, 4, 0], "size": [2, 2, 5]},  # 低立方体
#     {"type": "cuboid", "position": [8, 5, 2], "size": [3, 2, 6]},  # 中等高度障碍物
#     {"type": "cuboid", "position": [5, 9, 0], "size": [2, 3, 4]},  # 宽障碍物
#
#     # 右侧区域障碍物
#     {"type": "cuboid", "position": [11, 4, 0], "size": [2, 2, 8]},  # 高障碍物
#     {"type": "cuboid", "position": [10, 8, 3], "size": [2, 2, 5]},  # 悬浮感障碍物
#
#     # 左侧区域障碍物
#     {"type": "cuboid", "position": [2, 7, 0], "size": [1, 3, 6]},  # 窄长障碍物
#
#     # 路径中间的障碍物群
#     {"type": "cuboid", "position": [6, 6, 0], "size": [1, 1, 3]},
#     {"type": "cuboid", "position": [7, 7, 4], "size": [1, 1, 3]},
#     {"type": "cuboid", "position": [8, 6, 7], "size": [1, 1, 3]}
# ]
#
#
# class Node:
#     """节点类，包含位置、父节点和路径成本"""
#
#     def __init__(self, position, parent=None):
#         self.position = np.array(position, dtype=np.float64)
#         self.parent = parent
#         self.cost = 0.0  # 从起点到该节点的累计距离成本
#
#     def update_cost(self):
#         """更新节点成本（父节点成本 + 到父节点的距离）"""
#         if self.parent:
#             self.cost = self.parent.cost + np.linalg.norm(self.position - self.parent.position)
#
#
# class RRTStar:
#     def __init__(self, start, goal, space_bounds, obstacles,
#                  step_size=0.6, goal_tolerance=0.8, search_radius=2.0):
#         self.start = Node(start)
#         self.goal = Node(goal)
#         self.space_bounds = space_bounds
#         self.obstacles = obstacles
#         self.step_size = step_size
#         self.goal_tolerance = goal_tolerance
#         self.search_radius = search_radius  # 重连搜索半径
#         self.tree = [self.start]  # 初始化树
#
#     def get_random_point(self):
#         """生成随机点，15%概率直接采样目标点加速搜索"""
#         if np.random.random() < 0.15:
#             return self.goal.position
#         return np.random.uniform(0, self.space_bounds)
#
#     def find_nearest_node(self, random_point):
#         """找到树中距离随机点最近的节点"""
#         min_distance = float('inf')
#         nearest_node = None
#         for node in self.tree:
#             distance = np.linalg.norm(node.position - random_point)
#             if distance < min_distance:
#                 min_distance = distance
#                 nearest_node = node
#         return nearest_node
#
#     def find_near_nodes(self, new_node):
#         """找到新节点周围搜索半径内的所有节点"""
#         near_nodes = []
#         for node in self.tree:
#             if np.linalg.norm(node.position - new_node.position) < self.search_radius:
#                 near_nodes.append(node)
#         return near_nodes
#
#     def steer(self, from_node, to_point):
#         """从当前节点向目标点移动固定步长"""
#         direction = to_point - from_node.position
#         distance = np.linalg.norm(direction)
#
#         if distance < self.step_size:
#             new_node = Node(to_point, from_node)
#         else:
#             new_position = from_node.position + (direction / distance) * self.step_size
#             new_node = Node(new_position, from_node)
#
#         new_node.update_cost()  # 计算新节点成本
#         return new_node
#
#     def is_collision_free(self, point1, point2):
#         """检查两点连线是否与障碍物碰撞（增加采样点提高检测精度）"""
#         num_samples = 15  # 更多采样点以适应复杂障碍物
#         for i in range(num_samples + 1):
#             t = i / num_samples
#             sample_point = point1 + t * (point2 - point1)
#             if self.is_in_obstacle(sample_point):
#                 return False
#         return True
#
#     def is_in_obstacle(self, point):
#         """检查点是否在障碍物内部"""
#         for obs in self.obstacles:
#             if obs["type"] == "cuboid":
#                 pos = np.array(obs["position"])
#                 size = np.array(obs["size"])
#                 if np.all(pos <= point) and np.all(point <= pos + size):
#                     return True
#         return False
#
#     def choose_best_parent(self, new_node, near_nodes):
#         """从附近节点中选择最优父节点（最小成本）"""
#         if not near_nodes:
#             return new_node
#
#         min_cost = new_node.cost
#         best_parent = new_node.parent
#
#         for node in near_nodes:
#             if self.is_collision_free(node.position, new_node.position):
#                 candidate_cost = node.cost + np.linalg.norm(new_node.position - node.position)
#                 if candidate_cost < min_cost:
#                     min_cost = candidate_cost
#                     best_parent = node
#
#         new_node.parent = best_parent
#         new_node.update_cost()
#         return new_node
#
#     def rewire(self, new_node, near_nodes):
#         """重连附近节点到新节点（如果新路径成本更低）"""
#         for node in near_nodes:
#             if (self.is_collision_free(new_node.position, node.position) and
#                     new_node.cost + np.linalg.norm(node.position - new_node.position) < node.cost):
#                 node.parent = new_node
#                 node.update_cost()
#
#     def extend_tree(self):
#         """扩展树并优化路径，适应更复杂的障碍物环境"""
#         max_iterations = 1500  # 增加迭代次数以应对复杂环境
#         for _ in range(max_iterations):
#             random_point = self.get_random_point()
#             nearest_node = self.find_nearest_node(random_point)
#             new_node = self.steer(nearest_node, random_point)
#
#             # 严格的碰撞检测
#             if (self.is_in_obstacle(new_node.position) or
#                     not self.is_collision_free(nearest_node.position, new_node.position)):
#                 continue
#
#             # 优化步骤
#             near_nodes = self.find_near_nodes(new_node)
#             new_node = self.choose_best_parent(new_node, near_nodes)
#             self.tree.append(new_node)
#             self.rewire(new_node, near_nodes)
#
#             # 检查是否到达目标
#             if np.linalg.norm(new_node.position - self.goal.position) < self.goal_tolerance:
#                 self.goal.parent = new_node
#                 self.goal.update_cost()
#                 return self.goal
#
#         print("达到最大迭代次数，返回当前最优路径")
#         return self.find_closest_to_goal()
#
#     def find_closest_to_goal(self):
#         """如果未到达目标，返回离目标最近的节点"""
#         min_dist = float('inf')
#         closest_node = None
#         for node in self.tree:
#             dist = np.linalg.norm(node.position - self.goal.position)
#             if dist < min_dist:
#                 min_dist = dist
#                 closest_node = node
#         self.goal.parent = closest_node
#         self.goal.update_cost()
#         return self.goal
#
#     def reconstruct_path(self, goal_node):
#         """重建从起点到目标的最优路径"""
#         path = []
#         current_node = goal_node
#         while current_node is not None:
#             path.append(current_node.position)
#             current_node = current_node.parent
#         return path[::-1]
#
#
# def plot_results(rrt_star, path):
#     """可视化复杂环境下的路径规划结果"""
#     fig = plt.figure(figsize=(12, 10))
#     ax = fig.add_subplot(111, projection='3d')
#
#     # 绘制所有障碍物
#     for obs in rrt_star.obstacles:
#         pos = np.array(obs["position"])
#         size = np.array(obs["size"])
#         x, y, z = pos
#         dx, dy, dz = size
#         # 不同高度的障碍物使用不同透明度，增强层次感
#         alpha = 0.6 if dz < 5 else 0.4
#         ax.bar3d(x, y, z, dx, dy, dz, color='gray', alpha=alpha)
#
#     # 绘制树结构
#     for node in rrt_star.tree:
#         if node.parent:
#             ax.plot(
#                 [node.position[0], node.parent.position[0]],
#                 [node.position[1], node.parent.position[1]],
#                 [node.position[2], node.parent.position[2]],
#                 'b-', alpha=0.15
#             )
#
#     # 绘制起点、目标和最优路径
#     ax.scatter(rrt_star.start.position[0], rrt_star.start.position[1],
#                rrt_star.start.position[2], c='green', s=120, label='起点')
#     ax.scatter(rrt_star.goal.position[0], rrt_star.goal.position[1],
#                rrt_star.goal.position[2], c='red', s=120, label='目标')
#
#     if path:
#         path_array = np.array(path)
#         ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2],
#                 'r-', linewidth=2.5, label='最优路径')
#
#     # 设置坐标轴和标题
#     ax.set_xlabel('X 坐标', fontsize=10)
#     ax.set_ylabel('Y 坐标', fontsize=10)
#     ax.set_zlabel('Z 坐标', fontsize=10)
#     ax.set_xlim(0, space_bounds[0])
#     ax.set_ylim(0, space_bounds[1])
#     ax.set_zlim(0, space_bounds[2])
#     ax.legend(fontsize=10)
#     plt.title('复杂障碍物环境下的三维RRT*路径规划', fontsize=12)
#     plt.tight_layout()
#     plt.show()
#
#
# # 主程序
# if __name__ == "__main__":
#     # 起点和目标点（设置在更复杂的环境中）
#     start = [1, 1, 1]
#     goal = [14, 14, 8]
#
#     # 创建RRT*实例（调整参数适应复杂环境）
#     rrt_star = RRTStar(
#         start=start,
#         goal=goal,
#         space_bounds=space_bounds,
#         obstacles=obstacles,
#         step_size=0.6,
#         goal_tolerance=1.0,
#         search_radius=2.0
#     )
#
#     # 扩展树并寻找最优路径
#     goal_node = rrt_star.extend_tree()
#     path = rrt_star.reconstruct_path(goal_node)
#
#     # 输出路径信息并可视化
#     print(f"路径总长度: {goal_node.cost:.2f} 单位距离")
#     print(f"路径点数量: {len(path)}")
#     if len(path) > 1:
#         print(f"路径片段示例:")
#         print(f"  从 {path[0]} 到 {path[1]}")
#         print(f"  从 {path[-2]} 到 {path[-1]}")
#     plot_results(rrt_star, path)
