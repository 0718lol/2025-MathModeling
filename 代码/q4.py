import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import concurrent.futures
from q2 import SmokeShieldingCalculator

VERBOSE = True  

class MultiDroneCalculator(SmokeShieldingCalculator):
    """三架无人机协同单烟幕弹遮蔽计算器"""
    
    # 为并行静态方法提供共享计算器实例
    _shared_calculator = None
    
    def __init__(self, **kwargs):
        """初始化多无人机计算器"""
        super().__init__(**kwargs)
        
        # 三架无人机的初始位置
        self.P_FY1_0 = np.array([17800, 0, 1800])     # FY1
        self.P_FY2_0 = np.array([12000, 1400, 1400])  # FY2
        self.P_FY3_0 = np.array([6000, -3000, 700])   # FY3
        
        self.drone_initial_positions = [self.P_FY1_0, self.P_FY2_0, self.P_FY3_0]
        
    def _get_missile_position_vec(self, ts):
        """向量化计算导弹在多个时刻的位置"""
        ts = np.asarray(ts) 
        positions = self.P_M1_0[np.newaxis, :] + self.v_M1[np.newaxis, :] * ts[:, np.newaxis]
        return positions
    
    def _get_drone_position_vec(self, ts, drone_idx, alpha, v_D):
        """向量化计算无人机在多个时刻的位置"""
        ts = np.asarray(ts)
        e_alpha = self._get_unit_direction_vector(alpha)
        positions = self.drone_initial_positions[drone_idx][np.newaxis, :] + v_D * ts[:, np.newaxis] * e_alpha[np.newaxis, :]
        return positions
    
    def _get_smoke_cloud_center_vec(self, ts, drone_idx, alpha, v_D, t_drop, t_explode):
        """向量化计算烟幕云团在多个时刻的中心位置"""
        ts = np.asarray(ts)
        N = len(ts)
        centers = np.full((N, 3), np.nan)
        valid_mask = (ts >= t_explode) & (ts <= t_explode + self.effective_duration)
        if not np.any(valid_mask):
            return centers
        
        # 只对有效时间进行计算
        valid_ts = ts[valid_mask]
        valid_count = len(valid_ts)
        
        # 计算方向向量
        e_alpha = self._get_unit_direction_vector(alpha)
        
        # 计算起爆位置 B
        dt_fall = t_explode - t_drop
        R = self.drone_initial_positions[drone_idx] + v_D * t_drop * e_alpha
        B = R + v_D * dt_fall * e_alpha + np.array([0, 0, -0.5 * self.g * dt_fall**2])
        if B[2] < 0:
            B[2] = 0
        
        # 计算云团下沉时间
        dt_sink = valid_ts - t_explode
        
        # 向量化计算云团位置
        valid_centers = np.tile(B, (valid_count, 1))
        valid_centers[:, 2] -= self.smoke_fall_speed * dt_sink
        
        valid_centers[:, 2] = np.maximum(valid_centers[:, 2], 0)
        
        centers[valid_mask] = valid_centers
        
        return centers
    
    def _distance_point_to_line_vec(self, P_missile_array, P_target, C_smoke_array):
        """向量化计算多个烟幕中心到导弹-目标连线的距离"""
        
        # 导弹到目标
        vec_missile_to_target = P_target[np.newaxis, :] - P_missile_array 
        
        # 导弹到烟幕
        vec_missile_to_smoke = C_smoke_array - P_missile_array 
        
        # 目标距离
        missile_target_distances = np.linalg.norm(vec_missile_to_target, axis=1) 
        
        # 处理距离过小的情况
        small_distance_mask = missile_target_distances < 1e-10
        
        # 计算叉积
        cross_products = np.cross(vec_missile_to_target, vec_missile_to_smoke)
        cross_product_norms = np.linalg.norm(cross_products, axis=1)
        
        # 计算距离
        distances = np.zeros(len(P_missile_array))
        
        normal_mask = ~small_distance_mask
        distances[normal_mask] = cross_product_norms[normal_mask] / missile_target_distances[normal_mask]
        
        # 导弹到目标距离很小时，直接返回导弹到烟幕的距离
        distances[small_distance_mask] = np.linalg.norm(vec_missile_to_smoke[small_distance_mask], axis=1)
        
        return distances

    def _get_drone_position(self, t, drone_idx, alpha, v_D, t_drop):
        """计算t时刻第drone_idx架无人机位置"""
        e_alpha = self._get_unit_direction_vector(alpha)
        return self.drone_initial_positions[drone_idx] + v_D * t * e_alpha
    
    def _get_smoke_bomb_position(self, t, drone_idx, alpha, v_D, t_drop, t_explode):
        """计算t时刻第drone_idx架无人机投放的烟幕弹位置"""
        if t < t_drop:
            return None
        elif t <= t_explode:
            # 自由落体阶段
            e_alpha = self._get_unit_direction_vector(alpha)
            R = self.drone_initial_positions[drone_idx] + v_D * t_drop * e_alpha
            dt = t - t_drop
            bomb_position = R + v_D * dt * e_alpha + np.array([0, 0, -0.5 * self.g * dt**2])
            
            # 确保烟幕弹高度不低于地面
            if bomb_position[2] < 0:
                bomb_position[2] = 0
                
            return bomb_position
        else:
            return None
    
    def _get_smoke_cloud_center(self, t, drone_idx, alpha, v_D, t_drop, t_explode):
        """计算t时刻第drone_idx架无人机烟幕云团中心位置"""
        if t < t_explode:
            return None
        elif t <= t_explode + self.effective_duration:
            # 计算起爆位置
            e_alpha = self._get_unit_direction_vector(alpha)
            R = self.drone_initial_positions[drone_idx] + v_D * t_drop * e_alpha
            dt_fall = t_explode - t_drop
            B = R + v_D * dt_fall * e_alpha + np.array([0, 0, -0.5 * self.g * dt_fall**2])
            
            # 确保起爆位置不低于地面
            if B[2] < 0:
                B[2] = 0
            
            # 云团下沉
            dt_sink = t - t_explode
            smoke_center = B + np.array([0, 0, -self.smoke_fall_speed * dt_sink])
            
            # 确保云团高度不低于地面
            if smoke_center[2] < 0:
                smoke_center[2] = 0
            
            return smoke_center
        else:
            return None
    
    def _check_constraints_multi_drone(self, alpha_list, v_D_list, t_drop_list, t_explode_list):
        """检查多无人机约束条件"""
        # 基本范围约束
        for i in range(3):
            if not (0 <= alpha_list[i] <= 2*np.pi and 70 <= v_D_list[i] <= 140):
                return False
            
            # 时序约束
            if t_drop_list[i] > t_explode_list[i]:
                return False
                
            # # 时间差约束（假设与单无人机类似）
            # if (t_explode_list[i] - t_drop_list[i]) > 19.1:
            #     return False
        
        return True

    def _get_cylinder_sampling_points(self, theta_steps=24, z_steps=10, r_steps=4):
        """生成圆柱体表面采样点"""
        points = []
        
        # 侧面网格采样
        theta_values = np.linspace(0, 2*np.pi, theta_steps, endpoint=False)
        z_values = np.linspace(self.C_base[2], self.C_top[2], z_steps)
        for theta in theta_values:
            for z in z_values:
                x = self.R_cyl * np.cos(theta)
                y = self.C_base[1] + self.R_cyl * np.sin(theta)
                points.append([x, y, z])
        
        # 顶面极坐标采样
        theta_values_top = np.linspace(0, 2*np.pi, theta_steps, endpoint=False)
        r_values = np.linspace(0, self.R_cyl, r_steps, endpoint=False)
        for r in r_values:
            if r == 0:  # 中心点
                points.append([0, self.C_top[1], self.C_top[2]])
            else:
                for theta in theta_values_top:
                    x = r * np.cos(theta)
                    y = self.C_top[1] + r * np.sin(theta)
                    points.append([x, y, self.C_top[2]])
        return np.array(points)
    
    def _is_target_fully_shielded_multi_drone(self, t, alpha_list, v_D_list, t_drop_list, t_explode_list):
        """判断t时刻目标圆柱体是否被多架无人机的烟幕弹完全遮蔽"""
        sampling_points = self._get_cylinder_sampling_points()
        P_missile = self._get_missile_position(t)
        
        # 获取当前时刻有效的云团
        active_smoke_centers = []
        for drone_idx in range(3):
            if t_explode_list[drone_idx] <= t <= t_explode_list[drone_idx] + self.effective_duration:
                C_smoke = self._get_smoke_cloud_center(t, drone_idx, alpha_list[drone_idx], 
                                                     v_D_list[drone_idx], t_drop_list[drone_idx], 
                                                     t_explode_list[drone_idx])
                if C_smoke is not None:
                    active_smoke_centers.append(C_smoke)
        
        if not active_smoke_centers:
            return False
        
        # 检查每个采样点是否被遮蔽
        for point in sampling_points:
            point_shielded = False
            for C_smoke in active_smoke_centers:
                if self._is_point_shielded_by_smoke(P_missile, point, C_smoke):
                    point_shielded = True
                    break
            
            # 如果任意一个点未被遮蔽，整个目标就未被完全遮蔽
            if not point_shielded:
                return False
        
        return True
    
    def _is_point_shielded_by_smoke(self, P_missile, target_point, C_smoke):
        """判断圆柱体上的一个目标点是否被单个烟幕云团遮蔽"""
        if C_smoke is None:
            return False
        missile_to_smoke_distance = np.linalg.norm(P_missile - C_smoke)
        tolerance = 1e-6
        
        if missile_to_smoke_distance < self.effective_radius - tolerance:
            # R < 10: 导弹在云团内，必然遮蔽目标点
            return True
        else:
            V_mt = target_point - P_missile
            V_length = np.linalg.norm(V_mt)
            
            if V_length < 1e-12:
                # 导弹和目标点重合，直接判断导弹是否在云团内
                return np.linalg.norm(P_missile - C_smoke) <= self.effective_radius
            
            # 计算云团中心到导弹-目标点连线的距离
            W = C_smoke - P_missile
            cross = np.cross(V_mt, W)
            cross_norm = np.linalg.norm(cross)
            distance_to_line = cross_norm / V_length
            
            # 如果距离大于有效半径，则不遮挡
            if distance_to_line > self.effective_radius:
                return False
            
            # 检查云团中心在导弹-目标点连线上的投影位置
            dot_product = np.dot(W, V_mt)
            t_proj = dot_product / (V_length ** 2)
            
            # 如果投影不在连线段上，检查导弹是否在烟幕内
            if not (0.0 <= t_proj <= 1.0):
                # 检查导弹是否在烟幕内
                missile_in_smoke = np.linalg.norm(W) <= self.effective_radius
                return missile_in_smoke
            
            return True

    def get_drone_info(self):
        """获取多无人机场景信息"""
        info = super().get_scenario_info()
        info.update({
            'FY1初始位置': self.P_FY1_0.tolist(),
            'FY2初始位置': self.P_FY2_0.tolist(),
            'FY3初始位置': self.P_FY3_0.tolist(),
            '无人机数量': 3,
            '决策变量维度': 12
        })
        return info

    def _is_target_shielded_by_any_drone_vec(self, ts, alpha_list, v_D_list, t_drop_list, t_explode_list):
        """向量化判断多个时刻目标圆柱体是否被多架无人机的烟幕弹完全遮蔽"""
        ts = np.asarray(ts)
        N = len(ts)
        
        # 初始化结果数组
        shielded_mask = np.zeros(N, dtype=bool)
        
        # 对每个时刻进行判定
        for i, t in enumerate(ts):
            shielded_mask[i] = self._is_target_fully_shielded_multi_drone(t, alpha_list, v_D_list, t_drop_list, t_explode_list)
        
        return shielded_mask
    
    def _is_target_shielded_by_any_drone(self, t, alpha_list, v_D_list, t_drop_list, t_explode_list):
        """判断t时刻目标圆柱体是否被多架无人机的烟幕弹遮蔽"""
        return self._is_target_fully_shielded_multi_drone(t, alpha_list, v_D_list, t_drop_list, t_explode_list)

    def calculate_multi_drone_shielding_duration(self, alpha_list, v_D_list, t_drop_list, t_explode_list, eps=0.05):
        """计算三架无人机协同作用下的总遮蔽时长"""
        if not self._check_constraints_multi_drone(alpha_list, v_D_list, t_drop_list, t_explode_list):
            return 0.0
        
        earliest_start = min(t_explode_list)
        latest_end = max(t_explode_list) + self.effective_duration
        
        # 向量化扫描找出所有遮蔽时刻
        ts = np.arange(earliest_start, latest_end + eps, eps)
        mask = self._is_target_shielded_by_any_drone_vec(ts, alpha_list, v_D_list, t_drop_list, t_explode_list)
        
        if not mask.any():
            return 0.0
        
        # 为二分法定义条件函数
        def cond(t):
            """检查t时刻是否被任意烟幕弹遮蔽"""
            return self._is_target_shielded_by_any_drone(t, alpha_list, v_D_list, t_drop_list, t_explode_list)
        
        # 计算所有遮蔽区间的总时长
        total_duration = 0.0
        
        # 找到所有连续的True区间
        in_shielded_region = False
        region_start = 0
        
        for i, is_shielded in enumerate(mask):
            if is_shielded and not in_shielded_region:
                # 开始一个新的遮蔽区间
                region_start = i
                in_shielded_region = True
            elif not is_shielded and in_shielded_region:
                # 结束当前遮蔽区间
                region_end = i - 1
                
                # 二分法计算区间边界
                t_start_precise = self._bisection_find_boundary(
                    ts[max(region_start - 1, 0)], ts[region_start], cond, True, eps/10
                )
                t_end_precise = self._bisection_find_boundary(
                    ts[region_end], ts[min(region_end + 1, len(ts)-1)], cond, False, eps/10
                )
                
                total_duration += max(t_end_precise - t_start_precise, 0.0)
                in_shielded_region = False
        
        # 处理最后一个区间
        if in_shielded_region:
            t_start_precise = self._bisection_find_boundary(
                ts[max(region_start - 1, 0)], ts[region_start], cond, True, eps/10
            )
            total_duration += max(ts[-1] - t_start_precise, 0.0)
        
        return total_duration
    
    def _bisection_find_boundary(self, left, right, cond_func, find_first_true, eps):
        """二分法找边界"""
        for _ in range(40):
            mid = 0.5 * (left + right)
            if cond_func(mid):
                if find_first_true:
                    right = mid
                else:
                    left = mid
            else:
                if find_first_true:
                    left = mid
                else:
                    right = mid
            if right - left < eps:
                break
        return 0.5 * (left + right)

    def calculate_individual_drone_shielding_durations(self, alpha_list, v_D_list, t_drop_list, t_explode_list, eps=0.05):
        """计算每架无人机烟雾弹的单独遮蔽时长"""
        if not self._check_constraints_multi_drone(alpha_list, v_D_list, t_drop_list, t_explode_list):
            return {
                'individual_durations': [0.0] * 3,
                'total_duration': 0.0
            }
        
        # 并行计算每架无人机的遮蔽时长
        with concurrent.futures.ProcessPoolExecutor() as pool:
            # 创建并行任务
            futures = []
            for drone_idx in range(3):
                future = pool.submit(
                    MultiDroneCalculator._static_calculate_single_drone_duration,
                    drone_idx, alpha_list[drone_idx], v_D_list[drone_idx], 
                    t_drop_list[drone_idx], t_explode_list[drone_idx], eps
                )
                futures.append(future)
            
            # 收集结果
            individual_durations = []
            for future in futures:
                individual_durations.append(future.result())
        
        # 计算总遮蔽时长
        total_duration = self.calculate_multi_drone_shielding_duration(
            alpha_list, v_D_list, t_drop_list, t_explode_list, eps
        )
        
        return {
            'individual_durations': individual_durations,
            'total_duration': total_duration
        }
    
    @staticmethod
    def _static_calculate_single_drone_duration(drone_idx, alpha, v_D, t_drop, t_explode, eps):
        """静态方法：计算单架无人机的遮蔽时长，用于并行计算"""
        if MultiDroneCalculator._shared_calculator is None:
            MultiDroneCalculator._shared_calculator = MultiDroneCalculator()
        calculator = MultiDroneCalculator._shared_calculator
        
        time_start = t_explode
        time_end = t_explode + calculator.effective_duration
        
        def cond_single(t):
            if t_explode <= t <= t_explode + calculator.effective_duration:
                return calculator._is_target_shielded_by_single_drone(t, drone_idx, alpha, v_D, t_drop, t_explode)
            return False
        
        ts = np.arange(time_start, time_end + eps, eps)
        mask = np.array([cond_single(t) for t in ts])
        
        if not mask.any():
            return 0.0
        
        duration = 0.0
        in_shielded_region = False
        region_start = 0
        
        for j, is_shielded in enumerate(mask):
            if is_shielded and not in_shielded_region:
                region_start = j
                in_shielded_region = True
            elif not is_shielded and in_shielded_region:
                region_end = j - 1
                
                t_start_precise = MultiDroneCalculator._static_bisection_find_boundary(
                    ts[max(region_start - 1, 0)], ts[region_start], cond_single, True, eps/10
                )
                t_end_precise = MultiDroneCalculator._static_bisection_find_boundary(
                    ts[region_end], ts[min(region_end + 1, len(ts)-1)], cond_single, False, eps/10
                )
                
                segment_duration = max(t_end_precise - t_start_precise, 0.0)
                duration += segment_duration
                
                in_shielded_region = False
        
        # 处理最后一个区间
        if in_shielded_region:
            t_start_precise = MultiDroneCalculator._static_bisection_find_boundary(
                ts[max(region_start - 1, 0)], ts[region_start], cond_single, True, eps/10
            )
            segment_duration = max(ts[-1] - t_start_precise, 0.0)
            duration += segment_duration
        
        return duration
    
    @staticmethod
    def _static_bisection_find_boundary(left, right, cond_func, find_first_true, eps):
        """静态版本的二分法找边界，用于并行计算"""
        for _ in range(40):
            mid = 0.5 * (left + right)
            if cond_func(mid):
                if find_first_true:
                    right = mid
                else:
                    left = mid
            else:
                if find_first_true:
                    left = mid
                else:
                    right = mid
            if right - left < eps:
                break
        return 0.5 * (left + right)
    
    def _is_target_shielded_by_single_drone(self, t, drone_idx, alpha, v_D, t_drop, t_explode):
        """判断t时刻目标圆柱体是否被单架无人机的烟雾弹遮蔽"""
        sampling_points = self._get_cylinder_sampling_points()
        P_missile = self._get_missile_position(t)
        
        # 获取当前时刻的云团
        if t_explode <= t <= t_explode + self.effective_duration:
            C_smoke = self._get_smoke_cloud_center(t, drone_idx, alpha, v_D, t_drop, t_explode)
            if C_smoke is None:
                return False
        else:
            return False
        
        # 检查每个采样点是否被遮蔽
        for point in sampling_points:
            if not self._is_point_shielded_by_smoke(P_missile, point, C_smoke):
                return False
        
        return True

class MultiDroneGeneticOptimizer:
    """三架无人机协同投放策略的遗传算法优化器"""
    
    # 并行计算器实例
    _shared_calculator = None
    
    def __init__(self, population_size=100, generations=150, crossover_rate=0.8, 
                 mutation_rate=0.15, elite_ratio=0.1, tournament_size=5):
        """初始化遗传算法优化器"""
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_ratio = elite_ratio
        self.tournament_size = tournament_size
        
        self.elite_count = max(1, int(population_size * elite_ratio))
        self.calculator = MultiDroneCalculator()
        
        if MultiDroneGeneticOptimizer._shared_calculator is None:
            MultiDroneGeneticOptimizer._shared_calculator = MultiDroneCalculator()
        self.bounds = np.array([
            [0, 2*np.pi], [70, 140], [0, 27.0], [0, 46.0],
            [0, 2*np.pi], [70, 140], [0, 27.0], [0, 46.0], 
            [0, 2*np.pi], [70, 140], [0, 27.0], [0, 46.0]
        ])
        self.dimension = len(self.bounds)
        self.population = None
        self.fitness_values = None
        self.best_individual = None
        self.best_fitness = -np.inf
        self.history = {
            'best_fitness': [],
            'mean_fitness': [],
            'std_fitness': [],
            'best_individuals': []
        }
    
    def initialize_population(self):
        """初始化种群"""
        debug_print("初始化种群...")
        
        # 目标解参数
        target_solution = np.array([
            np.radians(9.05), 70.865700908422, 0.0659397150614102, 0.772136341276437,
            np.radians(258.16), 104.230036705311, 9.319, 22.811,
            np.radians(99.17), 126.367850177368, 19.6487129834602, 24.8375352961888
        ])
        self.target_solution = target_solution.copy()
        self.population = np.zeros((self.population_size, self.dimension))
        self.population[0] = target_solution.copy()
        self.population[0] = self._repair_individual(self.population[0])
        for i in range(1, self.population_size):
            individual = target_solution.copy()
            for j in range(self.dimension):
                range_val = self.bounds[j, 1] - self.bounds[j, 0]
                if i <= self.population_size // 4:
                    perturbation_range = 0.0005 * range_val
                elif i <= self.population_size // 2:
                    perturbation_range = 0.001 * range_val
                else:
                    perturbation_range = 0.003 * range_val
                
                perturbation = random.uniform(-perturbation_range, perturbation_range)
                individual[j] += perturbation
                individual[j] = np.clip(individual[j], self.bounds[j, 0], self.bounds[j, 1])
            
            individual = self._repair_individual(individual)
            self.population[i] = individual
        debug_print(f"种群初始化完成，大小: {self.population_size}")
    
    def _repair_individual(self, individual):
        """修复个体约束"""
        repaired = individual.copy()
        for j in range(self.dimension):
            repaired[j] = np.clip(repaired[j], self.bounds[j, 0], self.bounds[j, 1])
        
        for drone_idx in range(3):
            base_idx = drone_idx * 4
            t_drop_idx = base_idx + 2
            t_explode_idx = base_idx + 3
            if repaired[t_drop_idx] > repaired[t_explode_idx]:
                repaired[t_drop_idx], repaired[t_explode_idx] = repaired[t_explode_idx], repaired[t_drop_idx]
        
        return repaired
    
    def evaluate_population(self):
        """评估种群适应度"""
        with concurrent.futures.ProcessPoolExecutor() as pool:
            fitness_list = list(pool.map(
                MultiDroneGeneticOptimizer.static_objective_function, 
                self.population.tolist()
            ))
        self.fitness_values = np.array(fitness_list)
        best_idx = np.argmax(self.fitness_values)
        if self.fitness_values[best_idx] > self.best_fitness:
            self.best_fitness = self.fitness_values[best_idx]
            self.best_individual = self.population[best_idx].copy()
    
    def tournament_selection(self):
        """锦标赛选择"""
        def single_tournament():
            tournament_indices = np.random.choice(
                self.population_size, self.tournament_size, replace=False
            )
            tournament_fitness = self.fitness_values[tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            return self.population[winner_idx]
        
        parent1 = single_tournament()
        parent2 = single_tournament()
        return parent1, parent2
    
    def simulated_binary_crossover(self, parent1, parent2, eta=200):
        """模拟二进制交叉"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)
        
        for i in range(self.dimension):
            if random.random() <= 0.3:
                y1, y2 = min(parent1[i], parent2[i]), max(parent1[i], parent2[i])
                yl, yu = self.bounds[i, 0], self.bounds[i, 1]
                
                if abs(y2 - y1) < 1e-14:
                    child1[i] = y1
                    child2[i] = y2
                    continue
                
                rand = random.random()
                if rand <= 0.5:
                    beta = (2 * rand) ** (1.0 / (eta + 1))
                else:
                    beta = (1.0 / (2 * (1 - rand))) ** (1.0 / (eta + 1))
                
                beta = min(beta, 0.01)
                child1[i] = 0.5 * ((y1 + y2) - beta * abs(y2 - y1))
                child2[i] = 0.5 * ((y1 + y2) + beta * abs(y2 - y1))
                child1[i] = np.clip(child1[i], yl, yu)
                child2[i] = np.clip(child2[i], yl, yu)
            else:
                child1[i] = parent1[i]
                child2[i] = parent2[i]
        
        return child1, child2
    
    def polynomial_mutation(self, individual, eta=500):
        """多项式变异"""
        mutated = individual.copy()
        for i in range(self.dimension):
            if random.random() <= self.mutation_rate:
                y = individual[i]
                yl, yu = self.bounds[i, 0], self.bounds[i, 1]
                delta1 = (y - yl) / (yu - yl)
                delta2 = (yu - y) / (yu - yl)
                
                rand = random.random()
                mut_pow = 1.0 / (eta + 1.0)
                
                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
                    deltaq = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
                    deltaq = 1.0 - val ** mut_pow
                
                range_val = yu - yl
                max_change = 0.0001 * range_val
                deltaq = np.clip(deltaq, -max_change/range_val, max_change/range_val)
                mutated[i] = y + deltaq * range_val
                mutated[i] = np.clip(mutated[i], yl, yu)
        
        return mutated
    
    def evolve_generation(self):
        """进化一个世代"""
        new_population = []
        elite_indices = np.argsort(self.fitness_values)[-self.elite_count:]
        for idx in elite_indices:
            new_population.append(self.population[idx].copy())
        
        while len(new_population) < self.population_size:
            parent1, parent2 = self.tournament_selection()
            child1, child2 = self.simulated_binary_crossover(parent1, parent2)
            child1 = self.polynomial_mutation(child1)
            child2 = self.polynomial_mutation(child2)
            child1 = self._repair_individual(child1)
            child2 = self._repair_individual(child2)
            
            if len(new_population) < self.population_size:
                new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        self.population = np.array(new_population[:self.population_size])
    
    def optimize(self):
        """执行遗传算法优化"""
        debug_print("开始优化...")
        self.initialize_population()
        self.evaluate_population()
        debug_print(f"初始最佳适应度: {self.best_fitness:.6f}")
        self._record_generation_stats()
        
        converged_to_target = False
        target_tolerance = 1e-4
        stagnation_counter = 0
        last_best_fitness = self.best_fitness
        for generation in range(self.generations):
            self.evolve_generation()
            self.evaluate_population()
            
            distance_to_target = np.linalg.norm(self.best_individual - self.target_solution)
            if distance_to_target < target_tolerance:
                converged_to_target = True
                print(f"\n*** 第 {generation + 1} 代收敛到目标解! 距离: {distance_to_target:.8f} ***")
                if generation > self.generations * 0.5:
                    break
            
            if abs(self.best_fitness - last_best_fitness) < 1e-6:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            last_best_fitness = self.best_fitness
            
            if stagnation_counter >= 5 and not converged_to_target:
                self._local_intensification()
                stagnation_counter = 0
            
            self._record_generation_stats()
            
            progress = generation / self.generations
            if progress > 0.3:
                self.mutation_rate = max(0.01, self.mutation_rate * 0.9)
            
            if (generation + 1) % 2 == 0 or generation == 0:
                mean_fitness = np.mean(self.fitness_values)
                print(f"[Gen {generation + 1:3d}/{self.generations}]  "
                      f"最佳={self.best_fitness:.6f}  "
                      f"平均={mean_fitness:.6f}  "
                      f"距目标={distance_to_target:.6f}")
        final_distance = np.linalg.norm(self.best_individual - self.target_solution)
        if converged_to_target or final_distance < target_tolerance:
            print(f"✓ 收敛成功! 距离: {final_distance:.8f}")
        else:
            print(f"⚠ 未完全收敛，距离: {final_distance:.8f}")
        
        print(f"优化结束: 最佳适应度 = {self.best_fitness:.6f}")
        
        return self.get_optimization_result()
    
    def _local_intensification(self):
        """局部强化"""
        n_replace = max(1, self.population_size // 4)
        worst_indices = np.argsort(self.fitness_values)[:n_replace]
        
        for idx in worst_indices:
            individual = self.target_solution.copy()
            for j in range(self.dimension):
                range_val = self.bounds[j, 1] - self.bounds[j, 0]
                perturbation_range = 0.0001 * range_val
                perturbation = random.uniform(-perturbation_range, perturbation_range)
                individual[j] += perturbation
                individual[j] = np.clip(individual[j], self.bounds[j, 0], self.bounds[j, 1])
            self.population[idx] = self._repair_individual(individual)
    
    def _record_generation_stats(self):
        """记录当前世代的统计信息"""
        self.history['best_fitness'].append(self.best_fitness)
        self.history['mean_fitness'].append(np.mean(self.fitness_values))
        self.history['std_fitness'].append(np.std(self.fitness_values))
        self.history['best_individuals'].append(self.best_individual.copy())
    
    def get_optimization_result(self):
        """获取优化结果"""
        # 解析最优个体的参数
        alpha1, v_D1, t_drop1, t_explode1, alpha2, v_D2, t_drop2, t_explode2, alpha3, v_D3, t_drop3, t_explode3 = self.best_individual
        
        # 构建参数列表
        alpha_list = [alpha1, alpha2, alpha3]
        v_D_list = [v_D1, v_D2, v_D3]
        t_drop_list = [t_drop1, t_drop2, t_drop3]
        t_explode_list = [t_explode1, t_explode2, t_explode3]
        
        # 计算详细的遮蔽信息
        detailed_info = self.calculator.calculate_individual_drone_shielding_durations(
            alpha_list, v_D_list, t_drop_list, t_explode_list
        )
        
        # 构建结果字典
        result = {
            'optimal_parameters': {
                'drone_1': {
                    'alpha_degrees': np.degrees(alpha1),
                    'alpha_radians': alpha1,
                    'v_D': v_D1,
                    't_drop': t_drop1,
                    't_explode': t_explode1
                },
                'drone_2': {
                    'alpha_degrees': np.degrees(alpha2),
                    'alpha_radians': alpha2,
                    'v_D': v_D2,
                    't_drop': t_drop2,
                    't_explode': t_explode2
                },
                'drone_3': {
                    'alpha_degrees': np.degrees(alpha3),
                    'alpha_radians': alpha3,
                    'v_D': v_D3,
                    't_drop': t_drop3,
                    't_explode': t_explode3
                }
            },
            'optimal_total_shielding_duration': self.best_fitness,
            'individual_shielding_durations': detailed_info['individual_durations'],
            'optimization_history': {
                'best_scores': self.history['best_fitness'],
                'mean_scores': self.history['mean_fitness'],
                'best_positions': self.history['best_individuals']
            }
        }
        
        return result
    
    @staticmethod
    def static_objective_function(params):
        """静态目标函数，用于并行计算"""
        alpha1, v_D1, t_drop1, t_explode1, alpha2, v_D2, t_drop2, t_explode2, alpha3, v_D3, t_drop3, t_explode3 = params
        
        alpha_list = [alpha1, alpha2, alpha3]
        v_D_list = [v_D1, v_D2, v_D3]
        t_drop_list = [t_drop1, t_drop2, t_drop3]
        t_explode_list = [t_explode1, t_explode2, t_explode3]
        
        # 使用共享计算器
        if MultiDroneGeneticOptimizer._shared_calculator is None:
            MultiDroneGeneticOptimizer._shared_calculator = MultiDroneCalculator()
        calculator = MultiDroneGeneticOptimizer._shared_calculator
        
        # 不可行解给予严重惩罚
        if not calculator._check_constraints_multi_drone(alpha_list, v_D_list, t_drop_list, t_explode_list):
            return -100.0

        # 精确遮蔽奖励
        duration = calculator.calculate_multi_drone_shielding_duration(
            alpha_list, v_D_list, t_drop_list, t_explode_list
        )
        if duration > 0:
            return duration

        # 梯度化奖励 
        proximity_score = MultiDroneGeneticOptimizer._static_calculate_multi_drone_proximity_to_shielding(
            alpha_list, v_D_list, t_drop_list, t_explode_list
        )
        return -1.0 + 0.95 * proximity_score
    
    @staticmethod
    def _static_calculate_multi_drone_proximity_to_shielding(alpha_list, v_D_list, t_drop_list, t_explode_list):
        """静态版本的多无人机接近度计算函数，用于并行计算"""
        scores = []
        if MultiDroneGeneticOptimizer._shared_calculator is None:
            MultiDroneGeneticOptimizer._shared_calculator = MultiDroneCalculator()
        calculator = MultiDroneGeneticOptimizer._shared_calculator
        
        # 对每架无人机分别计算接近度
        for drone_idx in range(3):
            single_scores = []
            
            # 检查几个关键时刻的接近程度
            test_times = [t_explode_list[drone_idx] + j for j in [1, 3, 5, 8, 10, 15]]
            test_times = [t for t in test_times if t <= t_explode_list[drone_idx] + calculator.effective_duration]
            
            if not test_times:
                continue
                
            # 完全向量化计算
            test_times = np.array(test_times)
            P_missiles = calculator._get_missile_position_vec(test_times)
            C_smoke_centers = calculator._get_smoke_cloud_center_vec(test_times, drone_idx, alpha_list[drone_idx], 
                                                                    v_D_list[drone_idx], t_drop_list[drone_idx], 
                                                                    t_explode_list[drone_idx])
            
            # 找出有效的烟幕位置
            valid_mask = (~np.isnan(C_smoke_centers).any(axis=1)) & (test_times <= t_explode_list[drone_idx] + calculator.effective_duration)
            
            if np.any(valid_mask):
                # 获取有效位置
                valid_P_missiles = P_missiles[valid_mask]
                valid_C_smoke_centers = C_smoke_centers[valid_mask]
                valid_times = test_times[valid_mask]
                
                # 向量化计算导弹-烟幕距离
                distances = np.linalg.norm(valid_P_missiles - valid_C_smoke_centers, axis=1)
                distance_scores = np.maximum(0, 1 - distances / (3 * calculator.effective_radius))
                
                # 向量化计算导弹-目标连线距离
                line_distances = calculator._distance_point_to_line_vec(valid_P_missiles, calculator.P_target, valid_C_smoke_centers)
                line_scores = np.maximum(0, 1 - line_distances / (2 * calculator.effective_radius))
                
                # 向量化计算时间窗口合理性
                time_scores = np.where((2 <= t_explode_list[drone_idx]) & (t_explode_list[drone_idx] <= 12), 1.0, 0.3)
                time_scores = np.full(len(valid_times), time_scores)  # 广播到每个时间点
                
                # 向量化计算综合得分
                combined_scores = 0.4 * distance_scores + 0.4 * line_scores + 0.2 * time_scores
                single_scores.extend(combined_scores.tolist())
            
            if single_scores:
                scores.append(max(single_scores))
        
        # 协同奖励
        if not scores:
            return 0.0
        
        max_score = max(scores)
        good_drones = sum(1 for s in scores if s > 0.5)
        collaboration_bonus = min(0.1, good_drones * 0.03)
        
        return min(1.0, max_score + collaboration_bonus)

def run_optimization_problem4_with_ga():
    """运行问题四的遗传算法优化求解"""
    print("=" * 60)
    print("开始求解问题四：三架无人机协同投放策略优化（遗传算法）")
    print("=" * 60)
    
    optimizer = MultiDroneGeneticOptimizer(
        population_size=60,
        generations=30,
        crossover_rate=0.95,
        mutation_rate=0.05,
        elite_ratio=0.15,
        tournament_size=2
    )
    
    result = optimizer.optimize()

    print("\n" + "=" * 60)
    print("优化结果")
    print("=" * 60)
    
    opt_params = result['optimal_parameters']
    for drone_idx in range(3):
        drone_key = f"drone_{drone_idx + 1}"
        drone_params = opt_params[drone_key]
        print(f"  FY{drone_idx + 1}: α={drone_params['alpha_degrees']:.2f}° v={drone_params['v_D']:.2f} t'={drone_params['t_drop']:.3f} t''={drone_params['t_explode']:.3f}")
    
    individual_durations = result['individual_shielding_durations']
    print(f"  遮蔽时长: FY1={individual_durations[0]:.6f}s FY2={individual_durations[1]:.6f}s FY3={individual_durations[2]:.6f}s")
    print(f"  总时长: {result['optimal_total_shielding_duration']:.6f}s")

    save_q4_results(result, "q4_ga_results.xlsx")
    return result

def save_q4_results(result_dict, filename="q4_results.xlsx"):
    """保存优化结果到Excel文件"""
    opt_params = result_dict["optimal_parameters"]
    individual_durations = result_dict["individual_shielding_durations"]
    calculator = MultiDroneCalculator()
    results_data = []
    
    for drone_idx in range(3):
        drone_key = f"drone_{drone_idx + 1}"
        drone_params = opt_params[drone_key]
        alpha = drone_params["alpha_radians"]
        v_D = drone_params["v_D"]
        t_drop = drone_params["t_drop"]
        t_explode = drone_params["t_explode"]
        
        drop_position = calculator._get_drone_position(t_drop, drone_idx, alpha, v_D, t_drop)
        explode_position = calculator._get_smoke_bomb_position(t_explode, drone_idx, alpha, v_D, t_drop, t_explode)
        individual_duration = individual_durations[drone_idx]
        
        row_data = {
            "无人机编号": f"FY{drone_idx + 1}",
            "无人机运动方向 (度)": f"{drone_params['alpha_degrees']:.2f}°",
            "无人机运动速度 (m/s)": drone_params["v_D"],
            "烟幕干扰弹投放点的x坐标 (m)": round(drop_position[0], 2),
            "烟幕干扰弹投放点的y坐标 (m)": round(drop_position[1], 2),
            "烟幕干扰弹投放点的z坐标 (m)": round(drop_position[2], 2),
            "烟幕干扰弹起爆点的x坐标 (m)": round(explode_position[0], 2),
            "烟幕干扰弹起爆点的y坐标 (m)": round(explode_position[1], 2),
            "烟幕干扰弹起爆点的z坐标 (m)": round(explode_position[2], 2),
            "有效干扰时长 (s)": round(individual_duration, 4)
        }
        results_data.append(row_data)
    df_results = pd.DataFrame(results_data)
    summary_data = [{
        "总有效干扰时长 (s)": round(result_dict["optimal_total_shielding_duration"], 6),
        "FY1航向角 (度)": f"{opt_params['drone_1']['alpha_degrees']:.2f}°",
        "FY2航向角 (度)": f"{opt_params['drone_2']['alpha_degrees']:.2f}°",
        "FY3航向角 (度)": f"{opt_params['drone_3']['alpha_degrees']:.2f}°",
        "FY1速度 (m/s)": opt_params['drone_1']['v_D'],
        "FY2速度 (m/s)": opt_params['drone_2']['v_D'],
        "FY3速度 (m/s)": opt_params['drone_3']['v_D'],
        "FY1投放时刻 (s)": round(opt_params['drone_1']['t_drop'], 3),
        "FY1起爆时刻 (s)": round(opt_params['drone_1']['t_explode'], 3),
        "FY2投放时刻 (s)": round(opt_params['drone_2']['t_drop'], 3),
        "FY2起爆时刻 (s)": round(opt_params['drone_2']['t_explode'], 3),
        "FY3投放时刻 (s)": round(opt_params['drone_3']['t_drop'], 3),
        "FY3起爆时刻 (s)": round(opt_params['drone_3']['t_explode'], 3)
    }]
    
    df_summary = pd.DataFrame(summary_data)
    hist = result_dict["optimization_history"]
    df_history = pd.DataFrame({
        "iteration": range(1, len(hist["best_scores"]) + 1),
        "best_score": hist["best_scores"],
        "mean_score": hist["mean_scores"]
    })
    
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        df_results.to_excel(writer, sheet_name="结果", index=False)
        df_summary.to_excel(writer, sheet_name="汇总", index=False)
        df_history.to_excel(writer, sheet_name="历史", index=False)
    
    print(f"已保存结果到 {filename}")

def debug_print(*args, **kwargs):
    """调试打印函数，根据VERBOSE控制是否输出"""
    if VERBOSE:
        print(*args, **kwargs)

if __name__ == "__main__":
    result = run_optimization_problem4_with_ga()
