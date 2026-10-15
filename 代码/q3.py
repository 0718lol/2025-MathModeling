import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import concurrent.futures
from q2 import SmokeShieldingCalculator

VERBOSE = True

class MultiSmokeBombCalculator(SmokeShieldingCalculator):
    """三枚烟幕弹协同遮蔽计算器"""
    
    def __init__(self, **kwargs):
        """初始化多烟幕弹计算器"""
        super().__init__(**kwargs)

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
            if r == 0:
                points.append([0, self.C_top[1], self.C_top[2]])
            else:
                for theta in theta_values_top:
                    x = r * np.cos(theta)
                    y = self.C_top[1] + r * np.sin(theta)
                    points.append([x, y, self.C_top[2]])
        return np.array(points)

    def _get_missile_position_vec(self, ts):
        """向量化计算导弹在多个时刻的位置"""
        ts = np.asarray(ts) 
        positions = self.P_M1_0[np.newaxis, :] + self.v_M1[np.newaxis, :] * ts[:, np.newaxis]
        return positions
    
    def _get_smoke_cloud_center_vec(self, ts, alpha, v_D, t_drop, t_explode):
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
        R = self.P_FY1_0 + v_D * t_drop * e_alpha
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
    
    def _is_target_shielded_by_any_bomb_vec(self, ts, alpha, v_D, t_drop_list, t_explode_list):
        """判断多个时刻目标圆柱体是否被多个烟幕弹完全遮蔽"""
        ts = np.asarray(ts)
        N = len(ts)
        
        # 初始化结果数组
        shielded_mask = np.zeros(N, dtype=bool)
        
        # 对每个时刻进行判定
        for i, t in enumerate(ts):
            shielded_mask[i] = self._is_target_fully_shielded_multi_bomb(t, alpha, v_D, t_drop_list, t_explode_list)
        
        return shielded_mask
    
    def calculate_multi_shielding_duration(self, alpha, v_D, t_drop_list, t_explode_list, eps=0.05):
        """计算三枚烟幕弹协同作用下的总遮蔽时长"""
        if not self._check_constraints(alpha, v_D, t_drop_list, t_explode_list):
            return 0.0
        
        earliest_start = min(t_explode_list)
        latest_end = max(t_explode_list) + self.effective_duration
        
        # 向量化扫描找出所有遮蔽时刻
        ts = np.arange(earliest_start, latest_end + eps, eps)
        mask = self._is_target_shielded_by_any_bomb_vec(ts, alpha, v_D, t_drop_list, t_explode_list)
        
        if not mask.any():
            return 0.0
        
        # 为二分法定义条件函数
        def cond(t):
            """检查t时刻是否被任意烟幕弹遮蔽"""
            return self._is_target_shielded_by_any_bomb(t, alpha, v_D, t_drop_list, t_explode_list)
        
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
    
    def _check_constraints(self, alpha, v_D, t_drop_list, t_explode_list):
        """检查约束条件"""
        # 基本范围约束
        if not (0 <= alpha <= 2*np.pi and 70 <= v_D <= 140):
            return False
        
        # 时序约束
        for i in range(3):
            if t_drop_list[i] > t_explode_list[i]:
                return False
        
        # 时间间隔约束
        if t_drop_list[1] - t_drop_list[0] < 1.0:
            return False
        if t_drop_list[2] - t_drop_list[1] < 1.0:
            return False
        
        return True
    
    def _is_target_shielded_by_any_bomb(self, t, alpha, v_D, t_drop_list, t_explode_list):
        """判断t时刻目标圆柱体是否被多个烟幕弹完全遮蔽"""
        return self._is_target_fully_shielded_multi_bomb(t, alpha, v_D, t_drop_list, t_explode_list)

    def _is_target_fully_shielded_multi_bomb(self, t, alpha, v_D, t_drop_list, t_explode_list):
        """判断t时刻目标圆柱体是否被多个烟幕弹完全遮蔽"""
        sampling_points = self._get_cylinder_sampling_points()
        P_missile = self._get_missile_position(t)
        
        # 获取当前时刻有效的云团
        active_smoke_centers = []
        for i in range(len(t_drop_list)):  
            if t_explode_list[i] <= t <= t_explode_list[i] + self.effective_duration:
                C_smoke = self._get_smoke_cloud_center(t, alpha, v_D, t_drop_list[i], t_explode_list[i])
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

    def calculate_single_shielding_durations(self, alpha, v_D, t_drop_list, t_explode_list, 
                                     individual_bombs=None, eps=0.05):
        """单个烟幕弹遮蔽时长计算函数"""
        if not self._check_constraints(alpha, v_D, t_drop_list, t_explode_list):
            return {
                'individual_durations': [0.0] * len(t_drop_list),
                'total_duration': 0.0
            }
        
        if individual_bombs is None:
            individual_bombs = list(range(len(t_drop_list)))
        
        individual_durations = []
        detailed_time_segments = []
        
        for i in individual_bombs:
            time_start = t_explode_list[i]
            time_end = t_explode_list[i] + self.effective_duration
            
            def cond_single(t, bomb_idx=i):
                if t_explode_list[bomb_idx] <= t <= t_explode_list[bomb_idx] + self.effective_duration:
                    single_bomb_t_drop = [t_drop_list[bomb_idx]]
                    single_bomb_t_explode = [t_explode_list[bomb_idx]]
                    return self._is_target_shielded_by_any_bomb(t, alpha, v_D, single_bomb_t_drop, single_bomb_t_explode)
                return False
            
            ts = np.arange(time_start, time_end + eps, eps)
            single_bomb_t_drop = [t_drop_list[i]]
            single_bomb_t_explode = [t_explode_list[i]]
            mask = self._is_target_shielded_by_any_bomb_vec(ts, alpha, v_D, single_bomb_t_drop, single_bomb_t_explode)
            
            bomb_segments = []
            
            if not mask.any():
                duration_i = 0.0
            else:
                duration_i = 0.0
                
                in_shielded_region = False
                region_start = 0
                
                for j, is_shielded in enumerate(mask):
                    if is_shielded and not in_shielded_region:
                        region_start = j
                        in_shielded_region = True
                    elif not is_shielded and in_shielded_region:
                        region_end = j - 1
                        
                        t_start_precise = self._bisection_find_boundary(
                            ts[max(region_start - 1, 0)], ts[region_start], cond_single, True, eps/10
                        )
                        t_end_precise = self._bisection_find_boundary(
                            ts[region_end], ts[min(region_end + 1, len(ts)-1)], cond_single, False, eps/10
                        )
                        
                        segment_duration = max(t_end_precise - t_start_precise, 0.0)
                        duration_i += segment_duration
                        
                        bomb_segments.append({
                            'start_time': round(t_start_precise, 4),
                            'end_time': round(t_end_precise, 4),
                            'duration': round(segment_duration, 4)
                        })
                        
                        in_shielded_region = False
                
                # 处理最后一个区间
                if in_shielded_region:
                    t_start_precise = self._bisection_find_boundary(
                        ts[max(region_start - 1, 0)], ts[region_start], cond_single, True, eps/10
                    )
                    segment_duration = max(ts[-1] - t_start_precise, 0.0)
                    duration_i += segment_duration
                    
                    bomb_segments.append({
                        'start_time': round(t_start_precise, 4),
                        'end_time': round(ts[-1], 4),
                        'duration': round(segment_duration, 4)
                    })
            
            individual_durations.append(duration_i)
            detailed_time_segments.append(bomb_segments)
        
        # 计算总遮蔽时长
        total_duration = self.calculate_multi_shielding_duration(
            alpha, v_D, t_drop_list, t_explode_list, eps
        )
        
        return {
            'individual_durations': individual_durations,
            'total_duration': total_duration,
            'detailed_time_segments': detailed_time_segments
        }

class MultiSmokeBombOptimizer:
    """三枚烟幕弹协同投放策略的粒子群优化算法"""

    _shared_calculator = None
    
    def __init__(self, n_particles=50, n_iterations=100, w=0.9, c1=2.0, c2=2.0):
        """初始化粒子群优化器"""
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        self.calculator = MultiSmokeBombCalculator()

        if MultiSmokeBombOptimizer._shared_calculator is None:
            MultiSmokeBombOptimizer._shared_calculator = MultiSmokeBombCalculator()
        
        # 定义8维决策变量的边界
        self.bounds = np.array([
            [0, 2*np.pi],    # alpha
            [70, 140],       # v_D
            [0, 25.0],       # t'₁
            [0, 46.0],       # t''₁
            [1.0, 26.0],     # t'₂
            [1.0, 46.0],     # t''₂
            [2.0, 27.0],     # t'₃
            [2.0, 46.0]      # t''₃
        ])
        
        self.dimension = len(self.bounds)
        
        # 粒子群状态
        self.particles = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = -np.inf
        
        # 历史记录
        self.history = {
            'best_scores': [],
            'mean_scores': [],
            'best_positions': []
        }
    
    def gradient_objective_function(self, params):
        """梯度化适应度函数：建立分层奖励机制"""
        alpha, v_D, t_drop1, t_explode1, t_drop2, t_explode2, t_drop3, t_explode3 = params
        
        t_drop_list = [t_drop1, t_drop2, t_drop3]
        t_explode_list = [t_explode1, t_explode2, t_explode3]
        
        # 不可行解给予严重惩罚
        if not self.calculator._check_constraints(alpha, v_D, t_drop_list, t_explode_list):
            return -100.0
        for i in range(3):
            if (t_explode_list[i] - t_drop_list[i]) > 19.1:
                return -100.0
        if not (0 <= alpha <= 2*np.pi and 70 <= v_D <= 140):
            return -100.0

        # 精确遮蔽奖励
        duration = self.calculator.calculate_multi_shielding_duration(
            alpha, v_D, t_drop_list, t_explode_list
        )
        if duration > 0:
            return duration

        # 梯度化奖励
        proximity_score = self.calculate_multi_proximity_to_shielding(alpha, v_D, t_drop_list, t_explode_list)
        return -1.0 + 0.95 * proximity_score

    def objective_function(self, params):
        """目标函数：使用梯度化适应度函数"""
        return self.gradient_objective_function(params)
    
    def calculate_multi_proximity_to_shielding(self, alpha, v_D, t_drop_list, t_explode_list):
        """计算三枚烟幕弹接近遮蔽的程度，返回[0,1]"""
        scores = []
        
        # 对每枚烟幕弹分别计算接近度，然后取最大值
        for i in range(3):
            single_scores = []
            
            # 检查几个关键时刻的接近程度
            test_times = [t_explode_list[i] + j for j in [1, 3, 5, 8, 10, 15]]
            test_times = [t for t in test_times if t <= t_explode_list[i] + self.calculator.effective_duration]
            
            if not test_times:
                continue
                
            # 完全向量化计算
            test_times = np.array(test_times)
            P_missiles = self.calculator._get_missile_position_vec(test_times)  
            C_smoke_centers = self.calculator._get_smoke_cloud_center_vec(test_times, alpha, v_D, t_drop_list[i], t_explode_list[i])
            
            # 找出有效的烟幕位置
            valid_mask = (~np.isnan(C_smoke_centers).any(axis=1)) & (test_times <= t_explode_list[i] + self.calculator.effective_duration)
            
            if np.any(valid_mask):
                # 获取有效位置
                valid_P_missiles = P_missiles[valid_mask]
                valid_C_smoke_centers = C_smoke_centers[valid_mask]
                valid_times = test_times[valid_mask]
                
                # 向量化计算导弹-烟幕距离
                distances = np.linalg.norm(valid_P_missiles - valid_C_smoke_centers, axis=1)
                distance_scores = np.maximum(0, 1 - distances / (3 * self.calculator.effective_radius))
                
                # 向量化计算导弹-目标连线距离
                line_distances = self.calculator._distance_point_to_line_vec(valid_P_missiles, self.calculator.P_target, valid_C_smoke_centers)
                line_scores = np.maximum(0, 1 - line_distances / (2 * self.calculator.effective_radius))
                
                # 向量化计算时间窗口合理性
                time_scores = np.where((2 <= t_explode_list[i]) & (t_explode_list[i] <= 12), 1.0, 0.3)
                time_scores = np.full(len(valid_times), time_scores)  # 广播到每个时间点
                
                # 向量化计算综合得分
                combined_scores = 0.4 * distance_scores + 0.4 * line_scores + 0.2 * time_scores
                single_scores.extend(combined_scores.tolist())
            
            if single_scores:
                scores.append(max(single_scores))
        
        # 协同效应：取所有烟幕弹中的最高分，并给予小幅协同奖励
        if not scores:
            return 0.0
        
        max_score = max(scores)
        # 多个烟幕弹都有不错的分数，给予协同奖励
        good_bombs = sum(1 for s in scores if s > 0.5)
        collaboration_bonus = min(0.1, good_bombs * 0.03)
        
        return min(1.0, max_score + collaboration_bonus)

    @staticmethod
    def static_objective_function(params):
        """静态目标函数：用于并行计算"""
        alpha, v_D, t_drop1, t_explode1, t_drop2, t_explode2, t_drop3, t_explode3 = params
        
        t_drop_list = [t_drop1, t_drop2, t_drop3]
        t_explode_list = [t_explode1, t_explode2, t_explode3]
        
        # 使用共享计算器
        if MultiSmokeBombOptimizer._shared_calculator is None:
            MultiSmokeBombOptimizer._shared_calculator = MultiSmokeBombCalculator()
        calculator = MultiSmokeBombOptimizer._shared_calculator
        
        # 不可行解给予严重惩罚
        if not calculator._check_constraints(alpha, v_D, t_drop_list, t_explode_list):
            return -100.0
        for i in range(3):
            if (t_explode_list[i] - t_drop_list[i]) > 19.1:
                return -100.0
        if not (0 <= alpha <= 2*np.pi and 70 <= v_D <= 140):
            return -100.0

        # 精确遮蔽奖励
        duration = calculator.calculate_multi_shielding_duration(
            alpha, v_D, t_drop_list, t_explode_list
        )
        if duration > 0:
            return duration

        # 梯度化奖励 
        proximity_score = MultiSmokeBombOptimizer._static_calculate_multi_proximity_to_shielding(
            alpha, v_D, t_drop_list, t_explode_list
        )
        return -1.0 + 0.95 * proximity_score
    
    @staticmethod
    def _static_calculate_multi_proximity_to_shielding(alpha, v_D, t_drop_list, t_explode_list):
        """静态版本的多烟幕弹接近度计算函数，用于并行计算"""
        scores = []
        if MultiSmokeBombOptimizer._shared_calculator is None:
            MultiSmokeBombOptimizer._shared_calculator = MultiSmokeBombCalculator()
        calculator = MultiSmokeBombOptimizer._shared_calculator
        
        # 对每枚烟幕弹分别计算接近度
        for i in range(3):
            single_scores = []
            
            # 检查几个关键时刻的接近程度
            test_times = [t_explode_list[i] + j for j in [1, 3, 5, 8, 10, 15]]
            test_times = [t for t in test_times if t <= t_explode_list[i] + calculator.effective_duration]
            
            if not test_times:
                continue
                
            # 完全向量化计算
            test_times = np.array(test_times)
            # 使用共享计算器的向量化函数
            P_missiles = calculator._get_missile_position_vec(test_times)
            C_smoke_centers = calculator._get_smoke_cloud_center_vec(test_times, alpha, v_D, t_drop_list[i], t_explode_list[i])
            
            # 找出有效的烟幕位置
            valid_mask = (~np.isnan(C_smoke_centers).any(axis=1)) & (test_times <= t_explode_list[i] + calculator.effective_duration)
            
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
                time_scores = np.where((2 <= t_explode_list[i]) & (t_explode_list[i] <= 12), 1.0, 0.3)
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
        good_bombs = sum(1 for s in scores if s > 0.5)
        collaboration_bonus = min(0.1, good_bombs * 0.03)
        
        return min(1.0, max_score + collaboration_bonus)
    
    def _repair_particle(self, particle):
        """约束修复"""
        alpha, v_D, t_drop1, t_explode1, t_drop2, t_explode2, t_drop3, t_explode3 = particle
        
        # 修复时间间隔约束
        if t_drop2 - t_drop1 < 1.0:
            t_drop2 = t_drop1 + 1.0
        if t_drop3 - t_drop2 < 1.0:
            t_drop3 = t_drop2 + 1.0
            
        # 修复时序约束
        if t_drop1 > t_explode1:
            t_explode1, t_drop1 = t_drop1, t_explode1
        if t_drop2 > t_explode2:
            t_explode2, t_drop2 = t_drop2, t_explode2
        if t_drop3 > t_explode3:
            t_explode3, t_drop3 = t_drop3, t_explode3
            
        # 修复时间差约束
        for i, (td, te) in enumerate([(t_drop1, t_explode1), (t_drop2, t_explode2), (t_drop3, t_explode3)]):
            if (te - td) > 19.2:
                if i == 0:
                    t_explode1 = t_drop1 + 19.2
                elif i == 1:
                    t_explode2 = t_drop2 + 19.2
                else:
                    t_explode3 = t_drop3 + 19.2
        
        # 修复边界约束
        repaired = np.array([alpha, v_D, t_drop1, t_explode1, t_drop2, t_explode2, t_drop3, t_explode3])
        for j in range(self.dimension):
            repaired[j] = np.clip(repaired[j], self.bounds[j, 0], self.bounds[j, 1])
            
        # 重新检查修复后是否仍满足约束
        _, _, td1, te1, td2, te2, td3, te3 = repaired
        
        # 如果边界截断破坏了约束，进行最终调整
        if td2 - td1 < 1.0:
            td2 = min(td1 + 1.0, self.bounds[4, 1])
        if td3 - td2 < 1.0:
            td3 = min(td2 + 1.0, self.bounds[6, 1])
            
        repaired[2:] = [td1, te1, td2, te2, td3, te3]
        
        return repaired
    
    def initialize_particles(self):
        """初始化粒子群"""
        debug_print("初始化粒子群...")
        
        self.particles = np.zeros((self.n_particles, self.dimension))
        self.velocities = np.zeros((self.n_particles, self.dimension))
        self.personal_best_positions = np.zeros((self.n_particles, self.dimension))
        self.personal_best_scores = np.full(self.n_particles, -np.inf)
        
        # 随机初始化粒子位置
        for i in range(self.n_particles):
            for j in range(self.dimension):
                min_val, max_val = self.bounds[j]
                self.particles[i, j] = random.uniform(min_val, max_val)
            
            # 修复约束
            self.particles[i] = self._repair_particle(self.particles[i])
            
            # 根据粒子位置到边界的距离自适应调整初始速度
            for j in range(self.dimension):
                range_val = self.bounds[j, 1] - self.bounds[j, 0]
                self.velocities[i, j] = random.uniform(-0.1 * range_val, 0.1 * range_val)

        # 并行计算初始适应度
        with concurrent.futures.ProcessPoolExecutor() as pool:
            scores = list(pool.map(MultiSmokeBombOptimizer.static_objective_function, self.particles.tolist()))
        for i, score in enumerate(scores):
            self.personal_best_scores[i] = score
            self.personal_best_positions[i] = self.particles[i].copy()
            if score > self.global_best_score:
                self.global_best_score = score
                self.global_best_position = self.particles[i].copy()
        
        debug_print(f"初始化完成，最佳初始适应度: {self.global_best_score:.6f}")
    
    def update_particles(self):
        """更新粒子位置和速度"""
        for i in range(self.n_particles):
            # 更新速度
            r1, r2 = random.random(), random.random()
            
            cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.particles[i])
            social_component = self.c2 * r2 * (self.global_best_position - self.particles[i])
            
            self.velocities[i] = (self.w * self.velocities[i] + 
                                cognitive_component + social_component)
            
            # 限制速度大小
            for j in range(self.dimension):
                range_val = self.bounds[j, 1] - self.bounds[j, 0]
                max_velocity = 0.2 * range_val
                self.velocities[i, j] = np.clip(self.velocities[i, j], -max_velocity, max_velocity)
            
            # 更新位置
            self.particles[i] += self.velocities[i]
            
            # 边界约束
            for j in range(self.dimension):
                self.particles[i, j] = np.clip(self.particles[i, j], 
                                             self.bounds[j, 0], self.bounds[j, 1])
            
            # 修复约束
            self.particles[i] = self._repair_particle(self.particles[i])
    
    def evaluate_and_update_best(self):
        """评估粒子适应度并更新最佳位置"""
        with concurrent.futures.ProcessPoolExecutor() as pool:
            scores = list(pool.map(MultiSmokeBombOptimizer.static_objective_function, self.particles.tolist()))

        for i, score in enumerate(scores):
            if score > self.personal_best_scores[i]:
                self.personal_best_scores[i] = score
                self.personal_best_positions[i] = self.particles[i].copy()
            if score > self.global_best_score:
                self.global_best_score = score
                self.global_best_position = self.particles[i].copy()
        
        # 记录历史
        self.history['best_scores'].append(self.global_best_score)
        self.history['mean_scores'].append(np.mean(scores))
        self.history['best_positions'].append(self.global_best_position.copy())
    
    def optimize(self):
        """执行粒子群优化"""
        debug_print("开始三枚烟幕弹协同优化...")
        
        # 初始化
        self.initialize_particles()
        
        # 迭代优化
        for iteration in range(self.n_iterations):
            # 更新粒子
            self.update_particles()
            
            # 评估适应度
            self.evaluate_and_update_best()
            
            # 动态调整惯性权重
            self.w = 0.9 - 0.5 * iteration / self.n_iterations
            
            # 输出进度
            if (iteration + 1) % 5 == 0 or iteration == 0:
                print(f"[Iter {iteration + 1:3d}/{self.n_iterations}]  最佳={self.global_best_score:.6f}  平均={self.history['mean_scores'][-1]:.6f}")
        
        print(f"优化结束: 全局最佳适应度 = {self.global_best_score:.6f}")
        
        return self.get_optimization_result()
    
    def get_optimization_result(self):
        """获取优化结果"""
        alpha_opt, v_D_opt, t_drop1, t_explode1, t_drop2, t_explode2, t_drop3, t_explode3 = self.global_best_position
        
        # 计算详细的遮蔽信息
        t_drop_list = [t_drop1, t_drop2, t_drop3]
        t_explode_list = [t_explode1, t_explode2, t_explode3]
        
        detailed_info = self.calculator.calculate_single_shielding_durations(
            alpha_opt, v_D_opt, t_drop_list, t_explode_list
        )
        
        result = {
            'optimal_parameters': {
                'alpha_degrees': np.degrees(alpha_opt),
                'alpha_radians': alpha_opt,
                'v_D': v_D_opt,
                'smoke_bomb_1': {'t_drop': t_drop1, 't_explode': t_explode1},
                'smoke_bomb_2': {'t_drop': t_drop2, 't_explode': t_explode2},
                'smoke_bomb_3': {'t_drop': t_drop3, 't_explode': t_explode3}
            },
            'optimal_total_shielding_duration': self.global_best_score,
            'individual_shielding_durations': detailed_info['individual_durations'],
            'optimization_history': self.history
        }
        
        return result

# 辅助函数
def debug_print(*args, **kwargs):
    """调试打印函数，根据VERBOSE控制是否输出"""
    if VERBOSE:
        print(*args, **kwargs)

def save_q3_results(result_dict, filename="q3_results.xlsx"):
    """将问题三的优化结果保存到Excel文件"""
    opt_params = result_dict["optimal_parameters"]
    alpha = opt_params["alpha_radians"]
    v_D = opt_params["v_D"]
    
    calculator = MultiSmokeBombCalculator()
    
    t_drop_list = [opt_params[f"smoke_bomb_{i+1}"]["t_drop"] for i in range(3)]
    t_explode_list = [opt_params[f"smoke_bomb_{i+1}"]["t_explode"] for i in range(3)]
    
    detailed_info = calculator.calculate_single_shielding_durations(
        alpha, v_D, t_drop_list, t_explode_list
    )
    
    # 创建问题三结果数据表
    results_data = []
    
    for i in range(3):
        bomb_key = f"smoke_bomb_{i+1}"
        t_drop = opt_params[bomb_key]["t_drop"]
        t_explode = opt_params[bomb_key]["t_explode"]
        drop_position = calculator._get_drone_position(t_drop, alpha, v_D, t_drop)
        explode_position = calculator._get_smoke_bomb_position(t_explode, alpha, v_D, t_drop, t_explode)
        
        # 获取每枚烟幕弹的单独遮蔽时长
        individual_duration = detailed_info["individual_durations"][i]
        
        row_data = {
            "无人机运动方向": f"{opt_params['alpha_degrees']:.2f}°",
            "无人机运动速度 (m/s)": opt_params["v_D"],
            "烟幕干扰弹编号": i + 1,
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
    
    # 创建详细时间段数据表
    time_segments_data = []
    
    for i in range(3):
        bomb_segments = detailed_info["detailed_time_segments"][i]
        
        if bomb_segments:  # 如果有遮蔽时间段
            for segment_idx, segment in enumerate(bomb_segments):
                time_segments_data.append({
                    "烟幕弹编号": i + 1,
                    "时间段序号": segment_idx + 1,
                    "遮蔽开始时间 (s)": segment["start_time"],
                    "遮蔽结束时间 (s)": segment["end_time"],
                    "时间段时长 (s)": segment["duration"]
                })
        else:  # 如果没有遮蔽时间段
            time_segments_data.append({
                "烟幕弹编号": i + 1,
                "时间段序号": 0,
                "遮蔽开始时间 (s)": 0,
                "遮蔽结束时间 (s)": 0,
                "时间段时长 (s)": 0
            })
    
    df_time_segments = pd.DataFrame(time_segments_data)
    
    # 创建汇总信息数据表
    summary_data = [{
        "无人机运动方向": f"{opt_params['alpha_degrees']:.2f}°",
        "无人机运动速度 (m/s)": opt_params["v_D"],
        "烟幕弹1单独遮蔽时长 (s)": round(detailed_info["individual_durations"][0], 4),
        "烟幕弹2单独遮蔽时长 (s)": round(detailed_info["individual_durations"][1], 4),
        "烟幕弹3单独遮蔽时长 (s)": round(detailed_info["individual_durations"][2], 4),
        "总有效干扰时长 (s)": round(detailed_info["total_duration"], 4)
    }]
    
    df_summary = pd.DataFrame(summary_data)
    
    # 创建优化历史数据表
    hist = result_dict["optimization_history"]
    df_history = pd.DataFrame({
        "iteration": range(1, len(hist["best_scores"]) + 1),
        "best_score": hist["best_scores"],
        "mean_score": hist["mean_scores"]
    })
    
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        df_results.to_excel(writer, sheet_name="问题三结果", index=False)
        df_time_segments.to_excel(writer, sheet_name="详细时间段", index=False)
        df_summary.to_excel(writer, sheet_name="遮蔽时长汇总", index=False)
        df_history.to_excel(writer, sheet_name="优化历史", index=False)
    
    print(f"已保存问题三优化结果到 {filename}")

# 主函数
def run_optimization_problem3():
    """运行问题三的优化求解"""
    print("=" * 60)
    print("开始求解问题三：三枚烟幕干扰弹协同投放策略优化")
    print("=" * 60)
    
    optimizer = MultiSmokeBombOptimizer(
        n_particles=200,       
        n_iterations=50,      
        w=0.9,               
        c1=2.8,              
        c2=1.3               
    )
    result = optimizer.optimize()
    
    print("\n" + "=" * 60)
    print("问题三优化结果")
    print("=" * 60)
    
    opt_params = result['optimal_parameters']
    print(f"最优参数:")
    print(f"  航向角 α = {opt_params['alpha_degrees']:.2f}° ({opt_params['alpha_radians']:.4f} 弧度)")
    print(f"  无人机速度 v_D = {opt_params['v_D']:.2f} m/s")
    print(f"")
    print(f"  烟幕弹 1: 投放时刻 t'₁ = {opt_params['smoke_bomb_1']['t_drop']:.3f} s, 起爆时刻 t''₁ = {opt_params['smoke_bomb_1']['t_explode']:.3f} s")
    print(f"  烟幕弹 2: 投放时刻 t'₂ = {opt_params['smoke_bomb_2']['t_drop']:.3f} s, 起爆时刻 t''₂ = {opt_params['smoke_bomb_2']['t_explode']:.3f} s")
    print(f"  烟幕弹 3: 投放时刻 t'₃ = {opt_params['smoke_bomb_3']['t_drop']:.3f} s, 起爆时刻 t''₃ = {opt_params['smoke_bomb_3']['t_explode']:.3f} s")
    print(f"")
    print(f"遮蔽时长详情:")
    individual_durations = result['individual_shielding_durations']
    print(f"  烟幕弹 1 单独遮蔽时长: {individual_durations[0]:.6f} 秒")
    print(f"  烟幕弹 2 单独遮蔽时长: {individual_durations[1]:.6f} 秒")
    print(f"  烟幕弹 3 单独遮蔽时长: {individual_durations[2]:.6f} 秒")
    print(f"  总遮蔽时长: {result['optimal_total_shielding_duration']:.6f} 秒")
    
    print(f"\n约束条件验证:")
    t1_drop = opt_params['smoke_bomb_1']['t_drop']
    t2_drop = opt_params['smoke_bomb_2']['t_drop'] 
    t3_drop = opt_params['smoke_bomb_3']['t_drop']
    print(f"  时间间隔 t'₂ - t'₁ = {t2_drop - t1_drop:.3f} s (≥ 1.0)")
    print(f"  时间间隔 t'₃ - t'₂ = {t3_drop - t2_drop:.3f} s (≥ 1.0)")

    save_q3_results(result, "q3_results.xlsx")
    return result

if __name__ == "__main__":
    result = run_optimization_problem3()
