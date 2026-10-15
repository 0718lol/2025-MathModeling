import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import concurrent.futures
from copy import deepcopy
from q2 import SmokeShieldingCalculator

VERBOSE = True

class Problem5Calculator(SmokeShieldingCalculator):
    """问题五计算器"""
    def __init__(self, **kwargs):
        """初始化问题五计算器"""
        super().__init__(**kwargs)
        self.drone_initial_positions = [
            np.array([17800, 0, 1800]),      # FY1
            np.array([12000, 1400, 1400]),   # FY2  
            np.array([6000, -3000, 700]),    # FY3
            np.array([11000, 2000, 1800]),   # FY4
            np.array([13000, -2000, 1300])   # FY5 
        ]
        missile_positions = [
            np.array([20000, 0, 2000]),      # 导弹M1
            np.array([19000, 600, 2100]),    # 导弹M2  
            np.array([18000, -600, 1900])    # 导弹M3
        ]
        
        self.missile_initial_positions = []
        self.missile_velocities = []
        self.missile_unit_vectors = []
        
        for missile_pos in missile_positions:
            self.missile_initial_positions.append(missile_pos)
            
            # 计算导弹飞向目标的单位方向向量
            direction = self.P_target - missile_pos
            unit_vector = direction / np.linalg.norm(direction)
            velocity = self.missile_speed * unit_vector
            
            self.missile_unit_vectors.append(unit_vector)
            self.missile_velocities.append(velocity)
            
        self.num_drones = 5
        self.num_missiles = 3
        self.num_bombs_per_drone = 3
        
    def _get_missile_position(self, t, missile_idx):
        """计算t时刻第missile_idx枚导弹的位置"""
        return self.missile_initial_positions[missile_idx] + self.missile_velocities[missile_idx] * t
    
    def _get_drone_position(self, t, drone_idx, alpha, v_D):
        """计算t时刻第drone_idx架无人机的位置"""
        e_alpha = self._get_unit_direction_vector(alpha)
        return self.drone_initial_positions[drone_idx] + v_D * t * e_alpha
        
    def _get_smoke_cloud_center(self, t, drone_idx, bomb_idx, alpha, v_D, t_drop, t_explode):
        """计算t时刻第drone_idx架无人机第bomb_idx枚烟幕弹的云团中心位置"""
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

    def _check_constraints(self, solution):
        """检查解的约束条件"""
        alphas = solution[0:5]
        velocities = solution[5:10] 
        t_drops = solution[10:25].reshape(5, 3)  
        t_explodes = solution[25:40].reshape(5, 3) 
        
        # 速度约束
        for v in velocities:
            if not (70 <= v <= 140):
                return False
        
        # 时序约束和物理约束
        for drone_idx in range(5):
            drone_height = self.drone_initial_positions[drone_idx][2]
            max_fall_time = np.sqrt(2 * drone_height / self.g)
            
            for bomb_idx in range(3):
                t_drop = t_drops[drone_idx, bomb_idx]
                t_explode = t_explodes[drone_idx, bomb_idx]
                
                # 基本时序约束
                if t_drop > t_explode:
                    return False
                    
                # 烟幕弹需在落地前爆炸
                if (t_explode - t_drop) > max_fall_time:
                    return False
                
                # 连续投放间隔约束
                if bomb_idx > 0:  # 第2、3枚烟幕弹
                    if t_drops[drone_idx, bomb_idx] - t_drops[drone_idx, bomb_idx-1] < 1.0:
                        return False
        
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

    def _is_point_shielded_by_smoke(self, missile_pos, target_point, smoke_center):
        """判断目标点是否被单个烟幕云团遮蔽"""
        if smoke_center is None:
            return False
            
        missile_to_smoke_distance = np.linalg.norm(missile_pos - smoke_center)
        tolerance = 1e-6
        
        if missile_to_smoke_distance < self.effective_radius - tolerance:
            # 导弹在云团内，必然遮蔽目标点
            return True
        else:
            V_mt = target_point - missile_pos
            V_length = np.linalg.norm(V_mt)
            
            if V_length < 1e-12:
                # 导弹和目标点重合
                return np.linalg.norm(missile_pos - smoke_center) <= self.effective_radius
            
            # 计算云团中心到导弹-目标点连线的距离
            W = smoke_center - missile_pos
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
                missile_in_smoke = np.linalg.norm(W) <= self.effective_radius
                return missile_in_smoke
            
            return True

    def _is_target_fully_shielded_from_missile(self, t, missile_idx, solution):
        """判断t时刻目标圆柱体是否被完全遮蔽"""
        alphas = solution[0:5]
        velocities = solution[5:10] 
        t_drops = solution[10:25].reshape(5, 3)  
        t_explodes = solution[25:40].reshape(5, 3)
        
        sampling_points = self._get_cylinder_sampling_points()
        missile_pos = self._get_missile_position(t, missile_idx)
        
        # 获取当前时刻所有有效的烟幕云团
        active_smoke_centers = []
        for drone_idx in range(5):
            for bomb_idx in range(3):
                t_explode = t_explodes[drone_idx, bomb_idx]
                if t_explode <= t <= t_explode + self.effective_duration:
                    smoke_center = self._get_smoke_cloud_center(
                        t, drone_idx, bomb_idx, alphas[drone_idx], 
                        velocities[drone_idx], t_drops[drone_idx, bomb_idx], t_explode
                    )
                    if smoke_center is not None:
                        active_smoke_centers.append(smoke_center)
        
        if not active_smoke_centers:
            return False
        
        # 检查每个采样点是否被遮蔽
        for point in sampling_points:
            point_shielded = False
            for smoke_center in active_smoke_centers:
                if self._is_point_shielded_by_smoke(missile_pos, point, smoke_center):
                    point_shielded = True
                    break
            
            # 如果任意一个点未被遮蔽，整个目标就未被完全遮蔽
            if not point_shielded:
                return False
        
        return True

    def _is_target_fully_shielded_from_all_missiles(self, t, solution):
        """判断t时刻目标圆柱体是否被所有3枚导弹同时无法观测到"""
        for missile_idx in range(3):
            if not self._is_target_fully_shielded_from_missile(t, missile_idx, solution):
                return False
        return True

    def calculate_objective_function(self, solution, eps=0.1):
        """计算目标函数"""
        if not self._check_constraints(solution):
            return 0.0
        
        # 确定搜索时间范围
        t_explodes = solution[25:40].reshape(5, 3)
        earliest_explode = np.min(t_explodes)
        latest_explode = np.max(t_explodes)
        
        # 搜索范围
        time_start = max(0, earliest_explode - 1.0)
        time_end = latest_explode + self.effective_duration + 1.0
        
        # 粗略扫描
        ts = np.arange(time_start, time_end, eps)
        if len(ts) == 0:
            return 0.0
            
        shielded_mask = np.array([
            self._is_target_fully_shielded_from_all_missiles(t, solution) 
            for t in ts
        ])
        
        if not shielded_mask.any():
            return 0.0
        
        # 计算所有遮蔽区间的总时长
        total_duration = 0.0
        in_shielded_region = False
        region_start = 0
        
        for i, is_shielded in enumerate(shielded_mask):
            if is_shielded and not in_shielded_region:
                # 开始新的遮蔽区间
                region_start = i
                in_shielded_region = True
            elif not is_shielded and in_shielded_region:
                # 结束当前遮蔽区间
                region_end = i - 1
                segment_duration = (region_end - region_start + 1) * eps
                total_duration += segment_duration
                in_shielded_region = False
        
        # 处理最后一个区间
        if in_shielded_region:
            segment_duration = (len(ts) - region_start) * eps
            total_duration += segment_duration
        
        return total_duration

    def calculate_detailed_shielding_info(self, solution, eps=0.05):
        """详细计算遮蔽信息，包括各导弹的单独遮蔽情况"""
        if not self._check_constraints(solution):
            return {
                'total_duration': 0.0,
                'missile_individual_durations': [0.0, 0.0, 0.0],
                'intersection_duration': 0.0,
                'time_ranges': []
            }
        
        # 计算时间范围
        t_explodes = solution[25:40].reshape(5, 3)
        earliest_explode = np.min(t_explodes)
        latest_explode = np.max(t_explodes)
        time_start = max(0, earliest_explode - 1.0)
        time_end = latest_explode + self.effective_duration + 1.0
        
        ts = np.arange(time_start, time_end, eps)
        if len(ts) == 0:
            return {'total_duration': 0.0, 'missile_individual_durations': [0.0, 0.0, 0.0], 
                   'intersection_duration': 0.0, 'time_ranges': []}
        
        # 计算每枚导弹的遮蔽情况
        missile_masks = []
        for missile_idx in range(3):
            mask = np.array([
                self._is_target_fully_shielded_from_missile(t, missile_idx, solution) 
                for t in ts
            ])
            missile_masks.append(mask)
        
        # 计算各导弹的单独遮蔽时长
        individual_durations = []
        for mask in missile_masks:
            duration = np.sum(mask) * eps
            individual_durations.append(duration)
        
        # 计算交集
        intersection_mask = np.logical_and.reduce(missile_masks)
        intersection_duration = np.sum(intersection_mask) * eps
        
        return {
            'total_duration': intersection_duration,
            'missile_individual_durations': individual_durations,
            'intersection_duration': intersection_duration,
            'time_ranges': self._extract_time_ranges(ts, intersection_mask)
        }

    def _extract_time_ranges(self, ts, mask):
        """从掩码中提取连续的时间范围"""
        ranges = []
        if not mask.any():
            return ranges
            
        in_range = False
        start_idx = 0
        
        for i, is_active in enumerate(mask):
            if is_active and not in_range:
                start_idx = i
                in_range = True
            elif not is_active and in_range:
                ranges.append((ts[start_idx], ts[i-1]))
                in_range = False
        
        # 处理最后一个范围
        if in_range:
            ranges.append((ts[start_idx], ts[-1]))
            
        return ranges

def debug_print(*args, **kwargs):
    """调试打印函数"""
    if VERBOSE:
        print(*args, **kwargs) 

class Problem5PSO:
    """处理40维优化问题：最大化三枚导弹同时无法观测到目标的时长"""
    
    def __init__(self, calculator, num_particles=50, max_iterations=200):
        self.calculator = calculator
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        
        self.w = 0.7      
        self.c1 = 1.5     
        self.c2 = 1.5     
        
        self.dim = 40
        
        # 变量边界定义
        self.lower_bounds = np.zeros(40)
        self.upper_bounds = np.zeros(40)
        
        self.lower_bounds[0:5] = 0.0
        self.upper_bounds[0:5] = 2 * np.pi
        
        self.lower_bounds[5:10] = 70.0
        self.upper_bounds[5:10] = 140.0
        
        self.lower_bounds[10:25] = 0.0
        self.upper_bounds[10:25] = 30.0
        
        self.lower_bounds[25:40] = 0.0  
        self.upper_bounds[25:40] = 50.0
        
        # 初始化
        self.particles = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_values = None
        self.global_best_position = None
        self.global_best_value = -float('inf')
        
        self.history = []  # 存储优化历史

    def _initialize_particles(self):
        """初始化"""
        self.particles = np.zeros((self.num_particles, self.dim))
        self.velocities = np.zeros((self.num_particles, self.dim))
        self.personal_best_positions = np.zeros((self.num_particles, self.dim))
        self.personal_best_values = np.full(self.num_particles, -float('inf'))
        
        for i in range(self.num_particles):
            # 随机初始化
            particle = self._generate_random_solution()
            
            # 确保约束满足
            particle = self._repair_solution(particle)
            self.particles[i] = particle
            
            # 初始化速度
            velocity_range = (self.upper_bounds - self.lower_bounds) * 0.1
            self.velocities[i] = np.random.uniform(-velocity_range, velocity_range)
            
            # 评估适应度
            fitness = self.calculator.calculate_objective_function(particle)
            self.personal_best_positions[i] = particle.copy()
            self.personal_best_values[i] = fitness
            
            # 更新全局最优
            if fitness > self.global_best_value:
                self.global_best_value = fitness
                self.global_best_position = particle.copy()
    
    def _generate_random_solution(self):
        """生成随机解"""
        solution = np.zeros(40)
        solution[0:5] = np.random.uniform(0, 2*np.pi, 5)
        solution[5:10] = np.random.uniform(70, 140, 5)
        for drone_idx in range(5):
            for bomb_idx in range(3):
                t_drop_idx = 10 + drone_idx * 3 + bomb_idx
                t_explode_idx = 25 + drone_idx * 3 + bomb_idx
                
                # 投放时间
                if bomb_idx == 0:
                    t_drop = np.random.uniform(0, 15)
                else:
                    # 确保连续投放间隔至少1秒
                    prev_t_drop = solution[t_drop_idx - 1]
                    t_drop = prev_t_drop + np.random.uniform(1.0, 5.0)
                
                solution[t_drop_idx] = t_drop
                
                # 爆炸时间
                drone_height = self.calculator.drone_initial_positions[drone_idx][2]
                max_fall_time = np.sqrt(2 * drone_height / self.calculator.g)
                
                max_explode_delay = min(max_fall_time - 0.1, 20.0)  # 留有余量
                explode_delay = np.random.uniform(0.5, max_explode_delay)
                solution[t_explode_idx] = t_drop + explode_delay
                
        return solution
    
    def _repair_solution(self, solution):
        """修复解以满足约束条件"""
        repaired = solution.copy()
        
        # 边界约束
        repaired = np.clip(repaired, self.lower_bounds, self.upper_bounds)
        
        # 角度约束
        repaired[0:5] = repaired[0:5] % (2 * np.pi)
        
        # 时序约束修复
        for drone_idx in range(5):
            drone_height = self.calculator.drone_initial_positions[drone_idx][2]
            max_fall_time = np.sqrt(2 * drone_height / self.calculator.g)
            
            for bomb_idx in range(3):
                t_drop_idx = 10 + drone_idx * 3 + bomb_idx
                t_explode_idx = 25 + drone_idx * 3 + bomb_idx
                
                # 连续投放间隔约束
                if bomb_idx > 0:
                    prev_t_drop = repaired[t_drop_idx - 1]
                    if repaired[t_drop_idx] < prev_t_drop + 1.0:
                        repaired[t_drop_idx] = prev_t_drop + 1.0
                
                # 时序约束
                if repaired[t_explode_idx] <= repaired[t_drop_idx]:
                    repaired[t_explode_idx] = repaired[t_drop_idx] + 0.5
                
                # 物理约束
                max_explode_time = repaired[t_drop_idx] + max_fall_time - 0.1
                if repaired[t_explode_idx] > max_explode_time:
                    repaired[t_explode_idx] = max_explode_time
        
        return repaired
    
    def _update_inertia_weight(self, iteration):
        """动态更新惯性权重"""
        # 线性递减策略
        w_max = 0.9
        w_min = 0.4
        self.w = w_max - (w_max - w_min) * iteration / self.max_iterations
    
    def optimize(self, verbose=True):
        """执行优化"""
        debug_print("开始问题五的优化求解...")
        
        # 初始化
        self._initialize_particles()
        
        debug_print(f"初始全局最优值: {self.global_best_value:.4f}")
        
        # 主优化循环
        for iteration in range(self.max_iterations):
            self._update_inertia_weight(iteration)
            
            for i in range(self.num_particles):
                # 更新速度
                r1, r2 = np.random.random(2)
                
                cognitive_velocity = self.c1 * r1 * (self.personal_best_positions[i] - self.particles[i])
                social_velocity = self.c2 * r2 * (self.global_best_position - self.particles[i])
                
                self.velocities[i] = (self.w * self.velocities[i] + 
                                    cognitive_velocity + social_velocity)
                
                # 限制速度
                max_velocity = (self.upper_bounds - self.lower_bounds) * 0.2
                self.velocities[i] = np.clip(self.velocities[i], -max_velocity, max_velocity)
                
                # 更新位置
                self.particles[i] += self.velocities[i]
                
                # 修复解
                self.particles[i] = self._repair_solution(self.particles[i])
                
                # 评估适应度
                fitness = self.calculator.calculate_objective_function(self.particles[i])
                
                # 更新个体最优
                if fitness > self.personal_best_values[i]:
                    self.personal_best_values[i] = fitness
                    self.personal_best_positions[i] = self.particles[i].copy()
                    
                    # 更新全局最优
                    if fitness > self.global_best_value:
                        self.global_best_value = fitness
                        self.global_best_position = self.particles[i].copy()
            
            # 记录历史
            avg_fitness = np.mean(self.personal_best_values)
            self.history.append({
                'iteration': iteration,
                'best_fitness': self.global_best_value,
                'avg_fitness': avg_fitness,
                'inertia_weight': self.w
            })
            
            # 输出进度
            if verbose and (iteration + 1) % 10 == 0:
                debug_print(f"迭代 {iteration + 1:3d}: 最优值 = {self.global_best_value:.6f}, "
                          f"平均值 = {avg_fitness:.6f}, w = {self.w:.3f}")
        
        debug_print(f"优化完成! 最终最优值: {self.global_best_value:.6f}")
        return self.global_best_position, self.global_best_value
    
    def get_optimization_results(self):
        """获取优化结果的详细信息"""
        if self.global_best_position is None:
            return None
            
        # 解析决策变量
        solution = self.global_best_position
        alphas = solution[0:5]  
        velocities = solution[5:10]
        t_drops = solution[10:25].reshape(5, 3)
        t_explodes = solution[25:40].reshape(5, 3)
        
        # 计算详细遮蔽信息
        shielding_info = self.calculator.calculate_detailed_shielding_info(solution)
        
        return {
            'best_solution': solution,
            'best_objective_value': self.global_best_value,
            'drone_angles_deg': np.degrees(alphas),
            'drone_velocities': velocities,
            'drop_times': t_drops,
            'explode_times': t_explodes,
            'shielding_info': shielding_info,
            'optimization_history': self.history,
            'constraint_satisfied': self.calculator._check_constraints(solution)
        } 

def solve_problem5(num_particles=50, max_iterations=200, save_results=True, output_file='q5_results.xlsx'):
    """问题五主求解函数"""
    print("=" * 60)
    print("问题五：5架无人机协同多烟幕弹遮蔽3枚导弹优化求解")
    print("=" * 60)
    calculator = Problem5Calculator()
    print("\n场景信息：")
    print(f"- 无人机数量：{calculator.num_drones} 架")
    print(f"- 导弹数量：{calculator.num_missiles} 枚") 
    print(f"- 每架无人机烟幕弹数量：{calculator.num_bombs_per_drone} 枚")
    print(f"- 总烟幕弹数量：{calculator.num_drones * calculator.num_bombs_per_drone} 枚")
    print(f"- 决策变量维度：{calculator.num_drones * 2 + calculator.num_drones * calculator.num_bombs_per_drone * 2} 维")
    
    print(f"\n无人机初始位置：")
    for i, pos in enumerate(calculator.drone_initial_positions):
        print(f"  FY{i+1}: ({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f})")
        
    print(f"\n导弹初始位置：")
    for i, pos in enumerate(calculator.missile_initial_positions):
        print(f"  M{i+1}: ({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f})")
    
    # 创建求解器
    pso = Problem5PSO(calculator, num_particles=num_particles, max_iterations=max_iterations)
    
    print(f"\n开始优化...")
    # 执行优化
    start_time = pd.Timestamp.now()
    best_solution, best_value = pso.optimize(verbose=True)
    end_time = pd.Timestamp.now()
    
    computation_time = (end_time - start_time).total_seconds()
    
    # 获取详细结果
    results = pso.get_optimization_results()
    
    print(f"\n" + "=" * 60)
    print("优化结果总结")
    print(f"=" * 60)
    print(f"计算耗时：{computation_time:.2f} 秒")
    print(f"最优目标函数值：{best_value:.6f} 秒")
    print(f"约束条件满足：{'是' if results['constraint_satisfied'] else '否'}")
    
    # 输出优化的决策变量
    print(f"\n最优解的决策变量：")
    print(f"无人机飞行角度（度）：")
    for i, angle in enumerate(results['drone_angles_deg']):
        print(f"  FY{i+1}: {angle:.2f}°")
        
    print(f"\n无人机飞行速度（m/s）：")
    for i, velocity in enumerate(results['drone_velocities']):
        print(f"  FY{i+1}: {velocity:.2f}")
        
    print(f"\n烟幕弹投放时间（秒）：")
    for i in range(5):
        times_str = ", ".join([f"{t:.2f}" for t in results['drop_times'][i]])
        print(f"  FY{i+1}: [{times_str}]")
        
    print(f"\n烟幕弹爆炸时间（秒）：")
    for i in range(5):
        times_str = ", ".join([f"{t:.2f}" for t in results['explode_times'][i]])
        print(f"  FY{i+1}: [{times_str}]")
    
    # 输出遮蔽效果分析
    shielding_info = results['shielding_info']
    print(f"\n遮蔽效果分析：")
    print(f"总遮蔽时长（三导弹同时被遮蔽）：{shielding_info['total_duration']:.4f} 秒")
    print(f"各导弹单独遮蔽时长：")
    for i, duration in enumerate(shielding_info['missile_individual_durations']):
        print(f"  导弹M{i+1}: {duration:.4f} 秒")
    
    if shielding_info['time_ranges']:
        print(f"\n有效遮蔽时间段：")
        for i, (start, end) in enumerate(shielding_info['time_ranges']):
            duration = end - start
            print(f"  区间{i+1}: {start:.2f}s - {end:.2f}s (时长: {duration:.2f}s)")
    
    # 保存结果
    if save_results:
        print(f"\n正在保存结果到 {output_file}...")
        save_problem5_results(results, computation_time, output_file)
        print("结果保存完成！")
    
    return results

def save_problem5_results(results, computation_time, filename='q5_results.xlsx'):
    """保存问题五的优化结果到Excel文件"""
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        
        summary_data = {
            '项目': ['计算耗时(秒)', '最优目标函数值(秒)', '约束条件满足', '无人机数量', '导弹数量', '总烟幕弹数量'],
            '数值': [
                f"{computation_time:.2f}",
                f"{results['best_objective_value']:.6f}", 
                '是' if results['constraint_satisfied'] else '否',
                '5', '3', '15'
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='优化概要', index=False)
        
        decision_vars = []
        
        for i in range(5):
            decision_vars.append({
                '无人机': f'FY{i+1}',
                '变量类型': '飞行角度',
                '数值': f"{results['drone_angles_deg'][i]:.2f}",
                '单位': '度'
            })
            decision_vars.append({
                '无人机': f'FY{i+1}',
                '变量类型': '飞行速度', 
                '数值': f"{results['drone_velocities'][i]:.2f}",
                '单位': 'm/s'
            })
        
        for i in range(5):
            for j in range(3):
                decision_vars.append({
                    '无人机': f'FY{i+1}',
                    '变量类型': f'烟幕弹{j+1}投放时间',
                    '数值': f"{results['drop_times'][i,j]:.2f}",
                    '单位': '秒'
                })
                decision_vars.append({
                    '无人机': f'FY{i+1}',
                    '变量类型': f'烟幕弹{j+1}爆炸时间',
                    '数值': f"{results['explode_times'][i,j]:.2f}",
                    '单位': '秒'
                })
        
        decision_df = pd.DataFrame(decision_vars)
        decision_df.to_excel(writer, sheet_name='决策变量', index=False)
        
        shielding_data = []
        shielding_info = results['shielding_info']
        
        shielding_data.append({
            '项目': '总遮蔽时长(三导弹同时)',
            '时长(秒)': f"{shielding_info['total_duration']:.4f}",
            '说明': '三枚导弹同时无法观测到目标的总时长'
        })
        
        for i, duration in enumerate(shielding_info['missile_individual_durations']):
            shielding_data.append({
                '项目': f'导弹M{i+1}遮蔽时长',
                '时长(秒)': f"{duration:.4f}",
                '说明': f'导弹M{i+1}无法观测到目标的时长'
            })
        
        shielding_df = pd.DataFrame(shielding_data)
        shielding_df.to_excel(writer, sheet_name='遮蔽效果', index=False)
        
        if shielding_info['time_ranges']:
            time_ranges_data = []
            for i, (start, end) in enumerate(shielding_info['time_ranges']):
                time_ranges_data.append({
                    '区间序号': i+1,
                    '开始时间(秒)': f"{start:.2f}",
                    '结束时间(秒)': f"{end:.2f}",
                    '持续时长(秒)': f"{end-start:.2f}"
                })
            
            time_ranges_df = pd.DataFrame(time_ranges_data)
            time_ranges_df.to_excel(writer, sheet_name='遮蔽时间段', index=False)
        
        history_data = []
        for record in results['optimization_history']:
            history_data.append({
                '迭代次数': record['iteration'] + 1,
                '最优适应度': f"{record['best_fitness']:.6f}",
                '平均适应度': f"{record['avg_fitness']:.6f}",
                '惯性权重': f"{record['inertia_weight']:.3f}"
            })
        
        history_df = pd.DataFrame(history_data)
        history_df.to_excel(writer, sheet_name='优化历史', index=False)


# 主程序入口
if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    random.seed(42)
    
    results = solve_problem5(
        num_particles=50,      
        max_iterations=50,      
        save_results=True,     
        output_file='q5_results.xlsx'
    )
    
