import numpy as np
import pandas as pd
import random
import concurrent.futures
from functools import lru_cache

VERBOSE = True  

class SmokeShieldingCalculator:
    """物理模型和遮蔽时长计算"""
    
    def __init__(self, 
                 missile_speed=300,
                 smoke_fall_speed=3,
                 effective_radius=10,
                 effective_duration=20,
                 gravity=9.80665,
                 edge_sample_num=100):
        # 物理参数
        self.missile_speed = missile_speed
        self.smoke_fall_speed = smoke_fall_speed
        self.effective_radius = effective_radius
        self.effective_duration = effective_duration
        self.g = gravity
        self.edge_sample_num = edge_sample_num
        
        # 固定的场景参数
        self.P_M1_0 = np.array([20000, 0, 2000])  
        self.P_target = np.array([0, 0, 0])       
        self.P_FY1_0 = np.array([17800, 0, 1800])
        
        # 圆柱体目标参数
        self.C_base = np.array([0, 200, 0])
        self.C_top = np.array([0, 200, 10])
        self.R_cyl = 7
        self.H_cyl = 10
        
        # 导弹运动参数
        self._setup_missile_trajectory()
        
        # 计算圆柱体边缘点
        self._edge_points = self._get_cylinder_edge_points()
    
    def _setup_missile_trajectory(self):
        """设置导弹轨迹参数"""
        d_M1 = self.P_target - self.P_M1_0
        distance_M1 = np.linalg.norm(d_M1)
        self.u_M1 = d_M1 / distance_M1
        self.v_M1 = self.missile_speed * self.u_M1
    
    def _get_unit_direction_vector(self, alpha):
        """根据航向角计算单位方向向量"""
        return np.array([np.cos(alpha), np.sin(alpha), 0])
    
    def _get_missile_position(self, t):
        """计算t时刻导弹位置"""
        return self.P_M1_0 + self.v_M1 * t
    
    def _get_drone_position(self, t, alpha, v_D, t_drop):
        """计算t时刻无人机位置"""
        e_alpha = self._get_unit_direction_vector(alpha)
        return self.P_FY1_0 + v_D * t * e_alpha
    
    def _get_smoke_bomb_position(self, t, alpha, v_D, t_drop, t_explode):
        """计算t时刻烟幕弹位置"""
        if t < t_drop:
            return None
        elif t <= t_explode:
            # 自由落体阶段
            e_alpha = self._get_unit_direction_vector(alpha)
            R = self.P_FY1_0 + v_D * t_drop * e_alpha
            dt = t - t_drop
            bomb_position = R + v_D * dt * e_alpha + np.array([0, 0, -0.5 * self.g * dt**2])
            
            # 确保烟幕弹高度不低于地面
            if bomb_position[2] < 0:
                bomb_position[2] = 0
                
            return bomb_position
        else:
            return None
    
    def _get_smoke_cloud_center(self, t, alpha, v_D, t_drop, t_explode):
        """计算t时刻烟幕云团中心位置"""
        if t < t_explode:
            return None
        elif t <= t_explode + self.effective_duration:
            # 计算起爆位置
            e_alpha = self._get_unit_direction_vector(alpha)
            R = self.P_FY1_0 + v_D * t_drop * e_alpha
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
    
    def _get_cylinder_edge_points(self):
        """获取圆柱体上下底面边缘采样点"""
        thetas = np.linspace(0.0, 2.0 * np.pi, self.edge_sample_num, endpoint=False)
        
        x_circle = self.R_cyl * np.cos(thetas)
        y_circle = self.C_base[1] + self.R_cyl * np.sin(thetas)
        
        # 顶圆周
        z_top = np.full_like(x_circle, self.C_top[2])
        P_top = np.column_stack((x_circle, y_circle, z_top))
        
        # 底圆周
        z_bot = np.full_like(x_circle, self.C_base[2])
        P_bot = np.column_stack((x_circle, y_circle, z_bot))
        
        return np.vstack((P_top, P_bot))
    
    def _distance_point_to_line(self, P_missile, P_target, C_smoke):
        """计算烟幕中心到导弹-目标连线的距离"""
        vec_missile_to_target = P_target - P_missile
        vec_missile_to_smoke = C_smoke - P_missile
        
        missile_target_distance = np.linalg.norm(vec_missile_to_target)
        if missile_target_distance < 1e-10:
            return np.linalg.norm(vec_missile_to_smoke)
        
        cross_product = np.cross(vec_missile_to_target, vec_missile_to_smoke)
        cross_product_norm = np.linalg.norm(cross_product)
        
        return cross_product_norm / missile_target_distance
    
    @lru_cache(maxsize=None)
    def _has_missile_ever_penetrated(self, alpha, v_D, t_drop, t_explode, t_now, dt=0.05):
        """Y2: 从 t_explode 到 t_now 是否曾进入烟幕球体"""
        t = t_explode
        while t <= t_now:
            C_smoke = self._get_smoke_cloud_center(t, alpha, v_D, t_drop, t_explode)
            if C_smoke is None:
                return False
            dist = np.linalg.norm(self._get_missile_position(t) - C_smoke)
            if dist <= self.effective_radius + 1e-9:
                return True
            t += dt
        return False
    
    def _check_penetration_exit_shielding(self, P_missile, C_smoke, edge_points):
        """穿出时刻的向量夹角判定"""
        V_mt = edge_points - P_missile
        W = C_smoke - P_missile
        dot_ws = np.dot(W, V_mt.T)
        return not np.any(dot_ws <= 0)
    
    def _check_general_shielding(self, P_missile, C_smoke, edge_points):
        """一般情况的遮挡判定"""
        V_mt = edge_points - P_missile
        V_length_sq = np.einsum('ij,ij->i', V_mt, V_mt)
        W = C_smoke - P_missile
        cross = np.cross(V_mt, W)
        cross_norm = np.linalg.norm(cross, axis=1)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            distances = np.where(V_length_sq > 1e-12, cross_norm / np.sqrt(V_length_sq), 
                               np.linalg.norm(C_smoke - P_missile))
        
        if np.any(distances > self.effective_radius):
            return False
        
        # 方位判定
        dot_ws = np.dot(W, V_mt.T)
        with np.errstate(divide='ignore', invalid='ignore'):
            t_proj = np.where(V_length_sq > 1e-12, dot_ws / V_length_sq, 0.0)
        
        return not np.any((t_proj < 0.0) | (t_proj > 1.0))
    
    def _is_target_shielded(self, t, alpha, v_D, t_drop, t_explode):
        """判断t时刻目标是否被完全遮蔽"""
        # 只在烟幕云团存在期间进行判定
        if t < t_explode or t > t_explode + self.effective_duration:
            return False
        
        # 获取当前时刻的位置
        P_missile = self._get_missile_position(t)
        C_smoke = self._get_smoke_cloud_center(t, alpha, v_D, t_drop, t_explode)
        
        if C_smoke is None:
            return False
        
        # 判定是否曾经穿入云团
        Y2 = self._has_missile_ever_penetrated(alpha, v_D, t_drop, t_explode, t, dt=0.05)
        
        # 判断导弹和云团的位置关系
        missile_to_smoke_distance = np.linalg.norm(P_missile - C_smoke)
        tolerance = 1e-6
        
        if missile_to_smoke_distance < self.effective_radius - tolerance:
            # R < 10: 导弹在云团内，必然遮蔽
            return True
        elif abs(missile_to_smoke_distance - self.effective_radius) <= tolerance:
            # R = 10: 导弹在边界上
            if Y2:
                # 曾穿入且在边界：穿出时刻，需要向量夹角判定
                return self._check_penetration_exit_shielding(P_missile, C_smoke, self._edge_points)
            else:
                # 未穿入且在边界：按一般判定
                return self._check_general_shielding(P_missile, C_smoke, self._edge_points)
        else:
            # R > 10: 导弹在云团外
            if Y2:
                # 曾穿入但已在外部：按一般判定
                return self._check_general_shielding(P_missile, C_smoke, self._edge_points)
            else:
                # 未穿入且在外部：按一般判定
                return self._check_general_shielding(P_missile, C_smoke, self._edge_points)
    
    def calculate_shielding_duration(self, alpha, v_D, t_drop, t_explode, eps=1e-4):
        """计算给定参数下的总遮蔽时长"""
        # 约束检查
        if t_drop > t_explode:
            return 0.0
        
        if not (0 <= alpha <= 2*np.pi and 70 <= v_D <= 140):
            return 0.0
        
        # 使用二分法精确计算遮蔽区间
        def cond(t):
            return self._is_target_shielded(t, alpha, v_D, t_drop, t_explode)
        
        # 粗扫
        ts = np.arange(t_explode, t_explode + self.effective_duration + 0.05, 0.05)
        mask = np.array([cond(t) for t in ts])
        if not mask.any():
            return 0.0
        
        idx0 = np.argmax(mask)
        idx1 = len(mask) - 1 - np.argmax(mask[::-1])
        
        def bisection(left, right, target_first_true):
            """二分找第一次 True 或最后一次 True"""
            for _ in range(40):
                mid = 0.5 * (left + right)
                if cond(mid):
                    if target_first_true:
                        right = mid
                    else:
                        left = mid
                else:
                    if target_first_true:
                        left = mid
                    else:
                        right = mid
                if right - left < eps:
                    break
            return 0.5 * (left + right)
        
        # 第一个遮蔽精确时刻
        t_left_lo = ts[max(idx0 - 1, 0)]
        t_left_hi = ts[idx0]
        T0 = bisection(t_left_lo, t_left_hi, True)
        
        # 最后一个遮蔽精确时刻
        t_right_lo = ts[idx1]
        t_right_hi = ts[min(idx1 + 1, len(ts)-1)]
        T1 = bisection(t_right_lo, t_right_hi, False)
        
        return max(T1 - T0, 0.0)
    
    def get_scenario_info(self):
        """获取场景信息"""
        return {
            '导弹初始位置': self.P_M1_0.tolist(),
            '目标位置': self.P_target.tolist(),
            '无人机初始位置': self.P_FY1_0.tolist(),
            '圆柱体底面中心': self.C_base.tolist(),
            '圆柱体顶面中心': self.C_top.tolist(),
            '圆柱体半径': self.R_cyl,
            '圆柱体高度': self.H_cyl,
            '导弹速度': self.missile_speed,
            '烟幕下沉速度': self.smoke_fall_speed,
            '烟幕有效半径': self.effective_radius,
            '烟幕有效持续时间': self.effective_duration
        }

class ParticleSwarmOptimizer:
    """粒子群优化算法求解器"""
    
    # 用于并行计算的计算器实例
    _shared_calculator = None
    
    def __init__(self, n_particles=50, n_iterations=100, w=0.9, c1=2.0, c2=2.0):
        """初始化粒子群优化器"""
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # 创建计算器实例
        self.calculator = SmokeShieldingCalculator()
        
        # 设置共享计算器
        if ParticleSwarmOptimizer._shared_calculator is None:
            ParticleSwarmOptimizer._shared_calculator = SmokeShieldingCalculator()
        
        # 定义决策变量的边界
        self.bounds = np.array([
            [0, 2*np.pi],    # alpha
            [70, 140],       # v_D
            [0, 27.0],     # t_drop
            [0, 46.0]      # t_explode
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
        alpha, v_D, t_drop, t_explode = params
        
        # 不可行解给予严重惩罚
        if t_drop > t_explode:
            return -100.0
        if (t_explode - t_drop) > 19.1:
            return -100.0
        if not (0 <= alpha <= 2*np.pi and 70 <= v_D <= 140):
            return -100.0

        # 精确遮蔽奖励 
        duration = self.calculator.calculate_shielding_duration(alpha, v_D, t_drop, t_explode)
        if duration > 0:
            return duration

        # 梯度化奖励 
        proximity_score = self.calculate_proximity_to_shielding(alpha, v_D, t_drop, t_explode)
        return -1.0 + 0.95 * proximity_score

    def objective_function(self, params):
        """目标函数：使用梯度化适应度函数"""
        return self.gradient_objective_function(params)
    
    @staticmethod
    def static_objective_function(params):
        """静态目标函数：用于并行计算"""
        alpha, v_D, t_drop, t_explode = params
        
        # 不可行解给予严重惩罚
        if t_drop > t_explode:
            return -100.0
        if (t_explode - t_drop) > 19.1:
            return -100.0
        if not (0 <= alpha <= 2*np.pi and 70 <= v_D <= 140):
            return -100.0

        # 使用共享计算器
        if ParticleSwarmOptimizer._shared_calculator is None:
            ParticleSwarmOptimizer._shared_calculator = SmokeShieldingCalculator()
        calculator = ParticleSwarmOptimizer._shared_calculator
        
        # 精确遮蔽奖励
        duration = calculator.calculate_shielding_duration(alpha, v_D, t_drop, t_explode)
        if duration > 0:
            return duration

        # 梯度化奖励
        proximity_score = ParticleSwarmOptimizer._static_calculate_proximity_to_shielding(alpha, v_D, t_drop, t_explode)
        return -1.0 + 0.95 * proximity_score
    
    @staticmethod
    def _static_calculate_proximity_to_shielding(alpha, v_D, t_drop, t_explode):
        """静态版本的接近度计算函数，用于并行计算"""
        scores = []
        # 确保共享计算器存在
        if ParticleSwarmOptimizer._shared_calculator is None:
            ParticleSwarmOptimizer._shared_calculator = SmokeShieldingCalculator()
        calculator = ParticleSwarmOptimizer._shared_calculator
        
        # 检查几个关键时刻的接近程度
        test_times = [t_explode + i for i in [1, 3, 5, 8, 10, 15]]
        test_times = [t for t in test_times if t <= t_explode + calculator.effective_duration]
        
        if not test_times:
            return 0.0
        
        for t in test_times:
            if t > t_explode + calculator.effective_duration:
                continue
                
            P_missile = calculator._get_missile_position(t)
            C_smoke = calculator._get_smoke_cloud_center(t, alpha, v_D, t_drop, t_explode)
            
            if C_smoke is not None:
                # 导弹-烟幕距离接近度 (40%)
                distance = np.linalg.norm(P_missile - C_smoke)
                distance_score = max(0, 1 - distance / (3 * calculator.effective_radius))
                
                # 导弹-目标连线与烟幕的几何关系 (40%)
                line_distance = calculator._distance_point_to_line(P_missile, calculator.P_target, C_smoke)
                line_score = max(0, 1 - line_distance / (2 * calculator.effective_radius))
                
                # 时间窗口合理性 (20%)
                time_score = 1.0 if 2 <= t_explode <= 12 else 0.3
                
                combined_score = (0.4 * distance_score + 0.4 * line_score + 0.2 * time_score)
                scores.append(combined_score)
        
        return max(scores) if scores else 0.0
    
    def calculate_proximity_to_shielding(self, alpha, v_D, t_drop, t_explode):
        """计算接近遮蔽的程度，返回[0,1]"""
        scores = []
        
        # 检查几个关键时刻的接近程度
        test_times = [t_explode + i for i in [1, 3, 5, 8, 10, 15]]
        test_times = [t for t in test_times if t <= t_explode + self.calculator.effective_duration]
        
        if not test_times:
            return 0.0
        
        for t in test_times:
            if t > t_explode + self.calculator.effective_duration:
                continue
                
            P_missile = self.calculator._get_missile_position(t)
            C_smoke = self.calculator._get_smoke_cloud_center(t, alpha, v_D, t_drop, t_explode)
            
            if C_smoke is not None:
                # 导弹-烟幕距离接近度 (40%)
                distance = np.linalg.norm(P_missile - C_smoke)
                distance_score = max(0, 1 - distance / (3 * self.calculator.effective_radius))
                
                # 导弹-目标连线与烟幕的几何关系 (40%)
                line_distance = self.calculator._distance_point_to_line(P_missile, self.calculator.P_target, C_smoke)
                line_score = max(0, 1 - line_distance / (2 * self.calculator.effective_radius))
                
                # 时间窗口合理性 (20%)
                time_score = 1.0 if 2 <= t_explode <= 12 else 0.3
                
                combined_score = (0.4 * distance_score + 0.4 * line_score + 0.2 * time_score)
                scores.append(combined_score)
        
        return max(scores) if scores else 0.0
    
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
            
            # 确保时间约束
            if self.particles[i, 2] > self.particles[i, 3]:
                self.particles[i, 2], self.particles[i, 3] = self.particles[i, 3], self.particles[i, 2]
            
            # 初始化速度
            for j in range(self.dimension):
                range_val = self.bounds[j, 1] - self.bounds[j, 0]
                self.velocities[i, j] = random.uniform(-0.1 * range_val, 0.1 * range_val)
        
        # 并行计算初始适应度
        with concurrent.futures.ProcessPoolExecutor() as pool:
            scores = list(pool.map(ParticleSwarmOptimizer.static_objective_function, self.particles.tolist()))
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
            # 约束修复
            t_drop = self.particles[i, 2]
            t_explode = self.particles[i, 3]

            # 修复时序约束
            if t_drop > t_explode:
                t_drop, t_explode = t_explode, t_drop 
        
            # 修复时间差约束
            if (t_explode - t_drop) > 19.2:
                t_explode = t_drop + 19.2
                # 修复后可能超出上界，需要再次clip
                t_explode = np.clip(t_explode, self.bounds[3, 0], self.bounds[3, 1])

            # 更新粒子的时间参数
            self.particles[i, 2] = t_drop
            self.particles[i, 3] = t_explode
    
    def evaluate_and_update_best(self):
        """评估粒子适应度并更新最佳位置"""
        with concurrent.futures.ProcessPoolExecutor() as pool:
            scores = list(pool.map(ParticleSwarmOptimizer.static_objective_function, self.particles.tolist()))

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
        debug_print("开始粒子群优化...")
        
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
        alpha_opt, v_D_opt, t_drop_opt, t_explode_opt = self.global_best_position
        
        result = {
            'optimal_parameters': {
                'alpha_degrees': np.degrees(alpha_opt),
                'alpha_radians': alpha_opt,
                'v_D': v_D_opt,
                't_drop': t_drop_opt,
                't_explode': t_explode_opt
            },
            'optimal_shielding_duration': self.global_best_score,
            'optimization_history': self.history
        }
        
        return result

# 辅助函数
def save_pso_history(result_dict, filename="PSO_analysis.xlsx"):
    """将 PSO 优化结果保存到 Excel 文件"""
    optimal = result_dict["optimal_parameters"].copy()
    optimal["shielding_duration"] = result_dict["optimal_shielding_duration"]

    df_summary = pd.DataFrame([optimal])

    hist = result_dict["optimization_history"]
    df_history = pd.DataFrame({
        "iteration": range(1, len(hist["best_scores"]) + 1),
        "best_score": hist["best_scores"],
        "mean_score": hist["mean_scores"]
    })

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="summary", index=False)
        df_history.to_excel(writer, sheet_name="history", index=False)

    print(f"已保存优化结果到 {filename}")

def debug_print(*args, **kwargs):
    """调试打印函数，根据VERBOSE控制是否输出"""
    if VERBOSE:
        print(*args, **kwargs)

# 主函数
def run_optimization_problem2():
    """运行问题二的优化求解"""
    print("=" * 60)
    print("开始求解问题二：烟幕干扰弹投放策略优化")
    print("=" * 60)
    
    # 创建优化器
    optimizer = ParticleSwarmOptimizer(
        n_particles=200,      # 粒子数量
        n_iterations=100,     # 迭代次数
        w=0.9,              # 惯性权重
        c1=2.8,             # 个体学习因子
        c2=1.3              # 群体学习因子
    )
    result = optimizer.optimize()
    
    print("\n" + "=" * 60)
    print("问题二优化结果")
    print("=" * 60)
    
    opt_params = result['optimal_parameters']
    print(f"最优参数:")
    print(f"  航向角 α = {opt_params['alpha_degrees']:.2f}° ({opt_params['alpha_radians']:.4f} 弧度)")
    print(f"  无人机速度 v_D = {opt_params['v_D']:.2f} m/s")
    print(f"  投放时刻 t' = {opt_params['t_drop']:.3f} s")
    print(f"  起爆时刻 t'' = {opt_params['t_explode']:.3f} s")
    print(f"\n最大遮蔽时长: {result['optimal_shielding_duration']:.6f} 秒")
    save_pso_history(result, "PSO_analysis.xlsx")
    return result

if __name__ == "__main__":
    result = run_optimization_problem2()
