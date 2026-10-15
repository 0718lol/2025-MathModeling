import numpy as np
import pandas as pd

# 物理参数
g = 9.80665
missile_speed = 300
drone_speed = 120
smoke_fall_speed = 3
effective_radius = 10
effective_duration = 20

# 时间参数
t_drop = 1.5
interval = 3.6
t_exp = t_drop + interval

# 初始位置
P_M1_0 = np.array([20000, 0, 2000])
P_FY1_0 = np.array([17800, 0, 1800])
P_target = np.array([0, 0, 0])

# 真目标圆柱体参数
C_base = np.array([0, 200, 0])
C_top = np.array([0, 200, 10])
R_cyl = 7
H_cyl = 10

# 导弹运动参数
d_M1 = P_target - P_M1_0
distance_M1 = np.linalg.norm(d_M1)
u_M1 = d_M1 / distance_M1
v_M1 = missile_speed * u_M1

print(f"导弹M1运动参数：")
print(f"  初始位置: {P_M1_0}")
print(f"  飞行方向: {u_M1}")
print(f"  速度向量: {v_M1}")
print(f"  x方向速度分量: {v_M1[0]:.2f} m/s")
print(f"  z方向速度分量: {v_M1[2]:.2f} m/s")
print(f"  到假目标距离: {distance_M1:.2f} m")
print(f"  到达假目标时间: {distance_M1/missile_speed:.2f} s\n")

# 无人机运动参数 - 朝原点方向水平飞行
direction_to_target_horizontal = np.array([P_target[0] - P_FY1_0[0], P_target[1] - P_FY1_0[1], 0])
u_FY1 = direction_to_target_horizontal / np.linalg.norm(direction_to_target_horizontal)
v_FY1 = drone_speed * u_FY1

print(f"无人机FY1运动参数：")
print(f"  初始位置: {P_FY1_0}")
print(f"  飞行方向: {u_FY1}")
print(f"  速度向量: {v_FY1}")
print(f"  水平飞行速度: {drone_speed} m/s\n")

# 计算烟幕弹投放和起爆位置
P_drop = P_FY1_0 + v_FY1 * t_drop
print(f"烟幕弹投放参数：")
print(f"  投放时间: {t_drop} s")
print(f"  投放位置: {P_drop}")

dt_fall = t_exp - t_drop
P_exp = P_drop + v_FY1 * dt_fall + np.array([0, 0, -0.5 * g * dt_fall**2])
print(f"  起爆时间: {t_exp} s")
print(f"  起爆位置: {P_exp}\n")

# 导弹在起爆时刻的位置
P_missile_at_exp = P_M1_0 + v_M1 * t_exp
print(f"=== 起爆时刻导弹位置 ===")
print(f"导弹在烟幕弹起爆时({t_exp}s)的坐标: {P_missile_at_exp}\n")

# 分析导弹与烟幕弹的相对位置
smoke_exp_x = P_exp[0]
smoke_exp_z = P_exp[2]

t_missile_at_smoke_x = (smoke_exp_x - P_M1_0[0]) / v_M1[0]
P_missile_at_smoke_x = P_M1_0 + v_M1 * t_missile_at_smoke_x
missile_height_at_smoke_x = P_missile_at_smoke_x[2]

print(f"=== 导弹与烟幕弹高度比较 ===")
print(f"烟幕弹起爆时的x坐标: {smoke_exp_x:.2f} m")
print(f"导弹运动到该x坐标的时间: {t_missile_at_smoke_x:.2f} s")
print(f"导弹在该x坐标时的高度: {missile_height_at_smoke_x:.2f} m")
print(f"烟幕弹起爆时的高度: {smoke_exp_z:.2f} m")

height_difference = smoke_exp_z - missile_height_at_smoke_x
if height_difference > 0:
    print(f"烟幕弹在导弹轨迹上方 {height_difference:.2f} m")
elif height_difference < 0:
    print(f"烟幕弹在导弹轨迹下方 {abs(height_difference):.2f} m")
else:
    print(f"烟幕弹与导弹轨迹在同一高度")
print()

# 计算云团降落时间
target_height = 10
explosion_height = P_exp[2]

print(f"=== 云团降落分析 ===")
print(f"起爆位置高度: {explosion_height:.2f} m")

fall_distance = explosion_height - target_height

if fall_distance > 0:
    fall_time = fall_distance / smoke_fall_speed
    time_at_10m = t_exp + fall_time
    
    print(f"需要下降距离: {fall_distance:.2f} m")
    print(f"下降时间: {fall_time:.2f} s")
    print(f"云团中心降落到离地面10m时的总用时: {time_at_10m:.2f} s\n")
else:
    print(f"起爆位置已低于10m高度")
    time_at_10m = t_exp
    print(f"云团中心降落到离地面10m时的总用时: {time_at_10m:.2f} s\n")

def get_missile_position(t):
    """计算指定时刻导弹位置"""
    return P_M1_0 + v_M1 * t

def get_smoke_center(t):
    """计算指定时刻烟幕中心位置"""
    if t < t_drop:
        return None
    elif t < t_exp:
        # 自由落体阶段
        dt_current = t - t_drop
        return P_drop + v_FY1 * dt_current + np.array([0, 0, -0.5 * g * dt_current**2])
    else:
        # 云团下沉阶段
        dt_fall_current = t_exp - t_drop
        dt_sink = t - t_exp
        P_exp_calc = P_drop + v_FY1 * dt_fall_current + np.array([0, 0, -0.5 * g * dt_fall_current**2])
        return P_exp_calc + np.array([0, 0, -smoke_fall_speed * dt_sink])

def is_missile_in_smoke(t):
    """判断指定时刻导弹是否在烟幕球体内"""
    if t < t_exp:
        return False
    
    P_missile = get_missile_position(t)
    C_smoke = get_smoke_center(t)
    
    if C_smoke is None:
        return False
    
    distance = np.linalg.norm(P_missile - C_smoke)
    return distance <= effective_radius

def is_fully_shielded(t):
    """判断指定时刻圆柱体目标是否被完全遮挡"""
    if t < t_exp:
        return False
    
    P_missile = get_missile_position(t)
    C_smoke = get_smoke_center(t)
    
    if C_smoke is None:
        return False
    
    # 对圆柱体进行采样
    thetas = np.linspace(0.0, 2.0 * np.pi, 100, endpoint=False)
    
    x_circle = R_cyl * np.cos(thetas)
    y_circle = C_base[1] + R_cyl * np.sin(thetas)
    
    # 顶圆周
    z_top = np.full_like(x_circle, C_top[2])
    P_top = np.column_stack((x_circle, y_circle, z_top))
    
    # 底圆周
    z_bot = np.full_like(x_circle, C_base[2])
    P_bot = np.column_stack((x_circle, y_circle, z_bot))
    
    target_points = np.vstack((P_top, P_bot))
    R_sq = effective_radius * effective_radius
    
    # 检查所有采样点是否被遮挡
    for T in target_points:
        v = T - P_missile
        v_len_sq = np.dot(v, v)
        
        if v_len_sq < 1e-10:  # 避免数值误差
            # 导弹和目标重合的特殊情况
            distance = np.linalg.norm(C_smoke - P_missile)
        else:
            # 计算投影参数
            w = C_smoke - P_missile
            t_proj = np.dot(w, v) / v_len_sq
            
            # 计算烟幕中心到直线的最近点
            closest_point = P_missile + t_proj * v
            
            # 烟幕中心到直线的距离
            distance = np.linalg.norm(C_smoke - closest_point)
        
        # 烟幕中心到直线距离小于半径
        if distance > effective_radius:
            return False
        
        # 方位关系判断（仅当导弹不在烟幕内时）
        if v_len_sq >= 1e-10:
            inside_smoke = np.linalg.norm(P_missile - C_smoke) <= effective_radius
            
            if not inside_smoke:
                if not (0.0 <= t_proj <= 1.0):
                    return False

    return True  # 所有点都被遮挡

def binary_search_transition(t_start, t_end, condition_func, search_direction='first_true', tolerance=1e-8):
    """通用二分法搜索函数，用于找到条件函数状态转换的精确时刻"""
    left, right = t_start, t_end
    while right - left > tolerance:
        mid = (left + right) / 2
        if condition_func(mid):
            if search_direction == 'first_true':
                right = mid  # 向左搜索第一个真值点
            else:  
                left = mid   # 向右搜索最后一个真值点
        else:
            if search_direction == 'first_true':
                left = mid   # 向右搜索第一个真值点
            else:  
                right = mid  # 向左搜索最后一个真值点
    return (left + right) / 2

# 遮挡判定分析
t_start = 5.1
effective_duration = 20.0
t_end = t_start + effective_duration

print("\n=== 开始进行遮挡判定分析===")
print(f"分析时间窗口: 从 {t_start:.2f}s 到 {t_end:.2f}s")

# 粗略搜索找到所有关键时间段
dt_search = 0.1  # 粗略搜索步长
current_time = t_start

# 寻找所有关键时间段
shielding_start_rough = None
shielding_end_rough = None
missile_enter_rough = None
missile_exit_rough = None

last_shielded = False
last_in_smoke = False

print("\n--- 粗略搜索所有关键时间段 ---")

while current_time < t_end:
    current_shielded = is_fully_shielded(current_time)
    current_in_smoke = is_missile_in_smoke(current_time)
    
    # 检测遮挡开始
    if not last_shielded and current_shielded and shielding_start_rough is None:
        shielding_start_rough = (current_time - dt_search, current_time)
        print(f"检测到遮挡开始区间: [{shielding_start_rough[0]:.2f}s, {shielding_start_rough[1]:.2f}s]")
    
    # 检测遮挡结束
    if last_shielded and not current_shielded and shielding_end_rough is None:
        shielding_end_rough = (current_time - dt_search, current_time)
        print(f"检测到遮挡结束区间: [{shielding_end_rough[0]:.2f}s, {shielding_end_rough[1]:.2f}s]")
    
    # 检测导弹穿入烟幕
    if not last_in_smoke and current_in_smoke and missile_enter_rough is None:
        missile_enter_rough = (current_time - dt_search, current_time)
        print(f"检测到导弹穿入烟幕区间: [{missile_enter_rough[0]:.2f}s, {missile_enter_rough[1]:.2f}s]")
    
    # 检测导弹穿出烟幕
    if last_in_smoke and not current_in_smoke and missile_exit_rough is None:
        missile_exit_rough = (current_time - dt_search, current_time)
        print(f"检测到导弹穿出烟幕区间: [{missile_exit_rough[0]:.2f}s, {missile_exit_rough[1]:.2f}s]")
    
    last_shielded = current_shielded
    last_in_smoke = current_in_smoke
    current_time += dt_search

print("\n--- 精确计算所有关键时刻 ---")

# 使用二分法精确计算遮挡关键时刻
first_shielding_time = None
shielding_end_time = None

if shielding_start_rough is not None:
    first_shielding_time = binary_search_transition(
        shielding_start_rough[0], shielding_start_rough[1], 
        is_fully_shielded, 'first_true'
    )
    print(f"首次完全遮挡时刻: {first_shielding_time:.8f}s")

if shielding_end_rough is not None:
    shielding_end_time = binary_search_transition(
        shielding_end_rough[0], shielding_end_rough[1], 
        is_fully_shielded, 'last_true'
    )
    print(f"遮挡结束时刻: {shielding_end_time:.8f}s")

# 使用二分法精确计算导弹穿越时刻
missile_enter_time = None
missile_exit_time = None

if missile_enter_rough is not None:
    missile_enter_time = binary_search_transition(
        missile_enter_rough[0], missile_enter_rough[1], 
        is_missile_in_smoke, 'first_true'
    )
    print(f"导弹穿入烟幕时刻: {missile_enter_time:.8f}s")

if missile_exit_rough is not None:
    missile_exit_time = binary_search_transition(
        missile_exit_rough[0], missile_exit_rough[1], 
        is_missile_in_smoke, 'last_true'
    )
    print(f"导弹穿出烟幕时刻: {missile_exit_time:.8f}s")

# 计算最终结果
print("\n=== 分析完成 ===")

if first_shielding_time is not None and shielding_end_time is not None:
    total_shielded_duration = shielding_end_time - first_shielding_time
    print(f"问题一：烟幕干扰弹对M1的有效遮蔽时长为: {total_shielded_duration:.8f} 秒")
    print(f"  - 首次完全遮挡时刻: {first_shielding_time:.8f}s")
    print(f"  - 遮挡结束时刻: {shielding_end_time:.8f}s")
else:
    print("未检测到完全遮挡时间段")

# 导弹穿越分析
print("\n=== 导弹穿越烟幕分析 ===")
if missile_enter_time is not None and missile_exit_time is not None:
    missile_smoke_duration = missile_exit_time - missile_enter_time
    print(f"导弹在烟幕中的时长: {missile_smoke_duration:.8f} 秒")
    print(f"  - 穿入时刻: {missile_enter_time:.8f}s")
    print(f"  - 穿出时刻: {missile_exit_time:.8f}s")
else:
    print("导弹未穿越烟幕球体")

# 最终遮挡时间段计算
print("\n=== 最终遮挡时间段计算 ===")
if first_shielding_time is not None and missile_exit_time is not None:
    final_shielding_duration = missile_exit_time - first_shielding_time
    print(f"最终有效遮蔽时长（穿出时刻 - 首次遮挡时刻）: {final_shielding_duration:.8f} 秒")
    print(f"  - 首次完全遮挡时刻: {first_shielding_time:.8f}s")
    print(f"  - 导弹穿出烟幕时刻: {missile_exit_time:.8f}s")
    print(f"  - 时间差: {final_shielding_duration:.8f}s")
else:
    print("无法计算最终遮挡时长（缺少首次遮挡时刻或导弹穿出时刻）")

