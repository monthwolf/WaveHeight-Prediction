#f <= f_avg,w = wmin + (wmax - wmin(f-fmin))/(f_avg-fmin)
#f > f_avg,w=wmax
#c1 = (c1i - c1f)(Iter - iter)/Iter + c1f
#c2 = (c2i - c2f)(Iter - iter)/Iter + c2f

import numpy as np
import matplotlib.pyplot as plt

# 改进粒子群优化算法(IPSO)实现
# IPSO通过自适应惯性权重和动态学习因子改进传统PSO算法

# 粒子群算法全局参数
SWARM_SIZE = 500      # 种群大小，即粒子的数量
PARTICLE_DIM = 10     # 每个粒子的维度，即解空间的维度
WEIGHT_MAX = 0.9      # 惯性权重最大值
WEIGHT_MIN = 0.4      # 惯性权重最小值
C1_INITIAL = 2.5      # 个体学习因子初始值
C2_FINAL = 2.5        # 社会学习因子终值
C1_FINAL = 0.5        # 个体学习因子终值
C2_INITIAL = 0.5      # 社会学习因子初始值
MAX_ITERATIONS = 200  # 最大迭代次数

# 自适应惯性权重计算函数
def calculate_adaptive_weight(fitness_values, particle_index):
    """
    计算每个粒子的自适应惯性权重，根据适应度值自动调整
    
    根据粒子当前适应值与种群平均适应值的关系动态调整惯性权重：
    1. 适应度低于平均值时，使用较小权重增强局部搜索能力
    2. 适应度高于平均值时，使用较大权重增强全局探索能力
    
    参数:
        fitness_values: 所有粒子的适应度值数组
        particle_index: 当前粒子的索引
        
    返回:
        adaptive_weight: 计算得到的自适应惯性权重
    """
    # 计算所有粒子适应度的平均值
    fitness_avg = np.mean(fitness_values)
    fitness_min = np.min(fitness_values)
    
    # 根据粒子适应度与平均适应度的关系动态调整权重
    if fitness_values[particle_index] <= fitness_avg:
        # 当粒子适应值低于平均适应值时，根据公式计算较小的权重以增强局部搜索
        adaptive_weight = WEIGHT_MIN + (WEIGHT_MAX - WEIGHT_MIN * (fitness_values[particle_index] - fitness_min)) / (fitness_avg - fitness_min)
    else:
        # 当粒子适应值高于平均适应值时，使用最大权重以增强全局探索
        adaptive_weight = WEIGHT_MAX

    return adaptive_weight

# 动态学习因子计算函数
def calculate_learning_factors(current_iter):
    """
    根据当前迭代次数动态计算学习因子c1和c2
    
    随着迭代进行，个体学习因子c1逐渐减小，社会学习因子c2逐渐增大，
    实现从全局探索到局部开发的平衡转换
    
    参数:
        current_iter: 当前迭代次数
        
    返回:
        c1, c2: 当前迭代轮次的学习因子值
    """
    # 计算个体学习因子c1，随迭代次数增加而减小
    c1 = (C1_INITIAL - C1_FINAL) * (MAX_ITERATIONS - current_iter) / MAX_ITERATIONS + C1_FINAL
    # 计算社会学习因子c2，随迭代次数增加而增大
    c2 = (C2_INITIAL - C2_FINAL) * (MAX_ITERATIONS - current_iter) / MAX_ITERATIONS + C2_FINAL
    return c1, c2

# 生成随机位置或速度的函数
def generate_random_value():
    """
    生成随机的粒子位置或速度值
    
    返回:
        random_value: (-1,1)范围内的随机数
    """
    return (np.random.rand() - 0.5) * 2  # 返回(-1,1)范围内的随机数

# 初始化粒子群位置和速度
def initialize_swarm():
    """
    初始化粒子群的位置和速度矩阵
    
    随机初始化每个粒子的位置和速度，作为优化算法的起点
    
    返回:
        velocity_matrix: 粒子速度矩阵，大小为[SWARM_SIZE, PARTICLE_DIM]
        position_matrix: 粒子位置矩阵，大小为[SWARM_SIZE, PARTICLE_DIM]
    """
    # 创建速度和位置矩阵
    velocity_matrix = np.zeros((SWARM_SIZE, PARTICLE_DIM))
    position_matrix = np.zeros((SWARM_SIZE, PARTICLE_DIM))
    
    # 随机初始化每个粒子的速度和位置
    for i in range(SWARM_SIZE):
        for j in range(PARTICLE_DIM):
            velocity_matrix[i][j] = generate_random_value()
            position_matrix[i][j] = generate_random_value()
            
    return velocity_matrix, position_matrix
    
# 计算每个粒子的适应度值（优化目标函数）
def calculate_fitness(position_matrix):
    """
    计算每个粒子的适应度值
    
    这里使用粒子位置与预设最优位置的差的绝对值作为评价指标
    适应度值越小表示解越优
    
    参数:
        position_matrix: 粒子位置矩阵
        
    返回:
        fitness_values: 所有粒子的适应度值数组
    """
    # 设定目标位置，在实际应用中这通常是待优化问题的最优解或期望解
    target_position = np.ones(PARTICLE_DIM) * 2.3
    
    # 计算每个粒子与目标位置的距离作为适应度
    fitness_values = np.zeros(SWARM_SIZE)
    for i in range(SWARM_SIZE):
        # 计算粒子当前位置与目标位置的偏差
        position_error = position_matrix[i] - target_position
        # 将各维度上的绝对偏差累加作为总的适应度值
        fitness_values[i] = np.sum(np.abs(position_error))
        
    return fitness_values

# 获取全局最优粒子及其适应度
def find_global_best(fitness_values, position_matrix):
    """
    获取当前种群中适应度最小（最优）的粒子及其适应度值
    
    参数:
        fitness_values: 所有粒子的适应度值数组
        position_matrix: 粒子位置矩阵
        
    返回:
        global_best_position: 全局最优粒子的位置
        global_best_fitness: 全局最优粒子的适应度值
    """
    # 找出适应度最小的粒子索引
    best_index = np.argmin(fitness_values)
    # 获取全局最优适应度值
    global_best_fitness = fitness_values[best_index]
    # 获取全局最优粒子位置
    global_best_position = position_matrix[best_index].copy()
    
    return global_best_position, global_best_fitness

# 粒子群算法主函数
def run_ipso_algorithm():
    """
    执行改进的粒子群优化算法
    
    实现了自适应惯性权重和动态学习因子的IPSO算法，用于解决优化问题
    
    返回:
        global_best_position: 找到的全局最优解
        global_best_fitness: 全局最优解的适应度值
        iterations: 实际迭代次数
    """
    # 初始化粒子群
    velocity_matrix, position_matrix = initialize_swarm()
    
    # 计算初始适应度
    fitness_values = calculate_fitness(position_matrix)
    
    # 初始化全局最优解和个体最优解
    global_best_position, global_best_fitness = find_global_best(fitness_values, position_matrix)
    personal_best_positions = position_matrix.copy()  # 个体历史最优位置
    personal_best_fitness = fitness_values.copy()     # 个体历史最优适应度
    
    # 记录优化过程中的适应度变化
    fitness_history = [global_best_fitness]
    
    # 迭代优化过程
    for iteration in range(MAX_ITERATIONS):
        # 收敛条件：当全局最优适应度小于阈值时提前结束
        if global_best_fitness <= 0.001:
            print(f"已收敛，在第{iteration}次迭代达到目标精度")
            break
            
        # 更新每个粒子的速度和位置
        for i in range(SWARM_SIZE):
            # 计算自适应惯性权重
            weight = calculate_adaptive_weight(fitness_values, i)
            # 计算动态学习因子
            c1, c2 = calculate_learning_factors(iteration)
            
            for j in range(PARTICLE_DIM):
                # 计算认知项：粒子向个体历史最优位置学习的部分
                cognitive_component = c1 * np.random.rand() * (personal_best_positions[i][j] - position_matrix[i][j])
                # 计算社会项：粒子向群体最优位置学习的部分
                social_component = c2 * np.random.rand() * (global_best_position[j] - position_matrix[i][j])
                
                # 更新粒子速度：惯性项 + 认知项 + 社会项
                velocity_matrix[i][j] = weight * velocity_matrix[i][j] + cognitive_component + social_component
                
                # 更新粒子位置
                position_matrix[i][j] = position_matrix[i][j] + velocity_matrix[i][j]
        
        # 重新计算适应度
        fitness_values = calculate_fitness(position_matrix)
        
        # 更新全局最优解
        new_global_best_position, new_global_best_fitness = find_global_best(fitness_values, position_matrix)
        if new_global_best_fitness < global_best_fitness:
            global_best_fitness = new_global_best_fitness
            global_best_position = new_global_best_position.copy()
            
        fitness_history.append(global_best_fitness)
        
        # 更新个体最优解
        for i in range(SWARM_SIZE):
            if fitness_values[i] < personal_best_fitness[i]:
                personal_best_fitness[i] = fitness_values[i]
                personal_best_positions[i] = position_matrix[i].copy()
    
    # 绘制优化过程中的适应度变化
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history, 'b-', linewidth=2)
    plt.title('IPSO Optimization Progress')
    plt.xlabel('Iteration')
    plt.ylabel('Global Best Fitness')
    plt.grid(True)
    plt.show()
    
    return global_best_position, global_best_fitness, iteration + 1

# 主程序入口
if __name__ == "__main__":
    print("开始执行改进粒子群优化算法(IPSO)...")
    # 执行IPSO算法
    best_solution, best_fitness, iterations = run_ipso_algorithm()
    
    # 输出优化结果
    print("\n优化结果:")
    print(f"最优解: {best_solution}")
    print(f"最优适应度: {best_fitness:.6f}")
    print(f"迭代次数: {iterations}")
