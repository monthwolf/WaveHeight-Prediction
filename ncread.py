import netCDF4 as nc
import math

# 定义数据文件常量和所需变量
DATA_FILENAME = 'wavewindF.nc'  # 数据文件包含渤海2016年海浪和风场数据
PRESSURE_VAR = 'msl'            # 平均海平面气压变量名
U_WIND_VAR = 'u10'              # 10米高度的东西向风速分量变量名
V_WIND_VAR = 'v10'              # 10米高度的南北向风速分量变量名
WAVE_HEIGHT_VAR = 'swh'         # 有效波高变量名

# 使用netCDF4库打开数据文件
dataset = nc.Dataset(DATA_FILENAME)   

# 从数据集中提取各个变量数据
pressure_data = dataset[PRESSURE_VAR]     # 气压数据
u_wind_data = dataset[U_WIND_VAR]         # 东西向风速
v_wind_data = dataset[V_WIND_VAR]         # 南北向风速
wave_height_data = dataset[WAVE_HEIGHT_VAR]  # 海浪高度

# 创建用于存储处理后数据的列表
pressure_values = []        # 存储气压数据
wind_speed_values = []      # 存储风速大小(合成风速)
wave_height_values = []     # 存储海浪高度

# 循环提取并处理数据
# 数据集的结构为：时间(1464) x 纬度(3) x 经度(2)
TIME_STEPS = 1464   # 时间步数
LAT_POINTS = 3      # 纬度点数
LON_POINTS = 2      # 经度点数

for time_idx in range(TIME_STEPS):    # 遍历时间维度
    for lat_idx in range(LAT_POINTS):  # 遍历纬度维度
        for lon_idx in range(LON_POINTS):  # 遍历经度维度
            # 提取气压数据
            pressure_values.append(pressure_data[time_idx][lat_idx][lon_idx])
            
            # 计算合成风速，通过东西向风速和南北向风速的平方和再开方
            u_component = u_wind_data[time_idx][lat_idx][lon_idx]  
            v_component = v_wind_data[time_idx][lat_idx][lon_idx]
            wind_speed = math.sqrt(u_component**2 + v_component**2)
            wind_speed_values.append(wind_speed)
            
            # 提取海浪高度数据
            wave_height_values.append(wave_height_data[time_idx][lat_idx][lon_idx])

# 导出变量，使其可以被其他模块导入
msl = pressure_values       # 兼容旧代码的变量名
uv10 = wind_speed_values    # 兼容旧代码的变量名
swh = wave_height_values    # 兼容旧代码的变量名
