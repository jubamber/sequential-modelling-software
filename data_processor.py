
# 数据层：只负责数据的读取、清洗、转换。不涉及任何模型或UI代码

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import utils

def validate_and_load_local(file_path):
    """
    解析本地上传的 CSV/TXT 文件。
    支持：
    1. 自动识别逗号、空格、制表符等分隔符。
    2. 自动判断第一行是否为表头（Header）。
    3. 第一列作为索引：优先尝试转为时间，若非时间格式则打印警告并保留原样。
    4. 第二列作为数值：非数值行将被剔除。
    """
    try:
        # 1. 读取文件
        # sep=None + engine='python' 能够自动嗅探分隔符（逗号、空格等）
        # header=None: 先把所有内容读进来，后续手动判断哪一行是数据
        df_raw = pd.read_csv(file_path, sep=None, engine='python', header=None)
    except Exception as e:
        raise ValueError(f"文件读取失败，请检查文件格式是否损坏。\n底层错误: {e}")

    # 2. 列数校验
    if df_raw.shape[1] < 2:
        raise ValueError(f"识别到的列数不足（当前列数: {df_raw.shape[1]}），请确保数据至少包含两列（索引和数值），并使用一致的分隔符。")

    # 截取前两列：假设第一列是索引，第二列是数值
    df = df_raw.iloc[:, :2].copy()
    
    # 3. 智能判断第一行是否为表头
    # 逻辑：检查第二列（数值列）的第一行内容
    first_val_raw = df.iloc[0, 1]
    
    # 尝试将第一行第二列转为数字
    is_header = False
    try:
        pd.to_numeric(first_val_raw, float_precision='high')
    except (ValueError, TypeError):
        # 如果转换失败（例如它是字符串 "Value"），则认为是表头
        is_header = True
    
    # 如果第一行是表头，删去第一行
    if is_header:
        df = df.iloc[1:].copy()

    # 重新设置列名，方便后续处理
    df.columns = ['_raw_index', 'Value']

    # 4. 数据清洗：处理数值列
    # 强制转为数值，无法转换的变为 NaN
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    
    # 删除数值为空的行
    df.dropna(subset=['Value'], inplace=True)
    
    if df.empty:
        raise ValueError("清洗后有效数据为空，请检查文件内容是否包含有效数值。")

    # 5. 处理索引列 (Date/Index)
    raw_index_col = df['_raw_index']
    
    try:
        # 尝试转换为 datetime
        # errors='coerce' 会将无法转换的变为 NaT
        datetime_index = pd.to_datetime(raw_index_col, errors='coerce')
        
        # 校验转换成功率：如果超过一半的数据都无法转为时间，说明这可能不是时间列
        if datetime_index.notna().sum() < 0.5 * len(df):
            raise ValueError("Time conversion failed mostly")
            
        df.index = datetime_index
        df.index.name = 'Date'
        
        # 剔除时间转换失败的行（可选，保证时间序列的纯净性）
        # df = df[df.index.notna()] 
        
    except Exception:
        # 【修改点】不强制报错，而是打印警告并使用原始索引
        print("警告：第一列无法识别为标准时间格式(Date)，已保留原始索引。")
        df.index = raw_index_col
        df.index.name = 'Index'

    # 移除临时列
    df.drop(columns=['_raw_index'], inplace=True)

    # 6. 最终排序与输出
    try:
        df.sort_index(inplace=True)
    except TypeError:
        # 如果索引是混合类型（字符串+数字），排序可能会失败，此时忽略排序
        pass

    return df

def load_raw_data(dataset_name, local_file_path=None):
    """生成或加载原始数据 DataFrame"""
    if dataset_name == "📂 加载本地数据":
        if not local_file_path or not os.path.exists(local_file_path):
            raise ValueError("未找到上传的文件路径。")
        return validate_and_load_local(local_file_path)
    
    elif dataset_name == "Sine Wave (模拟)":
        x = np.linspace(0, 50, 1000)
        y = np.sin(x) + np.random.normal(0, 0.1, 1000) # 加点噪声
        # 【核心修改】：将 x 设为索引，明确列名为 Value
        # 这样 df.iloc[:, 0] 取到的就是 y (Value)，而不是 x
        df = pd.DataFrame(data=y, index=x, columns=['Value'])
        df.index.name = 'Date' # 保持索引名一致
    elif dataset_name == "AirPassengers (模拟)":
        # 模拟增长趋势 + 季节性
        x = np.linspace(0, 10, 1000)
        y = x * 0.5 + np.sin(x * 5) + np.random.normal(0, 0.2, 1000)
        # 【核心修改】：同上
        df = pd.DataFrame(data=y, index=x, columns=['Value'])
        df.index.name = 'Date'
    elif dataset_name == "AirPassengers":
        df = pd.read_csv(
            utils.get_resource_path("datasets/AirPassengers.csv"),
            header = 0, 
            parse_dates = [0], 
            names = ['Date', 'Value'], 
            index_col = 0
        )
    elif dataset_name == "Daily minimum temperatures in Melbourne":
        df = pd.read_csv(
            utils.get_resource_path("datasets/daily-minimum-temperatures-in-me.csv"), 
            on_bad_lines='skip', # 遇到格式错误的行直接跳过
            #skipfooter=1,       # 忽略最后一行
            header = 0, 
            parse_dates = [0], 
            names = ['Date', 'Value'], 
            index_col = 0
        )
    elif dataset_name == "Sunspots":
        df = pd.read_csv(
            utils.get_resource_path("datasets/sunspots.csv"), 
            header=0,                  # 指定第0行为表头
            usecols=[1, 2],            # 关键点：只读取第2列(Date)和第3列(Value)，忽略第1列的序列号
            parse_dates=[0],           # 解析读取后的第1列（即Date）为时间格式
            names=['Date', 'Value'],   # 将读取的两列重命名为标准格式
            index_col=0                # 将读取后的第1列（即Date）设为索引
        )
    elif dataset_name == "Mauna Loa CO2 Weekly":
        df = pd.read_csv(
            utils.get_resource_path("datasets/co2_weekly_16Aug2025.txt"), # 请替换为你实际的文件名
            sep=r'\s+',                # 使用正则表达式匹配任意长度的空白字符作为分隔符
            comment='#',               # 忽略以 # 开头的注释行
            header=None,               # 不使用文件自带的表头（因为格式不标准）
            skiprows=2,                # 忽略注释行之后的两行文字标题 ("Start of week...", "(yr, mon...)")
                                       # 注意：如果读取报错，可能需要根据实际文件调整此数值，或结合 on_bad_lines='skip'
            names=['Year', 'Month', 'Day', 'Decimal', 'Value', 'Days', '1yr', '10yr', 'Since1800'], # 手动定义所有列名
            usecols=[0, 1, 2, 4],      # 只读取第0,1,2列(年月日)和第4列(CO2 ppm值)
            parse_dates={'Date': [0, 1, 2]}, # 将读取的前三列合并解析名为 'Date' 的时间列
            index_col='Date',          # 将解析出的 'Date' 列设为索引
            na_values=[-999.99]        # 将 -999.99 识别为 NaN (空值)
        )
        # 2. 确保索引排序（插值前必须保证时间是顺序的）
        df = df.sort_index()

        # 3. 处理时间轴断裂（可选但推荐）
        # 如果数据中不仅有 NaN，还完全缺失了某些周的行，需要先重采样生成连续的时间轴
        # 'W' 代表按周重采样，根据实际数据也可以用 'W-SAT' (周六) 等
        df = df.resample('W').asfreq() 

        # 4. 使用插值填充中间缺失值
        # method='time' 会根据时间索引的距离进行插值，比 'linear' 更适合时间序列
        df['Value'] = df['Value'].interpolate(method='time')

        # 5. 处理开头或结尾可能残留的 NaN (如果开头就是缺失值，插值无法填充)
        # 使用 bfill (向后填充) 处理开头，ffill (向前填充) 处理结尾
        df['Value'] = df['Value'].bfill().ffill()

        # 6. 再次确认是否还有空值（用于调试）
        if df['Value'].isnull().any():
            print("警告：数据中仍存在无法填充的空值")
    elif dataset_name == "Arctic Oscillation Dataset":
        df = pd.read_csv(
        utils.get_resource_path("datasets/monthly.ao.index.b50.current.ascii"),
        sep='\s+',                 # 关键点1：使用正则表达式匹配任意数量的空格作为分隔符
        header=None,               # 关键点2：原数据没有标题行，所以设为None
        names=['Year', 'Month', 'Value'], # 手动指定列名
        parse_dates={'Date': [0, 1]},     # 关键点3：将第0列(Year)和第1列(Month)合并解析名为'Date'
        index_col='Date'           # 将合并后的日期列设为索引
    )
    else:
        # 这里可以扩展为读取本地 CSV
        df = pd.DataFrame({'Date': [], 'Value': []})
    return df


def _create_dataset(dataset, look_back=1):
    """辅助函数：创建时序窗口"""
    # look_back = 用过去多少个时间步来预测下一个时间步，它是时间序列建模的窗口大小，look_back越大，模型可以看到更多历史信息，但特征维度也会增加
    X, Y = [], []
    for i in range(len(dataset)-look_back): 
        # TODO 选取多列特征
        a = dataset[i:(i+look_back), 0] # 只取第一列的，如果features多维的话需要修改
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

def process_data(df, look_back, split_ratio, model_type):
    """
    根据模型类型决定数据处理方式：
    - LSTM/MLP: 使用 look_back 构建滑动窗口 X, Y
    - ARIMA/SARIMA/ES: 直接返回完整序列，不构建窗口
    """
    # 读取数据
    data = df['Value'].values.reshape(-1, 1)
    
    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(data)
    
    # 计算切分点
    train_size = int(len(dataset) * split_ratio)
    
    # --- 分支逻辑开始 ---
    
    # A. 统计学模型 (ARIMA, SARIMA, ETS) -> 不使用 Lookback
    if model_type in ["ARIMA", "SARIMA", "Exponential-Smoothing"]:
        # 训练集：直接截取
        train = dataset[0:train_size, :]
        # 测试集：紧接训练集之后 (不需要重叠，因为不需要窗口)
        test = dataset[train_size:len(dataset), :]
        
        # 对于 ARIMA 类模型，sktime 只需要 Y (序列本身)，X (特征) 可以为空
        # 为了保持 dict 结构一致，我们将序列赋值给 Y_train/Y_test
        # X_train/X_test 设为 None 或空占位符
        
        return {
            "X_train": None, 
            "Y_train": train, # (n_train, 1)
            "X_test": None, 
            "Y_test": test,   # (n_test, 1)
            "scaler": scaler,
            "full_dataset_scaled": dataset
        }
        
    # B. 深度学习模型 (LSTM, MLP) -> 使用 Lookback
    else:
        # 训练集
        train = dataset[0:train_size, :]
        
        # 测试集：为了不丢失开头的 look_back 数据，我们需要回溯
        test = dataset[train_size - look_back : len(dataset), :]
        
        # 创建时序数据
        X_train, Y_train = _create_dataset(train, look_back)
        X_test, Y_test = _create_dataset(test, look_back)
        
        # Reshape
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        
        return {
            "X_train": X_train, 
            "Y_train": Y_train,
            "X_test": X_test, 
            "Y_test": Y_test,
            "scaler": scaler,
            "full_dataset_scaled": dataset
        }