# 可视化层：专注于接收数据并生成模型训练图表，不关心数据怎么来的

import plotly.graph_objects as go
import numpy as np

def create_forecast_plot(model_type, dataset_name, original_data_scaled, train_predict, test_predict, future_predict, look_back, scaler):
    """
    新增参数: future_predict (numpy array or None)
    """
    # 反归一化原始数据
    original_data = scaler.inverse_transform(original_data_scaled)
    total_len = len(original_data)
    
    # 1. 准备训练集绘图数据 (保持原逻辑)
    if model_type in ["LSTM", "MLP"]:
        train_plot = np.empty_like(original_data_scaled)
        train_plot[:, :] = np.nan
        train_plot[look_back:len(train_predict)+look_back, :] = train_predict
        
        # 测试集索引计算
        test_start_idx = len(train_predict) + look_back
        test_end_idx = test_start_idx + len(test_predict)
        
    elif model_type in ["SARIMA", "ARIMA", "Exponential-Smoothing"]:
        train_plot = np.empty_like(original_data_scaled)
        train_plot[:, :] = np.nan
        train_plot[:len(train_predict), :] = train_predict
        
        test_start_idx = len(train_predict)
        test_end_idx = test_start_idx + len(test_predict)

    # 2. 准备测试集绘图数据
    test_plot = np.empty_like(original_data_scaled)
    test_plot[:, :] = np.nan
    
    # 简单的边界保护
    valid_end = min(test_end_idx, total_len)
    test_plot[test_start_idx:valid_end, :] = test_predict[:(valid_end-test_start_idx)]

    # 3. 绘图对象初始化
    fig = go.Figure()
    
    # 绘制基础层
    fig.add_trace(go.Scatter(y=original_data[:,0], mode='lines', name='真实数据', line=dict(color='lightgray')))
    fig.add_trace(go.Scatter(y=train_plot[:,0], mode='lines', name='训练集拟合', line=dict(color='blue')))
    fig.add_trace(go.Scatter(y=test_plot[:,0], mode='lines', name='测试集预测', line=dict(color='red', width=2)))
    
    # [新增] 4. 处理未来预测数据的绘制
    if future_predict is not None and len(future_predict) > 0:
        # X 轴坐标: 从原始数据最后一个点开始往后推
        # 为了视觉连续性，我们可以把测试集的最后一个点作为未来的起点
        
        # 这种方式生成的 X 轴是绝对索引： [100, 101, 102, ...]
        future_x = np.arange(total_len, total_len + len(future_predict))
        
        # 如果想让红线和绿线视觉上连接，可以在 future_predict 前面插一个点(测试集最后一个点)
        # 但为了逻辑简单，这里直接画，Plotly 在缩放时通常能看清
        
        fig.add_trace(go.Scatter(
            x=future_x,
            y=future_predict[:, 0],
            mode='lines+markers', # 加点方便看清步数
            name='未来预测 (Future)',
            line=dict(color='#00ff00', width=2, dash='dot'), # 绿色虚线
            marker=dict(size=4)
        ))

    # 设置布局
    fig.update_layout(
        title=f"时间序列预测结果 (含未来推演) - {dataset_name}",
        xaxis_title="时间步 (Time Step)",
        yaxis_title="数值 (Value)",
        template="plotly_dark",
        hovermode="x unified"
    )
    return fig


def create_data_preview_plot(dataset_name, df):
    """
    仅绘制原始数据的预览图
    :param df: pandas DataFrame, 包含原始数据
    """
    # 提取数据 (假设第一列是目标数据)
    data_values = df.iloc[:, 0].values

    fig = go.Figure()
    
    # 复用 create_forecast_plot 中的原始数据样式
    fig.add_trace(go.Scatter(
        y=data_values, 
        mode='lines', 
        name='真实数据', 
        line=dict(color='lightgray')
    ))
    
    # 复用统一的布局样式
    fig.update_layout(
        title=f"原始数据预览: {dataset_name}",
        xaxis_title="时间步 (Time Step)",
        yaxis_title="数值 (Value)",
        template="plotly_dark", # 保持暗色主题
        hovermode="x unified"
    )
    return fig