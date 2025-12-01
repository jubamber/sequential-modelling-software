# 模型层：负责定义模型结构、训练过程和评估指标

import os
from datetime import datetime
import numpy as np
import pandas as pd
import joblib # # 新增：用于保存 sklearn/sktime 风格的模型
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sktime.forecasting.arima import ARIMA, AutoARIMA
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from keras.callbacks import Callback
from sklearn.metrics import mean_absolute_error, mean_squared_error

import config


# 定义自定义回调函数
class GradioProgressCallback(Callback):
    """自定义回调，用于连接 Keras 训练过程和 Gradio 进度条"""
    def __init__(self, progress, total_epochs, start_progress=0.0, end_progress=1):
        super().__init__()
        self.progress = progress
        self.total_epochs = total_epochs
        self.start_progress = start_progress
        self.end_progress = end_progress

    def on_epoch_end(self, epoch, logs=None):
        current_epoch = epoch + 1 # epoch 是从 0 开始的，所以 +1
        fraction = current_epoch / self.total_epochs # 计算在训练阶段的相对进度 (0.0 -> 1.0)
        global_progress = self.start_progress + fraction * (self.end_progress - self.start_progress) # 映射到全局进度条范围
        
        # 获取当前的 loss 或 acc 用于显示
        loss = logs.get('loss', 0)
        #acc = logs.get('accuracy', 0)
        
        # 更新 Gradio 进度条
        self.progress(
            global_progress, 
            #desc=f"训练中: Epoch {current_epoch}/{self.total_epochs} - Loss: {loss:.4f} - Acc: {acc:.4f}"
            desc=f"训练中: Epoch {current_epoch}/{self.total_epochs} - Loss: {loss:.4f}"
        )


def build_model(model_type, look_back, p, d, q, auto_arima, P, D, Q, s):
    if model_type == "LSTM":
        model = Sequential()
        model.add(LSTM(50, input_shape=(1, look_back)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

    elif model_type == "MLP":
        model = Sequential()
        # 简单的 Dense 层作为对比
        model.add(Dense(10, input_shape=(1, look_back), activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

    elif model_type == "ARIMA":
        if auto_arima:
            #使用 AutoARIMA 自动寻找最优参数
            # sp: 季节性周期 (Seasonal Period)。
            #     如果你是月度数据且有年度周期，sp=12；季度数据 sp=4；
            #     如果是纯每日数据无明显周期或不想处理季节性，sp=1。
            # stepwise=True: 使用逐步搜索算法（速度快），False 则进行更详尽的搜索（速度慢但可能更准）。
            print("正在使用 AutoARIMA 自动搜索最优参数，这可能需要一些时间...")
            model = AutoARIMA(
                sp=1,                # 如果你的数据有周期性（如按月），请改为 12
                d=None,              # 让模型自动推断差分阶数 d
                start_p=0, max_p=10,  # p 的搜索范围
                start_q=0, max_q=10,  # q 的搜索范围
                suppress_warnings=True,
                stepwise=True        # 推荐 True，否则非常慢
            )
        else:
            # 使用自动 ARIMA 或指定参数，这里演示使用基础配置
            # suppress_warnings=True 防止在 UI 中弹出大量收敛警告
            model = ARIMA(order=(p, d, q), seasonal_order=(0, 0, 0, 0), suppress_warnings=True)

    elif model_type == "SARIMA":
        if auto_arima:
            #使用 AutoARIMA 自动寻找最优参数
            # sp: 季节性周期 (Seasonal Period)。
            #     如果你是月度数据且有年度周期，sp=12；季度数据 sp=4；
            #     如果是纯每日数据无明显周期或不想处理季节性，sp=1。
            # stepwise=True: 使用逐步搜索算法（速度快），False 则进行更详尽的搜索（速度慢但可能更准）。
            print("正在使用 AutoARIMA 自动搜索最优参数，这可能需要一些时间...")
            model = AutoARIMA(
                sp=s,                # 如果你的数据有周期性（如按月），请改为 12
                d=None,              # 让模型自动推断差分阶数 d
                start_p=0, max_p=10,  # p 的搜索范围
                start_q=0, max_q=10,  # q 的搜索范围
                suppress_warnings=True,
                stepwise=True        # 推荐 True，否则非常慢
            )
        else:
            # 使用自动 ARIMA 或指定参数，这里演示使用基础配置
            # suppress_warnings=True 防止在 UI 中弹出大量收敛警告
            model = ARIMA(order=(p, d, q), seasonal_order=(P, D, Q, s), suppress_warnings=True)
        
    elif model_type == "Exponential-Smoothing":
        # 指数平滑，设置 trend='add' 等参数
        model = ExponentialSmoothing(trend="add", seasonal=None, sp=1)
    
    return model


def train_model(model_type, model, X_train, Y_train, epochs, batch_size, progress_callback, dataset_name):
    save_dir = config.MODEL_SAVE_DIR
    # 生成当前时间戳字符串 (格式例如: 20231027103055)
    # %Y=年, %m=月, %d=日, %H=时, %M=分, %S=秒
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # 判断模型类型
    if model_type in ["LSTM", "MLP"]:
        # --- Keras 训练逻辑 ---
        history = model.fit(
            X_train, Y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[progress_callback]
        )
        save_path = os.path.join(save_dir, f"{model_type}_model_{timestamp}_{dataset_name}.keras")
        model.save(save_path)
    
    elif model_type in ["ARIMA", "SARIMA", "Exponential-Smoothing"]:
        # --- sktime 训练逻辑 ---
        # sktime 不需要 epochs 和 batch_size，也不支持 Keras 的 callback
        # 它的 fit 只需要 y (目标序列)，通常是一维数组
        
        # 更新进度条提示（因为没有 epoch 循环，直接更新一次）
        if progress_callback:
            progress_callback.progress(progress_callback.start_progress, desc="正在拟合统计模型 (Fitting)...")
        
        # Y_train 这里的形状通常是 (n_samples, 1)，sktime 需要 (n_samples,)
        y_flat = Y_train.flatten()
        
        # 训练
        model.fit(y=y_flat)
        
        # 更新进度条到结束
        if progress_callback:
            progress_callback.progress(progress_callback.end_progress, desc="拟合完成")
            
        # 保存模型 (Pickle/Joblib)
        save_path = os.path.join(save_dir, f"{model_type}_model_{timestamp}_{dataset_name}.pkl")
        joblib.dump(model, save_path)

    return save_path


def evaluate_model(model_type, model, X_train, X_test, Y_train, Y_test, scaler, future_steps=0):
    """预测并计算反归一化后的指标"""
    train_predict = None
    test_predict = None
    future_predict = None # 用于存储未来数据

    if model_type in ["LSTM", "MLP"]:
        # --- 1. 训练集预测 (保持单步预测，查看拟合程度) ---
        train_predict = model.predict(X_train, verbose=0)
        
        # --- 2. 测试集预测 (修改为：递归多步预测) ---
        # 说明：由于 X_test 包含了未来的真实数据（Teacher Forcing），
        # 真正的多步预测应该只使用 X_test 的第一个窗口，之后的输入都来源于模型自己的预测。
        
        test_predictions_list = []
        
        # 获取测试集的第一个输入窗口
        # X_test shape: (样本数, 1, look_back)
        # 我们取第0个样本，作为递归的起始点
        # current_input shape: (1, 1, look_back)
        current_input = X_test[0].reshape(1, 1, -1)
        
        # 循环次数等于测试集的长度
        steps_to_predict = len(X_test) + future_steps
        
        print(f"开始 {model_type} 递归预测，总步数: {steps_to_predict} (测试集: {len(X_test)}, 预测: {future_steps})")
        
        for _ in range(steps_to_predict):
            # (A) 预测下一步
            # pred shape: (1, 1)
            pred = model.predict(current_input, verbose=0)
            
            # (B) 存入结果列表
            pred_value = pred[0, 0]
            test_predictions_list.append(pred_value)
            
            # (C) 构造下一次的输入 (滚动窗口)
            # current_input[0, 0] 是一个长度为 look_back 的向量
            # 我们丢弃最旧的一个数据 (index 0)，在末尾追加最新的预测值
            old_window = current_input[0, 0]
            new_window = np.append(old_window[1:], pred_value)
            
            # (D) 重塑回模型需要的 (1, 1, look_back)
            current_input = new_window.reshape(1, 1, -1)
            
        # 转换为数组
        full_pred = np.array(test_predictions_list).reshape(-1, 1)
        
        # [修改] 切分 测试集部分 和 未来部分
        test_len = len(X_test)
        test_predict = full_pred[:test_len]      # 对齐 Y_test 的部分
        
        if future_steps > 0:
            future_predict = full_pred[test_len:] # 纯未来部分

        if train_predict.ndim == 3:
            train_predict = train_predict.reshape(-1, 1)

    elif model_type in ["ARIMA", "SARIMA", "Exponential-Smoothing"]:
        # --- sktime (ARIMA/Exponential Smoothing) 全局预测 ---
        
        # sktime 的逻辑：
        # 训练结束后，模型的“当前时间”是训练集的最后一个点 (t=0)
        # 负数表示“过去”（训练集范围），正数表示“未来”（测试集范围）
        
        # 1. 构造一个连续的时间视窗 (Forecasting Horizon)
        # 范围：从 [训练集开始] 到 [测试集结束]
        # 训练集长度: len(Y_train)
        # 测试集长度: len(Y_test)
        # 起点 (相对于训练结束): -len(Y_train) + 1  (例如训练100个，起点就是 -99)
        # 终点 (相对于训练结束): len(Y_test)       (例如测试20个，终点就是 20)
        
        # 1. 尝试计算全量预测，但增加错误捕获
        try:
            # 计算预测视野
            # 注意：如果这是加载的旧模型，-len(Y_train) 可能会超出旧模型的记忆范围
            fh_full = np.arange(-len(Y_train) + 1, len(Y_test) + future_steps + 1)
            
            full_predict = model.predict(fh=fh_full)
            
        except ValueError as e:
            # 【核心修复】捕获 "earlier to train starting point" 错误
            if "earlier to train starting point" in str(e):
                raise ValueError(
                    f"⚠️ 模型不匹配错误：\n"
                    f"您加载的 {model_type} 模型是基于旧数据训练的（长度可能较短或索引不同）。\n"
                    f"统计模型(ARIMA/SARIMA)无法直接用于结构/长度差异过大的新数据集。\n"
                    f"建议：请使用当初训练该模型时的同一数据集，或重新训练新模型。"
                )
            else:
                raise e # 抛出其他未知错误
        
        # 转换格式
        if hasattr(full_predict, "to_numpy"):
             full_predict = full_predict.to_numpy()
        full_predict = full_predict.reshape(-1, 1)
        
        # 3. 切分回 训练部分 和 测试部分
        # 为了计算评估指标 (MAE/RMSE) 和前端分色显示，我们需要把它们切开
        # 切分点就是训练集的长度
        # ARIMA 模型在开始的几个点可能会产生 NaN (由于差分)，这会导致 Visualizer 崩溃
        # 我们使用 pandas 的 bfill (向后填充) 和 ffill (向前填充) 来清洗数据
        if np.isnan(full_predict).any():
            full_predict = pd.Series(full_predict.flatten()).bfill().ffill().values
            
        full_predict = full_predict.reshape(-1, 1)
        
        # [修改] 切分数据: 训练 | 测试 | 未来
        train_len = len(Y_train)
        test_len = len(Y_test)
        
        train_predict = full_predict[:train_len]
        test_predict = full_predict[train_len : train_len + test_len]
        
        if future_steps > 0:
            future_predict = full_predict[train_len + test_len :]

    # --- 以下逻辑通用：反归一化与评估 ---
    
    # 确保预测结果不为 None
    if train_predict is None or test_predict is None:
        raise ValueError(f"预测失败，模型类型 {model_type} 未能生成有效的预测结果。")

    # 反归一化
    train_predict_inv = scaler.inverse_transform(train_predict)
    test_predict_inv = scaler.inverse_transform(test_predict)

    future_predict_inv = None
    if future_predict is not None and len(future_predict) > 0:
        future_predict_inv = scaler.inverse_transform(future_predict)
    
    # 修正评估时 Y_train/Y_test 的形状处理
    # scaler.inverse_transform 需要 (N, 1) 的形状，直接传入 [flatten] 会变成 (1, N) 导致维度错误
    y_train_inv = scaler.inverse_transform(Y_train.reshape(-1, 1))
    y_test_inv = scaler.inverse_transform(Y_test.reshape(-1, 1))
    
    # 计算指标
    # y_test_inv 是 (N, 1)，test_predict_inv 是 (N, 1)，可以直接 flatten 后对比
    mae = mean_absolute_error(y_test_inv.flatten(), test_predict_inv.flatten())
    rmse = np.sqrt(mean_squared_error(y_test_inv.flatten(), test_predict_inv.flatten()))
    return {
        "mae": mae,
        "rmse": rmse,
        "train_predict": train_predict_inv, 
        "test_predict": test_predict_inv,
        "future_predict": future_predict_inv # 返回未来数据
    }