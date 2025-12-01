# 交互层：Gradio，业务流程的控制器（Controller）。它将 UI 事件与后面几层逻辑串联起来
import glob
import os
import json

import gradio as gr
from fastapi import FastAPI
import data_processor
import model_engine
import visualizer
import utils
import config

def clean_model_history():
    """清理保存的模型文件"""
    save_dir = config.MODEL_SAVE_DIR
    
    # 确保目录存在
    if not os.path.exists(save_dir):
        return "### ⚠️ 目录不存在，无需清理。"
    
    # 查找常见后缀的模型文件
    # 根据你的 train_model 逻辑，模型保存为 .keras (Keras) 或 .pkl (Sktime)
    files_to_delete = glob.glob(os.path.join(save_dir, "*.keras")) + \
                      glob.glob(os.path.join(save_dir, "*.pkl")) + \
                      glob.glob(os.path.join(save_dir, "*.json"))
    
    deleted_count = 0
    errors = []
    
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            deleted_count += 1
        except Exception as e:
            errors.append(f"无法删除 {os.path.basename(file_path)}: {str(e)}")

    deleted_count = int(deleted_count / 2) if deleted_count > 0 else 0
    
    # 构建返回信息
    if len(errors) > 0:
        return f"### ⚠️ 清理完成，共删除 {deleted_count} 个模型文件。\n错误: {'; '.join(errors)}"
    elif deleted_count == 0:
        return "### ℹ️ 暂无已保存的模型文件可清理。"
    else:
        return f"### ✅ 成功清理历史模型缓存（共删除 {deleted_count} 个模型文件）。"
    

def get_metadata_path(model_path):
    """根据模型路径获取对应的 metadata.json 路径"""
    # 假设模型是 model.keras，元数据存为 model.json
    base, _ = os.path.splitext(model_path)
    return f"{base}_meta.json"

def save_pipeline_config(save_path, params):
    """保存训练参数到 json"""
    meta_path = get_metadata_path(save_path)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=4, ensure_ascii=False)

def load_pipeline_config(model_filename):
    """读取已保存的参数"""
    save_dir = config.MODEL_SAVE_DIR
    model_path = os.path.join(save_dir, model_filename)
    meta_path = get_metadata_path(model_path)
    
    if not os.path.exists(meta_path):
        return None
    
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_saved_model_list():
    """获取所有已保存的模型文件列表"""
    save_dir = config.MODEL_SAVE_DIR
    if not os.path.exists(save_dir):
        return []
    
    # 获取 .keras 和 .pkl 文件
    files = glob.glob(os.path.join(save_dir, "*.keras")) + \
            glob.glob(os.path.join(save_dir, "*.pkl"))
    # 按时间倒序排列
    files.sort(key=os.path.getmtime, reverse=True)
    return [os.path.basename(f) for f in files]
    
def run_pipeline(dataset_name, use_local_data, local_file_path, clean_dataset_name, # <--- 修改：新增 use_local_data
                 model_type, epochs, batch_size, look_back, split_ratio, 
                 p, d, q, auto_arima, P, D, Q, s, 
                 use_saved_model, saved_model_name, 
                 enable_future, future_steps, progress=gr.Progress()):
    """控制器函数：协调数据、模型和绘图"""
    try:

        # 确定用于保存的文件名标识
        # 如果是本地数据，用 clean_dataset_name (文件名)，否则用 dataset_name
        if use_local_data:
            # 如果勾选了本地数据，使用 clean_dataset_name (文件名) 作为标识
            actual_dataset_name_for_save = clean_dataset_name 
            # 确保有文件路径
            if not local_file_path:
                return "### ❌ 错误：请上传本地 CSV 文件。", None
        else:
            # 否则使用预置数据集名称
            actual_dataset_name_for_save = dataset_name
        
        # --- 模式分支 ---
        if use_saved_model and saved_model_name:
            progress(0.1, desc=f"正在加载模型: {saved_model_name}...")
            save_path = os.path.join(config.MODEL_SAVE_DIR, saved_model_name)
            
            # 1. 加载模型实体
            try:
                if saved_model_name.endswith(".keras"):
                    from keras.models import load_model
                    model = load_model(save_path)
                elif saved_model_name.endswith(".pkl"):
                    import joblib
                    model = joblib.load(save_path)
                else:
                    return f"### ⚠️ 不支持的文件格式: {saved_model_name}", None
            except Exception as e:
                return f"### ❌ 模型加载失败: {str(e)}", None
                
            file_size_str = utils.get_file_size_str(save_path)
            report_prefix = "### ♻️ 模型加载报告 (已保存模型)"
            
        else:
            # === 原有的训练逻辑 ===
            
            # 1. 数据准备
            progress(0, desc="加载与处理数据...")
            try:
                # 根据 use_local_data 的逻辑，load_source 可能是文件名也可能是路径
                # data_processor.load_raw_data 内部应该兼容 (如果 dataset_name 不在预置列表中且是路径)
                # 这里我们显式传参以防万一
                if use_local_data:
                    df = data_processor.load_raw_data("📂 加载本地数据", local_file_path) # 借用旧接口逻辑或直接传path
                else:
                    df = data_processor.load_raw_data(dataset_name)
            except Exception as e:
                return f"### ❌ 数据加载失败: {str(e)}", None
            data_pkg = data_processor.process_data(df, look_back, split_ratio, model_type)
            
            # 2. 构建模型
            progress(0.2, desc="构建模型...")
            model = model_engine.build_model(model_type, look_back, p, d, q, auto_arima, P, D, Q, s)
            
            # 3. 训练模型
            progress_cb = model_engine.GradioProgressCallback(progress, epochs, start_progress=0.4, end_progress=0.8)
            # 【修改点】传入 actual_dataset_name_for_save 作为保存文件名的一部分
            save_path = model_engine.train_model(model_type, model, data_pkg["X_train"], data_pkg["Y_train"], epochs, batch_size, progress_cb, actual_dataset_name_for_save)
            
            # --- [新增] 保存参数配置 ---
            # 将当前的所有参数打包保存，方便下次读取
            current_params = {
                "dataset_name": dataset_name, "use_local_data": use_local_data, # 保存是否使用了本地数据
                "model_type": model_type,
                "epochs": epochs, "batch_size": batch_size,
                "look_back": look_back, "split_ratio": split_ratio,
                "p": p, "d": d, "q": q, "auto_arima": auto_arima,
                "P": P, "D": D, "Q": Q, "s": s
            }
            save_pipeline_config(save_path, current_params)
            
            file_size_str = utils.get_file_size_str(save_path)
            report_prefix = "### 训练报告"

        # === 公共部分：评估与绘图 ===
        # 注意：为了评估和绘图，我们需要重新加载数据。
        # 哪怕是加载模型模式，我们也需要用当初保存参数里的 dataset_name 等配置来重新处理数据，
        # 这样才能保证 X_test 的形状和模型的输入匹配。
        
        progress(0.8, desc="准备评估数据...")
        # 【修改点】再次加载数据用于评估
        try:
            # 再次加载数据用于评估 (保证数据一致性)
            if use_local_data:
                df = data_processor.load_raw_data("📂 加载本地数据", local_file_path)
            else:
                df = data_processor.load_raw_data(dataset_name)
        except Exception as e:
            return f"### ❌ 评估阶段数据加载失败: {str(e)}", None
        data_pkg = data_processor.process_data(df, look_back, split_ratio, model_type)
        
        progress(0.9, desc="模型预测与评估...")
        eval_res = model_engine.evaluate_model(
            model_type, model, 
            data_pkg["X_train"], data_pkg["X_test"], 
            data_pkg["Y_train"], data_pkg["Y_test"], 
            data_pkg["scaler"],
            future_steps if enable_future else 0  # 传入需要额外预测的步数
        )
        
        report = f"""
        {report_prefix}
        - **数据集**: {actual_dataset_name_for_save}
        - **数据来源**: {'本地上传' if use_local_data else '系统预置'}
        - **模型类型**: {model_type}
        - **MAE**: {eval_res['mae']:.4f}
        - **RMSE**: {eval_res['rmse']:.4f}
        - **额外预测**: {'已启用 (' + str(future_steps) + '步)' if enable_future else '未启用'}
        - **模型路径**: `{save_path}`
        - **模型文件大小**: **{file_size_str}**
        """
        
        # [修改] 传入 future_predict 数据到绘图函数
        fig = visualizer.create_forecast_plot(
            model_type, actual_dataset_name_for_save,
            data_pkg["full_dataset_scaled"],
            eval_res["train_predict"], 
            eval_res["test_predict"],
            eval_res.get("future_predict", None),
            look_back, data_pkg["scaler"]
        )
        
        return report, fig
    except Exception as e:
        # 【新增】顶层错误捕获，防止 Gradio 崩溃
        import traceback
        traceback.print_exc() # 在控制台打印详细错误，方便调试
        error_msg = f"### ❌ 运行过程中发生错误\n\n**错误信息**: {str(e)}\n\n*提示：如果使用的是已保存的 ARIMA/SARIMA 模型，请确保加载的数据集与训练时一致。*"
        return error_msg, None

def create_ui():
    # 使用自定义 CSS 美化
    custom_css = """
    body { background-color: #0b0f19; } 
    .gradio-container { 
        font-family: 'Roboto', sans-serif; 
        margin-top: 1vh;  /* 顶部留白 */
    }
    /* 修复 Electron 中 Markdown 组件（如标题）出现不必要滚动条的问题 */
    .prose {
        overflow: visible !important; /* 强制内容可见，不裁剪也不滚动 */
    }
    /* 如果上述无效，可以尝试更暴力的隐藏滚动条样式 */
    .prose::-webkit-scrollbar {
        display: none; 
        width: 0 !important;
        height: 0 !important;
    }
    /* 针对所有 Markdown 类型的容器 */
    .gr-markdown, .markdown-text {
        overflow: visible !important;
    }
    /* 甚至可以直接针对标题标签 */
    h1, h2, h3, h4, h5, h6 {
        overflow: visible !important;
        margin-bottom: 0.2em !important; /* 有时增加一点下边距也能解决计算误差 */
    }

    /* 定义禁用按钮的样式 (可选) */
    .disabled-btn { opacity: 0.5; cursor: not-allowed; }

    /* 核心：针对 interactive=False 的组件应用样式 */
    /* 注意：Gradio 版本不同类名可能略有差异，这里覆盖了常见的禁用状态 */
    input:disabled, textarea:disabled, .disabled, .gr-disabled {
        opacity: 0.4 !important;
        cursor: not-allowed !important;
    }
    /* 针对滑块和容器的禁用层 */
    .pointer-events-none {
        opacity: 0.4 !important;
        pointer-events: none !important;
    }
    """
    
    with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="时间序列分析工作站") as demo:
        gr.Markdown("# 📈 时间序列分析工作站 (Ver. 1.0.0)\n\n")

        # === 状态变量 (State) ===
        # 用于存储本地文件的绝对路径
        local_file_path_state = gr.State(value=None)
        # 用于存储用于显示和保存的“干净”数据集名称 (例如: my_data)
        dataset_name_clean_state = gr.State(value="Sine Wave (模拟)")

        with gr.Column():
            gr.Markdown("### 📊 可视化窗口")
            plot_out = gr.Plot(label="可视化")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ 配置面板")
                    
                    # --- 模块 A: 已保存模型 (最高优先级) ---
                    with gr.Group():
                        use_saved_cb = gr.Checkbox(label="📂 选用已保存的模型", value=False)
                        saved_model_dd = gr.Dropdown(label="选择模型文件", choices=[], visible=False, interactive=True)

                    # --- 模块 B: 本地数据 (次优先级) ---
                    with gr.Group():
                        use_local_cb = gr.Checkbox(label="📂 上传本地数据 (CSV)", value=False)
                        file_uploader = gr.File(
                            label="拖拽上传 CSV 文件 (需符合 Date-Value 格式)", 
                            file_types=[".csv"], 
                            visible=False,
                            type="filepath"
                        )

                    # --- 模块 C: 预置数据集 (默认) ---
                    # 移除了 "📂 加载本地数据" 选项
                    dataset_dd = gr.Dropdown(
                        choices=["Sine Wave (模拟)", "AirPassengers (模拟)", "AirPassengers", "Daily minimum temperatures in Melbourne", "Sunspots", "Mauna Loa CO2 Weekly", "Arctic Oscillation Dataset"], 
                        value="Sine Wave (模拟)", 
                        label="预置数据集"
                    )

                    model_dd = gr.Dropdown(choices=["LSTM", "MLP", "ARIMA", "SARIMA", "Exponential-Smoothing"], value="LSTM", label="模型")

                    with gr.Group():
                        enable_future_cb = gr.Checkbox(label="🔮 是否额外预测数据", value=False, interactive=True)
                        future_steps_sl = gr.Slider(minimum=1, maximum=100, value=12, step=1, label="额外预测步数", visible=False, interactive=True)

                with gr.Column(scale=1):
                    gr.Markdown("### 🔧 参数设置")
                    with gr.Column(visible=True) as dl_params_group:
                        epochs_sl = gr.Slider(minimum=1, maximum=100, value=20, step=1, label="训练轮次 (Epochs)")
                        batch_sl = gr.Slider(minimum=1, maximum=64, value=16, step=1, label="Batch Size")

                    with gr.Column(visible=False) as arima_params_group:
                        auto_arima = gr.Checkbox(value=True, label="是否使用(S)ARIMA的自动参数推导")
                        with gr.Column(visible=False) as manual_params_container:
                            gr.Markdown("#### ARIMA 基础参数")
                            p_sl = gr.Slider(minimum=0, maximum=10, value=2, step=1, label="p")
                            d_sl = gr.Slider(minimum=0, maximum=10, value=1, step=1, label="d")
                            q_sl = gr.Slider(minimum=0, maximum=10, value=2, step=1, label="q")
                            with gr.Column(visible=False) as sarima_pdq_group:
                                gr.Markdown("#### SARIMA 季节性参数")
                                P_sl = gr.Slider(minimum=0, maximum=10, value=2, step=1, label="P")
                                D_sl = gr.Slider(minimum=0, maximum=10, value=1, step=1, label="D")
                                Q_sl = gr.Slider(minimum=0, maximum=10, value=2, step=1, label="Q")
                        with gr.Column(visible=False) as sarima_s_group:
                            s_sl = gr.Slider(minimum=0, maximum=100, value=12, step=1, label="s")

                    look_back_sl = gr.Slider(minimum=1, maximum=100, value=3, step=1, label="时间窗口 (Look Back)")
                    split_sl = gr.Slider(minimum=0.5, maximum=0.9, value=0.7, step=0.05, label="训练集占比")
                    
                # 定义参数列表，用于批量禁用/启用
                # 这些参数在加载模型时会被禁用（因为它们是模型结构的一部分）
                train_params_locked = [
                    model_dd, epochs_sl, batch_sl, look_back_sl, split_sl,
                    auto_arima, p_sl, d_sl, q_sl, P_sl, D_sl, Q_sl, s_sl
                ]
                # 这些参数在任何时候都应该允许用户修改
                predict_params = [enable_future_cb, future_steps_sl]

                with gr.Column(scale=1):
                    gr.Markdown("### ▶️ 运行按键")
                    # 按钮区
                    with gr.Row():
                        btn_run = gr.Button("🚀 开始训练与评估", variant="primary")
                        btn_stop = gr.Button("⏹️ 终止当前任务", variant="secondary")

                    btn_clear = gr.Button("🗑️ 清空已保存模型", variant="stop") 
                    metrics_out = gr.Markdown("### 等待指令...")


        # =========================================
        # 4. 交互逻辑 (Callbacks)
        # =========================================

        # --- A. 界面可见性控制函数 ---

        def update_lookback_visibility(model_type):
            return gr.update(visible=True) if model_type not in ["ARIMA", "SARIMA", "Exponential-Smoothing"] else gr.update(visible=False)

        def update_dl_params_visibility(model_type):
            return gr.update(visible=(model_type in ["LSTM", "MLP"]))

        def update_arima_container_visibility(model_type):
            return gr.update(visible=(model_type in ["ARIMA", "SARIMA"]))

        def update_sarima_sub_visibility(model_type, auto_mode):
            if model_type == "SARIMA":
                return gr.update(visible=not auto_mode), gr.update(visible=True)
            return gr.update(visible=False), gr.update(visible=False)

        def update_manual_params_visibility(auto_mode):
            return gr.update(visible=not auto_mode)
        
        # --- B. 数据源切换逻辑 (核心修改) ---

        def on_use_saved_change(use_saved):
            """
            修改后逻辑：
            1. 禁用/启用 'train_params_locked' (模型结构参数)。
            2. 始终保持 'predict_params' (预测步数) 为 Interactive=True。
            """
            # 1. 训练参数：根据是否使用保存模型来锁定
            params_interactive = not use_saved
            lock_updates = [gr.update(interactive=params_interactive) for _ in train_params_locked]
            
            # 2. 预测参数：始终允许修改
            # 注意：visibility 依然由 enable_future_cb 自身的逻辑控制，这里只管 interactive
            predict_updates = [
                gr.update(interactive=True), # enable_future_cb
                gr.update(interactive=True)  # future_steps_sl
            ]
            
            # 3. 其他UI组件
            file_list = get_saved_model_list() if use_saved else []
            if not file_list:
                saved_model_dd_update = gr.update(visible=use_saved, choices=[], value=None)
            else:
                # 保持当前值（如果在列表中），或者设为 None
                saved_model_dd_update = gr.update(visible=use_saved, choices=file_list)
            
            btn_text = "🚫 请先选择模型文件" if use_saved else "🚀 开始训练与评估"
            btn_update = gr.update(interactive=not use_saved, value=btn_text, variant="secondary" if use_saved else "primary")
            
            # 返回列表顺序必须与 outputs 定义一致：Lock Params + Predict Params + [Saved DD, Btn, Local CB, Dataset DD]
            return lock_updates + predict_updates + [
                saved_model_dd_update, 
                btn_update, 
                gr.update(interactive=True),
                gr.update(interactive=True)
            ]

        def on_use_local_change(use_local):
            """
            修改后逻辑：
            完全解耦。只看 use_local 的值，不看 use_saved。
            - 勾选本地: 显示上传框，禁用预置下拉
            - 取消本地: 隐藏上传框，启用预置下拉
            """
            if use_local:
                return gr.update(visible=True), gr.update(interactive=False)
            else:
                return gr.update(visible=False), gr.update(interactive=True)

        # --- C. 数据加载与预览 ---

        def update_preview_by_preset(dataset_name, use_local):
            """预置数据集改变 -> 更新预览 (仅当未使用本地数据时)"""
            if use_local: 
                return None # 不更新，保持当前本地数据的图
            
            try:
                df = data_processor.load_raw_data(dataset_name)
                fig = visualizer.create_data_preview_plot(dataset_name, df)
                return fig
            except:
                return None

        def on_file_upload(file_path):
            """文件上传完毕 -> 校验并更新预览"""
            if not file_path:
                return gr.update(), None, None, None
            
            file_name_clean = os.path.splitext(os.path.basename(file_path))[0]
            try:
                df = data_processor.validate_and_load_local(file_path)
                fig = visualizer.create_data_preview_plot(f"本地数据: {file_name_clean}", df)
                msg = f"### ✅ 成功加载本地数据: {os.path.basename(file_path)}\n样本数: {len(df)}"
                return msg, fig, file_name_clean, file_path
            except ValueError as e:
                return f"### ❌ 格式错误: {str(e)}", None, None, None

        # --- D. 约束与参数回填 ---

        def update_lookback_constraints(dataset_name, use_local, local_path, split_ratio, current_lookback):
            """根据当前选中的数据源（预置或本地）计算约束"""
            try:
                if use_local:
                    if not local_path: return gr.update(), "### ⚠️ 请先上传 CSV"
                    df = data_processor.load_raw_data("📂 加载本地数据", local_path)
                else:
                    df = data_processor.load_raw_data(dataset_name)
                
                total_len = len(df)
                train_size = int(total_len * split_ratio)
                new_max = max(1, train_size - 2)
                new_value = min(current_lookback, new_max)
                
                return gr.update(maximum=new_max, value=new_value), f"当前数据总长: {total_len}, 训练集: {train_size}"
            except:
                return gr.update(), ""

        def on_saved_model_select(filename):
            """选择模型 -> 回填参数 (预测参数不受影响，保持当前UI状态)"""
            if not filename: return [gr.skip()] * 15
            params = load_pipeline_config(filename)
            if not params: return [gr.update()] * 14 + [gr.update(interactive=True, value="⚠️ 元数据丢失")]
            
            return [
                gr.update(value=params.get("dataset_name"), interactive=True), 
                gr.update(value=params.get("model_type"), interactive=False),   
                params.get("epochs", 20), params.get("batch_size", 16), params.get("look_back", 3), params.get("split_ratio", 0.7),          
                params.get("p", 2), params.get("d", 1), params.get("q", 2),
                params.get("auto_arima", True),          
                params.get("P", 2), params.get("D", 1), params.get("Q", 2), params.get("s", 12),
                gr.update(interactive=True, value="🚀 加载模型并评估", variant="primary") 
            ]

        # =========================================
        # 5. 事件绑定 (Event Wiring)
        # =========================================
        
        # 1. “已保存模型” 勾选逻辑
        use_saved_cb.change(
            fn=on_use_saved_change,
            inputs=use_saved_cb,
            # Outputs 必须包含所有被修改的组件
            outputs=train_params_locked + predict_params + [saved_model_dd, btn_run, use_local_cb, dataset_dd]
        )
        
        # 2. “本地数据” 勾选 (去掉了 use_saved_cb 输入)
        use_local_cb.change(
            fn=on_use_local_change,
            inputs=[use_local_cb], # 只需要这一个输入
            outputs=[file_uploader, dataset_dd]
        )
        
        # 3. 文件上传逻辑
        upload_event = file_uploader.change(
            fn=on_file_upload,
            inputs=file_uploader,
            outputs=[metrics_out, plot_out, dataset_name_clean_state, local_file_path_state]
        )
        
        # 4. 预置数据集切换逻辑 (仅更新图)
        dataset_dd.change(
            fn=update_preview_by_preset,
            inputs=[dataset_dd, use_local_cb],
            outputs=plot_out
        )

        # 5. 参数可见性联动 (Model -> Params)
        model_dd.change(fn=update_dl_params_visibility, inputs=model_dd, outputs=dl_params_group)
        model_dd.change(fn=update_arima_container_visibility, inputs=model_dd, outputs=arima_params_group)
        model_dd.change(fn=update_sarima_sub_visibility, inputs=[model_dd, auto_arima], outputs=[sarima_pdq_group, sarima_s_group])
        model_dd.change(fn=update_lookback_visibility, inputs=model_dd, outputs=look_back_sl)
        auto_arima.change(fn=update_manual_params_visibility, inputs=auto_arima, outputs=manual_params_container)
        auto_arima.change(fn=update_sarima_sub_visibility, inputs=[model_dd, auto_arima], outputs=[sarima_pdq_group, sarima_s_group])
        enable_future_cb.change(fn=lambda x: gr.update(visible=x), inputs=enable_future_cb, outputs=future_steps_sl)

        # 6. 自动约束 Lookback (所有可能改变数据长度的操作都要触发)
        constraint_inputs = [dataset_dd, use_local_cb, local_file_path_state, split_sl, look_back_sl]
        
        # 绑定到 upload 结束
        upload_event.then(fn=update_lookback_constraints, inputs=constraint_inputs, outputs=[look_back_sl, metrics_out])
        # 绑定到 dataset 切换
        dataset_dd.change(fn=update_lookback_constraints, inputs=constraint_inputs, outputs=[look_back_sl, metrics_out])
        # 绑定到 split 变化
        split_sl.change(fn=update_lookback_constraints, inputs=constraint_inputs, outputs=[look_back_sl, metrics_out])

        # 7. 选择已保存模型文件
        saved_model_dd.change(
            fn=on_saved_model_select,
            inputs=saved_model_dd,
            outputs=[dataset_dd, model_dd, epochs_sl, batch_sl, look_back_sl, split_sl, p_sl, d_sl, q_sl, auto_arima, P_sl, D_sl, Q_sl, s_sl, btn_run]
        )

        # 8. 运行按钮
        # 注意：这里需要传入 use_local_cb 的值
        run_event = btn_run.click(
            fn=run_pipeline,
            inputs=[
                dataset_dd, use_local_cb, local_file_path_state, dataset_name_clean_state, # <--- Updated inputs
                model_dd, epochs_sl, batch_sl, look_back_sl, split_sl, 
                p_sl, d_sl, q_sl, auto_arima, P_sl, D_sl, Q_sl, s_sl,
                use_saved_cb, saved_model_dd,
                enable_future_cb, future_steps_sl
            ],
            outputs=[metrics_out, plot_out]
        )
        
        btn_stop.click(fn=lambda: ("### ⚠️ 任务已终止", None), outputs=[metrics_out, plot_out], cancels=[run_event])
        
        btn_clear.click(
            fn=lambda: (clean_model_history(), False, gr.update(visible=False, choices=[], value=None), gr.update(interactive=True)), 
            outputs=[metrics_out, use_saved_cb, saved_model_dd, btn_run]
        )

        # 初始化加载
        demo.load(fn=update_preview_by_preset, inputs=[dataset_dd, use_local_cb], outputs=plot_out)

    return demo

def register_shutdown(app: FastAPI):
    @app.get("/shutdown")
    def shutdown():
        print("Service shutting down...")
        utils.shutdown_server()
        return {"status": "ok"}