# 时间序列分析工作站 (Time Series Analysis Workstation)

Github Repository: https://github.com/jubamber/sequential-modelling-software

## 项目简介

**时间序列分析工作站** 是一款基于 Python 开发的轻量级本地桌面应用工具。该项目旨在为学习者、数据科学家及业务人员提供一个直观、易用的平台，用于探索、训练和评估多种**时间序列预测模型**，目前只支持**单变量**时间序列数据分析。

本项目后端集成 **FastAPI**，前端采用 **Gradio** 构建交互式界面，底层算法结合了 **TensorFlow/Keras**（用于深度学习模型）和 **Sktime**（用于统计学模型）。系统经过专门优化，大部分模型和数据集可在标准 CPU 环境下流畅运行，无需复杂的 GPU 配置。



## 核心功能

*   **多模型支持**：
    *   **深度学习**：LSTM (Long Short-Term Memory), MLP (Multilayer Perceptron)。
    *   **统计学模型**：ARIMA, SARIMA, Exponential Smoothing (指数平滑)。
*   **数据源灵活切换**：支持系统预置的经典数据集（如 AirPassengers, Sunspots 等），也支持用户上传本地 CSV 数据进行分析。
*   **全自动参数搜索**：支持 AutoARIMA，可自动推导最优的 p, d, q 及季节性参数。
*   **交互式可视化**：集成 Plotly，提供可缩放、交互式的训练集拟合曲线、测试集预测曲线及未来趋势推演。
*   **模型持久化**：支持训练后模型的自动保存、历史模型加载以及参数配置的回溯（Metadata）。
*   **未来预测**：除了验证测试集，还支持指定步数的推演数据生成（Future Forecasting）。



## 环境依赖与安装

推荐使用 **Conda** 或 **Python venv** 创建独立的虚拟环境来运行**Gradio**应用。

### 1. 系统要求
*   **Python 版本**：3.12.10 (推荐)
*   **操作系统**：Windows / macOS / Linux
*   **硬件**：支持标准 CPU（项目默认开启 OneDNN 优化），可选 GPU（需自行修改代码配置）。

### 2. 安装步骤

1.  **克隆或下载项目代码**：
    将所有源码文件放入项目根目录。

2.  **创建并激活虚拟环境**：
    ```bash
    # 使用 conda
    conda create -n ts_station python=3.12.10
    conda activate ts_station
    
    # 或者使用 venv
    python -m venv venv
    # Windows 激活
    venv\Scripts\activate
    # Linux/Mac 激活
    source venv/bin/activate
    ```

3.  **安装依赖库**：
    请确保目录下包含 `requirements.txt` 文件，运行以下命令：
    ```bash
    pip install -r requirements.txt
    ```



## 快速开始

1.  **启动程序**：
    在项目根目录下运行入口脚本：
    ```bash
    python main.py
    ```

2.  **访问界面**：
    程序启动后会自动清理 7860 端口。当控制台显示 `Running on local URL: http://127.0.0.1:7860` 时，打开浏览器访问该地址即可。

3.  **关闭程序**：
    在控制台按 `Ctrl+C`，如需彻底关闭服务，可直接关闭终端窗口或调用 `/shutdown` API。



## 数据格式规范

本系统对用户上传的本地数据有严格格式要求，以确保解析的准确性。

*   **文件格式**：CSV (.csv)
*   **列结构**：最好为包含两列的格式，其他格式不一定能正确解析。
    *   **第 1 列 (索引列)**：时间数据（Date/Time）。建议格式为 standard datetime (e.g., `YYYY-MM-DD` 或 `YYYY-MM-DD HH:MM:SS`)。
    *   **第 2 列 (数值列)**：目标观测值（Value）。必须为纯数值类型。
*   **表头 (Header)**：系统会自动识别是否存在表头。如果第一行不是数值，会被视为表头并自动剔除。

**CSV 示例**：
```csv
Date,Value
2023-01-01,100.5
2023-01-02,102.3
2023-01-03,99.8
...
```



## 使用指南

界面主要分为 **可视化窗口**（左侧/上方）和 **控制面板**（右侧/下方）。

### 1. 数据选择
*   **预置数据集**：在下拉菜单中选择 "Sine Wave", "AirPassengers" 等经典案例直接体验。
*   **本地数据**：勾选 "上传本地数据 (CSV)"，拖拽或点击上传符合规范的文件。系统会自动解析并更新预览图。

### 2. 模型配置
*   **深度学习 (LSTM/MLP)**：
    *   **Epochs**：训练轮次。
    *   **Batch Size**：批量大小。
    *   **Look Back**：时间窗口大小（即用过去多少个时间步预测下一个时间步）。
    *   **训练集占比**：划分训练集和测试集的比例。
*   **统计模型 (ARIMA/SARIMA)**：
    *   **Auto Parameter**：勾选 "是否使用(S)ARIMA的自动参数推导" 可让算法自动寻找最优参数（耗时较长但准确）。
    *   **手动参数**：取消自动勾选后，可手动调整 p, d, q (ARIMA) 以及 P, D, Q, s (SARIMA 季节性参数)。

### 3. 运行与评估
点击 **开始训练与评估** 按钮：
1.  **训练**：界面会显示进度条（针对 Keras 模型显示 Epoch 进度，针对 Sktime 模型显示 Fitting 状态）。
2.  **评估**：计算测试集上的 MAE (Mean Absolute Error) 和 RMSE (Root Mean Squared Error)。
3.  **绘图**：
    *   **灰色线**：真实数据。
    *   **蓝色线**：训练集拟合结果。
    *   **红色线**：测试集预测结果（递归预测，只使用预测的数据）。
    *   **绿色虚线**：未来推演数据（如启用）。

### 4. 终止当前任务

有些任务只使用CPU计算可能会运行很长时间（比如S参数较大时的自动搜寻最优参数SARIMA模型运行在较大的数据集上时），点击界面上的 **终止当前任务** 按钮可以停止当前的计算

### 5. 模型加载与重用
系统会自动将训练好的模型保存至 `saved_models` 目录。
*   勾选 "选用已保存的模型"。
*   在下拉框中选择历史模型文件。
*   系统会自动锁定与模型结构相关的参数（如 Look Back, Model Type），但允许修改 "额外预测步数"。
*   点击运行即可使用旧模型对数据进行推理。
*   注意：最好在训练时的数据集上使用，更换数据集使用已保存的模型很可能会有参数冲突导致错误

### 6. 清理缓存
点击 **清空已保存模型** 按钮可一键删除 `saved_models` 目录下的所有 `.keras`, `.pkl`, `.json` 文件，释放磁盘空间。



## 对于Release文件的说明

使用PyInstaller将原始Python文件和依赖数据打包为文件夹格式（Dir）后，嵌入Electron应用中，打包为Portable版本和基于NSIS的安装包版本

- **Sequential Modelling App 1.0.0.exe**：双击就可以使用，使用较为方便。但是由于打包体积较大，每次进入加载的时间很长
- **Sequential Modelling App Setup 1.0.0.exe**：需要安装，安装后启动时加载速度较另一种方式更快

这两种打包方式都较不成熟，**不是很推荐使用**，也不提供详细的打包代码，推荐使用自己搭建对应的Python环境直接运行



## 项目结构说明

```text
Project_Root/
├── main.py             # 入口层：程序启动、端口管理、Gradio挂载
├── config.py           # 配置层：环境变量、全局常量 (CPU优化设置)
├── interface.py        # 交互层：Gradio UI布局、事件回调、逻辑控制
├── model_engine.py     # 模型层：LSTM/ARIMA等模型定义、训练、预测逻辑
├── data_processor.py   # 数据层：本地/预置数据读取、清洗、归一化
├── visualizer.py       # 可视化层：基于 Plotly 的图表生成
├── utils.py            # 工具层：端口检测、路径处理、系统操作
├── datasets            # 所有内置的数据集原文件
└── saved_models/       # 运行时生成的模型文件存储目录
```



## 内置数据集的参考来源和简介

### 每日北极涛动指数

[CPC - Teleconnections: Arctic Oscillation](https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/ao.shtml)

每日北极涛动指数是通过将北纬 20°以北的每日（00Z）1000 毫巴高度异常投影到北极涛动的载荷模式上构建的。请注意，我们使用了全年月平均异常数据来获取北极涛动的载荷模式。由于北极涛动在寒冷季节变化最大，因此该载荷模式主要反映了寒冷季节北极涛动的特征。

### AirPassengers

[✈️Air Passengers Dataset✈️](https://www.kaggle.com/datasets/brmil07/air-passengers-dataset)

本数据集提供了 1949 年至 1960 年美国航空旅客月度总数。该数据集取自 R 语言内置的名为 AirPassengers 的数据集。分析人员通常采用各种统计技术，例如分解、平滑和预测模型，来分析数据中的模式、趋势和季节性波动。由于其历史性和一致的时间粒度，AirPassengers 数据集对于统计学、计量经济学和交通规划领域的研究人员、从业人员和学生而言，都是一项宝贵的资源。

### Sunspots

[Sunspots](https://www.kaggle.com/datasets/robervalt/sunspots)

太阳黑子是太阳光球层上的暂时性现象，表现为比周围区域更暗的斑点。它们是表面温度较低的区域，由磁场通量集中抑制对流引起。太阳黑子通常成对出现，磁极性相反。它们的数量随大约11年的太阳周期而变化。

### Daily Minimum Temperatures in Melbourne

[Daily Minimum Temperatures in Melbourne](https://www.kaggle.com/datasets/paulbrabban/daily-minimum-temperatures-in-melbourne)

### CO2 Mauna Loa Weekly

[CO2 Mauna Loa Weekly](https://www.kaggle.com/datasets/dan3dewey/co2-mauna-loa-weekly)

这些数据来自美国国家海洋和大气管理局环境科学研究实验室（NOAA ESRL）：

“这些数据免费向公众和科学界开放，我们相信其广泛传播将有助于加深理解并带来新的科学见解。提供这些数据并不等同于发表这些数据。NOAA 依靠用户的道德操守和诚信，以确保 ESRL 的工作得到应有的认可。……”



## 免责声明

本版本的软件仍存在一些有待解决的bug，但是经过测试不影响简单的使用

部分代码由大模型辅助生成，软件图标由大模型辅助生成，不涉及版权等问题



## 常见问题 (FAQ)

**Q: 为什么启动时提示端口被占用？**
A: 程序启动时会尝试自动清理 `7860` 端口。如果失败，请手动检查是否有其他 Python 进程或 Gradio 应用占用了该端口。

**Q: 为什么 ARIMA 模型训练速度很慢？**
A: 如果勾选了 "自动参数推导 (AutoARIMA)"，算法需要遍历多种参数组合以寻找最优解，这通常需要较长时间。如需快速验证，请取消勾选并手动指定 p, d, q。

**Q: 支持 GPU 加速吗？**
A: 项目默认配置为 CPU 优化 (`TF_ENABLE_ONEDNN_OPTS=1`) 以保证轻便性和兼容性。如需使用 NVIDIA GPU，请在 `config.py` 中移除相关 CPU 强制设置，并确保已安装 `tensorflow-gpu` 或对应的 CUDA 依赖。

**Q: "未来预测" 功能的原理是什么？**
A: 

*   **LSTM/MLP**：采用递归预测策略，将上一步的预测值作为下一步的输入。随着步数增加，误差可能会累积。
*   **ARIMA/SARIMA**：基于统计学规律直接计算未来时间步的期望值。

**Q: 我加载了保存的 ARIMA 模型，为什么报错？**
A: 统计模型（ARIMA/SARIMA）对输入数据的长度和索引非常敏感。如果在加载旧模型时使用了与训练时完全不同（或长度差异巨大）的数据集，sktime 可能会抛出 "earlier to train starting point" 错误。建议加载模型时使用与训练时一致的数据源。



## 许可证 

本项目遵循 MIT License 协议开源。有关详细信息，请参阅项目根目录下的 [LICENSE](LICENSE) 文件。 

Copyright (c) 2025 Jubamber
