# 配置层：负责环境变量和全局常量的设置

import os
import sys


def setup_env():
    # 环境变量设置

    """ # 检查是否已经开启了 UTF-8 模式
    # 这里的 -X utf8 标志或者环境变量 PYTHONUTF8 都会影响 sys.flags.utf8_mode
    if not sys.flags.utf8_mode:
        print("Re-launching with UTF-8 mode enabled...")
        
        # 设置环境变量
        os.environ["PYTHONUTF8"] = "1"
        
        # 获取当前的 python解释器路径 和 脚本路径及参数
        python = sys.executable
        cmd_args = [python] + sys.argv
        
        # 使用新的环境变量替换当前进程
        #这相当于关掉当前程序，用新的设置立刻重开一个
        os.execv(python, cmd_args) """

    # 使用tensorflow作为Keras后端，默认也为tensorflow,必须在设置之后再引入Keras
    os.environ["KERAS_BACKEND"] = "tensorflow"

    # 使用CPU优化tensorflow计算
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

    # 0代表显示所有信息，1表示不显示info，2表示不显示warning，3表示不显示error
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"

    # 强制输出控制台
    os.environ["PYTHONUNBUFFERED"] = "1"

    # 强制输出缓冲，实时在终端中输出
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    except:
        pass


# 项目配置
PORT = 7860
MODEL_SAVE_DIR = os.path.join(os.getcwd(), "saved_models")
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

