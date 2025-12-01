# 工具层：负责与操作系统交互，与业务逻辑完全解耦

import socket # 用于网络编程，可以创建 TCP/UDP 客户端或服务器，实现网络通信
import subprocess # 用于在 Python 程序中创建新进程、执行外部命令（shell 命令），并与这些进程进行交互（获取输出、发送输入、获取返回码等）
import sys
import os
import threading # 用于添加 FastAPI Shutdown API
import time


# 检测准备打开的目标端口是否被占用并关闭，以防上一个启动的应用没有正常关闭
def free_port(port):
    """检测端口是否被占用，如果被占用则杀掉对应进程"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))
    sock.close()
    
    if result == 0:
        print(f"[INFO] 端口 {port} 已被占用，尝试释放...")
        try:
            # 查找占用端口的进程 PID
            cmd = f"netstat -ano | findstr :{port}"
            output = subprocess.check_output(cmd, shell=True, text=True)
            for line in output.strip().split('\n'):
                parts = line.split()
                if len(parts) >= 5:
                    pid = parts[4]
                    # 跳过端口7860上的PIP为0的系统进程，可能只是用于监听的
                    if pid == "0":
                        print("[INFO] 跳过系统进程 PID 0")
                        continue
                    print(f"[INFO] 杀掉 PID {pid} 占用的端口 {port}")
                    os.system(f"taskkill /PID {pid} /F")
        except Exception as e:
            print(f"[WARN] 无法释放端口 {port}: {e}")
    else:
        print(f"[INFO] 端口 {port} 可用")


def shutdown_server():
    """执行强制关闭"""
    def kill_me():
        # 延迟一小会儿确保 HTTP 200 OK 能发回给 Electron
        time.sleep(0.5)
        print("Force exiting...")
        os._exit(0) # 使用 os._exit(0) 强制结束
    threading.Thread(target=kill_me).start()


def get_resource_path(relative_path):
    # 辅助函数：获取资源路径
    # 为了让代码能访问这些资源，PyInstaller 会在运行时动态创建一个 _MEIPASS 属性到 sys 模块上，指向临时目录
    if hasattr(sys, '_MEIPASS'): 
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


def get_file_size_str(file_path):
    """获取文件大小并转换为易读格式 (KB/MB)"""
    if not os.path.exists(file_path):
        return "未知大小"
    
    size_bytes = os.path.getsize(file_path)
    if size_bytes < 1024:
        return f"{size_bytes} Bytes"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"