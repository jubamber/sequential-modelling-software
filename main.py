# 入口层：唯一的程序入口

import config
# 必须最先初始化环境
config.setup_env()
import contextlib
import os

import utils
import interface

from fastapi import FastAPI
import uvicorn
import gradio as gr

if __name__ == "__main__":
    print("程序启动中...")
    
    # 1. 清理端口
    utils.free_port(config.PORT)
    
    # 2. 创建 UI
    demo = interface.create_ui()
    #demo.launch(server_name="127.0.0.1", server_port=config.PORT, show_api=False)
    
    # 3. 启动
    # 为了支持 shutdown API，我们需要访问底层的 FastAPI app
    # Gradio launch 后会启动服务器，为了注入 shutdown，我们可以这样做：
    # 简单方案：Gradio 启动后，我们很难动态添加 API 到它的内部 FastAPI
    # 所以我们采用 Gradio 提供的回调或者直接在 interface 中不定义 shutdown，
    # 而是通过标准 launch 参数。
    
    # 注意：FastAPI 的 shutdown 路由集成在 Gradio 中稍显复杂。
    # 最稳妥的方式是先定义 FastAPI，再 mount Gradio。
    
    app = FastAPI()
    
    # 注册关闭路由,可以使用/shutdown关闭
    interface.register_shutdown(app)
    
    # 挂载 Gradio
    # 使用 contextlib.redirect_stdout 将输出重定向到 os.devnull (空设备)
    # 这样在 mount_gradio_app 内部的 print("new", path) 就不会显示在控制台了
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
         app = gr.mount_gradio_app(
            app, 
            demo, 
            path="/"
        )


    # 打印服务运行成功通知
    url = f"http://127.0.0.1:{config.PORT}"
    print(f"Running on local URL:  {url}")

    # 使用 uvicorn 启动（这是更生产级的方式）
    # 注意：Electron 环境下不要开启 browser，并固定端口
    # log_level="warning" 会隐藏掉 Uvicorn 默认的那些 INFO 日志，让控制台更清爽
    try:
        uvicorn.run(
            app, 
            host="127.0.0.1", 
            port=config.PORT, 
            log_level="error",
            )
    
    except (KeyboardInterrupt, SystemExit):
        # 这里捕获键盘中断（Ctrl+C）
        pass
    finally:
        # 无论是因为 Ctrl+C 还是其他原因停止，服务结束时打印提示
        # 注意：Uvicorn 会捕获 Ctrl+C 信号进行优雅关闭，所以通常代码会执行到这里
        print("\nKeyboard interruption... closing server.")