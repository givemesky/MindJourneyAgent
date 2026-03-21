# 服务主入口：配置 FastAPI、中间件及路由挂载

# -*- coding: utf-8 -*-
import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# --- 1. 路径环境配置 ---
# 获取当前文件的绝对路径 (api_service/main.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目的根目录 (MindJourneyAgent/)
root_dir = os.path.dirname(current_dir)

# 将根目录添加到 sys.path，确保能导入 integrated_bridge 和 RAG 等模块
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# --- 2. 导入内部组件 ---
# 注意：这些导入必须在路径配置之后
from api_service.routes import chat
from api_service.dependencies import preload_models
from api_service.utils.logger import logger  # 假设你已在 utils 中简单定义了 logger

# --- 3. 初始化 FastAPI 应用 ---
app = FastAPI(
    title="MindJourney API Service",
    description="旅游偏好交互式推理与行程规划后端 (DPI-SynPT)",
    version="1.0.0"
)

# --- 4. 配置跨域中间件 (CORS) ---
# 允许 Android 手机通过局域网 IP 访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议指定具体的 IP
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 5. 挂载路由模块 ---
# 将 chat.py 中的路由挂载到 /v1 前缀下
# 最终访问路径为: http://IP:8000/v1/plan
app.include_router(chat.router, prefix="/v1", tags=["Conversation & DPI"])

# --- 6. 定义服务生命周期事件 ---
@app.on_event("startup")
async def startup_event():
    """
    在服务器启动时执行的逻辑。
    在这里预加载大模型，避免第一个用户请求时卡顿。
    """
    print("\n" + "="*50)
    print("🚀 [System] MindJourney API 后端正在启动...")
    print(f"📍 项目根目录: {root_dir}")
    print("📦 [Model] 正在初始化共享模型资源 (Qwen-7B + LoRA)...")
    print("="*50 + "\n")
    
    try:
        # 调用 dependencies.py 中的预加载函数
        preload_models()
        print("\n✅ [System] 服务就绪！模型已成功载入显存。")
    except Exception as e:
        print(f"\n❌ [Critical Error] 模型初始化失败: {str(e)}")
        # 在算力集群中，如果模型加载失败，通常建议直接退出
        os._exit(1)

@app.on_event("shutdown")
async def shutdown_event():
    """在服务器关闭时执行清理逻辑"""
    print("\n🛑 [System] 正在关闭 MindJourney 服务...")

# --- 7. 基础健康检查接口 ---
@app.get("/", tags=["Health Check"])
async def health_check():
    return {
        "status": "online",
        "service": "MindJourney API",
        "engine": "DPI-SynPT v1.0"
    }

# 挂载 Web 测试页面
# 去掉 async，让 FastAPI 自动处理同步文件读取
@app.get("/web_test", response_class=HTMLResponse, tags=["Debug Tool"])
def get_web_test():
    # 确保 test_ui.html 放在 api_service/ 目录下
    html_path = os.path.join(current_dir, "test_ui.html")
    if not os.path.exists(html_path):
        return f"<html><body><h1>❌ 找不到文件: {html_path}</h1></body></html>"
    
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

# --- 8. 启动脚本 ---
if __name__ == "__main__":
    # 建议直接传入 app 对象， host="0.0.0.0" 确保外网可访问
    uvicorn.run(
        app,  # 直接传入对象
        host="0.0.0.0", 
        port=8000
    )