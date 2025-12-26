#!/bin/bash

# 后端服务启动脚本

echo "正在启动人脸检测后端服务..."

# 检查是否已安装依赖
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
    source venv/bin/activate
    echo "安装依赖..."
    pip install -r requirements.txt
else
    echo "激活虚拟环境..."
    source venv/bin/activate
fi

# 启动 Flask 应用
echo "启动 Flask 应用..."
export FLASK_APP=app.py
export FLASK_ENV=development
flask run --host=0.0.0.0 --port=5001