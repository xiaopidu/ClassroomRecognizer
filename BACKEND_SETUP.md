# 后端服务设置指南

本项目采用前后端分离架构，人脸检测和识别功能在后端 Python 服务中运行，前端仅负责用户界面和结果展示。

## 目录结构

```
backend/
├── app.py              # Flask 后端主程序
├── requirements.txt     # Python 依赖包列表
├── start.sh            # 启动脚本
└── uploads/            # 临时上传文件目录
```

## 环境要求

- Python 3.8 或更高版本
- pip 包管理器
- 虚拟环境工具 (推荐)

## 安装步骤

### 1. 进入后端目录

```bash
cd backend
```

### 2. 创建虚拟环境（推荐）

```bash
python3 -m venv venv
```

### 3. 激活虚拟环境

**macOS/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### 4. 安装依赖

```bash
pip install -r requirements.txt
```

### 5. 启动后端服务

**方法一：使用启动脚本**
```bash
chmod +x start.sh
./start.sh
```

**方法二：手动启动**
```bash
export FLASK_APP=app.py
export FLASK_ENV=development
flask run --host=0.0.0.0 --port=5001
```

## 服务验证

启动成功后，后端服务将在 `http://localhost:5001` 运行。

### 健康检查端点
- `GET /health` - 检查服务和模型状态
- `GET /` - 服务基本信息

### API 端点
- `POST /api/detect` - 上传文件进行人脸检测
- `POST /api/detect-base64` - 发送 Base64 数据进行人脸检测

## 模型说明

后端使用 InsightFace 的 buffalo_l 模型，包含：
- 人脸检测模型
- 人脸识别模型（特征提取）
- 关键点检测

## 前端集成

前端通过以下 API 服务与后端通信：
- `src/services/apiService.ts` - API 调用封装
- 图像分析组件会自动调用后端接口

## 故障排除

### 1. 模型加载失败
确保已正确安装 InsightFace：
```bash
pip install insightface
```

### 2. 依赖安装问题
尝试单独安装关键依赖：
```bash
pip install insightface==0.7.3
pip install opencv-python==4.8.0.74
```

### 3. 端口冲突
如果 5001 端口被占用，可以修改启动端口。编辑 `app.py` 最后一行：
```python
app.run(host='0.0.0.0', port=YOUR_PORT, debug=True)
```
同时需要修改前端 `src/services/apiService.ts` 中的 `API_BASE_URL`。

### 4. 跨域问题
后端已配置 Flask-CORS，如仍有问题可检查网络设置。