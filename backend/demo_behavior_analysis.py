#!/usr/bin/env python3
"""
行为分析功能演示脚本
"""

import cv2
import numpy as np
import requests
import base64
import json
from PIL import Image
from io import BytesIO

def create_test_image():
    """创建一个测试图像"""
    # 创建一个白色背景的图像
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # 在图像上绘制一些简单的图形来模拟教室场景
    # 绘制几个矩形代表桌子
    cv2.rectangle(img, (50, 100), (250, 200), (200, 200, 200), -1)  # 桌子1
    cv2.rectangle(img, (350, 100), (550, 200), (200, 200, 200), -1)  # 桌子2
    cv2.rectangle(img, (50, 300), (250, 400), (200, 200, 200), -1)   # 桌子3
    
    # 绘制一些圆形代表学生的头部
    cv2.circle(img, (150, 150), 30, (0, 255, 0), -1)  # 学生1
    cv2.circle(img, (450, 150), 30, (0, 255, 0), -1)  # 学生2
    cv2.circle(img, (150, 350), 30, (0, 255, 0), -1)  # 学生3
    
    # 绘制一些线条代表学生的身体
    cv2.line(img, (150, 180), (150, 250), (0, 0, 255), 3)  # 学生1身体
    cv2.line(img, (450, 180), (450, 250), (0, 0, 255), 3)  # 学生2身体
    cv2.line(img, (150, 380), (150, 450), (0, 0, 255), 3)  # 学生3身体
    
    # 绘制一些线条代表学生的胳膊
    cv2.line(img, (150, 200), (120, 230), (255, 0, 0), 2)  # 学生1左臂
    cv2.line(img, (150, 200), (180, 230), (255, 0, 0), 2)  # 学生1右臂
    cv2.line(img, (450, 200), (420, 230), (255, 0, 0), 2)  # 学生2左臂
    cv2.line(img, (450, 200), (480, 230), (255, 0, 0), 2)  # 学生2右臂
    cv2.line(img, (150, 400), (120, 430), (255, 0, 0), 2)  # 学生3左臂
    cv2.line(img, (150, 400), (180, 430), (255, 0, 0), 2)  # 学生3右臂
    
    return img

def image_to_base64(image):
    """将OpenCV图像转换为Base64编码"""
    # 将BGR图像转换为RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 转换为PIL图像
    pil_image = Image.fromarray(rgb_image)
    
    # 保存到BytesIO对象
    buffer = BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)
    
    # 编码为Base64
    img_str = base64.b64encode(buffer.read()).decode()
    return f"data:image/png;base64,{img_str}"

def test_behavior_analysis():
    """测试行为分析功能"""
    print("=== 教室行为识别演示 ===")
    
    # 1. 创建测试图像
    print("1. 创建测试图像...")
    test_image = create_test_image()
    print("   ✓ 测试图像创建成功")
    
    # 2. 将图像转换为Base64
    print("2. 转换图像为Base64格式...")
    base64_image = image_to_base64(test_image)
    print("   ✓ 图像转换完成")
    
    # 3. 发送请求到后端API
    print("3. 发送行为分析请求到后端...")
    url = "http://localhost:5001/api/behavior-analyze-base64"
    
    payload = {
        "image": base64_image
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            print("   ✓ 请求成功")
            
            # 4. 解析并显示结果
            print("4. 分析结果:")
            print(f"   处理时间: {result['result']['processing_time']:.2f} 秒")
            print(f"   检测到的学生数量: {result['result']['student_count']}")
            
            if result['result']['behaviors']:
                print("   检测到的行为:")
                for i, behavior in enumerate(result['result']['behaviors']):
                    print(f"     学生 {i+1}:")
                    print(f"       头部姿态: {behavior.get('head_pose', '未知')}")
                    print(f"       手部活动: {behavior.get('hand_activity', '未知')}")
                    print(f"       置信度: {behavior.get('confidence', 0):.2f}")
                    
                    if 'desktop_objects' in behavior:
                        print(f"       桌面物品: {len(behavior['desktop_objects'])} 个")
            else:
                print("   未检测到明确的行为（这在测试图像中是正常的）")
                
        else:
            print(f"   ✗ 请求失败，状态码: {response.status_code}")
            print(f"   错误信息: {response.text}")
            
    except Exception as e:
        print(f"   ✗ 请求出错: {e}")

def test_health_check():
    """测试健康检查"""
    print("\n=== 健康检查 ===")
    url = "http://localhost:5001/health"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            result = response.json()
            print("   ✓ 健康检查通过")
            print(f"   模型加载状态: {'已加载' if result['model_loaded'] else '未加载'}")
            print(f"   服务状态: {result['status']}")
        else:
            print(f"   ✗ 健康检查失败，状态码: {response.status_code}")
    except Exception as e:
        print(f"   ✗ 健康检查出错: {e}")

if __name__ == "__main__":
    print("开始测试教室行为识别功能...")
    print("=" * 50)
    
    # 测试健康检查
    test_health_check()
    
    print()
    
    # 测试行为分析
    test_behavior_analysis()
    
    print("\n" + "=" * 50)
    print("演示完成!")
    print("\n接下来您可以:")
    print("1. 在前端界面中使用行为分析功能")
    print("2. 上传真实的教室视频进行分析")
    print("3. 查看分析结果，了解学生的行为状态")