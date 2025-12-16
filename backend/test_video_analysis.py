#!/usr/bin/env python3
"""
视频行为分析API测试脚本
"""

import requests
import json
import base64
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

def create_test_frame(frame_index):
    """创建测试帧"""
    # 创建一个简单的测试图像
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255  # 白色背景
    
    # 添加一些简单的图形元素
    # 绘制一个圆圈代表学生头部
    center_x = 320 + (frame_index % 10) * 5
    center_y = 240 + (frame_index % 5) * 10
    cv2.circle(img, (center_x, center_y), 30, (0, 255, 0), -1)
    
    # 添加帧编号
    cv2.putText(img, f"Frame {frame_index}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    return img

def frame_to_base64(frame):
    """将帧转换为Base64"""
    # 将BGR图像转换为RGB
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 转换为PIL图像
    pil_image = Image.fromarray(rgb_image)
    
    # 保存到BytesIO对象
    buffer = BytesIO()
    pil_image.save(buffer, format='JPEG', quality=80)
    buffer.seek(0)
    
    # 编码为Base64
    img_str = base64.b64encode(buffer.read()).decode()
    return f"data:image/jpeg;base64,{img_str}"

def test_video_analysis_api():
    """测试视频行为分析API"""
    print("=== 视频行为分析API测试 ===")
    
    # 1. 创建测试帧（模拟10帧视频）
    print("1. 创建测试帧...")
    frames = []
    for i in range(10):
        frame = create_test_frame(i)
        frames.append(frame)
    
    print(f"   ✓ 创建了 {len(frames)} 个测试帧")
    
    # 2. 转换为Base64
    print("2. 转换帧为Base64格式...")
    base64_frames = []
    for frame in frames:
        base64_frame = frame_to_base64(frame)
        base64_frames.append(base64_frame)
    
    print("   ✓ 转换完成")
    
    # 3. 发送到API
    print("3. 发送请求到视频行为分析API...")
    url = "http://localhost:5001/api/behavior-analyze-video-base64"
    
    payload = {
        "images": base64_frames
    }
    
    try:
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("   ✓ API调用成功")
            print(f"   处理时间: {result['result']['processing_time']:.2f} 秒")
            print(f"   帧数: {result['result']['frame_count']}")
            print(f"   学生总数: {sum(frame['student_count'] for frame in result['result']['frame_results'])}")
            
            # 显示汇总结果
            summary = result['result']['summary']
            print("\n   汇总分析结果:")
            print(f"     总帧数: {summary['total_frames']}")
            print("     行为统计:")
            for behavior, percentage in summary['behavior_percentages'].items():
                if percentage > 0:
                    print(f"       {behavior}: {percentage}%")
            
            print("     分析结论:")
            for i, conclusion in enumerate(summary['conclusions']):
                print(f"       {i+1}. {conclusion}")
                
        else:
            print(f"   ✗ API调用失败，状态码: {response.status_code}")
            print(f"   错误信息: {response.text}")
            
    except Exception as e:
        print(f"   ✗ 请求过程中出错: {e}")

def test_single_frame_analysis():
    """测试单帧行为分析API"""
    print("\n=== 单帧行为分析API测试 ===")
    
    # 1. 创建测试帧
    print("1. 创建测试帧...")
    frame = create_test_frame(0)
    base64_frame = frame_to_base64(frame)
    print("   ✓ 测试帧创建完成")
    
    # 2. 发送到API
    print("2. 发送请求到单帧行为分析API...")
    url = "http://localhost:5001/api/behavior-analyze-base64"
    
    payload = {
        "image": base64_frame
    }
    
    try:
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("   ✓ API调用成功")
            print(f"   处理时间: {result['result']['processing_time']:.2f} 秒")
            print(f"   检测到学生数: {result['result']['student_count']}")
            
            if result['result']['student_count'] > 0:
                first_behavior = result['result']['behaviors'][0]
                print(f"   示例学生行为: {first_behavior['head_pose']} / {first_behavior['hand_activity']}")
                
        else:
            print(f"   ✗ API调用失败，状态码: {response.status_code}")
            print(f"   错误信息: {response.text}")
            
    except Exception as e:
        print(f"   ✗ 请求过程中出错: {e}")

if __name__ == "__main__":
    print("开始测试视频行为分析功能...")
    print("=" * 50)
    
    # 测试单帧分析
    test_single_frame_analysis()
    
    # 测试视频分析
    test_video_analysis_api()
    
    print("\n" + "=" * 50)
    print("测试完成!")