#!/usr/bin/env python3
"""
测试YOLOv8模型下载和基本功能
"""

import cv2
import torch
from ultralytics import YOLO
import os

def test_pose_model():
    """测试姿态检测模型"""
    print("正在下载/加载YOLOv8 Pose模型...")
    try:
        # 加载姿态检测模型
        pose_model = YOLO('yolov8n-pose.pt')
        print("✓ YOLOv8 Pose模型加载成功")
        
        # 创建一个简单的测试图像
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        test_img[:] = (255, 255, 255)  # 白色背景
        
        # 进行一次简单的推理测试
        results = pose_model(test_img, verbose=False)
        print("✓ YOLOv8 Pose模型推理测试成功")
        
        return True
    except Exception as e:
        print(f"✗ YOLOv8 Pose模型测试失败: {e}")
        return False

def test_object_model():
    """测试物体检测模型"""
    print("正在下载/加载YOLOv8 Object Detection模型...")
    try:
        # 加载物体检测模型
        object_model = YOLO('yolov8n.pt')
        print("✓ YOLOv8 Object Detection模型加载成功")
        
        # 创建一个简单的测试图像
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        test_img[:] = (255, 255, 255)  # 白色背景
        
        # 进行一次简单的推理测试
        results = object_model(test_img, verbose=False)
        print("✓ YOLOv8 Object Detection模型推理测试成功")
        
        return True
    except Exception as e:
        print(f"✗ YOLOv8 Object Detection模型测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始测试YOLOv8模型下载和功能...")
    print("=" * 50)
    
    # 导入numpy
    import numpy as np
    
    # 测试姿态模型
    pose_success = test_pose_model()
    print()
    
    # 测试物体检测模型
    object_success = test_object_model()
    print()
    
    if pose_success and object_success:
        print("🎉 所有模型测试通过！")
        print("模型已成功下载并可以正常使用。")
    else:
        print("❌ 部分模型测试失败，请检查网络连接或重新运行脚本。")