#!/usr/bin/env python3
"""
下载YOLO11s物体检测模型
"""

from ultralytics import YOLO

print("正在下载YOLO11s模型...")
print("模型大小：约19MB")
print("预期精度提升：+8% mAP（相比YOLO11n）")
print("-" * 50)

try:
    # 下载并加载模型（首次会自动下载）
    model = YOLO('yolo11s.pt')
    print("\n✓ YOLO11s模型下载成功！")
    print(f"✓ 模型保存路径：{model.ckpt_path}")
    
    # 显示模型信息
    print("\n模型信息：")
    print(f"  - 参数量：9.44M")
    print(f"  - 权重大小：~19MB")
    print(f"  - mAP50：62.1%")
    print(f"  - 适用场景：小物体检测（手机、书本等）")
    
except Exception as e:
    print(f"\n✗ 下载失败：{e}")
    print("\n如果下载失败，请检查网络连接或手动下载：")
    print("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt")
