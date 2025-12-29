#!/usr/bin/env python3
"""
下载YOLO11m物体检测模型
"""

from ultralytics import YOLO

print("正在下载YOLO11m模型...")
print("模型大小：约38.8MB")
print("预期精度提升：+14% mAP（相比YOLO11n）")
print("预期精度提升：+6% mAP（相比YOLO11s）")
print("-" * 50)

try:
    # 下载并加载模型（首次会自动下载）
    model = YOLO('yolo11m.pt')
    print("\n✓ YOLO11m模型下载成功！")
    print(f"✓ 模型保存路径：{model.ckpt_path}")
    
    # 显示模型信息
    print("\n模型信息：")
    print(f"  - 参数量：~20M")
    print(f"  - 权重大小：38.8MB")
    print(f"  - mAP50：~68%")
    print(f"  - mAP50-95：~52%")
    print(f"  - 推理速度：2.4ms")
    print(f"  - 适用场景：高精度物体检测（遮挡、小物体）")
    
    print("\n相比YOLO11s的优势：")
    print("  ✓ 对遮挡场景更鲁棒")
    print("  ✓ 小物体检测率更高")
    print("  ✓ 边界框定位更准确")
    print("  ⚠ 速度稍慢（但单张图片几乎无感）")
    
except Exception as e:
    print(f"\n✗ 下载失败：{e}")
    print("\n如果下载失败，请检查网络连接或手动下载：")
    print("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt")
