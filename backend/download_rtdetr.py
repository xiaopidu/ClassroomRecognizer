#!/usr/bin/env python3
"""下载RT-DETR模型"""

import urllib.request
import ssl
import os
import sys

# 禁用SSL验证
ssl._create_default_https_context = ssl._create_unverified_context

url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/rtdetr-l.pt"
save_path = os.path.expanduser("~/.cache/ultralytics/rtdetr-l.pt")

os.makedirs(os.path.dirname(save_path), exist_ok=True)

print(f"🚀 开始下载 RT-DETR-L 模型...")
print(f"📥 下载链接: {url}")
print(f"💾 保存位置: {save_path}\n")

def progress_hook(block_num, block_size, total_size):
    """显示下载进度"""
    downloaded = block_num * block_size
    percent = min(downloaded * 100 / total_size, 100)
    mb_downloaded = downloaded / (1024 * 1024)
    mb_total = total_size / (1024 * 1024)
    
    # 创建进度条
    bar_length = 40
    filled_length = int(bar_length * percent / 100)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    
    print(f"\r进度: [{bar}] {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='', flush=True)

try:
    urllib.request.urlretrieve(url, save_path, progress_hook)
    print(f"\n\n✅ 下载完成!")
    
    # 检查文件大小
    file_size = os.path.getsize(save_path) / (1024 * 1024)
    print(f"📁 文件大小: {file_size:.1f} MB")
    print(f"📂 文件位置: {save_path}")
    
    # 验证模型
    print("\n🔍 验证模型文件...")
    from ultralytics import RTDETR
    model = RTDETR(save_path)
    print("✅ 模型验证成功! RT-DETR已准备就绪!")
    
    sys.exit(0)
    
except KeyboardInterrupt:
    print("\n\n⚠️  下载被用户中断")
    sys.exit(1)
except Exception as e:
    print(f"\n\n❌ 下载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
