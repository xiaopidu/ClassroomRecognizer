#!/usr/bin/env python3
"""
视频行为分析演示脚本
模拟分析300帧视频并生成可视化结果
"""

import cv2
import numpy as np
import requests
import base64
import json
from PIL import Image
from io import BytesIO
import time
import matplotlib
import matplotlib.pyplot as plt

# 设置matplotlib支持中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def create_classroom_scene_frame(frame_index):
    """
    创建一个教室场景的模拟帧
    根据帧索引改变学生的行为状态
    """
    # 创建一个教室背景
    img = np.ones((480, 640, 3), dtype=np.uint8) * 240  # 浅灰色背景
    
    # 绘制黑板
    cv2.rectangle(img, (50, 30), (590, 150), (30, 30, 30), -1)  # 黑板
    cv2.rectangle(img, (45, 25), (595, 155), (0, 0, 0), 2)      # 黑板边框
    
    # 绘制讲台
    cv2.rectangle(img, (250, 160), (390, 200), (139, 69, 19), -1)  # 讲台
    
    # 绘制课桌（3行4列）
    desk_positions = []
    for row in range(3):
        for col in range(4):
            x = 100 + col * 120
            y = 220 + row * 80
            desk_positions.append((x, y))
            # 绘制桌子
            cv2.rectangle(img, (x-40, y-20), (x+40, y+20), (139, 115, 85), -1)
            cv2.rectangle(img, (x-40, y-20), (x+40, y+20), (101, 67, 33), 2)
    
    # 绘制学生（根据帧索引改变行为）
    for i, (desk_x, desk_y) in enumerate(desk_positions):
        # 根据帧索引和学生索引决定行为
        time_factor = (frame_index // 10) % 30  # 每10帧改变一次行为模式
        student_state = (i + time_factor) % 6   # 6种不同的行为状态
        
        # 学生头部位置
        head_x = desk_x
        head_y = desk_y - 40
        
        # 根据状态绘制不同的学生姿态
        if student_state == 0:  # 抬头听课
            # 绘制头部
            cv2.circle(img, (head_x, head_y), 15, (0, 255, 0), -1)
            # 绘制身体（直立）
            cv2.line(img, (head_x, head_y + 15), (head_x, head_y + 50), (0, 0, 255), 3)
            # 绘制胳膊
            cv2.line(img, (head_x, head_y + 25), (head_x - 15, head_y + 35), (255, 0, 0), 2)
            cv2.line(img, (head_x, head_y + 25), (head_x + 15, head_y + 35), (255, 0, 0), 2)
            
        elif student_state == 1:  # 低头看手机
            # 绘制头部（低头）
            cv2.circle(img, (head_x, head_y + 10), 15, (0, 255, 255), -1)
            # 绘制身体
            cv2.line(img, (head_x, head_y + 25), (head_x, head_y + 60), (0, 0, 255), 3)
            # 绘制胳膊（拿着手机）
            cv2.line(img, (head_x, head_y + 35), (head_x - 20, head_y + 25), (255, 0, 0), 2)
            cv2.line(img, (head_x, head_y + 35), (head_x + 20, head_y + 25), (255, 0, 0), 2)
            # 绘制手机
            cv2.rectangle(img, (head_x - 25, head_y + 15), (head_x - 15, head_y + 5), (0, 0, 0), -1)
            
        elif student_state == 2:  # 认真记笔记
            # 绘制头部
            cv2.circle(img, (head_x, head_y), 15, (255, 0, 0), -1)
            # 绘制身体
            cv2.line(img, (head_x, head_y + 15), (head_x, head_y + 50), (0, 0, 255), 3)
            # 绘制胳膊（写字姿势）
            cv2.line(img, (head_x, head_y + 25), (head_x - 25, head_y + 35), (255, 0, 0), 2)
            cv2.line(img, (head_x, head_y + 25), (head_x + 15, head_y + 35), (255, 0, 0), 2)
            # 绘制笔
            cv2.line(img, (head_x - 25, head_y + 35), (head_x - 30, head_y + 50), (0, 0, 0), 2)
            
        elif student_state == 3:  # 打瞌睡
            # 绘制头部（趴着）
            cv2.circle(img, (head_x, head_y + 20), 15, (128, 128, 128), -1)
            # 绘制身体（趴着）
            cv2.line(img, (head_x, head_y + 35), (head_x, head_y + 50), (0, 0, 255), 3)
            cv2.line(img, (head_x, head_y + 35), (head_x - 20, head_y + 45), (0, 0, 255), 3)
            cv2.line(img, (head_x, head_y + 35), (head_x + 20, head_y + 45), (0, 0, 255), 3)
            
        elif student_state == 4:  # 回头说话
            # 绘制头部（转向侧面）
            cv2.circle(img, (head_x + 15, head_y), 15, (255, 255, 0), -1)
            # 绘制身体
            cv2.line(img, (head_x + 15, head_y + 15), (head_x + 15, head_y + 50), (0, 0, 255), 3)
            # 绘制胳膊
            cv2.line(img, (head_x + 15, head_y + 25), (head_x, head_y + 35), (255, 0, 0), 2)
            cv2.line(img, (head_x + 15, head_y + 25), (head_x + 30, head_y + 35), (255, 0, 0), 2)
            
        else:  # 正常状态
            # 绘制头部
            cv2.circle(img, (head_x, head_y), 15, (255, 165, 0), -1)
            # 绘制身体
            cv2.line(img, (head_x, head_y + 15), (head_x, head_y + 50), (0, 0, 255), 3)
            # 绘制胳膊
            cv2.line(img, (head_x, head_y + 25), (head_x - 15, head_y + 35), (255, 0, 0), 2)
            cv2.line(img, (head_x, head_y + 25), (head_x + 15, head_y + 35), (255, 0, 0), 2)
    
    # 添加帧编号
    cv2.putText(img, f"Frame: {frame_index}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return img

def frames_to_base64(frames):
    """将帧列表转换为Base64编码列表"""
    base64_frames = []
    for frame in frames:
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
        base64_frames.append(f"data:image/jpeg;base64,{img_str}")
    
    return base64_frames

def plot_behavior_analysis(summary):
    """绘制行为分析图表"""
    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('教室行为分析汇总报告', fontsize=16, fontweight='bold')
    
    # 1. 头部姿态分布饼图
    head_poses = ['looking_up', 'looking_down', 'neutral']
    head_pose_data = [max(0.1, summary['behavior_percentages'].get(pose, 0)) for pose in head_poses]  # 确保不为0
    colors_head = ['#00FF00', '#FF0000', '#FFFF00']
    
    # 检查数据有效性
    if sum(head_pose_data) > 0:
        wedges, texts, autotexts = ax1.pie(head_pose_data, labels=head_poses, autopct='%1.1f%%', colors=colors_head)
        # 设置标签字体
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    else:
        ax1.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax1.transAxes)
    ax1.set_title('头部姿态分布')
    
    # 2. 手部活动分布饼图
    hand_activities = ['writing', 'using_phone', 'resting']
    hand_activity_data = [max(0.1, summary['behavior_percentages'].get(activity, 0)) for activity in hand_activities]  # 确保不为0
    colors_hand = ['#FF0000', '#00FFFF', '#FF00FF']
    
    # 检查数据有效性
    if sum(hand_activity_data) > 0:
        wedges, texts, autotexts = ax2.pie(hand_activity_data, labels=hand_activities, autopct='%1.1f%%', colors=colors_hand)
        # 设置标签字体
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    else:
        ax2.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('手部活动分布')
    
    # 3. 桌面物品柱状图
    if summary['object_stats'] and any(v > 0 for v in summary['object_stats'].values()):
        objects = list(summary['object_stats'].keys())
        object_counts = list(summary['object_stats'].values())
        colors_objects = plt.cm.Set3(range(len(objects)))
        
        bars = ax3.bar(objects, object_counts, color=colors_objects)
        ax3.set_title('桌面物品统计')
        ax3.set_ylabel('出现帧数')
        ax3.tick_params(axis='x', rotation=45)
        
        # 在柱状图上添加数值标签
        for bar, count in zip(bars, object_counts):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom')
    else:
        ax3.text(0.5, 0.5, '无桌面物品检测', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('桌面物品统计')
    
    # 4. 分析结论文本
    ax4.axis('off')
    conclusions_text = '\n'.join([f'• {conclusion}' for conclusion in summary['conclusions']])
    ax4.text(0.1, 0.9, '分析结论:', fontsize=12, fontweight='bold', transform=ax4.transAxes)
    ax4.text(0.1, 0.8, conclusions_text, fontsize=10, transform=ax4.transAxes,
             verticalalignment='top')
    
    # 添加统计信息
    stats_text = f"""
    总帧数: {summary['total_frames']}
    平均每帧学生数: {summary['total_frames'] / max(1, summary['total_frames']):.1f}
    处理时间: {summary.get('processing_time', 0):.2f}秒
    """
    ax4.text(0.1, 0.3, stats_text, fontsize=10, transform=ax4.transAxes,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.savefig('behavior_analysis_report.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return 'behavior_analysis_report.png'

def demo_video_analysis():
    """演示视频行为分析"""
    print("=== 视频行为分析演示 ===")
    print("正在生成300帧模拟教室场景...")
    
    # 1. 生成300帧模拟视频
    frames = []
    for i in range(300):
        frame = create_classroom_scene_frame(i)
        frames.append(frame)
        
        # 每50帧显示一次进度
        if (i + 1) % 50 == 0:
            print(f"   已生成 {i + 1}/300 帧")
    
    print("   ✓ 300帧模拟视频生成完成")
    
    # 2. 转换为Base64编码
    print("2. 转换帧为Base64格式...")
    base64_frames = frames_to_base64(frames)
    print("   ✓ 转换完成")
    
    # 3. 发送请求到后端API（这里我们直接调用分析器，因为在演示中避免网络延迟）
    print("3. 分析300帧视频...")
    
    # 直接调用分析器进行演示
    from behavior_service import get_behavior_analyzer
    analyzer = get_behavior_analyzer()
    
    start_time = time.time()
    result = analyzer.analyze_video_frames(frames)
    processing_time = time.time() - start_time
    
    print(f"   ✓ 分析完成，耗时: {processing_time:.2f} 秒")
    
    # 4. 显示分析结果
    print("4. 分析结果:")
    summary = result['summary']
    print(f"   总帧数: {summary['total_frames']}")
    print(f"   处理时间: {result['processing_time']:.2f} 秒")
    
    print("   行为统计:")
    for behavior, percentage in summary['behavior_percentages'].items():
        if percentage > 0:
            print(f"     {behavior}: {percentage}%")
    
    print("   桌面物品统计:")
    for obj, percentage in summary['object_percentages'].items():
        print(f"     {obj}: {percentage}%")
    
    print("   分析结论:")
    for i, conclusion in enumerate(summary['conclusions']):
        print(f"     {i+1}. {conclusion}")
    
    # 5. 生成可视化图表
    print("5. 生成可视化分析报告...")
    try:
        chart_path = plot_behavior_analysis(summary)
        print(f"   ✓ 分析报告已保存为: {chart_path}")
    except Exception as e:
        print(f"   ! 图表生成失败: {e}")
        print("   但不影响主要功能演示")
    
    # 6. 显示一些关键帧的标注结果
    print("6. 关键帧标注示例:")
    frame_examples = [0, 75, 150, 225, 299]  # 显示第1, 76, 151, 226, 300帧
    for i, frame_idx in enumerate(frame_examples):
        frame_result = result['frame_results'][frame_idx]
        print(f"   帧 {frame_idx + 1}: 检测到 {frame_result['student_count']} 名学生")
        if frame_result['student_count'] > 0:
            first_behavior = frame_result['behaviors'][0]
            print(f"     示例学生行为: {first_behavior['head_pose']} / {first_behavior['hand_activity']}")
    
    return result

if __name__ == "__main__":
    print("开始演示视频行为分析功能...")
    print("=" * 60)
    
    # 运行演示
    result = demo_video_analysis()
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("\n分析结果已保存为 'behavior_analysis_report.png'")
    print("您可以查看该文件了解详细的分析报告")