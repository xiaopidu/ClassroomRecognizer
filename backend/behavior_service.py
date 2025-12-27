#!/usr/bin/env python3
"""
教室行为识别服务
使用YOLOv8 Pose + YOLOv8 Object Detection进行学生行为分析
"""

import cv2
import numpy as np
from ultralytics import YOLO, RTDETR
import logging
from typing import Dict, List, Any
import time
import base64
from io import BytesIO
from PIL import Image

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClassroomBehaviorAnalyzer:
    def __init__(self, behavior_params=None):
        """初始化行为分析器"""
        logger.info("正在加载检测模型...")
        
        # 加载姿态检测模型(保持YOLOv8)
        self.pose_model = YOLO('yolov8n-pose.pt')
        logger.info("✓ 姿态检测模型(YOLOv8 Pose)加载成功")
        
        # 加载物体检测模型(升级为RT-DETR)
        try:
            self.object_model = RTDETR('rtdetr-l.pt')
            logger.info("✓ 物体检测模型(RT-DETR-L)加载成功")
        except Exception as e:
            logger.warning(f"RT-DETR模型加载失败: {e}, 回退使用YOLOv8")
            self.object_model = YOLO('yolov8n.pt')
            logger.info("✓ 物体检测模型(YOLOv8)加载成功")
        
        # COCO数据集的类别标签
        self.coco_labels = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # 行为颜色映射
        self.behavior_colors = {
            "looking_up": (0, 255, 0),      # 绿色 - 抬头
            "looking_down": (0, 0, 255),    # 红色 - 低头
            "neutral": (255, 255, 0),       # 黄色 - 中性
            "writing": (255, 0, 0),         # 蓝色 - 写字
            "using_phone": (0, 255, 255),   # 青色 - 玩手机
            "resting": (255, 0, 255),       # 紫色 - 休息
            "unknown": (128, 128, 128)      # 灰色 - 未知
        }
        
        # 行为分析参数
        self.behavior_params = {
            "head_up_threshold": 2,        # 抬头阈值(正常坐姿算抬头)
            "head_down_threshold": 8,      # 低头阈值(明显低头才算)
            "writing_threshold": 30,       # 记笔记阈值(更敏感)
            "phone_threshold": -10,        # 玩手机阈倿(更敏感)
            "object_min_confidence": 0.2   # 物体检测最小置信度(降低以提高检测率)
        }
        
        # 如果提供了自定义参数，则更新默认参数
        if behavior_params:
            self.behavior_params.update(behavior_params)
        
        logger.info("行为分析器初始化完成")
        
    def update_params(self, params):
        """更新行为分析参数"""
        self.behavior_params.update(params)
        logger.info(f"行为分析参数已更新: {self.behavior_params}")
    
    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        分析单帧图像中的学生行为
        
        Args:
            frame: 输入图像帧
            
        Returns:
            包含行为分析结果的字典
        """
        start_time = time.time()
        
        # 进行姿态检测
        pose_results = self.pose_model(frame, verbose=False)
        
        # 进行物体检测
        object_results = self.object_model(frame, verbose=False)
        
        # 分析学生行为
        behavior_data = self._analyze_student_behaviors(pose_results, object_results)
        
        # 绘制行为标记
        annotated_frame = self._draw_behavior_annotations(frame.copy(), behavior_data, object_results)
        
        # 将标注后的图像转换为Base64
        annotated_image = self._frame_to_base64(annotated_frame)
        
        processing_time = time.time() - start_time
        
        return {
            "timestamp": time.time(),
            "processing_time": processing_time,
            "student_count": len(behavior_data),
            "behaviors": behavior_data,
            "annotated_image": annotated_image
        }
    
    def analyze_video_frames(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        分析视频帧序列中的学生行为并进行汇总
        
        Args:
            frames: 视频帧列表
            
        Returns:
            包含汇总分析结果的字典
        """
        start_time = time.time()
        
        # 存储每帧的分析结果
        frame_results = []
        
        # 分析每一帧
        for i, frame in enumerate(frames):
            result = self.analyze_frame(frame)
            result["frame_index"] = i
            frame_results.append(result)
        
        # 汇总分析结果
        summary = self._summarize_behavior_analysis(frame_results)
        
        processing_time = time.time() - start_time
        
        return {
            "timestamp": time.time(),
            "processing_time": processing_time,
            "frame_count": len(frames),
            "frame_results": frame_results,
            "summary": summary
        }
    
    def _analyze_student_behaviors(self, pose_results, object_results) -> List[Dict]:
        """
        分析学生行为
        
        Args:
            pose_results: 姿态检测结果
            object_results: 物体检测结果
            
        Returns:
            学生行为列表
        """
        behaviors = []
        
        # 处理姿态检测结果
        for result in pose_results:
            if result.keypoints is not None:
                for i, kpts in enumerate(result.keypoints.data):
                    # 分析每个检测到的人的姿态
                    behavior = self._analyze_single_person_pose(kpts)
                    if behavior:
                        # 获取边界框信息
                        if result.boxes is not None and i < len(result.boxes):
                            box = result.boxes[i]
                            xyxy = box.xyxy.cpu().numpy()[0]
                            bbox = {
                                "x1": int(xyxy[0]),
                                "y1": int(xyxy[1]),
                                "x2": int(xyxy[2]),
                                "y2": int(xyxy[3])
                            }
                            behavior["bbox"] = bbox
                            
                            # 检测该学生区域内的物品(与个人分析保持一致)
                            student_objects = self._analyze_desktop_objects_in_bbox(object_results, bbox)
                            behavior["desktop_objects"] = student_objects
                        else:
                            behavior["desktop_objects"] = []
                        
                        behaviors.append(behavior)
            
        return behaviors
    
    def _analyze_single_person_pose(self, keypoints) -> Dict:
        """
        分析单个人的姿态
        
        Args:
            keypoints: 关键点数据
            
        Returns:
            行为分析结果
        """
        # 提取关键点坐标 (17个关键点)
        # 0: 鼻子, 5: 左肩, 6: 右肩, 9: 左腕, 10: 右腕
        nose = keypoints[0][:2].cpu().numpy() if keypoints[0][2] > 0.5 else None
        left_shoulder = keypoints[5][:2].cpu().numpy() if keypoints[5][2] > 0.5 else None
        right_shoulder = keypoints[6][:2].cpu().numpy() if keypoints[6][2] > 0.5 else None
        left_wrist = keypoints[9][:2].cpu().numpy() if keypoints[9][2] > 0.5 else None
        right_wrist = keypoints[10][:2].cpu().numpy() if keypoints[10][2] > 0.5 else None
        
        # 如果关键点不可见，跳过分析
        if nose is None or left_shoulder is None or right_shoulder is None:
            return {}
        
        # 计算头部姿态（低头/抬头）
        head_pose = self._analyze_head_pose(nose, left_shoulder, right_shoulder)
        
        # 分析手部活动（记笔记/玩手机）
        hand_activity = self._analyze_hand_activity(left_wrist, right_wrist, nose)
        
        return {
            "head_pose": head_pose,  # "looking_up", "looking_down", "neutral"
            "hand_activity": hand_activity,  # "writing", "using_phone", "resting", "unknown"
            "confidence": float(keypoints[:, 2].mean().cpu().numpy()),  # 平均置信度
            "keypoints_visible": int(keypoints[:, 2].sum().cpu().numpy())  # 可见关键点数量
        }
    
    def _analyze_head_pose(self, nose, left_shoulder, right_shoulder) -> str:
        """
        分析头部姿态（低头/抬头）
        
        Args:
            nose: 鼻子坐标
            left_shoulder: 左肩坐标
            right_shoulder: 右肩坐标
            
        Returns:
            头部姿态标签
        """
        # 计算肩膀中点
        shoulder_center = (left_shoulder + right_shoulder) / 2
        
        # 计算鼻子与肩膀的相对位置
        vertical_diff = nose[1] - shoulder_center[1]  # y坐标差值
        
        if vertical_diff < self.behavior_params["head_up_threshold"]:  # 鼻子明显高于肩膀中心
            return "looking_up"
        elif vertical_diff > self.behavior_params["head_down_threshold"]:  # 鼻子明显低于肩膀中心
            return "looking_down"
        else:
            return "neutral"
    
    def _analyze_hand_activity(self, left_wrist, right_wrist, nose) -> str:
        """
        分析手部活动
        
        Args:
            left_wrist: 左手腕坐标
            right_wrist: 右手腕坐标
            nose: 鼻子坐标
            
        Returns:
            手部活动标签
        """
        # 如果手腕关键点不可见，无法判断
        if left_wrist is None and right_wrist is None:
            return "unknown"
        
        # 优先使用可见的手腕进行分析
        wrist = None
        if left_wrist is not None:
            wrist = left_wrist
        elif right_wrist is not None:
            wrist = right_wrist
        else:
            return "unknown"
        
        # 计算手腕与鼻子的相对位置
        vertical_diff = wrist[1] - nose[1]  # y坐标差值
        
        # 简单判断：手腕位置较低可能在记笔记，位置较高可能在玩手机
        if vertical_diff > self.behavior_params["writing_threshold"]:  # 手腕明显低于鼻子（在桌面上）
            return "writing"
        elif vertical_diff < self.behavior_params["phone_threshold"]:  # 手腕明显高于鼻子（举起手机）
            return "using_phone"
        else:
            return "resting"
    
    def _analyze_desktop_objects(self, object_results) -> List[Dict]:
        """
        分析桌面物品（书/电脑）
        
        Args:
            object_results: 物体检测结果
            
        Returns:
            桌面物品列表
        """
        desktop_objects = []
        
        for result in object_results:
            if result.boxes is not None:
                for box in result.boxes:
                    # 获取物体类别和置信度
                    class_id = int(box.cls.cpu().numpy()[0])
                    confidence = float(box.conf.cpu().numpy()[0])
                    label = self.coco_labels[class_id] if class_id < len(self.coco_labels) else "unknown"
                    
                    # 调试日志:输出所有检测到的物体
                    if confidence >= 0.15:  # 显示置信度>=0.15的所有物体
                        logger.info(f"检测到物体: {label}, 置信度: {confidence:.3f}")
                    
                    # 检查是否是我们关心的物品并且置信度足够高
                    if label in ["book", "laptop", "cell phone", "keyboard"] and confidence >= self.behavior_params["object_min_confidence"]:
                        # 获取边界框坐标
                        xyxy = box.xyxy.cpu().numpy()[0]
                        x1, y1, x2, y2 = xyxy
                        
                        desktop_objects.append({
                            "label": label,
                            "confidence": confidence,
                            "bbox": {
                                "x1": int(x1),
                                "y1": int(y1),
                                "x2": int(x2),
                                "y2": int(y2)
                            }
                        })
                        logger.info(f"✓ 添加到桌面物品: {label}, 置信度: {confidence:.3f}")
        
        return desktop_objects
    
    def _analyze_desktop_objects_in_bbox(self, object_results, bbox: Dict[str, int]) -> List[str]:
        """检测边界框区域内的物品
        
        Args:
            object_results: 物体检测结果
            bbox: 学生边界框 {'x1', 'y1', 'x2', 'y2'}
            
        Returns:
            检测到的物品标签列表
        """
        detected_objects = []
        
        for result in object_results:
            if result.boxes is not None:
                for box in result.boxes:
                    obj_bbox = box.xyxy.cpu().numpy()[0]
                    obj_bbox_dict = {
                        'x1': int(obj_bbox[0]),
                        'y1': int(obj_bbox[1]),
                        'x2': int(obj_bbox[2]),
                        'y2': int(obj_bbox[3])
                    }
                    
                    # 检查物体是否在学生区域内(IoU > 0.1)
                    if self._calculate_iou(bbox, obj_bbox_dict) > 0.1:
                        class_id = int(box.cls.cpu().numpy()[0])
                        confidence = float(box.conf.cpu().numpy()[0])
                        label = self.coco_labels[class_id] if class_id < len(self.coco_labels) else "unknown"
                        
                        # 检查是否是我们关心的物品并且置信度足够高
                        if label in ["book", "laptop", "cell phone", "keyboard"] and confidence >= self.behavior_params["object_min_confidence"]:
                            detected_objects.append({
                                "label": label,
                                "confidence": confidence
                            })
        
        return detected_objects
    
    def _calculate_iou(self, bbox1: Dict[str, int], bbox2: Dict[str, int]) -> float:
        """计算两个边界框的IoU
        
        Args:
            bbox1: 边界框1 {'x1', 'y1', 'x2', 'y2'}
            bbox2: 边界框2 {'x1', 'y1', 'x2', 'y2'}
            
        Returns:
            IoU值 (0-1)
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1['x1'], bbox1['y1'], bbox1['x2'], bbox1['y2']
        x1_2, y1_2, x2_2, y2_2 = bbox2['x1'], bbox2['y1'], bbox2['x2'], bbox2['y2']
        
        # 计算交集
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 计算并集
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _draw_behavior_annotations(self, frame: np.ndarray, behaviors: List[Dict], object_results) -> np.ndarray:
        """
        在图像上绘制行为分析结果
        
        Args:
            frame: 原始图像帧
            behaviors: 行为分析结果
            object_results: 物体检测结果
            
        Returns:
            标注后的图像
        """
        # 中英文映射
        behavior_labels = {
            'looking_up': '抬头',
            'looking_down': '低头',
            'neutral': '中性',
            'writing': '记笔记',
            'using_phone': '玩手机',
            'resting': '休息',
            'unknown': '未知'
        }
        
        # 绘制学生行为
        for behavior in behaviors:
            if "bbox" in behavior:
                # 绘制边界框
                bbox = behavior["bbox"]
                color = self.behavior_colors.get(behavior["head_pose"], (255, 255, 255))
                
                # 绘制学生边界框
                cv2.rectangle(frame, 
                            (bbox["x1"], bbox["y1"]), 
                            (bbox["x2"], bbox["y2"]), 
                            color, 2)
                
                # 绘制行为标签（中文）
                head_pose_label = behavior_labels.get(behavior["head_pose"], behavior["head_pose"])
                hand_activity_label = behavior_labels.get(behavior["hand_activity"], behavior["hand_activity"])
                label = f'{head_pose_label} / {hand_activity_label}'
                
                # 使用PIL绘制中文
                from PIL import ImageFont, ImageDraw
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)
                
                # 尝试加载中文字体
                font = None
                font_paths = [
                    # Windows 字体
                    "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
                    "C:/Windows/Fonts/simhei.ttf",  # 黑体
                    "C:/Windows/Fonts/simsun.ttc",  # 宋体
                    # macOS 字体
                    "/System/Library/Fonts/PingFang.ttc",
                    # Linux 字体
                    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                    "/usr/share/fonts/truetype/arphic/uming.ttc",
                ]
                
                for font_path in font_paths:
                    try:
                        font = ImageFont.truetype(font_path, 16)
                        break
                    except:
                        continue
                
                # 如果所有字体都加载失败，使用 OpenCV 绘制英文
                if font is None:
                    logger.warning("无法加载中文字体，使用 OpenCV 绘制英文")
                    # 使用 OpenCV 直接绘制英文
                    cv2.rectangle(frame, (bbox["x1"], bbox["y1"]-20), 
                                (bbox["x1"]+150, bbox["y1"]), (0, 0, 0), -1)
                    cv2.putText(frame, f"{behavior['head_pose']}/{behavior['hand_activity']}",
                              (bbox["x1"], bbox["y1"]-5), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.4, color, 1)
                    continue  # 跳过 PIL 绘制
                
                # 绘制文本
                text_position = (bbox["x1"], max(0, bbox["y1"] - 25))
                # 添加黑色背景，提高可见度
                try:
                    bbox_text = draw.textbbox(text_position, label, font=font)
                    draw.rectangle(bbox_text, fill=(0, 0, 0, 128))
                    draw.text(text_position, label, fill=color, font=font)
                except Exception as e:
                    logger.warning(f"绘制中文失败: {e}，使用OpenCV绘制")
                    # 备选方案: 使用 OpenCV
                    cv2.rectangle(frame, (bbox["x1"], bbox["y1"]-20), 
                                (bbox["x1"]+150, bbox["y1"]), (0, 0, 0), -1)
                    cv2.putText(frame, f"{behavior['head_pose']}/{behavior['hand_activity']}",
                              (bbox["x1"], bbox["y1"]-5), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.4, color, 1)
                    continue
                
                # 转回 OpenCV 格式
                frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # 绘制桌面物品
        desktop_objects = self._analyze_desktop_objects(object_results)
        for obj in desktop_objects:
            bbox = obj["bbox"]
            color = (0, 255, 0)  # 绿色
            
            # 绘制物品边界框
            cv2.rectangle(frame, 
                        (bbox["x1"], bbox["y1"]), 
                        (bbox["x2"], bbox["y2"]), 
                        color, 1)
            
            # 绘制物品标签
            label = f'{obj["label"]}: {obj["confidence"]:.2f}'
            cv2.putText(frame, label, 
                       (bbox["x1"], bbox["y2"] + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame
    
    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """
        将图像帧转换为Base64编码
        
        Args:
            frame: 图像帧
            
        Returns:
            Base64编码的图像
        """
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
    
    def _summarize_behavior_analysis(self, frame_results: List[Dict]) -> Dict:
        """
        汇总多帧行为分析结果
        
        Args:
            frame_results: 多帧分析结果
            
        Returns:
            汇总结果
        """
        # 统计各种行为的数量
        head_pose_stats = {
            "looking_up": 0,
            "looking_down": 0,
            "neutral": 0
        }
        
        hand_activity_stats = {
            "writing": 0,
            "using_phone": 0,
            "resting": 0,
            "unknown": 0
        }
        
        # 统计桌面物品
        object_stats = {}
        
        # 总帧数
        total_frames = len(frame_results)
        
        # 统计每帧的学生数，取平均值
        total_students_per_frame = []
        
        # 统计总的学生次数（用于计算百分比）
        total_student_instances = 0
        
        # 遍历所有帧的结果
        for frame_result in frame_results:
            frame_student_count = len(frame_result["behaviors"])
            total_students_per_frame.append(frame_student_count)
            
            for behavior in frame_result["behaviors"]:
                total_student_instances += 1
                
                # 统计头部姿态
                head_pose = behavior["head_pose"]
                if head_pose in head_pose_stats:
                    head_pose_stats[head_pose] += 1
                
                # 统计手部活动
                hand_activity = behavior["hand_activity"]
                if hand_activity in hand_activity_stats:
                    hand_activity_stats[hand_activity] += 1
                
                # 统计桌面物品
                for obj in behavior.get("desktop_objects", []):
                    label = obj["label"]
                    if label in object_stats:
                        object_stats[label] += 1
                    else:
                        object_stats[label] = 1
        
        # 计算平均学生数
        avg_student_count = round(sum(total_students_per_frame) / len(total_students_per_frame)) if total_students_per_frame else 0
        
        # 计算百分比（如果没有学生，设置为0）
        behavior_percentages = {}
        if total_student_instances > 0:
            # 头部姿态百分比
            for pose, count in head_pose_stats.items():
                behavior_percentages[pose] = round((count / total_student_instances) * 100, 2)
            
            # 手部活动百分比
            for activity, count in hand_activity_stats.items():
                behavior_percentages[activity] = round((count / total_student_instances) * 100, 2)
        else:
            # 没有检测到学生，所有百分比为0
            for pose in head_pose_stats.keys():
                behavior_percentages[pose] = 0.0
            for activity in hand_activity_stats.keys():
                behavior_percentages[activity] = 0.0
        
        # 物品百分比（相对于总学生次数，表示有多少比例的学生在使用该物品）
        object_percentages = {}
        if total_student_instances > 0:
            for obj, count in object_stats.items():
                object_percentages[obj] = round((count / total_student_instances) * 100, 2)
        else:
            for obj in object_stats.keys():
                object_percentages[obj] = 0.0
        
        # 合并统计数据（为了保持API兼容性）
        behavior_stats = {**head_pose_stats, **hand_activity_stats}
        
        # 生成分析结论
        conclusions = self._generate_conclusions(behavior_percentages, object_percentages)
        
        return {
            "behavior_stats": behavior_stats,
            "behavior_percentages": behavior_percentages,
            "object_stats": object_stats,
            "object_percentages": object_percentages,
            "conclusions": conclusions,
            "total_frames": total_frames,
            "avg_student_count": avg_student_count  # 添加平均学生数
        }
    
    def _generate_conclusions(self, behavior_percentages: Dict, object_percentages: Dict) -> List[str]:
        """
        根据统计数据生成分析结论
        
        Args:
            behavior_percentages: 行为百分比统计
            object_percentages: 物体百分比统计
            
        Returns:
            分析结论列表
        """
        conclusions = []
        
        # 分析注意力集中情况
        looking_up_pct = behavior_percentages.get("looking_up", 0)
        looking_down_pct = behavior_percentages.get("looking_down", 0)
        neutral_pct = behavior_percentages.get("neutral", 0)
        
        if looking_up_pct > 60:
            conclusions.append("大部分学生注意力集中，积极关注教学内容")
        elif looking_down_pct > 40:
            conclusions.append("较多学生注意力分散，可能存在走神现象")
        else:
            conclusions.append("学生注意力分布较为均衡")
        
        # 分析学习行为
        writing_pct = behavior_percentages.get("writing", 0)
        using_phone_pct = behavior_percentages.get("using_phone", 0)
        
        if writing_pct > 30:
            conclusions.append("多数学生在认真记笔记，学习积极性较高")
        elif using_phone_pct > 20:
            conclusions.append("部分学生使用手机，建议加强课堂纪律管理")
        else:
            conclusions.append("学生学习行为较为规范")
        
        # 分析桌面物品
        book_pct = object_percentages.get("book", 0)
        laptop_pct = object_percentages.get("laptop", 0)
        
        if book_pct > 50:
            conclusions.append("教室内以纸质教材为主")
        elif laptop_pct > 30:
            conclusions.append("教室内较多使用电子设备")
        else:
            conclusions.append("教室内学习用品配备适中")
        
        return conclusions

# 全局行为分析器实例
behavior_analyzer = None

# 全局行为分析参数
behavior_params = {}

def get_behavior_analyzer(params=None):
    """获取全局行为分析器实例"""
    global behavior_analyzer, behavior_params
    # 如果提供了新的参数，更新全局参数
    if params is not None:
        behavior_params = params
    
    # 如果分析器不存在或者有新参数，重新创建分析器
    if behavior_analyzer is None or params is not None:
        behavior_analyzer = ClassroomBehaviorAnalyzer(behavior_params)
    return behavior_analyzer

def update_behavior_params(params):
    """更新行为分析参数"""
    global behavior_params
    behavior_params.update(params)

if __name__ == "__main__":
    # 简单测试
    analyzer = ClassroomBehaviorAnalyzer()
    
    # 创建测试图像
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    test_frame[:] = (255, 255, 255)  # 白色背景
    
    # 进行分析
    result = analyzer.analyze_frame(test_frame)
    print("测试分析结果:", result)