#!/usr/bin/env python3
"""
基于边界框跟踪的个人行为分析服务
用户在第一帧手动框选目标学生，然后使用目标跟踪算法追踪该学生
"""

import cv2
import numpy as np
from ultralytics import YOLO, RTDETR
import logging
from typing import Dict, List, Any, Optional, Tuple
import time
from PIL import Image, ImageDraw, ImageFont
import base64
from io import BytesIO

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BBoxTrackerAnalyzer:
    """
    基于边界框跟踪的个人行为分析器
    """
    
    def __init__(self, behavior_params=None):
        """初始化分析器"""
        logger.info("正在初始化边界框跟踪分析器...")
        
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
        
        # 行为颜色映射（BGR格式）
        self.behavior_colors = {
            "looking_up": (0, 255, 0),      # 绿色 - 抬头
            "looking_down": (0, 0, 255),    # 红色 - 低头
            "neutral": (255, 255, 0),       # 黄色 - 中性
            "writing": (255, 0, 0),         # 蓝色 - 写字
            "using_phone": (0, 255, 255),   # 青色 - 玩手机
            "resting": (255, 0, 255),       # 紫色 - 休息
            "unknown": (128, 128, 128)      # 灰色 - 未知
        }
        
        # 行为中文标签
        self.behavior_labels = {
            'looking_up': '抬头',
            'looking_down': '低头',
            'neutral': '中性',
            'writing': '记笔记',
            'using_phone': '玩手机',
            'resting': '休息',
            'unknown': '未知'
        }
        
        # 行为分析参数
        self.behavior_params = behavior_params or {
            "head_up_threshold": 2,        # 正常坐姿算抬头
            "head_down_threshold": 8,      # 明显低头才算
            "writing_threshold": 30,       # 更敏感
            "phone_threshold": -10,        # 更敏感
        }
        
        logger.info("边界框跟踪分析器初始化完成")
    
    def analyze_with_bbox(
        self,
        frames: List[np.ndarray],
        target_student_name: str,
        initial_bbox: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        基于初始边界框分析学生行为
        
        Args:
            frames: 视频帧列表
            target_student_name: 目标学生姓名
            initial_bbox: 初始边界框 {'x': x, 'y': y, 'width': width, 'height': height}
            
        Returns:
            个人行为分析结果
        """
        start_time = time.time()
        
        logger.info(f"开始分析学生 {target_student_name}，共 {len(frames)} 帧")
        logger.info(f"初始边界框: {initial_bbox}")
        
        # 初始化OpenCV跟踪器（兼容不同版本）
        tracker = None
        tracker_name = ""
        
        # 尝试按优先级顺序创建跟踪器
        tracker_methods = [
            ('cv2.legacy.TrackerCSRT_create', lambda: cv2.legacy.TrackerCSRT_create()),
            ('cv2.TrackerCSRT_create', lambda: cv2.TrackerCSRT_create()),
            ('cv2.TrackerNano_create', lambda: cv2.TrackerNano_create()),  # OpenCV 4.5.4+
            ('cv2.TrackerVit_create', lambda: cv2.TrackerVit_create()),    # OpenCV 4.5.4+
            ('cv2.TrackerMIL_create', lambda: cv2.TrackerMIL_create()),    # 可用但较慢
            ('cv2.legacy.TrackerKCF_create', lambda: cv2.legacy.TrackerKCF_create()),
            ('cv2.TrackerKCF_create', lambda: cv2.TrackerKCF_create()),
        ]
        
        for method_name, method in tracker_methods:
            try:
                tracker = method()
                tracker_name = method_name
                logger.info(f"✓ 成功创建跟踪器: {tracker_name}")
                break
            except (AttributeError, cv2.error) as e:
                continue
        
        if tracker is None:
            raise RuntimeError("无法创建任何可用的OpenCV跟踪器，请安装 opencv-contrib-python")
        
        # 转换边界框格式 (x, y, width, height) - 必须是整数
        bbox = (
            int(initial_bbox['x']),
            int(initial_bbox['y']),
            int(initial_bbox['width']),
            int(initial_bbox['height'])
        )
        
        logger.info(f"转换后的边界框: {bbox}")
        
        # 在第一帧初始化跟踪器
        first_frame = frames[0]
        
        # 验证边界框是否在图像范围内
        img_height, img_width = first_frame.shape[:2]
        logger.info(f"第一帧尺寸: {img_width}x{img_height}")
        
        x, y, w, h = bbox
        if x < 0 or y < 0 or x + w > img_width or y + h > img_height:
            logger.error(f"边界框超出图像范围！图像: {img_width}x{img_height}, 边界框: {bbox}")
            # 裁剪边界框到图像范围内
            x = max(0, min(x, img_width - 1))
            y = max(0, min(y, img_height - 1))
            w = min(w, img_width - x)
            h = min(h, img_height - y)
            bbox = (x, y, w, h)
            logger.info(f"裁剪后的边界框: {bbox}")
        
        if w < 10 or h < 10:
            raise RuntimeError(f"边界框太小，无法跟踪: {bbox}")
        
        # 确保图像是uint8格式
        if first_frame.dtype != np.uint8:
            logger.warning(f"第一帧不是uint8格式: {first_frame.dtype}，正在转换...")
            first_frame = first_frame.astype(np.uint8)
        
        # 确保图像是3通道BGR格式
        if len(first_frame.shape) == 2:
            logger.warning("第一帧是灰度图，正在转换为BGR...")
            first_frame = cv2.cvtColor(first_frame, cv2.COLOR_GRAY2BGR)
        elif first_frame.shape[2] == 4:
            logger.warning("第一帧是RGBA格式，正在转换为BGR...")
            first_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGBA2BGR)
        
        logger.info(f"图像形状: {first_frame.shape}, 数据类型: {first_frame.dtype}")
        
        # 初始化跟踪器
        try:
            success = tracker.init(first_frame, bbox)
            if not success:
                logger.warning(f"{tracker_name} 初始化返回False，尝试其他跟踪器...")
                # 如果当前跟踪器初始化失败，尝试使用更简单的算法
                if 'MIL' not in tracker_name:
                    logger.info("尝试使用TrackerMIL作为后备方案...")
                    try:
                        tracker = cv2.TrackerMIL_create()
                        success = tracker.init(first_frame, bbox)
                        if success:
                            tracker_name = 'cv2.TrackerMIL_create (fallback)'
                            logger.info(f"✓ 后备跟踪器初始化成功: {tracker_name}")
                    except Exception as e:
                        logger.error(f"后备跟踪器也失败: {e}")
                        success = False
        except Exception as e:
            logger.error(f"跟踪器初始化异常: {str(e)}")
            # 尝试使用简化的跟踪方式：直接使用姿态检测而不使用跟踪器
            logger.info("跟踪器不可用，将使用基于姿态检测的区域匹配方式")
            success = None  # 标记为使用后备方案
        
        if success is False:
            raise RuntimeError(f"跟踪器初始化失败，边界框: {bbox}, 图像形状: {first_frame.shape}, 跟踪器: {tracker_name}")
        
        if success is None:
            logger.warning("⚠ 使用后备方案：基于姿态检测的区域匹配（不使用跟踪器）")
            tracker = None  # 设置为None表示使用后备方案
        else:
            logger.info(f"✓ 跟踪器初始化成功: {tracker_name}")
        
        # 存储每帧的分析结果
        frame_results = []
        frames_tracked = 0
        frames_lost = 0
        
        # 分析每一帧
        for i, frame in enumerate(frames):
            try:
                progress = (i + 1) / len(frames) * 100
                logger.info(f"[边界框跟踪] 进度: {progress:.1f}% ({i + 1}/{len(frames)})")
                
                if tracker is not None:
                    # 使用跟踪器
                    success, tracked_bbox = tracker.update(frame)
                    
                    if success:
                        # 转换bbox格式
                        x, y, w, h = [int(v) for v in tracked_bbox]
                        current_bbox = {
                            'x1': x,
                            'y1': y,
                            'x2': x + w,
                            'y2': y + h
                        }
                        
                        # 分析该区域的行为
                        result = self._analyze_bbox_region(
                            frame,
                            i,
                            current_bbox,
                            target_student_name
                        )
                        
                        frames_tracked += 1
                        frame_results.append(result)
                    else:
                        # 跟踪失败，尝试重新检测
                        logger.warning(f"帧 {i}: 跟踪失败，尝试重新检测")
                        frames_lost += 1
                else:
                    # 后备方案：使用姿态检测区域匹配
                    # 使用初始边界框区域，在其附近搜索姿态
                    x, y, w, h = bbox
                    # 扩大搜索范围
                    search_margin = 50
                    search_bbox = {
                        'x1': max(0, x - search_margin),
                        'y1': max(0, y - search_margin),
                        'x2': min(img_width, x + w + search_margin),
                        'y2': min(img_height, y + h + search_margin)
                    }
                    
                    result = self._analyze_bbox_region(
                        frame,
                        i,
                        search_bbox,
                        target_student_name
                    )
                    
                    if result.get('student_found', False):
                        frames_tracked += 1
                        frame_results.append(result)
                    else:
                        frames_lost += 1
                    
            except Exception as e:
                logger.error(f"帧 {i} 处理失败: {e}")
                frames_lost += 1
                continue
        
        # 汇总分析结果
        summary = self._summarize_bbox_analysis(frame_results, target_student_name)
        
        processing_time = time.time() - start_time
        
        logger.info(f"分析完成: 成功跟踪 {frames_tracked}/{len(frames)} 帧")
        
        return {
            "student_name": target_student_name,
            "timestamp": time.time(),
            "processing_time": processing_time,
            "total_frames": len(frames),
            "frames_with_student": frames_tracked,
            "frames_without_student": frames_lost,
            "frame_results": frame_results,
            "summary": summary
        }
    
    def _analyze_bbox_region(
        self,
        frame: np.ndarray,
        frame_index: int,
        bbox: Dict[str, int],
        student_name: str
    ) -> Dict[str, Any]:
        """
        分析边界框区域内的行为
        """
        # 1. 在边界框区域内检测姿态
        pose_results = self.pose_model(frame, verbose=False)
        
        # 找到与边界框重叠最大的姿态
        target_pose = None
        best_iou = 0
        pose_count = 0
        
        for result in pose_results:
            if result.boxes is not None and result.keypoints is not None:
                for i, box in enumerate(result.boxes):
                    pose_count += 1
                    pose_bbox = box.xyxy.cpu().numpy()[0].astype(int)
                    pose_bbox_dict = {
                        'x1': int(pose_bbox[0]),
                        'y1': int(pose_bbox[1]),
                        'x2': int(pose_bbox[2]),
                        'y2': int(pose_bbox[3])
                    }
                    
                    # 计算IoU
                    iou = self._calculate_iou(bbox, pose_bbox_dict)
                    
                    if frame_index == 0:  # 只在第一帧输出调试信息
                        logger.info(f"姿态#{pose_count}: bbox={pose_bbox_dict}, IoU={iou:.3f}")
                    
                    if iou > best_iou:
                        best_iou = iou
                        target_pose = result.keypoints.data[i]
        
        if frame_index == 0:
            logger.info(f"帧{frame_index}: 检测到{pose_count}个姿态, 最佳IoU={best_iou:.3f}, 边界框={bbox}")
        
        # 降低IoU阈值，从wjl0.3改为0.1，更容易匹配
        if target_pose is None or best_iou < 0.1:
            if frame_index == 0:
                logger.warning(f"帧{frame_index}: 未找到匹配的姿态（IoU < 0.1）")
            return {
                "frame_index": frame_index,
                "timestamp": frame_index * 30,
                "student_found": False,
                "pose_found": False,
                "debug_info": f"pose_count={pose_count}, best_iou={best_iou:.3f}"
            }
        
        # 2. 分析姿态行为
        behavior = self._analyze_single_person_pose(target_pose)
        
        # 第一帧输出详细诊断
        if frame_index == 0:
            logger.info(f"=== 第一帧行为分析结果 ===")
            logger.info(f"  头部姿态: {behavior['head_pose']}")
            logger.info(f"  手部活动: {behavior['hand_activity']}")
        
        behavior["bbox"] = bbox
        
        # 3. 物体检测
        object_results = self.object_model(frame, verbose=False)
        desktop_objects = self._analyze_desktop_objects_in_bbox(object_results, bbox)
        behavior["desktop_objects"] = desktop_objects
        
        # 4. 绘制标注（每10帧保存一次）
        annotated_image = None
        if frame_index % 10 == 0 or frame_index == 0:
            annotated_frame = self._draw_bbox_annotations(
                frame.copy(),
                behavior,
                student_name,
                bbox
            )
            annotated_image = self._frame_to_base64(annotated_frame)
        
        return {
            "frame_index": frame_index,
            "timestamp": frame_index * 30,
            "student_found": True,
            "pose_found": True,
            "student_name": student_name,
            "behavior": behavior,
            "annotated_image": annotated_image
        }
    
    def _analyze_single_person_pose(self, keypoints) -> Dict[str, Any]:
        """分析单个人的姿态"""
        kpts = keypoints.cpu().numpy()
        
        # 关键点索引（COCO格式）
        # 每个关键点格式: [x, y, confidence]
        nose = kpts[0]
        left_eye = kpts[1]
        right_eye = kpts[2]
        left_ear = kpts[3]
        right_ear = kpts[4]
        left_shoulder = kpts[5]
        right_shoulder = kpts[6]
        left_elbow = kpts[7]
        right_elbow = kpts[8]
        left_wrist = kpts[9]
        right_wrist = kpts[10]
        
        # 检查关键点置信度
        def is_visible(kpt, threshold=0.3):
            """\u68c0\u67e5\u5173\u952e\u70b9\u662f\u5426\u53ef\u89c1"""
            return len(kpt) > 2 and kpt[2] > threshold
        
        # 计算头部角度（需要鼻子和眼睛）
        if is_visible(nose) and (is_visible(left_eye) or is_visible(right_eye)):
            head_pose = self._calculate_head_pose(
                nose[:2], 
                left_eye[:2], 
                right_eye[:2], 
                left_ear[:2], 
                right_ear[:2]
            )
        else:
            logger.warning(f"头部关键点不可见: nose={nose[2] if len(nose)>2 else 0:.2f}, left_eye={left_eye[2] if len(left_eye)>2 else 0:.2f}, right_eye={right_eye[2] if len(right_eye)>2 else 0:.2f}")
            head_pose = "neutral"  # 默认为中性
        
        # 计算手部活动（需要肩膀和手腕）
        if (is_visible(left_shoulder) or is_visible(right_shoulder)) and \
           (is_visible(left_wrist) or is_visible(right_wrist)):
            hand_activity = self._calculate_hand_activity(
                left_shoulder[:2], 
                right_shoulder[:2],
                left_elbow[:2], 
                right_elbow[:2],
                left_wrist[:2], 
                right_wrist[:2]
            )
        else:
            logger.warning(f"手部关键点不可见: left_shoulder={left_shoulder[2] if len(left_shoulder)>2 else 0:.2f}, right_shoulder={right_shoulder[2] if len(right_shoulder)>2 else 0:.2f}, left_wrist={left_wrist[2] if len(left_wrist)>2 else 0:.2f}, right_wrist={right_wrist[2] if len(right_wrist)>2 else 0:.2f}")
            hand_activity = "neutral"  # 默认为中性
        
        return {
            "head_pose": head_pose,
            "hand_activity": hand_activity
        }
    
    def _calculate_head_pose(self, nose, left_eye, right_eye, left_ear, right_ear) -> str:
        """计算头部姿态
            
        在正常教室场景中:
        - 学生坐姿看前方(抬头听课): nose_diff在2-8范围内(鼻子略低于眼睛)
        - 学生低头看书/手机: nose_diff > 8 (鼻子明显低于眼睛)
        - 学生仰头: nose_diff < 2 (几乎不会出现)
        """
        # 计算眼睛中点的y坐标
        eye_center_y = (left_eye[1] + right_eye[1]) / 2
            
        # 计算鼻子相对于眼睛的位置差值
        nose_diff = nose[1] - eye_center_y
            
        # 重新定义判断逻辑:
        # - nose_diff <= head_down_threshold: 正常坐姿看前方,算作「抬头听课」
        # - nose_diff > head_down_threshold: 明显低头,算作「低头」
        # 注意: head_up_threshold暂时不使用,因为教室场景中几乎不会出现真正的仰头
        if nose_diff <= self.behavior_params["head_down_threshold"]:
            logger.info(f"头部: looking_up (nose_diff={nose_diff:.1f} <= {self.behavior_params['head_down_threshold']})")
            return "looking_up"
        else:
            logger.info(f"头部: looking_down (nose_diff={nose_diff:.1f} > {self.behavior_params['head_down_threshold']})")
            return "looking_down"
    
    def _calculate_hand_activity(self, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist) -> str:
        """计算手部活动"""
        # 判断是否在写字（手在桌面附近）
        avg_wrist_y = (left_wrist[1] + right_wrist[1]) / 2
        avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        
        # 计算手腕相对于肩膀的位置差值
        wrist_diff = avg_wrist_y - avg_shoulder_y
        
        # writing_threshold是正数（如100），表示手腕在肩膀下方100像素
        # phone_threshold是负数（如-50），表示手腕在肩膀上方50像素
        if wrist_diff > self.behavior_params["writing_threshold"]:
            logger.info(f"手部: writing (wrist_diff={wrist_diff:.1f} > {self.behavior_params['writing_threshold']})")
            return "writing"
        elif wrist_diff < self.behavior_params["phone_threshold"]:
            logger.info(f"手部: using_phone (wrist_diff={wrist_diff:.1f} < {self.behavior_params['phone_threshold']})")
            return "using_phone"
        else:
            logger.info(f"手部: neutral (wrist_diff={wrist_diff:.1f})")
            return "neutral"
    
    def _analyze_desktop_objects_in_bbox(self, object_results, bbox: Dict[str, int]) -> List[str]:
        """检测边界框区域内的物体"""
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
                    
                    # 检查物体是否在学生区域内
                    if self._calculate_iou(bbox, obj_bbox_dict) > 0.1:
                        class_id = int(box.cls.cpu().numpy()[0])
                        if class_id == 67:  # cell phone
                            detected_objects.append("cell_phone")
                        elif class_id == 63:  # laptop
                            detected_objects.append("laptop")
                        elif class_id == 73:  # book
                            detected_objects.append("book")
        
        return detected_objects
    
    def _calculate_iou(self, bbox1: Dict[str, int], bbox2: Dict[str, int]) -> float:
        """计算两个边界框的IoU"""
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
    
    def _draw_bbox_annotations(
        self,
        frame: np.ndarray,
        behavior: Dict[str, Any],
        student_name: str,
        bbox: Dict[str, int]
    ) -> np.ndarray:
        """绘制边界框和行为标注"""
        # 获取行为颜色
        head_pose = behavior.get("head_pose", "unknown")
        color = self.behavior_colors.get(head_pose, (255, 255, 255))
        
        # 绘制边界框
        cv2.rectangle(
            frame,
            (bbox['x1'], bbox['y1']),
            (bbox['x2'], bbox['y2']),
            color,
            3
        )
        
        # 使用PIL绘制中文
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # 加载中文字体
        try:
            font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 20)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 20)
            except:
                font = ImageFont.load_default()
        
        # 构建标签文本
        head_label = self.behavior_labels.get(head_pose, head_pose)
        hand_activity = behavior.get("hand_activity", "neutral")
        hand_label = self.behavior_labels.get(hand_activity, hand_activity)
        
        label = f'{student_name}: {head_label}/{hand_label}'
        
        # 绘制文字背景
        text_position = (bbox['x1'], max(0, bbox['y1'] - 30))
        bbox_text = draw.textbbox(text_position, label, font=font)
        draw.rectangle(bbox_text, fill=(0, 0, 0, 180))
        
        # 绘制文字
        draw.text(text_position, label, fill=color, font=font)
        
        # 转回OpenCV格式
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """将帧转换为Base64字符串"""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{img_base64}"
    
    def _summarize_bbox_analysis(
        self,
        frame_results: List[Dict],
        student_name: str
    ) -> Dict[str, Any]:
        """汇总边界框分析结果"""
        if not frame_results:
            return {
                "behavior_percentages": {},
                "attention_score": 0,
                "conclusions": ["未能成功跟踪到学生"]
            }
        
        # 统计各种行为
        head_pose_stats = {}
        hand_activity_stats = {}
        desktop_object_stats = {}  # 新增:物体统计
        total_frames = len(frame_results)
        
        for result in frame_results:
            if result.get("pose_found") and "behavior" in result:
                behavior = result["behavior"]
                
                # 统计头部姿态
                head_pose = behavior.get("head_pose", "unknown")
                head_pose_stats[head_pose] = head_pose_stats.get(head_pose, 0) + 1
                
                # 统计手部活动
                hand_activity = behavior.get("hand_activity", "neutral")
                hand_activity_stats[hand_activity] = hand_activity_stats.get(hand_activity, 0) + 1
                
                # 统计桌面物品
                desktop_objects = behavior.get("desktop_objects", [])
                for obj in desktop_objects:
                    desktop_object_stats[obj] = desktop_object_stats.get(obj, 0) + 1
        
        # 计算百分比 - 分别统计头部和手部
        head_percentages = {}
        for pose, count in head_pose_stats.items():
            head_percentages[pose] = round((count / total_frames) * 100, 2)
        
        hand_percentages = {}
        for activity, count in hand_activity_stats.items():
            hand_percentages[activity] = round((count / total_frames) * 100, 2)
        
        # 计算物体出现百分比
        object_percentages = {}
        for obj, count in desktop_object_stats.items():
            object_percentages[obj] = round((count / total_frames) * 100, 2)
        
        # 合并到behavior_percentages(为了保持API兼容)
        behavior_percentages = {**head_percentages, **hand_percentages}
        
        # 计算认真程度评分
        looking_up_pct = behavior_percentages.get("looking_up", 0)
        writing_pct = behavior_percentages.get("writing", 0)
        using_phone_pct = behavior_percentages.get("using_phone", 0)
        laptop_pct = object_percentages.get("laptop", 0)
        
        # 看电脑不扣分,但玩手机扣分
        attention_score = max(0, min(100, looking_up_pct * 0.6 + writing_pct * 0.3 - using_phone_pct * 0.3))
        
        # 生成结论
        conclusions = []
        if looking_up_pct > 60:
            conclusions.append(f"{student_name}课堂专注度较高，抬头听课比例达到{looking_up_pct:.1f}%")
        elif looking_up_pct > 30:
            conclusions.append(f"{student_name}课堂专注度一般，抬头听课比例为{looking_up_pct:.1f}%")
        else:
            conclusions.append(f"{student_name}课堂专注度较低，建议提高听课效率")
        
        if writing_pct > 20:
            conclusions.append(f"记笔记比例为{writing_pct:.1f}%，学习态度积极")
        
        # 新增:笔记本电脑使用提示
        if laptop_pct > 10:
            conclusions.append(f"使用笔记本电脑比例为{laptop_pct:.1f}%，可能在进行电子笔记或编程学习")
        
        if using_phone_pct > 10:
            conclusions.append(f"使用手机比例为{using_phone_pct:.1f}%，建议减少分心行为")
        
        return {
            "behavior_percentages": behavior_percentages,
            "head_percentages": head_percentages,  # 单独返回头部姿态百分比
            "hand_percentages": hand_percentages,  # 单独返回手部活动百分比
            "object_percentages": object_percentages,  # 新增:物体百分比
            "attention_score": round(attention_score, 2),
            "recognition_accuracy": round((total_frames / len(frame_results)) * 100, 2) if frame_results else 0,
            "conclusions": conclusions
        }
