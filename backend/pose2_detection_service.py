#!/usr/bin/env python3
"""
独立的姿态检测服务（测试版）
使用YOLO11s-pose检测人体关键点
与原有的behavior_service完全分离
使用耳朵-眼睛连线角度法判断抬头/低头
"""

import cv2
import numpy as np
from ultralytics import YOLO
import logging
from typing import Dict, List, Any
import time
import base64
from io import BytesIO
from PIL import Image
import math

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PoseDetectionService:
    """纯粹的姿态检测服务，只负责检测关键点"""
    
    def __init__(self):
        """初始化姿态检测模型"""
        logger.info("正在加载YOLO11m-pose姿态检测模型...")
        
        try:
            self.pose_model = YOLO('yolo11m-pose.pt')
            logger.info("✓ YOLO11m-pose模型加载成功")
        except Exception as e:
            logger.error(f"姿态模型加载失败: {e}")
            raise
        
        # 初始化物体检测模型
        logger.info("正在加载YOLO11m物体检测模型...")
        try:
            self.object_model = YOLO('yolo11m.pt')
            logger.info("✓ YOLO11m物体检测模型加载成功")
        except Exception as e:
            logger.error(f"物体检测模型加载失败: {e}")
            raise
        
        # COCO姿态关键点定义（17个点）
        self.keypoint_names = [
            'nose',           # 0: 鼻子
            'left_eye',       # 1: 左眼
            'right_eye',      # 2: 右眼
            'left_ear',       # 3: 左耳
            'right_ear',      # 4: 右耳
            'left_shoulder',  # 5: 左肩
            'right_shoulder', # 6: 右肩
            'left_elbow',     # 7: 左肘
            'right_elbow',    # 8: 右肘
            'left_wrist',     # 9: 左手腕
            'right_wrist',    # 10: 右手腕
            'left_hip',       # 11: 左髋
            'right_hip',      # 12: 右髋
            'left_knee',      # 13: 左膝
            'right_knee',     # 14: 右膝
            'left_ankle',     # 15: 左脚踝
            'right_ankle'     # 16: 右脚踝
        ]
        
        # 骨架连接定义（用于绘制骨架）
        self.skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],  # 腿部
            [6, 12], [7, 13],  # 躯干到髋部
            [6, 7],  # 肩膀
            [6, 8], [7, 9], [8, 10], [9, 11],  # 手臂
            [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]  # 头部和颈部
        ]
        
        # 关键点颜色（BGR格式）
        self.keypoint_colors = {
            'nose': (0, 255, 0),           # 绿色
            'eye': (255, 0, 0),            # 蓝色
            'ear': (0, 255, 255),          # 黄色
            'shoulder': (255, 0, 255),     # 紫色
            'elbow': (255, 128, 0),        # 橙色
            'wrist': (0, 128, 255),        # 浅蓝色
            'hip': (128, 0, 255),          # 紫红色
            'knee': (0, 255, 128),         # 青绿色
            'ankle': (255, 255, 0)         # 青色
        }
        
        # COCO物体检测类别（教室相关）
        self.target_classes = {
            63: 'laptop',       # 笔记本电脑
            67: 'cell phone',   # 手机
            73: 'book'          # 书
        }
        
        logger.info("姿态检测服务初始化完成")
    
    def detect_poses(self, image: np.ndarray, conf_threshold: float = 0.5,
                     looking_up_threshold: float = -2, looking_down_threshold: float = 0) -> Dict[str, Any]:
        """
        检测图像中的人体姿态关键点
        
        Args:
            image: 输入图像 (BGR格式)
            conf_threshold: 置信度阈值
            looking_up_threshold: 抬头判断阈值
            looking_down_threshold: 低头判断阈值
            
        Returns:
            检测结果字典
        """
        start_time = time.time()
        
        # 执行姿态检测（提高输入分辨率以改善小目标检测）
        results = self.pose_model(
            image, 
            verbose=False, 
            conf=conf_threshold,
            imgsz=1280,  # 提高到1280以检测更多远距离学生
            iou=0.5,     # NMS IOU阈值，避免过度抑制
            max_det=50   # 增加最大检测数量（默认300，但对pose通常够用）
        )
        
        # 解析检测结果
        detected_persons = []
        
        for result in results:
            if result.keypoints is None:
                continue
                
            # 遍历检测到的每个人
            for idx, keypoints in enumerate(result.keypoints.data):
                person_data = {
                    'person_id': idx,
                    'keypoints': [],
                    'bbox': None,
                    'confidence': 0.0
                }
                
                # 提取关键点信息
                for kp_idx, kp in enumerate(keypoints):
                    x, y, conf = float(kp[0]), float(kp[1]), float(kp[2])
                    
                    keypoint_info = {
                        'index': kp_idx,
                        'name': self.keypoint_names[kp_idx],
                        'x': x,
                        'y': y,
                        'confidence': conf,
                        'visible': conf > 0.5
                    }
                    person_data['keypoints'].append(keypoint_info)
                
                # 判断头部姿态（抬头/低头）
                head_pose = self._analyze_head_pose_ear_eye(
                    person_data['keypoints'], 
                    looking_up_threshold, 
                    looking_down_threshold
                )
                person_data['head_pose'] = head_pose
                
                # 获取边界框
                if result.boxes is not None and idx < len(result.boxes):
                    box = result.boxes[idx]
                    xyxy = box.xyxy.cpu().numpy()[0]
                    person_data['bbox'] = {
                        'x1': int(xyxy[0]),
                        'y1': int(xyxy[1]),
                        'x2': int(xyxy[2]),
                        'y2': int(xyxy[3])
                    }
                    person_data['confidence'] = float(box.conf.cpu().numpy()[0])
                
                detected_persons.append(person_data)
        
        processing_time = time.time() - start_time
        
        return {
            'timestamp': time.time(),
            'processing_time': processing_time,
            'person_count': len(detected_persons),
            'persons': detected_persons
        }
    
    def _analyze_head_pose_ear_eye(self, keypoints: List[Dict], 
                               looking_up_threshold: float = 0,  # 抬头阈值（默认0）
                               looking_down_threshold: float = 2) -> str:  # 低头阈值（默认2）
        """
        使用耳朵-眼睛连线角度法判断头部姿态
        
        参数说明：
        - looking_up_threshold: 抬头判定阈值（默认-2，表示耳朵在直线下方2像素）
        - looking_down_threshold: 低头判定阈值（默认0，表示耳朵在直线上）
        - 要求：looking_up_threshold <= looking_down_threshold
    
    逻辑：
    - relative_position 为正：耳朵在直线上方
    - relative_position 为负：耳朵在直线下方
    - relative_position < looking_up_threshold → 抬头
    - relative_position > looking_down_threshold → 低头
        
    Returns:
        'looking_up' | 'looking_down' | 'neutral' | 'unknown'
    """
        # 提取需要的关键点
        # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
        left_eye = None
        right_eye = None
        left_ear = None
        right_ear = None
        
        for kp in keypoints:
            if kp['name'] == 'left_eye' and kp['visible']:
                left_eye = (kp['x'], kp['y'])
            elif kp['name'] == 'right_eye' and kp['visible']:
                right_eye = (kp['x'], kp['y'])
            elif kp['name'] == 'left_ear' and kp['visible']:
                left_ear = (kp['x'], kp['y'])
            elif kp['name'] == 'right_ear' and kp['visible']:
                right_ear = (kp['x'], kp['y'])
        
        # 至少需要两只眼睛和一只耳朵
        if not (left_eye and right_eye and (left_ear or right_ear)):
            return 'unknown'
        
        # 选择可见的耳朵（优先左耳，因为摄像头在角落看到的通常是人的左耳）
        ear = left_ear if left_ear else right_ear
        
        # 计算两眼连线的方程：y = kx + b
        # k = (y2 - y1) / (x2 - x1)
        x1, y1 = left_eye
        x2, y2 = right_eye
        
        # 避免除零错误
        if abs(x2 - x1) < 1e-6:
            # 两眼X坐标几乎相同（完全正面），退回到简单的Y坐标比较
            eye_y = (y1 + y2) / 2
            vertical_diff = ear[1] - eye_y
            
            # 修正逻辑：耳朵在下方（vertical_diff > 0）→ 抬头
            #          耳朵在上方（vertical_diff < 0）→ 低头
            if vertical_diff > 15:  # 对应looking_down_threshold
                return 'looking_up'
            elif vertical_diff < -5:  # 对应looking_up_threshold
                return 'looking_down'
            else:
                return 'neutral'
        
        # 计算斜率和截距
        k = (y2 - y1) / (x2 - x1)
        b = y1 - k * x1
        
        # 计算耳朵点到两眼连线的有向距离
        # 直线方程：y = kx + b
        ear_x, ear_y = ear
        
        # 在耳朵X坐标处，直线的Y值
        line_y = k * ear_x + b
        
        # 有向距离（注意符号定义）
        # 在图像坐标系中（Y轴向下）：
        # 耳朵在直线上方（ear_y < line_y） → 正值
        # 耳朵在直线下方（ear_y > line_y） → 负值
        relative_position = line_y - ear_y # 因为(0,0)在左上角
        
        # # 临时调试日志
        # logger.info(f"=== 头部姿态检测调试信息 ===")
        # logger.info(f"左眼坐标: ({x1:.1f}, {y1:.1f})")
        # logger.info(f"右眼坐标: ({x2:.1f}, {y2:.1f})")
        # logger.info(f"眼睛直线方程: y = {k:.3f}x + {b:.1f}")
        # logger.info(f"耳朵坐标: ({ear_x:.1f}, {ear_y:.1f})")
        # logger.info(f"直线在耳朵X位置的Y值: {line_y:.1f}")
        # logger.info(f"relative_position: {relative_position:.2f}")
        # logger.info(f"抬头阈值 looking_up_threshold: {looking_up_threshold}")
        # logger.info(f"低头阈值 looking_down_threshold: {looking_down_threshold}")
        
        # 判断逻辑：
        # looking_up_threshold = -2 （耳朵在下方2像素）
        # looking_down_threshold = 0 （耳朵在直线上）
        if relative_position < looking_up_threshold:  # < -2 （耳朵在下方很多）
            result = 'looking_up'  # 抬头
        elif relative_position > looking_down_threshold:  # > 0 （耳朵在上方）
            result = 'looking_down'  # 低头
        else:  # -2 到 0 之间
            result = 'neutral'
        
        # logger.info(f"判定结果: {result}")
        # logger.info(f"==========================\n")
        
        return result

    def draw_keypoints_on_image(
        self, 
        image: np.ndarray, 
        detection_result: Dict[str, Any],
        draw_skeleton: bool = True,
        draw_bbox: bool = True
    ) -> np.ndarray:
        """
        在图像上绘制检测到的关键点
        
        Args:
            image: 输入图像
            detection_result: 检测结果
            draw_skeleton: 是否绘制骨架连线
            draw_bbox: 是否绘制边界框
            
        Returns:
            标注后的图像
        """
        annotated_image = image.copy()
        
        for person in detection_result['persons']:
            keypoints = person['keypoints']
            head_pose = person.get('head_pose', 'unknown')
            
            # 绘制边界框
            if draw_bbox and person['bbox']:
                bbox = person['bbox']
                cv2.rectangle(
                    annotated_image,
                    (bbox['x1'], bbox['y1']),
                    (bbox['x2'], bbox['y2']),
                    (0, 255, 0),
                    2
                )
                
                # 显示头部姿态（中文 + 颜色编码）
                # 注意：这里的颜色是RGB格式（PIL使用），不是BGR
                head_pose_labels = {
                    'looking_up': ('抬头', (0, 255, 0)),      # 绿色 (RGB)
                    'looking_down': ('低头', (255, 0, 0)),    # 红色 (RGB)
                    'neutral': ('中性', (255, 255, 0)),       # 黄色 (RGB)
                    'unknown': ('未知', (128, 128, 128))      # 灰色 (RGB)
                }
                
                pose_text, pose_color = head_pose_labels.get(head_pose, ('未知', (128, 128, 128)))
                
                # 使用PIL绘制中文
                from PIL import ImageFont, ImageDraw
                pil_img = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)
                
                # 尝试加载中文字体（加大字体大小）
                font = None
                font_paths = [
                    "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
                    "C:/Windows/Fonts/simhei.ttf",  # 黑体
                    "/System/Library/Fonts/PingFang.ttc",  # macOS
                    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # Linux
                ]
                
                for font_path in font_paths:
                    try:
                        font = ImageFont.truetype(font_path, 28)  # 从20加大到28
                        break
                    except:
                        continue
                
                if font:
                    # 绘制黑色背景
                    text_position = (bbox['x1'], bbox['y1'] - 10)
                    bbox_text = draw.textbbox(text_position, pose_text, font=font)
                    draw.rectangle(bbox_text, fill=(0, 0, 0, 180))
                    # 绘制彩色文字
                    draw.text(text_position, pose_text, fill=pose_color, font=font)
                    annotated_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                else:
                    # 备选：使用英文
                    cv2.putText(
                        annotated_image,
                        head_pose,
                        (bbox['x1'], bbox['y1'] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,  # 加大字体
                        pose_color,
                        2
                    )
            
            # 绘制骨架连线（如果启用）
            if draw_skeleton:
                for connection in self.skeleton:
                    # COCO格式使用1-based索引，需要转换为0-based
                    idx1 = connection[0] - 1
                    idx2 = connection[1] - 1
                    
                    if idx1 < len(keypoints) and idx2 < len(keypoints):
                        kp1 = keypoints[idx1]
                        kp2 = keypoints[idx2]
                        
                        # 只有两个关键点都可见时才绘制连线
                        if kp1['visible'] and kp2['visible']:
                            cv2.line(
                                annotated_image,
                                (int(kp1['x']), int(kp1['y'])),
                                (int(kp2['x']), int(kp2['y'])),
                                (255, 255, 255),  # 白色连线
                                2
                            )
            
            # 绘制关键点
            for kp in keypoints:
                if not kp['visible']:
                    continue
                
                x, y = int(kp['x']), int(kp['y'])
                
                # 根据关键点类型选择颜色
                if 'nose' in kp['name']:
                    color = self.keypoint_colors['nose']
                elif 'eye' in kp['name']:
                    color = self.keypoint_colors['eye']
                elif 'ear' in kp['name']:
                    color = self.keypoint_colors['ear']
                elif 'shoulder' in kp['name']:
                    color = self.keypoint_colors['shoulder']
                elif 'elbow' in kp['name']:
                    color = self.keypoint_colors['elbow']
                elif 'wrist' in kp['name']:
                    color = self.keypoint_colors['wrist']
                elif 'hip' in kp['name']:
                    color = self.keypoint_colors['hip']
                elif 'knee' in kp['name']:
                    color = self.keypoint_colors['knee']
                elif 'ankle' in kp['name']:
                    color = self.keypoint_colors['ankle']
                else:
                    color = (255, 255, 255)
                
                # 绘制关键点圆圈
                cv2.circle(annotated_image, (x, y), 5, color, -1)
                cv2.circle(annotated_image, (x, y), 6, (255, 255, 255), 1)  # 白色边框
                
                # 显示关键点名称（可选）
                # cv2.putText(
                #     annotated_image,
                #     kp['name'],
                #     (x + 8, y),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.3,
                #     color,
                #     1
                # )
        
        return annotated_image

    def detect_objects(self, image: np.ndarray, conf_threshold: float = 0.3) -> Dict[str, Any]:
        """
        检测图像中的物体（笔记本、手机、书等）
        
        Args:
            image: 输入图像 (BGR格式)
            conf_threshold: 置信度阈值
            
        Returns:
            检测结果字典
        """
        start_time = time.time()
        
        # 执行物体检测（优化参数以提高小物体检测能力）
        results = self.object_model(
            image,
            verbose=False,
            conf=conf_threshold,
            classes=list(self.target_classes.keys()),  # 只检测教室相关物体
            imgsz=1280,      # 提高输入分辨率，增强小物体检测
            iou=0.4,         # 进一步降低IOU阈值，允许更多重叠检测（书本常被遮挡）
            max_det=100,     # 增加最大检测数，适应教室多物体场景
            agnostic_nms=False,  # 使用类别特定的NMS
            augment=True     # 启用测试时数据增强（TTA），提高检测鲁棒性
        )
        
        detected_objects = []
        
        for result in results:
            if result.boxes is None:
                continue
            
            for box in result.boxes:
                class_id = int(box.cls.cpu().numpy()[0])
                class_name = self.target_classes.get(class_id, 'unknown')
                confidence = float(box.conf.cpu().numpy()[0])
                xyxy = box.xyxy.cpu().numpy()[0]
                
                obj_data = {
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': {
                        'x1': int(xyxy[0]),
                        'y1': int(xyxy[1]),
                        'x2': int(xyxy[2]),
                        'y2': int(xyxy[3])
                    },
                    'center': {
                        'x': int((xyxy[0] + xyxy[2]) / 2),
                        'y': int((xyxy[1] + xyxy[3]) / 2)
                    }
                }
                
                detected_objects.append(obj_data)
        
        processing_time = time.time() - start_time
        
        # 统计各类物体数量
        object_counts = {}
        for obj in detected_objects:
            class_name = obj['class_name']
            object_counts[class_name] = object_counts.get(class_name, 0) + 1
        
        return {
            'timestamp': time.time(),
            'processing_time': processing_time,
            'object_count': len(detected_objects),
            'objects': detected_objects,
            'object_counts': object_counts
        }
    
    def draw_objects_on_image(self, image: np.ndarray, detection_result: Dict[str, Any]) -> np.ndarray:
        """
        在图像上绘制检测到的物体
        
        Args:
            image: 输入图像
            detection_result: 检测结果
            
        Returns:
            标注后的图像
        """
        annotated_image = image.copy()
        
        # 物体类别颜色 (BGR)
        class_colors = {
            'laptop': (0, 255, 0),      # 绿色
            'cell phone': (0, 0, 255),  # 红色
            'book': (0, 255, 255)       # 黄色 (BGR格式)
        }
        
        for obj in detection_result['objects']:
            bbox = obj['bbox']
            class_name = obj['class_name']
            confidence = obj['confidence']
            
            # 获取颜色
            color = class_colors.get(class_name, (128, 128, 128))
            
            # 绘制边界框
            cv2.rectangle(
                annotated_image,
                (bbox['x1'], bbox['y1']),
                (bbox['x2'], bbox['y2']),
                color,
                2
            )
            
            # 绘制标签（使用PIL绘制中文）
            from PIL import ImageFont, ImageDraw
            pil_img = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            
            # 中文翻译
            class_name_cn = {
                'laptop': '笔记本电脑',
                'cell phone': '手机',
                'book': '书'
            }.get(class_name, class_name)
            
            label_text = f"{class_name_cn} {confidence:.2f}"
            
            # 尝试加载中文字体
            font = None
            font_paths = [
                "C:/Windows/Fonts/msyh.ttc",
                "C:/Windows/Fonts/simhei.ttf",
                "/System/Library/Fonts/PingFang.ttc",
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            ]
            
            for font_path in font_paths:
                try:
                    font = ImageFont.truetype(font_path, 20)
                    break
                except:
                    continue
            
            if font:
                # RGB颜色转换
                rgb_color = (color[2], color[1], color[0])
                
                # 绘制背景
                text_position = (bbox['x1'], bbox['y1'] - 25)
                bbox_text = draw.textbbox(text_position, label_text, font=font)
                draw.rectangle(bbox_text, fill=(0, 0, 0, 180))
                
                # 绘制文字
                draw.text(text_position, label_text, fill=rgb_color, font=font)
                annotated_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            else:
                # 备选：使用英文
                cv2.putText(
                    annotated_image,
                    f"{class_name} {confidence:.2f}",
                    (bbox['x1'], bbox['y1'] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )
        
        return annotated_image
    
    def analyze_objects_base64(
        self,
        image_base64: str,
        conf_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        分析Base64编码的图像中的物体
        
        Args:
            image_base64: Base64编码的图像
            conf_threshold: 置信度阈值
            
        Returns:
            分析结果
        """
        try:
            # 解码Base64图像
            image_data = base64.b64decode(image_base64.split(',')[1])
            image_bytes = BytesIO(image_data)
            image_pil = Image.open(image_bytes)
            image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            
            # 检测物体
            detection_result = self.detect_objects(image_np, conf_threshold)
            
            # 绘制物体
            annotated_image = self.draw_objects_on_image(image_np, detection_result)
            
            # 转换为Base64
            _, buffer = cv2.imencode('.jpg', annotated_image)
            annotated_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return {
                'success': True,
                'annotated_image': f"data:image/jpeg;base64,{annotated_base64}",
                'detection_result': detection_result
            }
        except Exception as e:
            logger.error(f"物体检测分析失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }
    
    def analyze_image_base64(
        self, 
        image_base64: str,
        conf_threshold: float = 0.5,
        draw_skeleton: bool = True,
        draw_bbox: bool = True,
        looking_up_threshold: float = -2,
        looking_down_threshold: float = 0
    ) -> Dict[str, Any]:
        """
        分析Base64编码的图像
        
        Args:
            image_base64: Base64编码的图像
            conf_threshold: 检测置信度阈值
            draw_skeleton: 是否绘制骨架
            draw_bbox: 是否绘制边界框
            
        Returns:
            包含检测结果和标注图像的字典
        """
        try:
            # 解码Base64图像
            if image_base64.startswith('data:image'):
                image_base64 = image_base64.split(',')[1]
            
            image_bytes = base64.b64decode(image_base64)
            image_pil = Image.open(BytesIO(image_bytes))
            image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            
            # 执行检测
            detection_result = self.detect_poses(
                image_np, 
                conf_threshold,
                looking_up_threshold,
                looking_down_threshold
            )
            
            # 绘制关键点
            annotated_image = self.draw_keypoints_on_image(
                image_np,
                detection_result,
                draw_skeleton=draw_skeleton,
                draw_bbox=draw_bbox
            )
            
            # 转换为Base64
            annotated_base64 = self._image_to_base64(annotated_image)
            
            return {
                'success': True,
                'detection_result': detection_result,
                'annotated_image': annotated_base64
            }
            
        except Exception as e:
            logger.error(f"图像分析失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def analyze_behavior_base64(self, image_base64: str, 
                                pose_conf_threshold: float = 0.15,
                                object_conf_threshold: float = 0.25,
                                draw_skeleton: bool = True,
                                draw_bbox: bool = True,
                                looking_up_threshold: float = -2,
                                looking_down_threshold: float = 0) -> Dict[str, Any]:
        """
        组合姿态和物体检测，分析学生行为
        
        Args:
            image_base64: Base64编码的图片
            pose_conf_threshold: 姿态检测置信度
            object_conf_threshold: 物体检测置信度
            draw_skeleton: 是否绘制骨架
            draw_bbox: 是否绘制边界框
            looking_up_threshold: 抬头阈值
            looking_down_threshold: 低头阈值
            
        Returns:
            分析结果字典
        """
        try:
            # 解码图片
            image = self._base64_to_image(image_base64)
            
            # 执行姿态检测
            pose_result = self.detect_poses(
                image, 
                conf_threshold=pose_conf_threshold,
                looking_up_threshold=looking_up_threshold,
                looking_down_threshold=looking_down_threshold
            )
            
            # 执行物体检测
            object_result = self.detect_objects(
                image,
                conf_threshold=object_conf_threshold
            )
            
            # 分析每个人的行为
            behaviors = self._analyze_behaviors(
                pose_result['persons'],
                object_result['objects'],
                image.shape,
                looking_down_threshold
            )
            
            # 绘制标注图像
            annotated_image = self._draw_behaviors(
                image.copy(),
                behaviors,
                draw_skeleton,
                draw_bbox
            )
            
            # 转换为Base64
            annotated_base64 = self._image_to_base64(annotated_image)
            
            # 统计行为分布
            behavior_stats = self._calculate_behavior_stats(behaviors)
            
            return {
                'success': True,
                'detection_result': {
                    'person_count': len(behaviors),
                    'behaviors': behaviors,
                    'behavior_stats': behavior_stats,
                    'processing_time': pose_result['processing_time'] + object_result['processing_time']
                },
                'annotated_image': annotated_base64
            }
            
        except Exception as e:
            logger.error(f"行为分析失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def analyze_behavior_frame(self, image: np.ndarray, 
                              pose_conf_threshold: float = 0.15,
                              object_conf_threshold: float = 0.25,
                              draw_skeleton: bool = True,
                              draw_bbox: bool = True,
                              looking_up_threshold: float = -2,
                              looking_down_threshold: float = 0) -> Dict[str, Any]:
        """
        对单帧进行行为分析（用于视频处理）
        
        Args:
            image: OpenCV图像 (BGR格式)
            pose_conf_threshold: 姿态检测置信度
            object_conf_threshold: 物体检测置信度
            draw_skeleton: 是否绘制骨架
            draw_bbox: 是否绘制边界框
            looking_up_threshold: 抬头阈值
            looking_down_threshold: 低头阈值
            
        Returns:
            分析结果字典
        """
        try:
            # 执行姿态检测
            pose_result = self.detect_poses(
                image, 
                conf_threshold=pose_conf_threshold,
                looking_up_threshold=looking_up_threshold,
                looking_down_threshold=looking_down_threshold
            )
            
            # 执行物体检测
            object_result = self.detect_objects(
                image,
                conf_threshold=object_conf_threshold
            )
            
            # 分析每个人的行为
            behaviors = self._analyze_behaviors(
                pose_result['persons'],
                object_result['objects'],
                image.shape,
                looking_down_threshold
            )
            
            # 绘制标注图像
            annotated_image = None
            if draw_skeleton or draw_bbox:
                annotated_image = self._draw_behaviors(
                    image.copy(),
                    behaviors,
                    draw_skeleton,
                    draw_bbox
                )
            
            # 统计行为分布
            behavior_stats = self._calculate_behavior_stats(behaviors)
            
            return {
                'success': True,
                'behaviors': behaviors,
                'behavior_stats': behavior_stats,
                'annotated_image': annotated_image,
                'processing_time': pose_result['processing_time'] + object_result['processing_time']
            }
            
        except Exception as e:
            logger.error(f"帧行为分析失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """将OpenCV图像转换为Base64编码"""
        # BGR转RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # 保存到BytesIO
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG', quality=90)
        buffer.seek(0)
        
        # 编码为Base64
        img_str = base64.b64encode(buffer.read()).decode()
        return f"data:image/jpeg;base64,{img_str}"
    
    def _base64_to_image(self, image_base64: str) -> np.ndarray:
        """将Base64编码转换为OpenCV图像"""
        # 移除data:image/jpeg;base64,前缀
        if 'base64,' in image_base64:
            image_base64 = image_base64.split('base64,')[1]
        
        # 解码Base64
        img_data = base64.b64decode(image_base64)
        
        # 转换为PIL Image
        pil_image = Image.open(BytesIO(img_data))
        
        # 转换为NumPy数组 (RGB)
        rgb_image = np.array(pil_image)
        
        # RGB转BGR (OpenCV格式)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        return bgr_image
    
    def _analyze_behaviors(self, persons: List[Dict], objects: List[Dict], 
                          image_shape: tuple, looking_down_threshold: float) -> List[Dict]:
        """
        分析每个人的行为
        
        判断逻辑：
        1. 抬头 → 听讲/看黑板
        2. 低头 + 检测到laptop → 看电脑
        3. 低头 + 检测到cell phone + 严重低头 → 看手机
        4. 低头 + 未检测到电子设备 → 看书/记笔记
        """
        behaviors = []
        
        for person in persons:
            behavior = {
                'person_id': person['person_id'],
                'bbox': person['bbox'],
                'head_pose': person['head_pose'],
                'behavior': 'unknown',
                'confidence': 0.0,
                'detected_objects': []
            }
            
            head_pose = person['head_pose']
            
            # 抬头：听讲
            if head_pose == 'looking_up':
                behavior['behavior'] = 'listening'
                behavior['confidence'] = 0.9
            
            # 低头：需要进一步判断
            elif head_pose == 'looking_down':
                # 获取人物边界框信息
                person_bbox = person['bbox']
                person_center = {
                    'x': (person_bbox['x1'] + person_bbox['x2']) / 2,
                    'y': (person_bbox['y1'] + person_bbox['y2']) / 2
                }
                
                # 获取鼻子和肩膠位置（用于判断物体是否在面前）
                nose_y = None
                shoulder_y = None
                for kp in person['keypoints']:
                    if kp['name'] == 'nose' and kp['visible']:
                        nose_y = kp['y']
                    elif kp['name'] in ['left_shoulder', 'right_shoulder'] and kp['visible']:
                        shoulder_y = kp['y']
                
                # 在人物附近查找物体
                nearby_objects = self._find_nearby_objects(
                    objects, person_bbox, nose_y, shoulder_y, image_shape
                )
                
                behavior['detected_objects'] = nearby_objects
                
                # 判断逻辑
                has_laptop = any(obj['class_name'] == 'laptop' for obj in nearby_objects)
                has_phone = any(obj['class_name'] == 'cell phone' for obj in nearby_objects)
                
                if has_laptop:
                    behavior['behavior'] = 'using_computer'
                    behavior['confidence'] = 0.85
                elif has_phone:
                    # 检查是否严重低头（看手机通常头很低）
                    # 这里简单判断，实际需要根据角度
                    behavior['behavior'] = 'using_phone'
                    behavior['confidence'] = 0.75
                else:
                    # 未检测到电子设备，很可能是看书/记笔记
                    behavior['behavior'] = 'reading_writing'
                    behavior['confidence'] = 0.7
            
            # 中性姿态
            else:
                behavior['behavior'] = 'neutral'
                behavior['confidence'] = 0.5
            
            behaviors.append(behavior)
        
        return behaviors
    
    def _find_nearby_objects(self, objects: List[Dict], person_bbox: Dict,
                            nose_y: float, shoulder_y: float, 
                            image_shape: tuple) -> List[Dict]:
        """
        查找人物附近的物体
        
        判断物体是否在“面前”：
        1. 物体中心的y坐标 > 鼻子的y坐标（物体在头部下方）
        2. 物体中心的x坐标在人物边界框宽度范围内
        """
        nearby = []
        
        if nose_y is None:
            return nearby
        
        person_x1 = person_bbox['x1']
        person_x2 = person_bbox['x2']
        person_width = person_x2 - person_x1
        
        for obj in objects:
            obj_center_x = obj['center']['x']
            obj_center_y = obj['center']['y']
            
            # 物体在头部下方
            if obj_center_y > nose_y:
                # 物体在人物边界框宽度范围内（略宽松）
                if person_x1 - person_width * 0.3 < obj_center_x < person_x2 + person_width * 0.3:
                    nearby.append(obj)
        
        return nearby
    
    def _draw_behaviors(self, image: np.ndarray, behaviors: List[Dict],
                       draw_skeleton: bool, draw_bbox: bool) -> np.ndarray:
        """绘制行为标注"""
        # 行为颜色映射（BGR格式）
        behavior_colors = {
            'listening': (0, 255, 0),          # 绿色 - 听讲
            'using_computer': (0, 255, 255),   # 黄色 - 看电脑
            'using_phone': (0, 0, 255),        # 红色 - 看手机
            'reading_writing': (255, 255, 0),  # 天蓝色 - 看书/记笔记
            'neutral': (128, 128, 128),        # 灰色 - 中性
            'unknown': (200, 200, 200)         # 浅灰色 - 未知
        }
        
        behavior_labels_cn = {
            'listening': '听讲',
            'using_computer': '看电脑',
            'using_phone': '看手机',
            'reading_writing': '看书/记笔记',
            'neutral': '中性',
            'unknown': '未知'
        }
        
        # 先绘制所有边界框（使用OpenCV，快速）
        for behavior in behaviors:
            bbox = behavior['bbox']
            behavior_type = behavior['behavior']
            color_bgr = behavior_colors.get(behavior_type, (200, 200, 200))
            
            # 绘制边界框
            if draw_bbox:
                cv2.rectangle(
                    image,
                    (bbox['x1'], bbox['y1']),
                    (bbox['x2'], bbox['y2']),
                    color_bgr,
                    3
                )
        
        # 只转换一次PIL来绘制所有文字
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(pil_image)
        
        try:
            # 尝试加载中文字体
            font = ImageFont.truetype("msyh.ttc", 28)
        except:
            font = ImageFont.load_default()
        
        # 绘制所有文字标签
        for behavior in behaviors:
            bbox = behavior['bbox']
            behavior_type = behavior['behavior']
            confidence = behavior['confidence']
            color_bgr = behavior_colors.get(behavior_type, (200, 200, 200))
            
            # 绘制行为标签
            label = behavior_labels_cn.get(behavior_type, behavior_type)
            label_text = f"{label} {confidence:.0%}"
            
            # 将BGR颜色转换为RGB（PIL使用RGB）
            color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
            
            # 绘制黑色背景
            text_bbox = draw.textbbox((bbox['x1'], bbox['y1'] - 35), label_text, font=font)
            draw.rectangle(text_bbox, fill=(0, 0, 0))  # 黑色背景
            
            # 绘制彩色文字
            draw.text((bbox['x1'], bbox['y1'] - 35), label_text, fill=color_rgb, font=font)
        
        # 转换回 OpenCV 格式
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return image
    
    def _calculate_behavior_stats(self, behaviors: List[Dict]) -> Dict:
        """统计行为分布"""
        stats = {
            'listening': 0,
            'using_computer': 0,
            'using_phone': 0,
            'reading_writing': 0,
            'neutral': 0,
            'unknown': 0
        }
        
        for behavior in behaviors:
            behavior_type = behavior['behavior']
            if behavior_type in stats:
                stats[behavior_type] += 1
        
        return stats


# 全局服务实例
_pose_detection_service = None


def get_pose_detection_service() -> PoseDetectionService:
    """获取姿态检测服务单例"""
    global _pose_detection_service
    if _pose_detection_service is None:
        _pose_detection_service = PoseDetectionService()
    return _pose_detection_service


if __name__ == "__main__":
    # 简单测试
    service = PoseDetectionService()
    
    # 创建测试图像
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:] = (255, 255, 255)  # 白色背景
    
    # 执行检测
    result = service.detect_poses(test_image)
    print("测试检测结果:", result)
