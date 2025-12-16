#!/usr/bin/env python3
"""
个人行为分析服务
结合InsightFace人脸识别和YOLOv8姿态检测，分析指定学生的行为
"""

import cv2
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import logging
from typing import Dict, List, Any, Optional, Tuple
import time

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IndividualBehaviorAnalyzer:
    def __init__(self, face_app, behavior_params=None):
        """
        初始化个人行为分析器
        
        Args:
            face_app: InsightFace人脸分析应用实例
            behavior_params: 行为分析参数
        """
        logger.info("正在初始化个人行为分析器...")
        
        # 人脸识别
        self.face_app = face_app
        
        # 加载姿态检测模型
        self.pose_model = YOLO('yolov8n-pose.pt')
        logger.info("✓ 姿态检测模型加载成功")
        
        # 加载物体检测模型
        self.object_model = YOLO('yolov8n.pt')
        logger.info("✓ 物体检测模型加载成功")
        
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
            "head_up_threshold": 2,        # 正常坐姿算抬头
            "head_down_threshold": 8,      # 明显低头才算
            "writing_threshold": 30,       # 更敏感
            "phone_threshold": -10,        # 更敏感
            "object_min_confidence": 0.5
        }
        
        if behavior_params:
            self.behavior_params.update(behavior_params)
        
        logger.info("个人行为分析器初始化完成")
    
    def analyze_individual_video(
        self, 
        frames: List[np.ndarray], 
        target_student_name: str,
        student_registry: List[Dict]
    ) -> Dict[str, Any]:
        """
        分析指定学生在视频中的行为
        
        Args:
            frames: 视频帧列表
            target_student_name: 目标学生姓名
            student_registry: 学生注册信息列表
            
        Returns:
            个人行为分析结果
        """
        start_time = time.time()
        
        # 获取目标学生的特征
        target_descriptors = self._get_student_descriptors(target_student_name, student_registry)
        if not target_descriptors:
            return {
                "error": f"未找到学生 {target_student_name} 的注册信息",
                "student_name": target_student_name,
                "frames_analyzed": 0
            }
        
        logger.info(f"开始分析学生 {target_student_name} 的行为，共 {len(frames)} 帧")
        
        # 存储每帧的分析结果
        frame_results = []
        frames_with_student = 0
        frames_without_student = 0
        
        # 分析每一帧
        for i, frame in enumerate(frames):
            try:
                # 每帧都打印进度
                progress = (i + 1) / len(frames) * 100
                logger.info(f"[个人分析] 进度: {progress:.1f}% ({i + 1}/{len(frames)})")
                
                result = self._analyze_frame_for_student(
                    frame, 
                    i,
                    target_student_name,
                    target_descriptors
                )
                
                if result["student_found"]:
                    frames_with_student += 1
                    frame_results.append(result)
                else:
                    frames_without_student += 1
                    
            except Exception as e:
                logger.error(f"帧 {i} 处理失败: {e}")
                frames_without_student += 1
                continue
        
        # 汇总分析结果
        summary = self._summarize_individual_analysis(frame_results, target_student_name)
        
        processing_time = time.time() - start_time
        
        logger.info(f"分析完成: 找到学生的帧数 {frames_with_student}/{len(frames)}")
        
        return {
            "student_name": target_student_name,
            "timestamp": time.time(),
            "processing_time": processing_time,
            "total_frames": len(frames),
            "frames_with_student": frames_with_student,
            "frames_without_student": frames_without_student,
            "frame_results": frame_results,
            "summary": summary
        }
    
    def _get_student_descriptors(
        self, 
        student_name: str, 
        student_registry: List[Dict]
    ) -> Optional[List[np.ndarray]]:
        """获取学生的人脸特征向量"""
        for student in student_registry:
            if student.get("name") == student_name:
                descriptors = student.get("descriptors", [])
                if descriptors:
                    # 将列表转换为numpy数组
                    return [np.array(desc) for desc in descriptors]
        return None
    
    def _analyze_frame_for_student(
        self,
        frame: np.ndarray,
        frame_index: int,
        target_student_name: str,
        target_descriptors: List[np.ndarray]
    ) -> Dict[str, Any]:
        """
        在单帧中分析目标学生的行为
        """
        # 1. 人脸识别 - 找到目标学生
        faces = self.face_app.get(frame)
        
        target_face = None
        max_similarity = -1
        
        for face in faces:
            # 计算与目标学生的相似度
            similarity = self._calculate_similarity(face.embedding, target_descriptors)
            
            if similarity > 0.5 and similarity > max_similarity:  # 相似度阈值
                max_similarity = similarity
                target_face = face
        
        if not target_face:
            return {
                "frame_index": frame_index,
                "student_found": False,
                "timestamp": frame_index * 30  # 每30秒一帧
            }
        
        # 2. 获取人脸边界框
        face_bbox = target_face.bbox.astype(int)
        
        # 3. 姿态检测
        pose_results = self.pose_model(frame, verbose=False)
        
        # 4. 匹配姿态到目标学生
        target_pose, pose_bbox = self._match_pose_to_bbox(pose_results, face_bbox)
        
        if target_pose is None:
            return {
                "frame_index": frame_index,
                "student_found": True,
                "student_name": target_student_name,
                "face_similarity": float(max_similarity),
                "pose_found": False,
                "timestamp": frame_index * 30
            }
        
        # 5. 分析姿态行为
        behavior = self._analyze_single_person_pose(target_pose)
        behavior["bbox"] = pose_bbox
        behavior["face_similarity"] = float(max_similarity)
        
        # 6. 物体检测
        object_results = self.object_model(frame, verbose=False)
        desktop_objects = self._analyze_desktop_objects(object_results)
        behavior["desktop_objects"] = desktop_objects
        
        # 7. 绘制标注（仅保存部分帧）
        annotated_image = None
        # 每10帧保存一次图片，或者第一帧和最后一帧
        if frame_index % 10 == 0 or frame_index == 0:
            annotated_frame = self._draw_individual_annotations(
                frame.copy(), 
                behavior, 
                target_student_name,
                face_bbox
            )
            
            # 8. 转换为Base64
            from behavior_service import ClassroomBehaviorAnalyzer
            temp_analyzer = ClassroomBehaviorAnalyzer()
            annotated_image = temp_analyzer._frame_to_base64(annotated_frame)
        
        return {
            "frame_index": frame_index,
            "timestamp": frame_index * 30,
            "student_found": True,
            "pose_found": True,
            "student_name": target_student_name,
            "behavior": behavior,
            "annotated_image": annotated_image
        }
    
    def _calculate_similarity(
        self, 
        embedding: np.ndarray, 
        target_descriptors: List[np.ndarray]
    ) -> float:
        """计算人脸特征的余弦相似度"""
        max_sim = -1
        for desc in target_descriptors:
            # 余弦相似度
            sim = np.dot(embedding, desc) / (np.linalg.norm(embedding) * np.linalg.norm(desc))
            max_sim = max(max_sim, sim)
        return float(max_sim)
    
    def _match_pose_to_bbox(
        self, 
        pose_results, 
        face_bbox: np.ndarray
    ) -> Tuple[Optional[Any], Optional[Dict]]:
        """
        根据人脸bbox匹配对应的姿态检测结果
        """
        best_match = None
        best_iou = 0
        best_bbox = None
        
        for result in pose_results:
            if result.boxes is not None and result.keypoints is not None:
                for i, box in enumerate(result.boxes):
                    pose_bbox = box.xyxy.cpu().numpy()[0].astype(int)
                    
                    # 计算IoU（重叠度）
                    iou = self._calculate_iou(face_bbox, pose_bbox)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_match = result.keypoints.data[i]
                        best_bbox = {
                            "x1": int(pose_bbox[0]),
                            "y1": int(pose_bbox[1]),
                            "x2": int(pose_bbox[2]),
                            "y2": int(pose_bbox[3])
                        }
        
        if best_iou > 0.3:  # IoU阈值
            return best_match, best_bbox
        
        return None, None
    
    def _calculate_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """计算两个边界框的IoU（交并比）"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0
        
        return intersection / union
    
    def _analyze_single_person_pose(self, keypoints) -> Dict:
        """分析单个人的姿态（复用behavior_service的逻辑）"""
        from behavior_service import ClassroomBehaviorAnalyzer
        temp_analyzer = ClassroomBehaviorAnalyzer()
        return temp_analyzer._analyze_single_person_pose(keypoints)
    
    def _analyze_desktop_objects(self, object_results) -> List[Dict]:
        """分析桌面物品（复用behavior_service的逻辑）"""
        from behavior_service import ClassroomBehaviorAnalyzer
        temp_analyzer = ClassroomBehaviorAnalyzer()
        return temp_analyzer._analyze_desktop_objects(object_results)
    
    def _draw_individual_annotations(
        self,
        frame: np.ndarray,
        behavior: Dict,
        student_name: str,
        face_bbox: np.ndarray
    ) -> np.ndarray:
        """绘制个人行为标注"""
        from PIL import Image, ImageDraw, ImageFont
        
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
        
        # 绘制姿态边界框
        if "bbox" in behavior:
            bbox = behavior["bbox"]
            color = self.behavior_colors.get(behavior["head_pose"], (255, 255, 255))
            
            cv2.rectangle(frame,
                        (bbox["x1"], bbox["y1"]),
                        (bbox["x2"], bbox["y2"]),
                        color, 3)  # 加粗边框
            
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
            
            # 绘制学生姓名
            name_position = (bbox["x1"], max(0, bbox["y1"] - 60))
            name_text = f"👤 {student_name}"
            bbox_name = draw.textbbox(name_position, name_text, font=font)
            draw.rectangle(bbox_name, fill=(255, 100, 0, 200))
            draw.text(name_position, name_text, fill=(255, 255, 255), font=font)
            
            # 绘制行为标签
            head_pose_label = behavior_labels.get(behavior["head_pose"], behavior["head_pose"])
            hand_activity_label = behavior_labels.get(behavior["hand_activity"], behavior["hand_activity"])
            label = f'{head_pose_label} / {hand_activity_label}'
            
            text_position = (bbox["x1"], max(0, bbox["y1"] - 30))
            bbox_text = draw.textbbox(text_position, label, font=font)
            draw.rectangle(bbox_text, fill=(0, 0, 0, 180))
            draw.text(text_position, label, fill=color, font=font)
            
            # 转回 OpenCV 格式
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        return frame
    
    def _summarize_individual_analysis(
        self, 
        frame_results: List[Dict],
        student_name: str
    ) -> Dict:
        """汇总个人行为分析结果"""
        if not frame_results:
            return {
                "error": f"在所有帧中都未找到学生 {student_name}",
                "behavior_percentages": {
                    "looking_up": 0.0,
                    "looking_down": 0.0,
                    "neutral": 0.0,
                    "writing": 0.0,
                    "using_phone": 0.0,
                    "resting": 0.0,
                    "unknown": 0.0
                },
                "recognition_score": 0,
                "total_frames_analyzed": 0
            }
        
        # 统计行为
        head_pose_stats = {"looking_up": 0, "looking_down": 0, "neutral": 0}
        hand_activity_stats = {"writing": 0, "using_phone": 0, "resting": 0, "unknown": 0}
        
        total_frames = len(frame_results)
        total_similarity = 0
        
        for result in frame_results:
            if result.get("pose_found") and "behavior" in result:
                behavior = result["behavior"]
                
                # 统计头部姿态
                head_pose = behavior["head_pose"]
                if head_pose in head_pose_stats:
                    head_pose_stats[head_pose] += 1
                
                # 统计手部活动
                hand_activity = behavior["hand_activity"]
                if hand_activity in hand_activity_stats:
                    hand_activity_stats[hand_activity] += 1
                
                # 累计相似度
                total_similarity += behavior.get("face_similarity", 0)
        
        # 计算百分比
        behavior_percentages = {}
        for pose, count in head_pose_stats.items():
            behavior_percentages[pose] = round((count / total_frames) * 100, 2)
        for activity, count in hand_activity_stats.items():
            behavior_percentages[activity] = round((count / total_frames) * 100, 2)
        
        # 计算认真程度评分
        looking_up_pct = behavior_percentages.get("looking_up", 0)
        writing_pct = behavior_percentages.get("writing", 0)
        using_phone_pct = behavior_percentages.get("using_phone", 0)
        
        attention_score = max(0, min(100, 
            looking_up_pct * 0.6 + writing_pct * 0.3 - using_phone_pct * 0.3
        ))
        
        # 平均识别准确度
        avg_similarity = total_similarity / total_frames if total_frames > 0 else 0
        
        return {
            "behavior_stats": {**head_pose_stats, **hand_activity_stats},
            "behavior_percentages": behavior_percentages,
            "attention_score": round(attention_score, 2),
            "recognition_accuracy": round(avg_similarity * 100, 2),
            "total_frames_analyzed": total_frames,
            "conclusions": self._generate_individual_conclusions(behavior_percentages, attention_score)
        }
    
    def _generate_individual_conclusions(
        self, 
        behavior_percentages: Dict, 
        attention_score: float
    ) -> List[str]:
        """生成个人分析结论"""
        conclusions = []
        
        looking_up = behavior_percentages.get("looking_up", 0)
        looking_down = behavior_percentages.get("looking_down", 0)
        writing = behavior_percentages.get("writing", 0)
        using_phone = behavior_percentages.get("using_phone", 0)
        
        # 总体评价
        if attention_score >= 70:
            conclusions.append(f"整体表现优秀，认真程度评分 {attention_score:.0f}/100，学习态度积极")
        elif attention_score >= 50:
            conclusions.append(f"整体表现良好，认真程度评分 {attention_score:.0f}/100，有一定的学习专注度")
        else:
            conclusions.append(f"需要改进，认真程度评分 {attention_score:.0f}/100，建议提高课堂参与度")
        
        # 抬头听课
        if looking_up > 60:
            conclusions.append(f"课堂注意力集中，抬头听课占比 {looking_up:.1f}%，专注度高")
        elif looking_up < 30:
            conclusions.append(f"抬头听课时间较少（{looking_up:.1f}%），建议提高课堂专注度")
        
        # 记笔记
        if writing > 30:
            conclusions.append(f"学习主动性强，记笔记时间占比 {writing:.1f}%")
        
        # 使用手机
        if using_phone > 15:
            conclusions.append(f"使用手机时间较多（{using_phone:.1f}%），建议减少手机使用")
        
        return conclusions
