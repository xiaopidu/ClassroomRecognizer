export interface Student {
  id: string;
  name: string;
  photoUrl: string;
  descriptors: Float32Array[]; // Storing face descriptors
  createdAt: number;
}

export interface RecognitionParams {
  minConfidence: number; // For detection
  similarityThreshold: number; // InsightFace uses Cosine Similarity (Higher is better)
  minFaceSize: number; // In pixels
  iouThreshold: number; // Intersection over union
  maskMode: boolean; // Optimization for masked faces
  networkSize: number; // Input size for the neural net
}

export interface ImageQualityReport {
  isValid: boolean;
  score: number;
  issues: string[];
  lighting: 'good' | 'poor' | 'harsh';
  angle: 'frontal' | 'profile' | 'tilted';
}

export interface BehaviorReport {
  timestamp: string;
  studentCount: number;
  attentionScore: number; // 0 - 100
  behaviors: {
    action: string; // e.g., "Looking at Blackboard", "Using Phone"
    count: number;
    description: string; // Specific details like "Student in back row"
  }[];
  summary: string; // Overall atmosphere description
}

export interface SingleStudentBehaviorReport {
  timestamp: string;
  focusScore: number; // 0-100
  isDistracted: boolean;
  action: string; // e.g., "Writing notes", "Sleeping"
  posture: string; // e.g., "Upright", "Leaning back"
  expression: string; // e.g., "Focused", "Bored"
  summary: string;
  head_percentages?: { [key: string]: number }; // 单独的头部姿态百分比
  hand_percentages?: { [key: string]: number }; // 单独的手部活动百分比
  object_percentages?: { [key: string]: number }; // 物体出现百分比(laptop, book, cell_phone)
}

// 视频行为分析相关类型
export interface VideoAnalysisParams {
  start_time: number; // 开始时间（秒）
  duration: number; // 分析时长（秒）
  pose_conf_threshold: number; // 姿态检测置信度
  object_conf_threshold: number; // 物体检测置信度
  looking_up_threshold: number; // 抬头阈值
  looking_down_threshold: number; // 低头阈值
  target_student_bbox?: { x: number; y: number; w: number; h: number }; // 目标学生边界框（可选）
}

export interface VideoBehaviorStats {
  total_frames: number;
  listening: number; // 听讲帧数
  using_computer: number;
  using_phone: number;
  reading_writing: number;
  neutral: number;
  duration_seconds: number;
}

// Augment window to include face-api types since we load it via script
declare global {
  interface Window {
    faceapi: any;
  }
}

export enum AppTab {
  ANALYZE = 'ANALYZE',
  REGISTER = 'REGISTER',
  IMAGE_RECOGNITION = 'IMAGE_RECOGNITION',
  POSE_TEST = 'POSE_TEST',  // 姿态检测测试页面
  POSE_VIDEO = 'POSE_VIDEO',  // 新增：行为分析（视频）
}