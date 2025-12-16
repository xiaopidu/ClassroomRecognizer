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
}