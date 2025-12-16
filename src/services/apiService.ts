// 后端 API 服务
const API_BASE_URL = 'http://localhost:5001';

export interface FaceDetectionResult {
  id: number;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
  confidence: number;
  landmarks: [number, number][]; // 关键点
  embedding: number[]; // 特征向量
  recognition?: {
    name: string;
    confidence: number;
  }; // 识别结果
}

export interface DetectionResponse {
  success: boolean;
  faces: FaceDetectionResult[];
  count: number;
  params_used?: {
    min_confidence: number;
    network_size: number;
    min_face_size: number;
  };
}

export interface HealthCheckResponse {
  status: string;
  model_loaded: boolean;
  params?: {
    min_confidence: number;
    network_size: number;
    min_face_size: number;
  };
}

export interface ParamsResponse {
  success: boolean;
  params: {
    min_confidence: number;
    network_size: number;
    min_face_size: number;
  };
  message?: string;
}

/**
 * 检查后端服务健康状态
 */
export async function checkHealth(): Promise<HealthCheckResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('健康检查失败:', error);
    throw new Error('无法连接到后端服务');
  }
}

/**
 * 上传图像文件进行人脸检测
 * @param imageFile 图像文件
 */
export async function detectFacesFromFile(imageFile: File): Promise<DetectionResponse> {
  try {
    const formData = new FormData();
    formData.append('image', imageFile);

    const response = await fetch(`${API_BASE_URL}/api/detect`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      let errorMessage = `HTTP error! status: ${response.status}`;
      try {
        const errorData = await response.json();
        errorMessage = errorData.error || errorMessage;
      } catch (e) {
        // If we can't parse the error response, use the text
        try {
          const errorText = await response.text();
          errorMessage = errorText || errorMessage;
        } catch (e) {
          // If we can't get the text, keep the original message
        }
      }
      throw new Error(errorMessage);
    }

    return await response.json();
  } catch (error: any) {
    console.error('人脸检测失败:', error);
    const errorMessage = error instanceof Error ? error.message : '未知错误';
    throw new Error(`人脸检测失败: ${errorMessage}`);
  }
}

/**
 * 发送 Base64 图像数据进行人脸检测
 * @param imageData Base64 图像数据
 */
export async function detectFacesFromBase64(imageData: string): Promise<DetectionResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/detect-base64`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ image: imageData }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('人脸检测失败:', error);
    throw new Error(`人脸检测失败: ${error instanceof Error ? error.message : '未知错误'}`);
  }
}

/**
 * 获取后端服务状态信息
 */
export async function getServiceInfo(): Promise<any> {
  try {
    const response = await fetch(`${API_BASE_URL}/`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('获取服务信息失败:', error);
    throw new Error('无法连接到后端服务');
  }
}

/**
 * 获取当前检测参数
 */
export async function getCurrentParams(): Promise<ParamsResponse> {
  try {
    console.log('发送获取参数请求');
    
    const response = await fetch(`${API_BASE_URL}/api/params`);
    
    // 检查响应状态
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status} ${response.statusText}`);
    }
    
    const text = await response.text();
    console.log('原始响应文本:', text);
    
    let responseData;
    try {
      responseData = JSON.parse(text);
    } catch (parseError) {
      throw new Error(`响应不是有效的JSON格式: ${text.substring(0, 100)}...`);
    }
    
    console.log('解析后的响应数据:', responseData);
    
    if (!responseData.success) {
      throw new Error(responseData.error || '获取参数失败');
    }

    return responseData;
  } catch (error) {
    console.error('获取参数失败:', error);
    if (error instanceof TypeError && error.message.includes('fetch')) {
      throw new Error('网络连接失败，请检查后端服务是否正常运行');
    }
    throw new Error(`获取参数失败: ${error instanceof Error ? error.message : '未知错误'}`);
  }
}

/**
 * 设置检测参数
 * @param params 参数对象
 */
export async function setDetectionParams(params: {
  min_confidence?: number;
  network_size?: number;
  min_face_size?: number;
}): Promise<ParamsResponse> {
  try {
    console.log('发送参数更新请求:', params);
    
    const response = await fetch(`${API_BASE_URL}/api/params`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(params),
    });

    // 检查响应状态
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status} ${response.statusText}`);
    }
    
    const text = await response.text();
    console.log('原始响应文本:', text);
    
    let responseData;
    try {
      responseData = JSON.parse(text);
    } catch (parseError) {
      throw new Error(`响应不是有效的JSON格式: ${text.substring(0, 100)}...`);
    }
    
    console.log('解析后的响应数据:', responseData);
    
    if (!responseData.success) {
      throw new Error(responseData.error || '参数更新失败');
    }

    return responseData;
  } catch (error) {
    console.error('设置参数失败:', error);
    if (error instanceof TypeError && error.message.includes('fetch')) {
      throw new Error('网络连接失败，请检查后端服务是否正常运行');
    }
    throw new Error(`设置参数失败: ${error instanceof Error ? error.message : '未知错误'}`);
  }
}


/**
 * 更新注册学生数据
 * @param students 学生数据数组
 */
export async function updateRegisteredStudents(students: any[]): Promise<any> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/students`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ students }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('更新学生数据失败:', error);
    throw new Error(`更新学生数据失败: ${error instanceof Error ? error.message : '未知错误'}`);
  }
}

// 行为分析参数接口
export interface BehaviorParams {
  head_up_threshold?: number;
  head_down_threshold?: number;
  writing_threshold?: number;
  phone_threshold?: number;
  object_min_confidence?: number;
}

export interface BehaviorParamsResponse {
  success: boolean;
  params: BehaviorParams;
  message?: string;
  updated_fields?: string[];
}

/**
 * 获取当前行为分析参数
 */
export async function getCurrentBehaviorParams(): Promise<BehaviorParamsResponse> {
  try {
    console.log('发送获取行为分析参数请求');
    
    const response = await fetch(`${API_BASE_URL}/api/behavior-params`);
    
    // 检查响应状态
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status} ${response.statusText}`);
    }
    
    const text = await response.text();
    console.log('原始响应文本:', text);
    
    let responseData;
    try {
      responseData = JSON.parse(text);
    } catch (parseError) {
      throw new Error(`响应不是有效的JSON格式: ${text.substring(0, 100)}...`);
    }
    
    console.log('解析后的响应数据:', responseData);
    
    if (!responseData.success) {
      throw new Error(responseData.error || '获取行为分析参数失败');
    }

    return responseData;
  } catch (error) {
    console.error('获取行为分析参数失败:', error);
    if (error instanceof TypeError && error.message.includes('fetch')) {
      throw new Error('网络连接失败，请检查后端服务是否正常运行');
    }
    throw new Error(`获取行为分析参数失败: ${error instanceof Error ? error.message : '未知错误'}`);
  }
}

/**
 * 设置行为分析参数
 * @param params 参数对象
 */
export async function setBehaviorParams(params: BehaviorParams): Promise<BehaviorParamsResponse> {
  try {
    console.log('发送行为分析参数更新请求:', params);
    
    const response = await fetch(`${API_BASE_URL}/api/behavior-params`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(params),
    });

    // 检查响应状态
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status} ${response.statusText}`);
    }
    
    const text = await response.text();
    console.log('原始响应文本:', text);
    
    let responseData;
    try {
      responseData = JSON.parse(text);
    } catch (parseError) {
      throw new Error(`响应不是有效的JSON格式: ${text.substring(0, 100)}...`);
    }
    
    console.log('解析后的响应数据:', responseData);
    
    if (!responseData.success) {
      throw new Error(responseData.error || '行为分析参数更新失败');
    }

    return responseData;
  } catch (error) {
    console.error('设置行为分析参数失败:', error);
    if (error instanceof TypeError && error.message.includes('fetch')) {
      throw new Error('网络连接失败，请检查后端服务是否正常运行');
    }
    throw new Error(`设置行为分析参数失败: ${error instanceof Error ? error.message : '未知错误'}`);
  }
}

// 视频行为分析接口
export interface StudentBehavior {
  head_pose: string;
  hand_activity: string;
  confidence: number;
  keypoints_visible: number;
  bbox: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
  };
  desktop_objects: Array<{
    label: string;
    confidence: number;
    bbox: {
      x1: number;
      y1: number;
      x2: number;
      y2: number;
    };
  }>;
}

export interface FrameResult {
  frame_index: number;
  timestamp: number;
  processing_time: number;
  student_count: number;
  behaviors: StudentBehavior[];
  annotated_image: string;
}

export interface BehaviorAnalysisSummary {
  behavior_stats: Record<string, number>;
  behavior_percentages: Record<string, number>;
  object_stats: Record<string, number>;
  object_percentages: Record<string, number>;
  conclusions: string[];
  total_frames: number;
  processing_time?: number;
}

export interface VideoAnalysisResult {
  timestamp: number;
  processing_time: number;
  frame_count: number;
  frame_results: FrameResult[];
  summary: BehaviorAnalysisSummary;
}

// 分析单帧图像中的学生行为
export async function analyzeClassroomBehaviorBase64(imageBase64: string): Promise<any> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/behavior-analyze-base64`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ image: imageBase64 }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data.result;
  } catch (error) {
    console.error('Error analyzing classroom behavior:', error);
    throw error;
  }
}

// 分析视频帧序列中的学生行为并进行汇总
export async function analyzeVideoFramesBase64(imagesBase64: string[]): Promise<VideoAnalysisResult> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/behavior-analyze-video-base64`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ images: imagesBase64 }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data.result;
  } catch (error) {
    console.error('Error analyzing video frames:', error);
    throw error;
  }
}

// 分析视频帧序列中的学生行为并进行汇总，生成图表
export async function analyzeVideoFramesBase64WithChart(imagesBase64: string[]): Promise<VideoAnalysisResult & { chart_image: string }> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/behavior-analyze-video-base64-with-chart`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ images: imagesBase64 }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data.result;
  } catch (error) {
    console.error('Error analyzing video frames with chart:', error);
    throw error;
  }
}

// 个人行为分析结果接口
export interface IndividualBehaviorResult {
  student_name: string;
  timestamp: number;
  processing_time: number;
  total_frames: number;
  frames_with_student: number;
  frames_without_student: number;
  frame_results: Array<{
    frame_index: number;
    timestamp: number;
    student_found: boolean;
    pose_found?: boolean;
    student_name?: string;
    face_similarity?: number;
    behavior?: StudentBehavior;
    annotated_image?: string;
  }>;
  summary: {
    behavior_stats: Record<string, number>;
    behavior_percentages: Record<string, number>;
    attention_score: number;
    recognition_accuracy: number;
    total_frames_analyzed: number;
    conclusions: string[];
  };
}

// 分析指定学生的行为（结合人脸识别和姿态检测）
export async function analyzeIndividualBehavior(
  imagesBase64: string[],
  targetStudent: string
): Promise<IndividualBehaviorResult> {
  try {
    console.log(`开始分析学生 ${targetStudent} 的行为，共 ${imagesBase64.length} 帧`);
    
    const response = await fetch(`${API_BASE_URL}/api/behavior-analyze-individual`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        images: imagesBase64,
        target_student: targetStudent
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    
    if (!data.success) {
      throw new Error(data.error || '个人行为分析失败');
    }
    
    console.log(`学生 ${targetStudent} 行为分析完成`);
    return data.result;
  } catch (error) {
    console.error('个人行为分析出错:', error);
    throw error;
  }
}
