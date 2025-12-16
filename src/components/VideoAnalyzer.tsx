import React, { useState, useRef, useEffect } from 'react';
import { analyzeVideoFramesBase64, getCurrentBehaviorParams, setBehaviorParams, analyzeIndividualBehavior, IndividualBehaviorResult } from '../services/apiService';
import { Student } from '../types';

interface StudentBehavior {
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

interface FrameResult {
  frame_index: number;
  timestamp: number;
  processing_time: number;
  student_count: number;
  behaviors: StudentBehavior[];
  annotated_image: string;
}

interface BehaviorAnalysisSummary {
  behavior_stats: Record<string, number>;
  behavior_percentages: Record<string, number>;
  object_stats: Record<string, number>;
  object_percentages: Record<string, number>;
  conclusions: string[];
  total_frames: number;
  processing_time?: number;
}

interface VideoAnalysisResult {
  timestamp: number;
  processing_time: number;
  frame_count: number;
  frame_results: FrameResult[];
  summary: BehaviorAnalysisSummary;
}

interface VideoAnalyzerProps {
  students: Student[];
  params: any;
  onSnapshotTaken?: (imageData: string) => void;
}

const VideoAnalyzer: React.FC<VideoAnalyzerProps> = ({ students, params, onSnapshotTaken }) => {
  // 视频相关状态
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoDuration, setVideoDuration] = useState<number>(0);
  const [trimmedVideoUrl, setTrimmedVideoUrl] = useState<string | null>(null);
  
  // 分析参数
  const [trimStart, setTrimStart] = useState(0);
  const [trimDuration, setTrimDuration] = useState(300);
  const [analysisMode, setAnalysisMode] = useState<'class' | 'individual'>('class'); // 分析模式
  const [targetStudent, setTargetStudent] = useState<string>(''); // 指定同学姓名
  
  // 框选相关状态
  const [isSelectingBBox, setIsSelectingBBox] = useState(false); // 是否正在框选
  const [isDrawing, setIsDrawing] = useState(false); // 是否正在拖动绘制
  const [bboxStart, setBboxStart] = useState<{x: number, y: number} | null>(null);
  const [bboxEnd, setBboxEnd] = useState<{x: number, y: number} | null>(null);
  const [selectedBBox, setSelectedBBox] = useState<{x: number, y: number, width: number, height: number} | null>(null);
  const [firstFrameData, setFirstFrameData] = useState<string | null>(null); // 第一帧图像数据
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  // 分析状态
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [analysisResult, setAnalysisResult] = useState<VideoAnalysisResult | IndividualBehaviorResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  // 行为分析参数
  const [behaviorParams, setBehaviorParamsState] = useState({
    head_up_threshold: 2,   // 正常坐姿算抬头
    head_down_threshold: 8,  // 明显低头才算
    writing_threshold: 30,
    phone_threshold: -10,
    object_min_confidence: 0.5
  });
  const [isParamsLoading, setIsParamsLoading] = useState(false);
  const [paramUpdateStatus, setParamUpdateStatus] = useState<{success: boolean, message: string} | null>(null);
  
  const videoRef = useRef<HTMLVideoElement>(null);

  // 行为标签的中文翻译
  const getBehaviorLabel = (behavior: string): string => {
    const labelMap: Record<string, string> = {
      'looking_up': '抬头',
      'looking_down': '低头',
      'neutral': '中性',
      'writing': '记笔记',
      'using_phone': '玩手机',
      'resting': '休息',
      'unknown': '未知'
    };
    return labelMap[behavior] || behavior;
  };

  // 加载行为分析参数
  useEffect(() => {
    const loadBehaviorParams = async () => {
      setIsParamsLoading(true);
      try {
        const response = await getCurrentBehaviorParams();
        if (response.success) {
          setBehaviorParamsState(response.params);
        }
      } catch (err) {
        console.error('加载行为分析参数失败:', err);
      } finally {
        setIsParamsLoading(false);
      }
    };
    
    loadBehaviorParams();
  }, []);

  // 处理视频上传
  const handleVideoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setVideoFile(file);
      setAnalysisResult(null);
      setError(null);
      
      const url = URL.createObjectURL(file);
      setTrimmedVideoUrl(url);
      
      // 获取视频时长
      const video = document.createElement('video');
      video.src = url;
      video.preload = 'metadata';
      video.onloadedmetadata = () => {
        setVideoDuration(video.duration);
      };
    }
  };

  // 保存行为参数
  const saveBehaviorParams = async () => {
    setIsParamsLoading(true);
    setParamUpdateStatus(null);
    try {
      const response = await setBehaviorParams(behaviorParams);
      if (response.success) {
        setParamUpdateStatus({ success: true, message: '参数保存成功' });
        setTimeout(() => setParamUpdateStatus(null), 3000);
      } else {
        setParamUpdateStatus({ success: false, message: '参数保存失败' });
      }
    } catch (err) {
      console.error('保存行为参数失败:', err);
      setParamUpdateStatus({ success: false, message: '保存失败' });
    } finally {
      setIsParamsLoading(false);
    }
  };

  // 开始分析
  const handleAnalyze = async () => {
    if (!videoFile) {
      setError('请先上传视频文件');
      return;
    }

    // 个人分析模式将在后续步骤中进行框选

    setIsAnalyzing(true);
    setProgress(0);
    setError(null);
    setAnalysisResult(null);

    try {
      await trimVideo();
      await analyzeVideoFrames();
    } catch (err) {
      console.error('分析过程中出错:', err);
      setError(err instanceof Error ? err.message : '分析过程中发生未知错误');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const trimVideo = (): Promise<void> => {
    return new Promise((resolve, reject) => {
      if (!videoFile) {
        reject(new Error('没有视频文件'));
        return;
      }

      setProgress(10);
      
      const video = document.createElement('video');
      video.src = URL.createObjectURL(videoFile);
      video.preload = 'metadata';
      
      video.onloadedmetadata = () => {
        if (trimStart >= video.duration) {
          setError(`开始时间(${trimStart}秒)超过视频总时长(${video.duration.toFixed(1)}秒)`);
          reject(new Error('开始时间超过视频总时长'));
          return;
        }
        
        setTrimmedVideoUrl(video.src);
        setProgress(30);
        resolve();
      };
      
      video.onerror = () => {
        reject(new Error('视频加载失败'));
      };
    });
  };

  const analyzeVideoFrames = async () => {
    if (!trimmedVideoUrl || !videoFile) {
      throw new Error('没有可用的视频片段');
    }

    setProgress(40);
    
    try {
      const video = document.createElement('video');
      video.src = trimmedVideoUrl;
      video.preload = 'metadata';
      
      await new Promise<void>((resolve, reject) => {
        video.onloadedmetadata = () => resolve();
        video.onerror = () => reject(new Error('视频加载失败'));
      });
      
      // 如果是个人分析模式，先显示第一帧让用户框选
      let bbox = null;
      if (analysisMode === 'individual') {
        console.log('个人分析模式：先显示第一帧让用户框选...');
        
        // 提取第一帧用于框选
        const canvas = document.createElement('canvas');
        canvas.width = 640;
        canvas.height = 480;
        const ctx = canvas.getContext('2d');
        
        if (!ctx) {
          throw new Error('无法创建canvas上下文');
        }
        
        // 设置视频到开始位置
        await new Promise<void>((resolve) => {
          video.currentTime = trimStart;
          video.onseeked = () => resolve();
        });
        
        // 绘制第一帧
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const firstFrame = canvas.toDataURL('image/jpeg', 0.8);
        
        // 显示框选界面
        setFirstFrameData(firstFrame);
        setIsSelectingBBox(true);
        setProgress(50);
        
        console.log('等待用户框选目标学生...');
        
        // 等待用户完成框选
        bbox = await waitForBBoxSelection();
        
        if (!bbox) {
          throw new Error('未选择目标区域');
        }
        
        console.log('用户已框选:', bbox);
        setProgress(55);
      }
      
      const endTime = Math.min(trimStart + trimDuration, video.duration);
      const actualDuration = endTime - trimStart;
      const frameRate = 1 / 10; // 每10秒1帧(减少处理负担,避免超时)
      const totalFrames = Math.floor(actualDuration * frameRate);
      const frameInterval = 1 / frameRate; // 10秒
      
      console.log(`提取帧: 从${trimStart}秒开始, 时长${trimDuration}秒, 共${totalFrames}帧 (每10秒1帧)`);
      
      const frames: string[] = [];
      const canvas = document.createElement('canvas');
      canvas.width = 640;
      canvas.height = 480;
      const ctx = canvas.getContext('2d');
      
      if (!ctx) {
        throw new Error('无法创建canvas上下文');
      }
      
      for (let i = 0; i < totalFrames; i++) {
        const currentTime = trimStart + (i * frameInterval);
        
        await new Promise<void>((resolve) => {
          video.currentTime = currentTime;
          video.onseeked = () => resolve();
        });
        
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const frameData = canvas.toDataURL('image/jpeg', 0.8);
        frames.push(frameData);
        
        const extractProgress = 60 + Math.floor((i / totalFrames) * 20);
        setProgress(extractProgress);
      }
      
      console.log(`成功提取${frames.length}帧`);
      setProgress(80);
      
      // 根据分析模式调用不同API
      let result;
      if (analysisMode === 'individual') {
        console.log('开始基于边界框的个人分析:', bbox);
        
        // 调用边界框追踪分析API
        result = await analyzeWithBBox(frames, bbox!);
      } else {
        // 全班行为分析：只做姿态检测
        console.log('开始全班分析');
        result = await analyzeVideoFramesBase64(frames);
      }
      
      setProgress(90);
      setAnalysisResult(result);
      setProgress(100);
      
      setTimeout(() => {
        setProgress(0);
      }, 2000);
      
    } catch (err) {
      console.error('帧分析过程中出错:', err);
      throw new Error('帧分析过程中发生错误: ' + (err instanceof Error ? err.message : '未知错误'));
    }
  };

  // Canvas鼠标事件处理
  const handleCanvasMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isSelectingBBox || !canvasRef.current || selectedBBox) return; // 已有选框则不再响应
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
    const y = ((e.clientY - rect.top) / rect.height) * canvas.height;
    
    setIsDrawing(true); // 开始绘制
    setBboxStart({x, y});
    setBboxEnd(null);
  };

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    // 只有在按下鼠标拖动时才绘制
    if (!isSelectingBBox || !isDrawing || !bboxStart || !canvasRef.current || selectedBBox) return;
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
    const y = ((e.clientY - rect.top) / rect.height) * canvas.height;
    
    setBboxEnd({x, y});
    
    // 实时绘制矩形框
    drawBBoxOnCanvas(bboxStart, {x, y});
  };

  const handleCanvasMouseUp = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isSelectingBBox || !isDrawing || !bboxStart || !canvasRef.current || selectedBBox) return;
    
    setIsDrawing(false); // 停止绘制
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
    const y = ((e.clientY - rect.top) / rect.height) * canvas.height;
    
    const endPoint = {x, y};
    setBboxEnd(endPoint);
    
    // 计算边界框
    const bbox = {
      x: Math.min(bboxStart.x, endPoint.x),
      y: Math.min(bboxStart.y, endPoint.y),
      width: Math.abs(endPoint.x - bboxStart.x),
      height: Math.abs(endPoint.y - bboxStart.y)
    };
    
    // 不自动设置，等待用户点击确认按钮
    // setSelectedBBox(bbox);
    
    // 保存临时边界框，绘制并显示确认按钮
    drawFinalBBox(bboxStart, endPoint);
    // 使用临时状态保存
    (window as any).tempBBox = bbox;
    console.log('绘制的边界框:', bbox);
  };

  // 在Canvas上绘制边界框
  const drawBBoxOnCanvas = (start: {x: number, y: number}, end: {x: number, y: number}) => {
    if (!canvasRef.current || !firstFrameData) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // 加载第一帧图像
    const img = new Image();
    img.src = firstFrameData;
    img.onload = () => {
      // 绘制图像
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      
      // 绘制矩形框
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 3;
      ctx.setLineDash([]);
      
      const width = end.x - start.x;
      const height = end.y - start.y;
      ctx.strokeRect(start.x, start.y, width, height);
      
      // 添加半透明填充
      ctx.fillStyle = 'rgba(59, 130, 246, 0.1)';
      ctx.fillRect(start.x, start.y, width, height);
    };
  };

  // 绘制最终固定的边界框（松开鼠标后）
  const drawFinalBBox = (start: {x: number, y: number}, end: {x: number, y: number}) => {
    if (!canvasRef.current || !firstFrameData) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // 加载第一帧图像
    const img = new Image();
    img.src = firstFrameData;
    img.onload = () => {
      // 绘制图像
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      
      // 绘制固定的矩形框（更粗的线条）
      ctx.strokeStyle = '#10b981'; // 绿色
      ctx.lineWidth = 4;
      ctx.setLineDash([]);
      
      const width = end.x - start.x;
      const height = end.y - start.y;
      ctx.strokeRect(start.x, start.y, width, height);
      
      // 添加半透明填充
      ctx.fillStyle = 'rgba(16, 185, 129, 0.15)';
      ctx.fillRect(start.x, start.y, width, height);
    };
  };

  // 确认选框并开始分析
  const handleConfirmBBox = () => {
    const tempBBox = (window as any).tempBBox;
    if (tempBBox) {
      setSelectedBBox(tempBBox);
      console.log('确认边界框:', tempBBox);
    }
  };

  // 重新选择边界框
  const handleResetBBox = () => {
    setIsDrawing(false);
    setBboxStart(null);
    setBboxEnd(null);
    (window as any).tempBBox = null;
    
    // 重新显示第一帧
    if (firstFrameData && canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        const img = new Image();
        img.src = firstFrameData;
        img.onload = () => {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        };
      }
    }
  };

  // 当firstFrameData变化时，在Canvas上显示第一帧
  useEffect(() => {
    if (firstFrameData && canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      
      const img = new Image();
      img.src = firstFrameData;
      img.onload = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      };
    }
  }, [firstFrameData]);

  // 等待用户完成边界框选择
  const waitForBBoxSelection = (): Promise<{x: number, y: number, width: number, height: number} | null> => {
    return new Promise((resolve) => {
      let resolved = false;
      
      // 定义一个检查间隔
      const checkInterval = setInterval(() => {
        // 使用回调获取最新状态
        setSelectedBBox(currentBBox => {
          if (currentBBox && !resolved) {
            resolved = true;
            clearInterval(checkInterval);
            setIsSelectingBBox(false);
            resolve(currentBBox);
          }
          return currentBBox;
        });
      }, 100);
      
      // 60秒超时
      setTimeout(() => {
        if (!resolved) {
          clearInterval(checkInterval);
          setIsSelectingBBox(false);
          console.warn('选框选择超时');
          resolve(null);
        }
      }, 60000);
    });
  };

  // 调用后端边界框追踪分析API
  const analyzeWithBBox = async (
    frames: string[], 
    bbox: {x: number, y: number, width: number, height: number}
  ): Promise<any> => {
    try {
      // 创建超时控制器（5分钟）
      const controller = new AbortController();
      const timeoutId = setTimeout(() => {
        controller.abort();
      }, 5 * 60 * 1000); // 5分钟超时

      try {
        const response = await fetch('http://localhost:5001/api/behavior-analyze-bbox', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            images: frames,
            target_student: '框选学生', // 简单命名
            initial_bbox: bbox
          }),
          signal: controller.signal // 添加超时控制
        });

        clearTimeout(timeoutId); // 清除超时定时器

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || '边界框追踪分析请求失败');
        }

        const data = await response.json();
        if (!data.success) {
          throw new Error(data.error || '边界框追踪分析失败');
        }

        return data.result;
      } catch (fetchError: any) {
        clearTimeout(timeoutId);
        if (fetchError.name === 'AbortError') {
          throw new Error('请求超时（5分钟），请尝试减少分析时长或联系管理员');
        }
        throw fetchError;
      }
    } catch (error) {
      console.error('边界框追踪分析出错:', error);
      throw error;
    }
  };

  // 格式化时间
  const formatTime = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
  };

  // 类型守卫：检查是否为个人分析结果
  const isIndividualResult = (result: VideoAnalysisResult | IndividualBehaviorResult): result is IndividualBehaviorResult => {
    return 'student_name' in result && 'frames_with_student' in result;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* 页面标题 */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">视频行为分析系统</h1>
          <p className="text-gray-600">基于 YOLOv8 Pose + YOLOv8 Object Detection 本地模型分析教室视频中的学生行为</p>
        </div>

        {/* 主要内容区域 */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* 左侧：视频上传和配置 */}
          <div className="lg:col-span-2 space-y-6">
            {/* 视频上传卡片 */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">1. 上传视频</h2>
              
              <div className="space-y-4">
                <div>
                  <input
                    type="file"
                    accept="video/*"
                    onChange={handleVideoUpload}
                    className="block w-full text-sm text-gray-900
                      file:mr-4 file:py-3 file:px-6
                      file:rounded-lg file:border-0
                      file:text-sm file:font-semibold
                      file:bg-blue-600 file:text-white
                      hover:file:bg-blue-700
                      file:cursor-pointer cursor-pointer
                      border border-gray-300 rounded-lg
                      focus:outline-none focus:ring-2 focus:ring-blue-500"
                    disabled={isAnalyzing}
                  />
                </div>
                
                {videoFile && videoDuration > 0 && (
                  <div className="flex items-center gap-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
                    <div className="flex-1">
                      <p className="text-sm font-medium text-gray-900">{videoFile.name}</p>
                      <p className="text-sm text-gray-600 mt-1">
                        总时长: {formatTime(videoDuration)} | 
                        大小: {(videoFile.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* 视频预览 */}
            {trimmedVideoUrl && (
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h2 className="text-xl font-semibold text-gray-900 mb-4">视频预览</h2>
                <video
                  ref={videoRef}
                  src={trimmedVideoUrl}
                  controls
                  className="w-full rounded-lg border border-gray-200"
                />
              </div>
            )}

            {/* 分析设置 */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">2. 分析设置</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-900 mb-2">
                    开始时间 (秒)
                  </label>
                  <input
                    type="number"
                    value={trimStart}
                    onChange={(e) => setTrimStart(Number(e.target.value))}
                    min="0"
                    max={videoDuration}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg 
                      focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                      text-gray-900"
                    disabled={isAnalyzing}
                  />
                  {videoDuration > 0 && (
                    <p className="text-xs text-gray-500 mt-1">
                      当前: {formatTime(trimStart)} / 总时长: {formatTime(videoDuration)}
                    </p>
                  )}
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-900 mb-2">
                    分析模式
                  </label>
                  <div className="flex gap-2 mb-4">
                    <button
                      onClick={() => {
                        setAnalysisMode('class');
                        setTrimDuration(300);
                      }}
                      className={`flex-1 px-4 py-3 rounded-lg font-medium text-sm transition-all ${
                        analysisMode === 'class'
                          ? 'bg-blue-600 text-white shadow-md'
                          : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                      }`}
                      disabled={isAnalyzing}
                    >
                      全班行为分析（5分钟）
                    </button>
                    <button
                      onClick={() => {
                        setAnalysisMode('individual');
                        setTrimDuration(2700);
                      }}
                      className={`flex-1 px-4 py-3 rounded-lg font-medium text-sm transition-all ${
                        analysisMode === 'individual'
                          ? 'bg-blue-600 text-white shadow-md'
                          : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                      }`}
                      disabled={isAnalyzing}
                    >
                      指定同学行为分析（45分钟）
                    </button>
                  </div>
                  
                  {/* 框选指定同学提示 */}
                  {analysisMode === 'individual' && (
                    <div className="mt-3">
                      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                        <div className="flex items-start gap-3">
                          <svg className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                          <div className="flex-1">
                            <h4 className="text-sm font-medium text-blue-900 mb-1">
                              框选操作说明
                            </h4>
                            <p className="text-xs text-blue-800 leading-relaxed">
                              点击"开始分析"后，将在视频第一帧显示画面，请用鼠标在视频画面上<strong>拖动框选</strong>目标学生区域。<br/>
                              系统将自动追踪该区域内的学生行为，无需人脸识别。
                            </p>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              <div className="mt-6">
                <button
                  onClick={handleAnalyze}
                  disabled={isAnalyzing || !videoFile}
                  className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 
                    text-white font-semibold py-4 px-6 rounded-lg 
                    disabled:opacity-50 disabled:cursor-not-allowed
                    transition-all duration-200 shadow-lg hover:shadow-xl
                    transform hover:-translate-y-0.5"
                >
                  {isAnalyzing ? '分析中...' : '开始分析'}
                </button>
              </div>
            </div>

            {/* 进度条 */}
            {isAnalyzing && (
              <div className="bg-white rounded-xl shadow-lg p-6">
                <div className="flex justify-between text-sm font-medium text-gray-900 mb-2">
                  <span>分析进度</span>
                  <span>{progress}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                  <div 
                    className="bg-gradient-to-r from-blue-600 to-indigo-600 h-3 rounded-full transition-all duration-300 ease-out" 
                    style={{ width: `${progress}%` }}
                  ></div>
                </div>
              </div>
            )}

            {/* 框选界面 */}
            {isSelectingBBox && firstFrameData && (
              <div className="bg-white rounded-xl shadow-lg p-6">
                <div className="mb-4">
                  <h2 className="text-xl font-semibold text-gray-900 mb-2">请框选目标学生</h2>
                  <p className="text-sm text-gray-600">
                    在下方视频画面上<strong className="text-blue-600">按住鼠标左键拖动</strong>，框选目标学生所在区域。
                    松开鼠标后点击<strong className="text-green-600">"确认选框"按钮</strong>开始追踪分析。
                  </p>
                </div>
                
                <div className="relative border-2 border-blue-400 rounded-lg overflow-hidden">
                  <canvas
                    ref={canvasRef}
                    width={640}
                    height={480}
                    onMouseDown={handleCanvasMouseDown}
                    onMouseMove={handleCanvasMouseMove}
                    onMouseUp={handleCanvasMouseUp}
                    className="w-full cursor-crosshair"
                    style={{ display: 'block' }}
                  />
                </div>
                
                {/* 显示确认和重选按钮 */}
                {bboxEnd && !selectedBBox && (
                  <div className="mt-4 flex gap-3">
                    <button
                      onClick={handleConfirmBBox}
                      className="flex-1 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 
                        text-white font-semibold py-3 px-6 rounded-lg 
                        transition-all duration-200 shadow-lg hover:shadow-xl
                        transform hover:-translate-y-0.5 flex items-center justify-center gap-2"
                    >
                      <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                      确认选框
                    </button>
                    <button
                      onClick={handleResetBBox}
                      className="flex-1 bg-gray-500 hover:bg-gray-600 
                        text-white font-semibold py-3 px-6 rounded-lg 
                        transition-all duration-200 shadow-lg hover:shadow-xl
                        transform hover:-translate-y-0.5 flex items-center justify-center gap-2"
                    >
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                      </svg>
                      重新选择
                    </button>
                  </div>
                )}
                
                {/* 已确认选框提示 */}
                {selectedBBox && (
                  <div className="mt-4 p-3 bg-green-50 border border-green-200 rounded-lg">
                    <div className="flex items-center gap-2 text-green-800">
                      <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                      <span className="font-medium">已确认目标区域，正在启动追踪分析...</span>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* 错误信息 */}
            {error && (
              <div className="bg-red-50 border-l-4 border-red-500 text-red-900 px-6 py-4 rounded-lg">
                <div className="flex items-center">
                  <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                  <span className="font-medium">{error}</span>
                </div>
              </div>
            )}
          </div>

          {/* 右侧：行为分析参数 */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-xl shadow-lg p-6 sticky top-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">行为分析参数</h2>
              
              {paramUpdateStatus && (
                <div className={`mb-4 p-3 rounded-lg text-sm ${
                  paramUpdateStatus.success 
                    ? 'bg-green-50 text-green-800 border border-green-200' 
                    : 'bg-red-50 text-red-800 border border-red-200'
                }`}>
                  {paramUpdateStatus.message}
                </div>
              )}
              
              <div className="space-y-6">
                {/* 头部姿态参数 */}
                <div>
                  <h3 className="text-sm font-semibold text-gray-900 mb-3">头部姿态阈值</h3>
                  
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm text-gray-700 mb-2">
                        抬头阈值: <span className="font-medium text-blue-600">{behaviorParams.head_up_threshold}</span>
                      </label>
                      <input
                        type="range"
                        min="-100"
                        max="0"
                        step="1"
                        value={behaviorParams.head_up_threshold}
                        onChange={(e) => setBehaviorParamsState(prev => ({
                          ...prev,
                          head_up_threshold: Number(e.target.value)
                        }))}
                        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                        disabled={isParamsLoading}
                      />
                      <p className="text-xs text-gray-500 mt-2">
                        <span className="font-medium">推荐值: -2</span> | 
                        调大(趋向0)→识别抬头更严格，需要头抬得更高 | 
                        调小(趋向-100)→识别抬头更宽松，轻微抬头也算
                      </p>
                    </div>
                    
                    <div>
                      <label className="block text-sm text-gray-700 mb-2">
                        低头阈值: <span className="font-medium text-blue-600">{behaviorParams.head_down_threshold}</span>
                      </label>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        step="1"
                        value={behaviorParams.head_down_threshold}
                        onChange={(e) => setBehaviorParamsState(prev => ({
                          ...prev,
                          head_down_threshold: Number(e.target.value)
                        }))}
                        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                        disabled={isParamsLoading}
                      />
                      <p className="text-xs text-gray-500 mt-2">
                        <span className="font-medium">推荐值: 3</span> | 
                        调大(趋向100)→识别低头更宽松，轻微低头也算 | 
                        调小(趋向0)→识别低头更严格，需要头低得更多
                      </p>
                    </div>
                  </div>
                </div>

                <div className="border-t border-gray-200 my-4"></div>

                {/* 手部活动参数 */}
                <div>
                  <h3 className="text-sm font-semibold text-gray-900 mb-3">手部活动阈值</h3>
                  
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm text-gray-700 mb-2">
                        记笔记阈值: <span className="font-medium text-blue-600">{behaviorParams.writing_threshold}</span>
                      </label>
                      <input
                        type="range"
                        min="0"
                        max="200"
                        step="1"
                        value={behaviorParams.writing_threshold}
                        onChange={(e) => setBehaviorParamsState(prev => ({
                          ...prev,
                          writing_threshold: Number(e.target.value)
                        }))}
                        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                        disabled={isParamsLoading}
                      />
                      <p className="text-xs text-gray-500 mt-2">
                        <span className="font-medium">推荐值: 30</span> | 
                        调大(趋向200)→识别记笔记更严格，手必须在更低位置 | 
                        调小(趋向0)→识别记笔记更宽松，手在桌面附近即算
                      </p>
                    </div>
                    
                    <div>
                      <label className="block text-sm text-gray-700 mb-2">
                        玩手机阈值: <span className="font-medium text-blue-600">{behaviorParams.phone_threshold}</span>
                      </label>
                      <input
                        type="range"
                        min="-200"
                        max="0"
                        step="1"
                        value={behaviorParams.phone_threshold}
                        onChange={(e) => setBehaviorParamsState(prev => ({
                          ...prev,
                          phone_threshold: Number(e.target.value)
                        }))}
                        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                        disabled={isParamsLoading}
                      />
                      <p className="text-xs text-gray-500 mt-2">
                        <span className="font-medium">推荐值: -10</span> | 
                        调大(趋向0)→识别玩手机更严格，手必须在更高位置 | 
                        调小(趋向-200)→识别玩手机更宽松，手稍微抬高即算
                      </p>
                    </div>
                    
                    <div>
                      <label className="block text-sm text-gray-700 mb-2">
                        物体检测置信度: <span className="font-medium text-blue-600">{behaviorParams.object_min_confidence.toFixed(2)}</span>
                      </label>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        value={behaviorParams.object_min_confidence}
                        onChange={(e) => setBehaviorParamsState(prev => ({
                          ...prev,
                          object_min_confidence: Number(e.target.value)
                        }))}
                        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                        disabled={isParamsLoading}
                      />
                      <p className="text-xs text-gray-500 mt-2">
                        <span className="font-medium">推荐值: 0.30</span> | 
                        调大(趋向1.0)→只识别高置信度物体，结果更准确但可能漏检 | 
                        调小(趋向0.0)→识别更多物体，但可能产生误检
                      </p>
                    </div>
                  </div>
                </div>

                <button
                  onClick={saveBehaviorParams}
                  disabled={isParamsLoading}
                  className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 px-4 rounded-lg 
                    disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {isParamsLoading ? '保存中...' : '保存参数'}
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* 分析结果 */}
        {analysisResult && (
          <div className="mt-6 bg-white rounded-xl shadow-lg p-6">
            <div className="flex justify-between items-center mb-6">
              <div>
                <h2 className="text-2xl font-bold text-gray-900">
                  {analysisMode === 'class' ? '全班行为分析结果' : `${targetStudent} - 行为分析结果`}
                </h2>
                {analysisMode === 'individual' && (
                  <p className="text-sm text-gray-500 mt-1">
                    重点分析该同学在45分钟内的行为表现
                  </p>
                )}
              </div>
              <div className="flex items-center gap-4">
                <div className="text-sm text-gray-600">
                  {isIndividualResult(analysisResult) ? (
                    // 个人分析统计
                    <>
                      分析帧数: {analysisResult.total_frames} | 
                      处理时间: {analysisResult.processing_time.toFixed(2)}秒 | 
                      识别准确度: {analysisResult.summary.recognition_accuracy?.toFixed(1)}%
                    </>
                  ) : (
                    // 全班分析统计
                    <>
                      分析帧数: {analysisResult.frame_count} | 
                      处理时间: {analysisResult.processing_time.toFixed(2)}秒
                    </>
                  )}
                </div>
                {analysisMode === 'individual' && (
                  <button
                    onClick={() => {
                      // 重置状态，重新开始框选
                      setAnalysisResult(null);
                      setSelectedBBox(null);
                      setBboxStart(null);
                      setBboxEnd(null);
                      setIsDrawing(false);
                      setProgress(0);
                      setError(null);
                      // 重新显示框选界面
                      if (trimmedVideoUrl) {
                        const video = document.createElement('video');
                        video.src = trimmedVideoUrl;
                        video.preload = 'metadata';
                        video.onloadedmetadata = async () => {
                          const canvas = document.createElement('canvas');
                          canvas.width = 640;
                          canvas.height = 480;
                          const ctx = canvas.getContext('2d');
                          if (ctx) {
                            video.currentTime = trimStart;
                            video.onseeked = () => {
                              ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                              const firstFrame = canvas.toDataURL('image/jpeg', 0.8);
                              setFirstFrameData(firstFrame);
                              setIsSelectingBBox(true);
                            };
                          }
                        };
                      }
                    }}
                    className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg 
                      transition-colors flex items-center gap-2 shadow-md hover:shadow-lg"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                    重新选择
                  </button>
                )}
              </div>
            </div>

            {/* 统计摘要 */}
            {analysisMode === 'class' ? (
              // 全班分析指标
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-6 rounded-xl border border-blue-200">
                  <div className="text-3xl font-bold text-blue-700 mb-1">
                    {analysisResult.frame_count}
                  </div>
                  <div className="text-sm font-medium text-gray-700">总帧数</div>
                </div>
                
                <div className="bg-gradient-to-br from-green-50 to-green-100 p-6 rounded-xl border border-green-200">
                  <div className="text-3xl font-bold text-green-700 mb-1">
                    {analysisResult.summary.avg_student_count || Math.round(
                      analysisResult.frame_results.reduce((sum, frame) => sum + frame.student_count, 0) / 
                      analysisResult.frame_count
                    )}
                  </div>
                  <div className="text-sm font-medium text-gray-700">平均学生数</div>
                </div>
                
                <div className="bg-gradient-to-br from-yellow-50 to-yellow-100 p-6 rounded-xl border border-yellow-200">
                  <div className="text-3xl font-bold text-yellow-700 mb-1">
                    {(analysisResult.summary.behavior_percentages.looking_up || 0).toFixed(1)}%
                  </div>
                  <div className="text-sm font-medium text-gray-700">抬头比例</div>
                </div>
                
                <div className="bg-gradient-to-br from-red-50 to-red-100 p-6 rounded-xl border border-red-200">
                  <div className="text-3xl font-bold text-red-700 mb-1">
                    {(analysisResult.summary.behavior_percentages.using_phone || 0).toFixed(1)}%
                  </div>
                  <div className="text-sm font-medium text-gray-700">使用手机比例</div>
                </div>
              </div>
            ) : (
              // 个人分析指标
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-6 rounded-xl border border-purple-200">
                  <div className="text-3xl font-bold text-purple-700 mb-1">
                    {isIndividualResult(analysisResult) && analysisResult.summary.attention_score !== undefined
                      ? analysisResult.summary.attention_score.toFixed(0)
                      : (() => {
                          // 备用计算（如果后端没有返回）
                          const lookingUp = analysisResult.summary.behavior_percentages.looking_up || 0;
                          const writing = analysisResult.summary.behavior_percentages.writing || 0;
                          const usingPhone = analysisResult.summary.behavior_percentages.using_phone || 0;
                          const score = Math.max(0, Math.min(100, 
                            lookingUp * 0.6 + writing * 0.3 - usingPhone * 0.3
                          ));
                          return score.toFixed(0);
                        })()}
                  </div>
                  <div className="text-sm font-medium text-gray-700">认真程度评分</div>
                </div>
                
                <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-6 rounded-xl border border-blue-200">
                  <div className="text-3xl font-bold text-blue-700 mb-1">
                    {(analysisResult.summary.behavior_percentages.looking_up || 0).toFixed(1)}%
                  </div>
                  <div className="text-sm font-medium text-gray-700">抬头听课时长</div>
                </div>
                
                <div className="bg-gradient-to-br from-green-50 to-green-100 p-6 rounded-xl border border-green-200">
                  <div className="text-3xl font-bold text-green-700 mb-1">
                    {(analysisResult.summary.behavior_percentages.writing || 0).toFixed(1)}%
                  </div>
                  <div className="text-sm font-medium text-gray-700">记笔记时长</div>
                </div>
                
                <div className="bg-gradient-to-br from-red-50 to-red-100 p-6 rounded-xl border border-red-200">
                  <div className="text-3xl font-bold text-red-700 mb-1">
                    {(analysisResult.summary.behavior_percentages.using_phone || 0).toFixed(1)}%
                  </div>
                  <div className="text-sm font-medium text-gray-700">玩手机时长</div>
                </div>
              </div>
            )}

            {/* 行为分析详情 */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
              {/* 头部姿态分布 */}
              <div className="bg-gray-50 p-6 rounded-xl border border-gray-200">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">头部姿态分布</h3>
                <div className="space-y-3">
                  {(() => {
                    // 优先使用head_percentages,如果没有则从behavior_percentages中筛选
                    const headData = isIndividualResult(analysisResult) && analysisResult.summary.head_percentages
                      ? analysisResult.summary.head_percentages
                      : Object.fromEntries(
                          Object.entries(analysisResult.summary.behavior_percentages)
                            .filter(([key]) => ['looking_up', 'looking_down', 'neutral'].includes(key))
                        );
                    
                    return Object.entries(headData).map(([behavior, percentage]) => (
                      <div key={behavior}>
                        <div className="flex justify-between text-sm mb-1">
                          <span className="font-medium text-gray-700">{getBehaviorLabel(behavior)}</span>
                          <span className="text-gray-900 font-semibold">{Number(percentage).toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2.5">
                          <div 
                            className="bg-blue-600 h-2.5 rounded-full transition-all"
                            style={{ width: `${percentage}%` }}
                          ></div>
                        </div>
                      </div>
                    ));
                  })()}
                </div>
              </div>

              {/* 手部活动分布 */}
              <div className="bg-gray-50 p-6 rounded-xl border border-gray-200">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">手部活动分布</h3>
                <div className="space-y-3">
                  {(() => {
                    // 优先使用hand_percentages,如果没有则从behavior_percentages中筛选
                    const handData = isIndividualResult(analysisResult) && analysisResult.summary.hand_percentages
                      ? analysisResult.summary.hand_percentages
                      : Object.fromEntries(
                          Object.entries(analysisResult.summary.behavior_percentages)
                            .filter(([key]) => ['writing', 'using_phone', 'resting', 'neutral'].includes(key))
                        );
                    
                    return Object.entries(handData).map(([behavior, percentage]) => (
                      <div key={behavior}>
                        <div className="flex justify-between text-sm mb-1">
                          <span className="font-medium text-gray-700">{getBehaviorLabel(behavior)}</span>
                          <span className="text-gray-900 font-semibold">{Number(percentage).toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2.5">
                          <div 
                            className="bg-green-600 h-2.5 rounded-full transition-all"
                            style={{ width: `${percentage}%` }}
                          ></div>
                        </div>
                      </div>
                    ));
                  })()}
                </div>
              </div>
            </div>

            {/* 分析结论 */}
            {analysisResult.summary.conclusions && analysisResult.summary.conclusions.length > 0 && (
              <div className="bg-blue-50 p-6 rounded-xl border border-blue-200">
                <h3 className="text-lg font-semibold text-gray-900 mb-3">分析结论</h3>
                <ul className="space-y-2">
                  {analysisResult.summary.conclusions.map((conclusion, index) => (
                    <li key={index} className="flex items-start">
                      <svg className="w-5 h-5 text-blue-600 mr-2 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                      <span className="text-gray-800">{conclusion}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default VideoAnalyzer;
