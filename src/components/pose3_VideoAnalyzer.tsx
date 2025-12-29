import React, { useState, useRef, useEffect } from 'react';
import { 
  Upload, Play, Pause, AlertCircle, CheckCircle, 
  Loader2, Video as VideoIcon, Users, User, Clock 
} from 'lucide-react';
import Slider from 'rc-slider';
import 'rc-slider/assets/index.css';

const Pose3VideoAnalyzer: React.FC = () => {
  // 视频相关状态
  const [selectedVideo, setSelectedVideo] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoDuration, setVideoDuration] = useState<number>(0);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  
  // 分析参数（复用姿态检测参数）
  const [confThreshold, setConfThreshold] = useState<number>(0.15);
  const [objectConfThreshold, setObjectConfThreshold] = useState<number>(0.25);
  const [headPoseThresholds, setHeadPoseThresholds] = useState<number[]>([0, 2]);  // [抬头阈值, 低头阈值]
  
  // 分析模式：class 全班5分钟 | individual 个人45分钟
  const [analysisMode, setAnalysisMode] = useState<'class' | 'individual'>('class');
  
  // 个人分析：选择学生
  const [isSelectingStudent, setIsSelectingStudent] = useState<boolean>(false);
  const [selectedStudentBbox, setSelectedStudentBbox] = useState<any>(null);
  const [firstFrameImage, setFirstFrameImage] = useState<string | null>(null);
  
  // 框选状态
  const [isDrawing, setIsDrawing] = useState<boolean>(false);
  const [bboxStart, setBboxStart] = useState<{x: number, y: number} | null>(null);
  const [bboxEnd, setBboxEnd] = useState<{x: number, y: number} | null>(null);
  
  // 分析状态
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [analysisProgress, setAnalysisProgress] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);
  
  // 进度轮询
  const progressIntervalRef = useRef<NodeJS.Timeout | null>(null);
  
  // 分析结果
  const [analysisResult, setAnalysisResult] = useState<any>(null);
  const [outputVideoUrl, setOutputVideoUrl] = useState<string | null>(null);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);  // 用于截取首帧
  const drawingCanvasRef = useRef<HTMLCanvasElement>(null);  // 用于弹窗中的绘制
  
  // 处理视频上传
  const handleVideoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith('video/')) {
      setSelectedVideo(file);
      const url = URL.createObjectURL(file);
      setVideoUrl(url);
      setError(null);
      setAnalysisResult(null);
      setOutputVideoUrl(null);
      setSelectedStudentBbox(null);
      setFirstFrameImage(null);
    } else {
      setError('请上传有效的视频文件');
    }
  };
  
  // 视频加载完成
  const handleVideoLoaded = () => {
    if (videoRef.current) {
      setVideoDuration(videoRef.current.duration);
      setCurrentTime(0);
    }
  };
  
  // 时间更新
  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };
  
  // 播放/暂停
  const togglePlay = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };
  
  // 跳转到指定时间
  const seekToTime = (time: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = time;
      setCurrentTime(time);
    }
  };
  
  // 格式化时间
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };
  
  // 获取首帧用于选择学生
  const captureFirstFrame = async () => {
    console.log('[captureFirstFrame] 开始执行');
    if (!videoRef.current || !canvasRef.current) {
      console.log('[captureFirstFrame] videoRef 或 canvasRef 不存在');
      return;
    }
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    if (!ctx) {
      console.log('[captureFirstFrame] 无法获取 canvas context');
      return;
    }
    
    // 如果视频已经在目标时间点，直接绘制；否则需要等待跳转
    const needSeek = Math.abs(video.currentTime - currentTime) > 0.1;
    
    if (needSeek) {
      // 跳转到起始时间
      video.currentTime = currentTime;
      
      // 等待视频跳转完成
      await new Promise<void>(resolve => {
        const onSeeked = () => {
          video.removeEventListener('seeked', onSeeked);
          resolve();
        };
        video.addEventListener('seeked', onSeeked);
        
        // 添加超时保护，避免永久等待
        setTimeout(() => {
          video.removeEventListener('seeked', onSeeked);
          resolve();
        }, 1000);
      });
    }
    
    // 绘制到canvas
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    
    // 转换为base64
    const imageData = canvas.toDataURL('image/jpeg');
    console.log('[captureFirstFrame] 截取首帧成功，图片大小:', imageData.length);
    setFirstFrameImage(imageData);
    setIsSelectingStudent(true);
    console.log('[captureFirstFrame] 状态已设置: isSelectingStudent=true');
  };
  
  // 画布上点击选择学生
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isSelectingStudent || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    
    // 假设学生边界框大小约为 150x200
    const bbox = {
      x: Math.max(0, x - 75),
      y: Math.max(0, y - 100),
      w: 150,
      h: 200
    };
    
    setSelectedStudentBbox(bbox);
    setIsSelectingStudent(false);
  };
  
  // 开始分析
  const startAnalysis = async () => {
    if (!selectedVideo) {
      setError('请先上传视频');
      return;
    }
    
    if (analysisMode === 'individual' && !selectedStudentBbox) {
      setError('请先选择目标学生');
      return;
    }
    
    setIsAnalyzing(true);
    setError(null);
    setAnalysisProgress(0);
    setAnalysisResult(null);
    setOutputVideoUrl(null);
    
    // 开始轮询进度
    // 全班分析和个人追踪都需要进度条
    progressIntervalRef.current = setInterval(async () => {
      try {
        const response = await fetch(`http://localhost:5001/api/video-analysis-progress?mode=${analysisMode}`);
        const data = await response.json();
        if (data.current !== undefined) {
          setAnalysisProgress(data.current);
        }
      } catch (err) {
        console.error('进度查询失败:', err);
      }
    }, 1000); // 每秒查询一次
    
    try {
      const formData = new FormData();
      formData.append('video', selectedVideo);
      formData.append('start_time', currentTime.toString());
      formData.append('duration', (analysisMode === 'class' ? 300 : 2700).toString()); // 5分钟或45分钟
      formData.append('pose_conf_threshold', confThreshold.toString());
      formData.append('object_conf_threshold', objectConfThreshold.toString());
      formData.append('looking_up_threshold', headPoseThresholds[0].toString());  // 抬头阈值
      formData.append('looking_down_threshold', headPoseThresholds[1].toString());  // 低头阈值
      formData.append('mode', analysisMode);
      
      if (analysisMode === 'individual' && selectedStudentBbox) {
        formData.append('target_bbox', JSON.stringify(selectedStudentBbox));
      }
      
      const response = await fetch('http://localhost:5001/api/video-behavior-analyze', {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      
      if (data.success) {
        setAnalysisResult(data.result);
        if (data.video_url) {
          setOutputVideoUrl(`http://localhost:5001${data.video_url}`);
        }
        setAnalysisProgress(100);
      } else {
        setError(data.error || '分析失败');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '未知错误');
    } finally {
      setIsAnalyzing(false);
      setAnalysisProgress(0);
      
      // 停止轮询
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
        progressIntervalRef.current = null;
      }
    }
  };
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-blue-50 to-indigo-50 py-8 px-4">
      <div className="max-w-7xl mx-auto">
        {/* 页面标题 */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">行为分析（视频）</h1>
          <p className="text-gray-600">
            上传视频，选择时间点，对全班或特定学生进行行为分析
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* 左侧：参数设置 */}
          <div className="lg:col-span-1 space-y-6">
            {/* 视频上传 */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <span className="w-1 h-6 bg-purple-600 rounded"></span>
                视频上传
              </h2>
              
              <div className="space-y-4">
                <label className="block">
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-purple-500 transition-colors cursor-pointer">
                    <Upload className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                    <p className="text-sm text-gray-600 mb-2">
                      点击上传视频或拖拽视频文件到此处
                    </p>
                    <p className="text-xs text-gray-500">
                      支持 MP4, AVI, MOV 等格式
                    </p>
                    <input
                      type="file"
                      accept="video/*"
                      onChange={handleVideoUpload}
                      className="hidden"
                    />
                  </div>
                </label>
                
                {selectedVideo && (
                  <div className="mt-4">
                    <p className="text-sm text-green-600 flex items-center gap-2">
                      <CheckCircle className="w-4 h-4" />
                      已加载: {selectedVideo.name}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      时长: {formatTime(videoDuration)}
                    </p>
                  </div>
                )}
              </div>
            </div>

            {/* 分析模式选择 */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <span className="w-1 h-6 bg-blue-600 rounded"></span>
                分析模式
              </h2>
              
              <div className="space-y-3">
                <label className="flex items-center gap-3 p-3 border-2 rounded-lg cursor-pointer hover:bg-blue-50 transition-colors"
                  style={{ borderColor: analysisMode === 'class' ? '#3b82f6' : '#e5e7eb' }}>
                  <input
                    type="radio"
                    name="mode"
                    checked={analysisMode === 'class'}
                    onChange={() => {
                      setAnalysisMode('class');
                      setAnalysisResult(null); // 切换模式时清除之前的结果
                      setOutputVideoUrl(null);
                    }}
                    className="w-4 h-4 text-blue-600"
                  />
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <Users className="w-5 h-5 text-blue-600" />
                      <span className="font-semibold text-gray-900">全班分析（5分钟）</span>
                    </div>
                    <p className="text-xs text-gray-500 mt-1">
                      从选定时间点开始，分析5分钟内所有学生的行为
                    </p>
                  </div>
                </label>
                
                <label className="flex items-center gap-3 p-3 border-2 rounded-lg cursor-pointer hover:bg-green-50 transition-colors"
                  style={{ borderColor: analysisMode === 'individual' ? '#10b981' : '#e5e7eb' }}>
                  <input
                    type="radio"
                    name="mode"
                    checked={analysisMode === 'individual'}
                    onChange={() => {
                      setAnalysisMode('individual');
                      setAnalysisResult(null); // 切换模式时清除之前的结果
                      setOutputVideoUrl(null);
                      setSelectedStudentBbox(null); // 同时清除之前选择的学生框
                      setFirstFrameImage(null);
                    }}
                    className="w-4 h-4 text-green-600"
                  />
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <User className="w-5 h-5 text-green-600" />
                      <span className="font-semibold text-gray-900">个人追踪（45分钟）</span>
                    </div>
                    <p className="text-xs text-gray-500 mt-1">
                      追踪特定学生45分钟，统计行为时长
                    </p>
                  </div>
                </label>
              </div>
            </div>

            {/* 检测参数 */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <span className="w-1 h-6 bg-indigo-600 rounded"></span>
                检测参数
              </h2>
              
              <div className="space-y-4">
                {/* 姿态检测置信度 */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    姿态置信度: <span className="text-indigo-600 font-semibold">{confThreshold.toFixed(2)}</span>
                  </label>
                  <input
                    type="range"
                    min="0.1"
                    max="0.9"
                    step="0.05"
                    value={confThreshold}
                    onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                  />
                </div>

                {/* 物体检测置信度 */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    物体置信度: <span className="text-indigo-600 font-semibold">{objectConfThreshold.toFixed(2)}</span>
                  </label>
                  <input
                    type="range"
                    min="0.1"
                    max="0.8"
                    step="0.05"
                    value={objectConfThreshold}
                    onChange={(e) => setObjectConfThreshold(parseFloat(e.target.value))}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                  />
                </div>

                {/* 抬头/低头阈值 */}
                <div className="pt-4 border-t border-gray-200">
                  <h3 className="text-sm font-semibold text-gray-700 mb-3">抬头/低头阈值</h3>
                  <div className="px-2">
                    <div className="flex justify-between items-center mb-3">
                      <span className="text-xs font-medium text-green-600">
                        抬头: {headPoseThresholds[0]}
                      </span>
                      <span className="text-xs font-medium text-red-600">
                        低头: {headPoseThresholds[1]}
                      </span>
                    </div>
                    
                    <Slider
                      range
                      min={-20}
                      max={20}
                      value={headPoseThresholds}
                      onChange={(value) => setHeadPoseThresholds(value as number[])}
                      trackStyle={[{ backgroundColor: '#6366f1', height: 6 }]}
                      handleStyle={[
                        { borderColor: '#22c55e', backgroundColor: '#22c55e', width: 18, height: 18, marginTop: -6 },  // 绿色 - 抬头
                        { borderColor: '#ef4444', backgroundColor: '#ef4444', width: 18, height: 18, marginTop: -6 }   // 红色 - 低头
                      ]}
                      railStyle={{ backgroundColor: '#e5e7eb', height: 6 }}
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* 右侧：视频显示和控制 */}
          <div className="lg:col-span-2 space-y-6">
            {/* 错误提示 */}
            {error && (
              <div className="bg-red-50 border-l-4 border-red-500 text-red-900 px-6 py-4 rounded-lg">
                <div className="flex items-center">
                  <AlertCircle className="w-5 h-5 mr-2" />
                  <span className="font-medium">{error}</span>
                </div>
              </div>
            )}

            {/* 视频播放器 */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">视频预览</h2>
              
              <div className="relative bg-gray-900 rounded-lg overflow-hidden">
                {videoUrl ? (
                  <>
                    <video
                      ref={videoRef}
                      src={videoUrl}
                      onLoadedMetadata={handleVideoLoaded}
                      onTimeUpdate={handleTimeUpdate}
                      className="w-full"
                      style={{ maxHeight: '500px' }}
                    />
                    
                    {/* 隐藏的canvas用于截取首帧 */}
                    <canvas ref={canvasRef} style={{ display: 'none' }} />
                    
                    {/* 视频控制 */}
                    <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-4">
                      <div className="flex items-center gap-4">
                        <button
                          onClick={togglePlay}
                          className="text-white hover:text-blue-400 transition-colors"
                        >
                          {isPlaying ? <Pause className="w-6 h-6" /> : <Play className="w-6 h-6" />}
                        </button>
                        
                        <div className="flex-1">
                          <input
                            type="range"
                            min="0"
                            max={videoDuration}
                            step="0.1"
                            value={currentTime}
                            onChange={(e) => seekToTime(parseFloat(e.target.value))}
                            className="w-full h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer accent-blue-500"
                          />
                        </div>
                        
                        <span className="text-white text-sm font-mono">
                          {formatTime(currentTime)} / {formatTime(videoDuration)}
                        </span>
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="flex items-center justify-center h-96">
                    <div className="text-center text-gray-400">
                      <VideoIcon className="w-16 h-16 mx-auto mb-4" />
                      <p>请上传视频文件</p>
                    </div>
                  </div>
                )}
              </div>
              
              {/* 时间点选择和学生选择 */}
              {videoUrl && (
                <div className="mt-4 space-y-3">
                  <div className="flex items-center gap-3">
                    <Clock className="w-5 h-5 text-gray-600" />
                    <span className="text-sm font-medium text-gray-700">
                      当前选择的起始时间: <span className="text-blue-600">{formatTime(currentTime)}</span>
                    </span>
                  </div>
                  
                  {analysisMode === 'individual' && (
                    <div>
                      <button
                        onClick={() => {
                          console.log('[按钮点击] 选择目标学生');
                          console.log('[按钮点击] 当前状态 - isSelectingStudent:', isSelectingStudent, 'firstFrameImage存在:', !!firstFrameImage);
                          // 直接调用captureFirstFrame，不需要提前重置状态
                          // captureFirstFrame内部会设置isSelectingStudent和firstFrameImage
                          captureFirstFrame();
                        }}
                        disabled={isSelectingStudent}
                        className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
                      >
                        {isSelectingStudent ? '请在画面上点击选择学生...' : '选择目标学生'}
                      </button>
                      
                      {selectedStudentBbox && (
                        <div className="bg-green-50 border border-green-200 rounded-lg p-3 mt-2">
                          <p className="text-sm text-green-800 font-medium flex items-center gap-2 mb-1">
                            <CheckCircle className="w-4 h-4" />
                            已选择目标学生
                          </p>
                          <p className="text-xs text-green-700 font-mono">
                            位置: ({Math.round(selectedStudentBbox.x)}, {Math.round(selectedStudentBbox.y)}) | 
                            大小: {Math.round(selectedStudentBbox.w)} × {Math.round(selectedStudentBbox.h)}
                          </p>
                        </div>
                      )}
                    </div>
                  )}
                  
                  {/* 开始分析按钮 */}
                  <button
                    onClick={startAnalysis}
                    disabled={isAnalyzing || !selectedVideo || (analysisMode === 'individual' && !selectedStudentBbox)}
                    className="w-full bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 
                      text-white font-semibold py-3 px-6 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed
                      transition-all duration-200 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5
                      flex items-center justify-center gap-2"
                  >
                    {isAnalyzing ? (
                      <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        分析中... {analysisProgress > 0 && `${analysisProgress}%`}
                      </>
                    ) : (
                      <>
                        <Play className="w-5 h-5" />
                        开始{analysisMode === 'class' ? '全班' : '个人'}分析
                      </>
                    )}
                  </button>
                  
                  {/* 进度条 */}
                  {isAnalyzing && (
                    <div className="mt-4">
                      <div className="flex justify-between text-sm text-gray-600 mb-2">
                        <span>{analysisMode === 'class' ? '全班分析' : '个人追踪'}进度</span>
                        <span className="font-semibold text-purple-600">{analysisProgress}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                        <div 
                          className="bg-gradient-to-r from-purple-600 to-indigo-600 h-full rounded-full transition-all duration-300 ease-out"
                          style={{ width: `${analysisProgress}%` }}
                        ></div>
                      </div>
                      <p className="text-xs text-gray-500 mt-2">
                        ⚡ 正在处理视频帧，请耐心等待...
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* 分析结果 */}
            {analysisResult && (
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h2 className="text-xl font-semibold text-gray-900 mb-4">分析结果</h2>
                
                {analysisMode === 'class' ? (
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-blue-50 rounded-lg p-4">
                        <p className="text-sm text-gray-600">总帧数</p>
                        <p className="text-2xl font-bold text-blue-600">{analysisResult.total_frames}</p>
                      </div>
                      <div className="bg-green-50 rounded-lg p-4">
                        <p className="text-sm text-gray-600">分析时长</p>
                        <p className="text-2xl font-bold text-green-600">{formatTime(analysisResult.duration_seconds)}</p>
                      </div>
                    </div>
                    
                    {outputVideoUrl && (
                      <div className="mt-4">
                        <a
                          href={outputVideoUrl}
                          download="behavior_analysis_result.mp4"
                          className="inline-flex items-center gap-2 px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-semibold transition-colors"
                        >
                          <VideoIcon className="w-5 h-5" />
                          下载标注视频
                        </a>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="space-y-4">
                    <h3 className="text-lg font-semibold text-gray-800">行为时长统计</h3>
                    

                    
                    <div className="space-y-3">
                      <div className="flex justify-between items-center p-3 bg-green-50 rounded-lg">
                        <span className="text-gray-700 font-medium">👂 听讲</span>
                        <span className="text-green-600 font-bold">
                          {(analysisResult.behavior_minutes?.listening_minutes || 0).toFixed(2)} 分钟
                        </span>
                      </div>
                      
                      <div className="flex justify-between items-center p-3 bg-yellow-50 rounded-lg">
                        <span className="text-gray-700 font-medium">💻 看电脑</span>
                        <span className="text-yellow-600 font-bold">
                          {(analysisResult.behavior_minutes?.using_computer_minutes || 0).toFixed(2)} 分钟
                        </span>
                      </div>
                      
                      <div className="flex justify-between items-center p-3 bg-red-50 rounded-lg">
                        <span className="text-gray-700 font-medium">📱 看手机</span>
                        <span className="text-red-600 font-bold">
                          {(analysisResult.behavior_minutes?.using_phone_minutes || 0).toFixed(2)} 分钟
                        </span>
                      </div>
                      
                      <div className="flex justify-between items-center p-3 bg-blue-50 rounded-lg">
                        <span className="text-gray-700 font-medium">📖 看书/记笔记</span>
                        <span className="text-blue-600 font-bold">
                          {(analysisResult.behavior_minutes?.reading_writing_minutes || 0).toFixed(2)} 分钟
                        </span>
                      </div>
                      
                      <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                        <span className="text-gray-700 font-medium">🧘 中性/其他</span>
                        <span className="text-gray-600 font-bold">
                          {(analysisResult.behavior_minutes?.neutral_minutes || 0).toFixed(2)} 分钟
                        </span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* 首帧选择学生画布 */}
            {isSelectingStudent && firstFrameImage && (
              <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4" onClick={(e) => {
                // 点击背景时关闭弹窗
                if (e.target === e.currentTarget) {
                  setIsSelectingStudent(false);
                  setBboxStart(null);
                  setBboxEnd(null);
                  setFirstFrameImage(null); // 重置首帧图像，避免下次无法弹出
                }
              }}>
                <div className="bg-white rounded-xl p-6 max-w-4xl w-full">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">请拖动框选目标学生</h3>
                  <p className="text-sm text-gray-600 mb-4">
                    在下方画面上<strong className="text-blue-600">按住鼠标左键拖动</strong>框选目标学生，松开后点击<strong className="text-green-600">"确认选框"</strong>。
                  </p>
                  
                  <div className="relative inline-block border-2 border-blue-400 rounded-lg overflow-hidden">
                    {/* 背景图片 */}
                    <img
                      src={firstFrameImage}
                      alt="视频首帧"
                      draggable={false}
                      onDragStart={(e) => e.preventDefault()}
                      className="max-w-full"
                      style={{ display: 'block', maxHeight: '70vh', userSelect: 'none' }}
                      onLoad={(e) => {
                        const img = e.currentTarget;
                        if (drawingCanvasRef.current) {
                          drawingCanvasRef.current.width = img.naturalWidth;
                          drawingCanvasRef.current.height = img.naturalHeight;
                        }
                      }}
                    />
                    {/* Canvas 覆盖层 */}
                    <canvas
                      ref={drawingCanvasRef}
                      onMouseDown={(e) => {
                        const canvas = drawingCanvasRef.current;
                        if (!canvas) return;
                        const rect = canvas.getBoundingClientRect();
                        const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
                        const y = ((e.clientY - rect.top) / rect.height) * canvas.height;
                        setIsDrawing(true);
                        setBboxStart({x, y});
                        setBboxEnd(null);
                      }}
                      onMouseMove={(e) => {
                        if (!isDrawing || !bboxStart || !drawingCanvasRef.current) return;
                        const canvas = drawingCanvasRef.current;
                        const rect = canvas.getBoundingClientRect();
                        const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
                        const y = ((e.clientY - rect.top) / rect.height) * canvas.height;
                        setBboxEnd({x, y});
                        
                        // 绘制框
                        const ctx = canvas.getContext('2d');
                        if (ctx) {
                          ctx.clearRect(0, 0, canvas.width, canvas.height);
                          ctx.strokeStyle = '#3b82f6';
                          ctx.lineWidth = 3;
                          ctx.strokeRect(bboxStart.x, bboxStart.y, x - bboxStart.x, y - bboxStart.y);
                          ctx.fillStyle = 'rgba(59, 130, 246, 0.1)';
                          ctx.fillRect(bboxStart.x, bboxStart.y, x - bboxStart.x, y - bboxStart.y);
                        }
                      }}
                      onMouseUp={(e) => {
                        if (!isDrawing || !bboxStart || !drawingCanvasRef.current) return;
                        setIsDrawing(false);
                        const canvas = drawingCanvasRef.current;
                        const rect = canvas.getBoundingClientRect();
                        const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
                        const y = ((e.clientY - rect.top) / rect.height) * canvas.height;
                        const endPoint = {x, y};
                        setBboxEnd(endPoint);
                        
                        // 绘制最终框
                        const ctx = canvas.getContext('2d');
                        if (ctx) {
                          ctx.clearRect(0, 0, canvas.width, canvas.height);
                          ctx.strokeStyle = '#10b981';
                          ctx.lineWidth = 4;
                          ctx.strokeRect(bboxStart.x, bboxStart.y, x - bboxStart.x, y - bboxStart.y);
                          ctx.fillStyle = 'rgba(16, 185, 129, 0.15)';
                          ctx.fillRect(bboxStart.x, bboxStart.y, x - bboxStart.x, y - bboxStart.y);
                        }
                      }}
                      className="absolute top-0 left-0 w-full h-full cursor-crosshair"
                      style={{ pointerEvents: 'auto' }}
                    />
                  </div>
                  
                  {/* 按钮 */}
                  <div className="mt-4 flex gap-3">
                    {bboxEnd && (
                      <button
                        onClick={() => {
                          if (bboxStart && bboxEnd) {
                            const bbox = {
                              x: Math.min(bboxStart.x, bboxEnd.x),
                              y: Math.min(bboxStart.y, bboxEnd.y),
                              w: Math.abs(bboxEnd.x - bboxStart.x),
                              h: Math.abs(bboxEnd.y - bboxStart.y)
                            };
                            setSelectedStudentBbox(bbox);
                            setIsSelectingStudent(false);
                            setBboxStart(null);
                            setBboxEnd(null);
                          }
                        }}
                        className="flex-1 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 
                          text-white font-semibold py-3 px-6 rounded-lg transition-all shadow-lg hover:shadow-xl
                          transform hover:-translate-y-0.5 flex items-center justify-center gap-2"
                      >
                        <CheckCircle className="w-5 h-5" />
                        确认选框
                      </button>
                    )}
                    <button
                      onClick={() => {
                        console.log('[取消按钮] 点击取消');
                        setIsSelectingStudent(false);
                        setBboxStart(null);
                        setBboxEnd(null);
                        setFirstFrameImage(null); // 重置首帧图像，避免下次无法弹出
                        console.log('[取消按钮] 已重置所有状态');
                        if (drawingCanvasRef.current) {
                          const ctx = drawingCanvasRef.current.getContext('2d');
                          if (ctx) ctx.clearRect(0, 0, drawingCanvasRef.current.width, drawingCanvasRef.current.height);
                        }
                      }}
                      className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg"
                    >
                      取消
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Pose3VideoAnalyzer;
