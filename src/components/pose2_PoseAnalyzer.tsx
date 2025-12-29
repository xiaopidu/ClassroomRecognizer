import React, { useState, useRef } from 'react';
import { Upload, Image as ImageIcon, Loader2, CheckCircle, AlertCircle } from 'lucide-react';
import Slider from 'rc-slider';
import 'rc-slider/assets/index.css';

interface PoseTestAnalyzerProps {
  // 预留接口，暂时不需要参数
}

const PoseTestAnalyzer: React.FC<PoseTestAnalyzerProps> = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [analyzedImage, setAnalyzedImage] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isObjectDetecting, setIsObjectDetecting] = useState(false);  // 物体检测状态
  const [isBehaviorAnalyzing, setIsBehaviorAnalyzing] = useState(false);  // 行为检测状态
  const [error, setError] = useState<string | null>(null);
  const [detectionResult, setDetectionResult] = useState<any>(null);
  
  // 检测参数
  const [confThreshold, setConfThreshold] = useState(0.15);  // 降低默认阈值以检测更多目标
  const [drawSkeleton, setDrawSkeleton] = useState(true);
  const [drawBBox, setDrawBBox] = useState(true);
  
  // 抬头/低头判断阈值（使用数组表示范围）
  const [headPoseThresholds, setHeadPoseThresholds] = useState<number[]>([0, 2]);  // [抬头阈值, 低头阈值]
  
  // 物体检测参数
  const [objectConfThreshold, setObjectConfThreshold] = useState(0.25);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  // 处理图片上传
  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // 检查文件类型
    if (!file.type.startsWith('image/')) {
      setError('请上传图片文件');
      return;
    }

    // 读取图片为Base64
    const reader = new FileReader();
    reader.onload = (event) => {
      const imageData = event.target?.result as string;
      setSelectedImage(imageData);
      setAnalyzedImage(null);
      setDetectionResult(null);
      setError(null);
    };
    reader.onerror = () => {
      setError('图片读取失败');
    };
    reader.readAsDataURL(file);
  };

  // 分析图片
  const handleAnalyze = async () => {
    if (!selectedImage) {
      setError('请先上传图片');
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setAnalyzedImage(null);
    setDetectionResult(null);

    try {
      const response = await fetch('http://localhost:5001/api/pose-detect-test', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: selectedImage,
          conf_threshold: confThreshold,
          draw_skeleton: drawSkeleton,
          draw_bbox: drawBBox,
          looking_up_threshold: headPoseThresholds[0],
          looking_down_threshold: headPoseThresholds[1]
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || '分析请求失败');
      }

      const data = await response.json();
      
      if (data.success) {
        setAnalyzedImage(data.annotated_image);
        setDetectionResult(data.detection_result);
      } else {
        throw new Error(data.error || '分析失败');
      }
    } catch (err) {
      console.error('分析错误:', err);
      setError(err instanceof Error ? err.message : '未知错误');
    } finally {
      setIsAnalyzing(false);
    }
  };

  // 物体检测分析
  const handleObjectDetect = async () => {
    if (!selectedImage) {
      setError('请先上传图片');
      return;
    }

    setIsObjectDetecting(true);  // 使用单独的状态
    setError(null);
    setAnalyzedImage(null);
    setDetectionResult(null);

    try {
      const response = await fetch('http://localhost:5001/api/object-detect-test', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: selectedImage,
          conf_threshold: objectConfThreshold  // 使用单独的阈值
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || '物体检测请求失败');
      }

      const data = await response.json();
      
      if (data.success) {
        setAnalyzedImage(data.annotated_image);
        setDetectionResult(data.detection_result);
      } else {
        throw new Error(data.error || '物体检测失败');
      }
    } catch (err) {
      console.error('物体检测错误:', err);
      setError(err instanceof Error ? err.message : '未知错误');
    } finally {
      setIsObjectDetecting(false);
    }
  };

  // 行为检测分析
  const handleBehaviorAnalyze = async () => {
    if (!selectedImage) {
      setError('请先上传图片');
      return;
    }

    setIsBehaviorAnalyzing(true);
    setError(null);
    setAnalyzedImage(null);
    setDetectionResult(null);

    try {
      const response = await fetch('http://localhost:5001/api/behavior-analyze-test', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: selectedImage,
          pose_conf_threshold: confThreshold,
          object_conf_threshold: objectConfThreshold,
          draw_skeleton: drawSkeleton,
          draw_bbox: drawBBox,
          looking_up_threshold: headPoseThresholds[0],
          looking_down_threshold: headPoseThresholds[1]
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || '行为检测请求失败');
      }

      const data = await response.json();
      
      if (data.success) {
        setAnalyzedImage(data.annotated_image);
        setDetectionResult(data.detection_result);
      } else {
        throw new Error(data.error || '行为检测失败');
      }
    } catch (err) {
      console.error('行为检测错误:', err);
      setError(err instanceof Error ? err.message : '未知错误');
    } finally {
      setIsBehaviorAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* 页面标题 */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">行为分析（标定）</h1>
          <p className="text-gray-600">
            基于 YOLO11m-pose 模型检测人体17个关键点 | 使用耳朵-眼睛连线角度法判断抬头/低头 | YOLO11m 物体检测
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* 左侧：上传和参数控制 */}
          <div className="lg:col-span-1 space-y-6">
            {/* 图片上传 */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <Upload className="w-5 h-5" />
                上传图片
              </h2>
              
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                className="block w-full text-sm text-gray-900
                  file:mr-4 file:py-3 file:px-6
                  file:rounded-lg file:border-0
                  file:text-sm file:font-semibold
                  file:bg-blue-600 file:text-white
                  hover:file:bg-blue-700
                  file:cursor-pointer cursor-pointer
                  border border-gray-300 rounded-lg
                  focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              
              {selectedImage && (
                <div className="mt-4">
                  <p className="text-sm text-green-600 flex items-center gap-2">
                    <CheckCircle className="w-4 h-4" />
                    图片已加载
                  </p>
                </div>
              )}
            </div>

            {/* 行为检测区域 */}
            <div className="bg-white rounded-xl shadow-lg p-6 border-2 border-purple-200">
              <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <span className="w-1 h-6 bg-purple-600 rounded"></span>
                行为检测
              </h2>
              
              <div className="space-y-4">
                {/* 行为检测说明 */}
                <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                  <h3 className="text-sm font-semibold text-purple-800 mb-2">检测逻辑：</h3>
                  <div className="text-xs text-gray-700 space-y-1">
                    <p>• <span className="font-semibold">抬头</span>：听讲/看黑板</p>
                    <p>• <span className="font-semibold text-green-700">看电脑</span>：低头 + 检测到笔记本电脑</p>
                    <p>• <span className="font-semibold text-red-700">看手机</span>：严重低头 + 检测到手机</p>
                    <p>• <span className="font-semibold text-yellow-700">看书/记笔记</span>：低头 + 未检测到电子设备</p>
                  </div>
                  <p className="text-xs text-gray-500 mt-3">
                    💡 组合姿态检测和物体检测，自动判断学生行为
                  </p>
                </div>
                
                {/* 行为检测按钮 */}
                <div className="pt-2">
                  <button
                    onClick={handleBehaviorAnalyze}
                    disabled={!selectedImage || isBehaviorAnalyzing}
                    className="w-full bg-gradient-to-r from-purple-600 to-indigo-600 
                      hover:from-purple-700 hover:to-indigo-700 
                      text-white font-semibold py-3 px-6 rounded-lg 
                      disabled:opacity-50 disabled:cursor-not-allowed
                      transition-all duration-200 shadow-lg hover:shadow-xl
                      transform hover:-translate-y-0.5
                      flex items-center justify-center gap-2"
                  >
                    {isBehaviorAnalyzing ? (
                      <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        行为分析中...
                      </>
                    ) : (
                      <>
                        <ImageIcon className="w-5 h-5" />
                        行为检测
                      </>
                    )}
                  </button>
                </div>
                
                <p className="text-xs text-gray-500">
                  ⚠️ 使用下方姿态检测和物体检测的参数设置
                </p>
              </div>
            </div>

            {/* 姿态检测区域 */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <span className="w-1 h-6 bg-blue-600 rounded"></span>
                姿态检测参数
              </h2>
              
              <div className="space-y-4">
                {/* 置信度阈值 */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    置信度阈值: <span className="text-blue-600 font-semibold">{confThreshold.toFixed(2)}</span>
                  </label>
                  <input
                    type="range"
                    min="0.1"
                    max="0.9"
                    step="0.05"
                    value={confThreshold}
                    onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    调高可减少误检，调低可检测更多关键点
                  </p>
                </div>

                {/* 绘制选项 */}
                <div className="space-y-2">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={drawSkeleton}
                      onChange={(e) => setDrawSkeleton(e.target.checked)}
                      className="w-4 h-4 text-blue-600 rounded focus:ring-2 focus:ring-blue-500"
                    />
                    <span className="text-sm font-medium text-gray-700">绘制骨架连线</span>
                  </label>
                  
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={drawBBox}
                      onChange={(e) => setDrawBBox(e.target.checked)}
                      className="w-4 h-4 text-blue-600 rounded focus:ring-2 focus:ring-blue-500"
                    />
                    <span className="text-sm font-medium text-gray-700">绘制边界框</span>
                  </label>
                </div>
                
                {/* 抬头/低头阈值 */}
                <div className="pt-4 border-t border-gray-200">
                  <h3 className="text-sm font-semibold text-gray-700 mb-3">抬头/低头阈值</h3>
                  
                  <div className="space-y-4">
                    <div className="px-2">
                      <div className="flex justify-between items-center mb-3">
                        <span className="text-xs font-medium text-green-600">
                          抬头: {headPoseThresholds[0]}
                        </span>
                        <span className="text-xs text-gray-500">
                          范围: {headPoseThresholds[0]} ~ {headPoseThresholds[1]}
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
                        trackStyle={[{ backgroundColor: '#3b82f6', height: 6 }]}
                        handleStyle={[
                          { 
                            borderColor: '#22c55e',  // 绿色 - 抬头
                            backgroundColor: '#22c55e',
                            width: 18,
                            height: 18,
                            marginTop: -6
                          },
                          { 
                            borderColor: '#ef4444',  // 红色 - 低头
                            backgroundColor: '#ef4444',
                            width: 18,
                            height: 18,
                            marginTop: -6
                          }
                        ]}
                        railStyle={{ backgroundColor: '#e5e7eb', height: 6 }}
                      />
                      
                      <div className="flex justify-between text-xs text-gray-400 mt-2">
                        <span>-20</span>
                        <span>0</span>
                        <span>20</span>
                      </div>
                      
                      <p className="text-xs text-gray-500 mt-3">
                        <span className="inline-flex items-center gap-1">
                          <span className="w-3 h-3 rounded-full bg-green-500"></span>
                          耳朵在连线下方多少像素算抬头
                        </span>
                        <br/>
                        <span className="inline-flex items-center gap-1">
                          <span className="w-3 h-3 rounded-full bg-red-500"></span>
                          耳朵在连线上方多少像素算低头
                        </span>
                      </p>
                    </div>
                  </div>
                </div>
                
                {/* 关键点说明 */}
                <div className="pt-4 border-t border-gray-200">
                  <h3 className="text-sm font-semibold text-gray-700 mb-3">关键点说明</h3>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="flex items-center gap-1">
                      <div className="w-2 h-2 rounded-full bg-green-500"></div>
                      <span className="text-gray-700">鼻子</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <div className="w-2 h-2 rounded-full bg-blue-500"></div>
                      <span className="text-gray-700">眼睛</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <div className="w-2 h-2 rounded-full bg-yellow-400"></div>
                      <span className="text-gray-700">耳朵</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <div className="w-2 h-2 rounded-full bg-purple-500"></div>
                      <span className="text-gray-700">肩膠</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <div className="w-2 h-2 rounded-full bg-orange-500"></div>
                      <span className="text-gray-700">手腕</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <div className="w-2 h-2 rounded-full" style={{backgroundColor: '#0080ff'}}></div>
                      <span className="text-gray-700">肘部</span>
                    </div>
                  </div>
                  <p className="text-xs text-gray-500 mt-2">
                    共17个关键点，白色线条连接形成人体骨架
                  </p>
                </div>
                
                {/* 姿态检测按钮 */}
                <div className="pt-4">
                  <button
                    onClick={handleAnalyze}
                    disabled={!selectedImage || isAnalyzing}
                    className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 
                      hover:from-blue-700 hover:to-indigo-700 
                      text-white font-semibold py-3 px-6 rounded-lg 
                      disabled:opacity-50 disabled:cursor-not-allowed
                      transition-all duration-200 shadow-lg hover:shadow-xl
                      transform hover:-translate-y-0.5
                      flex items-center justify-center gap-2"
                  >
                    {isAnalyzing ? (
                      <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        姿态分析中...
                      </>
                    ) : (
                      <>
                        <ImageIcon className="w-5 h-5" />
                        姿态检测
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>

            {/* 物体检测区域 */}
            <div className="bg-white rounded-xl shadow-lg p-6 border-2 border-green-200">
              <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <span className="w-1 h-6 bg-green-600 rounded"></span>
                物体检测参数
              </h2>
              
              <div className="space-y-4">
                {/* 物体检测置信度 */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    检测置信度: <span className="text-green-600 font-semibold">{objectConfThreshold.toFixed(2)}</span>
                  </label>
                  <input
                    type="range"
                    min="0.1"
                    max="0.8"
                    step="0.05"
                    value={objectConfThreshold}
                    onChange={(e) => setObjectConfThreshold(parseFloat(e.target.value))}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-green-600"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>0.1 (灵敏)</span>
                    <span>推荐: 0.20-0.30</span>
                    <span>0.8 (严格)</span>
                  </div>
                  <p className="text-xs text-gray-500 mt-2">
                    💡 已优化：使用1280分辨率增强小物体检测<br/>
                    ⚠️ 过低的置信度（&lt;0.2）会增加误检
                  </p>
                </div>
                
                {/* 物体检测说明 */}
                <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                  <h3 className="text-sm font-semibold text-green-800 mb-2">可检测物体：</h3>
                  <div className="grid grid-cols-1 gap-2 text-xs">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded bg-green-500"></div>
                      <span className="text-gray-700">笔记本电脑</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded bg-red-500"></div>
                      <span className="text-gray-700">手机</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded bg-yellow-400"></div>
                      <span className="text-gray-700">书</span>
                    </div>
                  </div>
                </div>
                
                {/* 物体检测按钮 */}
                <div className="pt-2">
                  <button
                    onClick={handleObjectDetect}
                    disabled={!selectedImage || isObjectDetecting}
                    className="w-full bg-gradient-to-r from-green-600 to-emerald-600 
                      hover:from-green-700 hover:to-emerald-700 
                      text-white font-semibold py-3 px-6 rounded-lg 
                      disabled:opacity-50 disabled:cursor-not-allowed
                      transition-all duration-200 shadow-lg hover:shadow-xl
                      transform hover:-translate-y-0.5
                      flex items-center justify-center gap-2"
                  >
                    {isObjectDetecting ? (
                      <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        物体检测中...
                      </>
                    ) : (
                      <>
                        <ImageIcon className="w-5 h-5" />
                        物体检测
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* 右侧：图片显示区域 */}
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

            {/* 图片显示 */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                {analyzedImage ? '检测结果' : '原始图片'}
              </h2>
              
              <div className="relative bg-gray-100 rounded-lg overflow-hidden min-h-[400px] flex items-center justify-center">
                {analyzedImage ? (
                  <img
                    src={analyzedImage}
                    alt="检测结果"
                    className="max-w-full h-auto"
                  />
                ) : selectedImage ? (
                  <img
                    src={selectedImage}
                    alt="原始图片"
                    className="max-w-full h-auto"
                  />
                ) : (
                  <div className="text-center py-20">
                    <ImageIcon className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-500">请上传图片开始检测</p>
                  </div>
                )}
              </div>
            </div>
    
            {/* 检测统计 */}
            {detectionResult && (
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h2 className="text-xl font-semibold text-gray-900 mb-4">检测统计</h2>
                    
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-700">检测人数</span>
                    <span className="text-2xl font-bold text-blue-600">
                      {detectionResult.person_count}
                    </span>
                  </div>
                      
                  <div className="flex justify-between items-center">
                    <span className="text-gray-700">处理时间</span>
                    <span className="text-lg font-semibold text-gray-900">
                      {(detectionResult.processing_time * 1000).toFixed(0)}ms
                    </span>
                  </div>
                </div>
    
                {/* 行为统计 */}
                {detectionResult.behavior_stats && (
                  <div className="mt-4 pt-4 border-t border-gray-200">
                    <h3 className="text-sm font-semibold text-gray-700 mb-3">行为分布：</h3>
                    <div className="space-y-2">
                      {detectionResult.behavior_stats.listening > 0 && (
                        <div className="flex justify-between items-center text-sm">
                          <span className="text-gray-700">👂 听讲</span>
                          <span className="font-semibold text-green-600">{detectionResult.behavior_stats.listening}人</span>
                        </div>
                      )}
                      {detectionResult.behavior_stats.using_computer > 0 && (
                        <div className="flex justify-between items-center text-sm">
                          <span className="text-gray-700">💻 看电脑</span>
                          <span className="font-semibold text-green-700">{detectionResult.behavior_stats.using_computer}人</span>
                        </div>
                      )}
                      {detectionResult.behavior_stats.using_phone > 0 && (
                        <div className="flex justify-between items-center text-sm">
                          <span className="text-gray-700">📱 看手机</span>
                          <span className="font-semibold text-red-600">{detectionResult.behavior_stats.using_phone}人</span>
                        </div>
                      )}
                      {detectionResult.behavior_stats.reading_writing > 0 && (
                        <div className="flex justify-between items-center text-sm">
                          <span className="text-gray-700">📖 看书/记笔记</span>
                          <span className="font-semibold text-yellow-600">{detectionResult.behavior_stats.reading_writing}人</span>
                        </div>
                      )}
                      {detectionResult.behavior_stats.neutral > 0 && (
                        <div className="flex justify-between items-center text-sm">
                          <span className="text-gray-700">❓ 中性</span>
                          <span className="font-semibold text-gray-500">{detectionResult.behavior_stats.neutral}人</span>
                        </div>
                      )}
                    </div>
                  </div>
                )}
    
                {/* 详细关键点信息 */}
                {detectionResult.persons && detectionResult.persons.length > 0 && (
                  <div className="mt-4 pt-4 border-t border-gray-200">
                    <h3 className="text-sm font-semibold text-gray-700 mb-2">关键点可见性</h3>
                    <div className="space-y-2">
                      {detectionResult.persons.map((person: any, idx: number) => (
                        <div key={idx} className="text-xs">
                          <p className="font-medium text-gray-800">
                            Person {person.person_id + 1}: {person.keypoints.filter((kp: any) => kp.visible).length}/17 个点可见
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default PoseTestAnalyzer;
