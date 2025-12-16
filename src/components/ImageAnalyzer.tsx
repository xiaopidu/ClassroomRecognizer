import React, { useState, useRef, useEffect } from 'react';
import { RecognitionParams, Student } from '../types';
import { detectFacesFromFile, detectFacesFromBase64 } from '../services/apiService';
import { Download, Upload, Play, SquareDashedMousePointer, RotateCcw, Video, Pause, SkipBack, SkipForward } from 'lucide-react';
import * as XLSX from 'xlsx';

interface ImageAnalyzerProps {
  students: Student[];
  params: RecognitionParams;
  snapshotImageData: string | null;
  onSnapshotFromVideo?: (imageData: string) => void;
}

const ImageAnalyzer: React.FC<ImageAnalyzerProps> = ({ students, params, snapshotImageData, onSnapshotFromVideo }) => {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [detectionsCount, setDetectionsCount] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [lastBase64Image, setLastBase64Image] = useState<string | null>(null); // Store last processed image
  const [detectionResults, setDetectionResults] = useState<any[]>([]); // Store detection results for Excel report
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // 视频处理状态
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [isVideoProcessing, setIsVideoProcessing] = useState(false);
  const [videoProcessingProgress, setVideoProcessingProgress] = useState(0);
  const [attendanceData, setAttendanceData] = useState<any[]>([]); // 出勤数据
  
  // 视频帧处理结果
  const [frameResults, setFrameResults] = useState<{time: number, imageData: string, detections: any[]}[]>([]);
  
  // 视频截取状态
  const [showVideoTrimControls, setShowVideoTrimControls] = useState(false);
  const [videoDuration, setVideoDuration] = useState(0);
  const [trimStart, setTrimStart] = useState(0);
  const [trimDuration, setTrimDuration] = useState(30); // 默认时长30秒
  const [isVideoTrimmed, setIsVideoTrimmed] = useState(false); // 新增状态：是否已截取
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const videoFileInputRef = useRef<HTMLInputElement>(null);

  // 选择帧进行预览
  const [selectedFrame, setSelectedFrame] = useState<{time: number, imageData: string, detections: any[]} | null>(null);

  // Handle snapshot from video
  useEffect(() => {
    if (snapshotImageData) {
      setImageUrl(snapshotImageData);
      setImageFile(null);
      setDetectionsCount(0);
      setLastBase64Image(null);
    }
  }, [snapshotImageData]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImageFile(file);
      setImageUrl(URL.createObjectURL(file));
      setDetectionsCount(0);
      setLastBase64Image(null);
    }
  };

  const handleImageUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setImageFile(file);
      const imageUrl = URL.createObjectURL(file);
      setImageUrl(imageUrl);
      setDetectionsCount(0);
      setLastBase64Image(null);
      
      try {
        // 读取文件为 Base64 用于后端处理
        const reader = new FileReader();
        reader.onload = async (e) => {
          const base64Image = e.target?.result as string;
          setLastBase64Image(base64Image); // Store for re-detection
          
          await processImageDetection(base64Image);
        };
        reader.readAsDataURL(file);
      } catch (error) {
        console.error('处理图像时出错:', error);
        setDetectionsCount(0);
      }
    }
  };

  // Process image detection with given base64 image
  const processImageDetection = async (base64Image: string) => {
    try {
      setIsProcessing(true);
      
      // 发送到后端进行人脸检测
      console.log('正在发送图像到后端进行人脸检测...');
      const result = await detectFacesFromBase64(base64Image);
      console.log('后端检测结果:', result);
      
      if (result.success) {
        // 转换后端结果为前端格式
        const detectedFaces = result.faces.map(face => ({
          id: face.id,
          detectionScore: face.confidence,
          box: {
            x: face.bbox[0],
            y: face.bbox[1],
            width: face.bbox[2] - face.bbox[0],
            height: face.bbox[3] - face.bbox[1]
          },
          landmarks: face.landmarks.map(([x, y]) => ({ x, y })),
          recognition: face.recognition // 添加识别结果
        }));
        
        setDetectionsCount(detectedFaces.length);
        setDetectionResults(detectedFaces); // 存储检测结果用于Excel报告
        drawFaceOverlays(detectedFaces);
        console.log(`检测到 ${detectedFaces.length} 个人脸`);
      } else {
        console.error('人脸检测失败:', result);
        setDetectionsCount(0);
        setDetectionResults([]); // 清空检测结果
      }
    } catch (error) {
      console.error('人脸检测出错:', error);
      setDetectionsCount(0);
      setDetectionResults([]); // 清空检测结果
    } finally {
      setIsProcessing(false);
    }
  };

  // Re-detect faces with current parameters
  const handleReDetect = async () => {
    if (!lastBase64Image) {
      alert('没有可重新检测的图像，请先上传一张图片');
      return;
    }
    
    await processImageDetection(lastBase64Image);
  };

  // 绘制检测到的人脸
  const drawFaceOverlays = (detections: any[], forDownload: boolean = false) => {
    if (!canvasRef.current || !imageRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // 用于跟踪已绘制的学生，确保一个人只绘制一次
    const drawnStudents = new Set<string>();
    
    // Draw face detections
    detections.forEach((detection, index) => {
      const box = detection.box || detection.detection?.box;
      if (!box) return;
      
      // 检查是否已经有识别结果且该学生已经被绘制
      if (detection.recognition) {
        const studentName = detection.recognition.name;
        // 允许显示所有识别结果，但在标签上标注重复
        if (drawnStudents.has(studentName)) {
          // 如果该学生已经被绘制，添加重复标记
          detection.isDuplicate = true;
        } else {
          // 标记该学生已被绘制
          drawnStudents.add(studentName);
        }
      }
      
      // Draw bounding box with gradient color based on confidence
      const confidence = detection.detectionScore || detection.detection?.score || 0;
      let boxColor = '#10b981'; // Green for high confidence
      
      // Color coding based on confidence
      if (confidence < 0.3) {
        boxColor = '#ef4444'; // Red for low confidence
      } else if (confidence < 0.6) {
        boxColor = '#f59e0b'; // Yellow for medium confidence
      }
      
      ctx.strokeStyle = boxColor;
      ctx.lineWidth = 2;
      ctx.strokeRect(box.x, box.y, box.width, box.height);
      
      // Prepare label text
      let labelText = `人脸 ${index + 1} (${(confidence * 100).toFixed(1)}%)`;
      
      // Add recognition result if available
      if (detection.recognition) {
        const recognition = detection.recognition;
        labelText = `${recognition.name} (${(recognition.confidence * 100).toFixed(1)}%)`;
        // 如果是重复识别，添加标记
        if (detection.isDuplicate) {
          labelText += " [重复]";
        }
      }
      
      // Draw label background
      const textWidth = ctx.measureText(labelText).width;
      ctx.fillStyle = boxColor + '80'; // Add transparency
      ctx.fillRect(box.x, box.y - 25, textWidth + 15, 25); // 增大背景框高度以适应更大字体
      
      // Draw label text with larger font (+5px)
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 17px Arial'; // 从12px增加到17px
      ctx.fillText(labelText, box.x + 5, box.y - 7); // 调整y坐标以适应更大字体
      
      // Draw landmarks if available (only in preview mode, not for download)
      if (!forDownload) {
        const landmarks = detection.landmarks || detection.landmarks?.positions;
        if (landmarks && Array.isArray(landmarks)) {
          ctx.fillStyle = '#f59e0b';
          landmarks.forEach(point => {
            if (point && typeof point.x === 'number' && typeof point.y === 'number') {
              ctx.beginPath();
              ctx.arc(point.x, point.y, 2, 0, Math.PI * 2);
              ctx.fill();
            }
          });
        }
      }
      
      // Draw face dimensions with larger font (+5px)
      ctx.fillStyle = '#94a3b8';
      ctx.font = '15px Arial'; // 从10px增加到15px
      ctx.fillText(`${Math.round(box.width)}×${Math.round(box.height)}px`, box.x + 5, box.y + box.height + 20); // 调整y坐标
    });
  };

  const downloadResults = async () => {
    if (!canvasRef.current || !imageRef.current) return;
    
    // 创建一个新的canvas来合并图像和检测结果
    const combinedCanvas = document.createElement('canvas');
    const ctx = combinedCanvas.getContext('2d');
    if (!ctx) return;
    
    // 设置canvas尺寸与原图一致
    combinedCanvas.width = imageRef.current.naturalWidth;
    combinedCanvas.height = imageRef.current.naturalHeight;
    
    // 绘制原始图像
    ctx.drawImage(imageRef.current, 0, 0, combinedCanvas.width, combinedCanvas.height);
    
    // 创建临时canvas来绘制不包含关键点的检测结果
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = canvasRef.current.width;
    tempCanvas.height = canvasRef.current.height;
    const tempCtx = tempCanvas.getContext('2d');
    
    if (tempCtx) {
      // 保存当前canvas引用
      const originalCanvasRef = canvasRef.current;
      
      // 临时替换canvas引用以便在临时canvas上绘制
      // 注意：这种方法可能不太理想，让我们采用另一种方法
      
      // 清理临时canvas
      tempCanvas.remove();
    }
    
    // 更好的方法：直接在合并canvas上绘制检测结果（不包含关键点）
    drawFaceOverlaysForDownload(ctx, detectionResults);
    
    // 生成包含参数信息的文件名
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const networkSize = params.networkSize;
    const minConfidence = params.minConfidence;
    
    // 获取图像尺寸信息
    const imageDimensions = `${combinedCanvas.width}x${combinedCanvas.height}`;
    
    const imageFilename = `face-recognition-${timestamp}-net${networkSize}-conf${Math.round(minConfidence * 100)}-img${imageDimensions}.png`;
    
    // 创建图片下载链接
    const imageLink = document.createElement('a');
    imageLink.download = imageFilename;
    imageLink.href = combinedCanvas.toDataURL('image/png');
    imageLink.click();
    
    // 清理canvas元素
    combinedCanvas.remove();
    
    // 创建并下载Excel文件
    downloadExcelReport(timestamp, networkSize, minConfidence, imageDimensions);
  };
  
  // 专门为下载创建的绘制函数，不包含人脸关键点
  const drawFaceOverlaysForDownload = (ctx: CanvasRenderingContext2D, detections: any[]) => {
    // 用于跟踪已绘制的学生，确保一个人只绘制一次
    const drawnStudents = new Set<string>();
    
    // Draw face detections without landmarks
    detections.forEach((detection, index) => {
      const box = detection.box || detection.detection?.box;
      if (!box) return;
      
      // 检查是否已经有识别结果且该学生已经被绘制
      if (detection.recognition) {
        const studentName = detection.recognition.name;
        // 允许显示所有识别结果，但在标签上标注重复
        if (drawnStudents.has(studentName)) {
          // 如果该学生已经被绘制，添加重复标记
          detection.isDuplicate = true;
        } else {
          // 标记该学生已被绘制
          drawnStudents.add(studentName);
        }
      }
      
      // Draw bounding box with gradient color based on confidence
      const confidence = detection.detectionScore || detection.detection?.score || 0;
      let boxColor = '#10b981'; // Green for high confidence
      
      // Color coding based on confidence
      if (confidence < 0.3) {
        boxColor = '#ef4444'; // Red for low confidence
      } else if (confidence < 0.6) {
        boxColor = '#f59e0b'; // Yellow for medium confidence
      }
      
      ctx.strokeStyle = boxColor;
      ctx.lineWidth = 2;
      ctx.strokeRect(box.x, box.y, box.width, box.height);
      
      // Prepare label text
      let labelText = `人脸 ${index + 1} (${(confidence * 100).toFixed(1)}%)`;
      
      // Add recognition result if available
      if (detection.recognition) {
        const recognition = detection.recognition;
        labelText = `${recognition.name} (${(recognition.confidence * 100).toFixed(1)}%)`;
        // 如果是重复识别，添加标记
        if (detection.isDuplicate) {
          labelText += " [重复]";
        }
      }
      
      // Draw label background
      const textWidth = ctx.measureText(labelText).width;
      ctx.fillStyle = boxColor + '80'; // Add transparency
      ctx.fillRect(box.x, box.y - 25, textWidth + 15, 25); // 增大背景框高度以适应更大字体
      
      // Draw label text with larger font (+5px)
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 17px Arial'; // 从12px增加到17px
      ctx.fillText(labelText, box.x + 5, box.y - 7); // 调整y坐标以适应更大字体
      
      // Draw face dimensions with larger font (+5px)
      ctx.fillStyle = '#94a3b8';
      ctx.font = '15px Arial'; // 从10px增加到15px
      ctx.fillText(`${Math.round(box.width)}×${Math.round(box.height)}px`, box.x + 5, box.y + box.height + 20); // 调整y坐标
    });
  };
  
  const downloadExcelReport = (timestamp: string, networkSize: number, minConfidence: number, imageDimensions: string) => {
    // 准备Excel数据
    const worksheetData = [
      ['人脸识别报告'],
      [`生成时间: ${new Date().toLocaleString('zh-CN')}`],
      [`检测参数: 网络尺寸=${networkSize}, 最小置信度=${minConfidence}`],
      [`图像尺寸: ${imageDimensions}`],
      [], // 空行
      ['学生姓名', '出勤状态', '可信度', '检测时间']
    ];
    
    // 创建一个映射来跟踪检测到的学生
    const detectedStudents = new Map<string, { confidence: number }>();
    
    // 处理检测到的人脸
    detectionResults.forEach(detection => {
      if (detection.recognition) {
        const name = detection.recognition.name;
        const confidence = detection.recognition.confidence;
        detectedStudents.set(name, { confidence });
      }
    });
    
    // 添加所有注册学生的数据
    students.forEach(student => {
      const studentName = student.name;
      if (detectedStudents.has(studentName)) {
        // 学生被检测到
        const detectionInfo = detectedStudents.get(studentName)!;
        worksheetData.push([
          studentName,
          '出勤',
          (detectionInfo.confidence * 100).toFixed(2) + '%',
          new Date().toLocaleString('zh-CN')
        ]);
      } else {
        // 学生未被检测到
        worksheetData.push([
          studentName,
          '缺勤',
          'N/A',
          new Date().toLocaleString('zh-CN')
        ]);
      }
    });
    
    // 添加未识别的人脸（不在学生列表中的检测结果）
    detectionResults.forEach((detection, index) => {
      if (!detection.recognition) {
        worksheetData.push([
          `未知人员-${index + 1}`,
          '出勤（未识别）',
          (detection.detectionScore * 100).toFixed(2) + '%',
          new Date().toLocaleString('zh-CN')
        ]);
      }
    });
    
    // 创建工作表
    const ws = XLSX.utils.aoa_to_sheet(worksheetData);
    
    // 设置列宽
    ws['!cols'] = [
      { wch: 15 }, // 学生姓名
      { wch: 15 }, // 出勤状态
      { wch: 10 }, // 可信度
      { wch: 20 }  // 检测时间
    ];
    
    // 创建工作簿
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, '人脸识别结果');
    
    // 生成文件名
    const excelFilename = `face-recognition-report-${timestamp}-net${networkSize}-conf${Math.round(minConfidence * 100)}-img${imageDimensions}.xlsx`;
    
    // 导出Excel文件
    XLSX.writeFile(wb, excelFilename);
  };
  
  // 视频预览功能
  const captureVideoPreview = async () => {
    if (!videoRef.current) return;
    
    const video = videoRef.current;
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    if (!ctx) return;
    
    // 设置画布尺寸
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // 绘制第一帧
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // 转换为Base64图像
    const imageData = canvas.toDataURL('image/jpeg');
    
    // 设置为预览图像
    setImageUrl(imageData);
  };

  // 处理视频上传
  const handleVideoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (file.type.startsWith('video/')) {
        setVideoFile(file);
        const url = URL.createObjectURL(file);
        setVideoUrl(url);
        setImageUrl(url); // 同时设置imageUrl以显示视频预览
        setDetectionsCount(0);
        setLastBase64Image(null);
        setAttendanceData([]); // 清除之前的出勤数据
        setShowVideoTrimControls(true); // 显示视频截取控件
      } else {
        alert('请选择视频文件');
      }
    }
  };
  
  // 视频加载完成时获取时长并设置默认截取范围
  const handleVideoLoadedMetadata = () => {
    if (videoRef.current) {
      const duration = videoRef.current.duration;
      setVideoDuration(duration);
      
      // 设置默认截取范围（最多5分钟）
      setTrimStart(0);
      setTrimDuration(Math.min(duration, 30)); // 默认时长30秒或视频总时长
      
      // 捕获预览帧
      captureVideoPreview();
    }
  };
  
  // 更新截取起始时间
  const handleTrimStartChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const start = parseFloat(e.target.value);
    setTrimStart(start);
  };
  
  // 更新截取时长
  const handleTrimDurationChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const duration = parseFloat(e.target.value);
    // 确保时长不超过视频剩余时间和5分钟限制
    const maxDuration = Math.min(videoDuration - trimStart, 300);
    if (duration <= maxDuration) {
      setTrimDuration(duration);
    }
  };
  
  // 处理视频截取
  const handleTrimVideo = async () => {
    if (!videoRef.current || !videoFile) return;
    
    // 隐藏截取控件
    setShowVideoTrimControls(false);
    
    // 设置截取状态
    setIsVideoTrimmed(true);
    
    // 创建截取后的视频预览
    await createTrimmedVideoPreview();
  };
  
  // 创建截取后的视频预览
  const createTrimmedVideoPreview = async () => {
    if (!videoFile) return;
    
    try {
      // 创建一个表示截取范围的对象
      const trimmedVideoInfo = {
        file: videoFile,
        startTime: trimStart,
        duration: trimDuration
      };
      
      // 在实际应用中，这里应该创建真正的截取视频
      // 但由于浏览器API限制，我们采用一种简化的方案：
      // 1. 继续显示原视频
      // 2. 但在处理时只处理指定的时间段
      // 3. 在UI上给出视觉提示表明已截取
      
      // 给用户一个视觉反馈，表明截取已完成
      console.log(`视频截取完成: 起始时间=${trimStart}s, 时长=${trimDuration}s`);
      
      // 注释掉弹窗，改为只在控制台输出日志
      // alert(`视频截取完成！
// 起始时间: ${trimStart.toFixed(1)}s
// 时长: ${trimDuration.toFixed(1)}s
//
// 请点击"开始识别"按钮进行人脸识别`);
      
    } catch (error) {
      console.error('创建截取视频预览失败:', error);
      // 注释掉错误弹窗，改为只在控制台输出错误
      // alert('视频截取预览创建失败: ' + (error instanceof Error ? error.message : '未知错误'));
    }
  };
  
  // 处理视频帧率控制和逐帧检测
  const processVideoFrames = async () => {
    if (!videoRef.current || !videoUrl) return;
    
    setIsVideoProcessing(true);
    setVideoProcessingProgress(0);
    setAttendanceData([]);
    setFrameResults([]); // 清除之前的帧结果
    
    const video = videoRef.current;
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    if (!ctx) {
      alert('无法创建画布上下文');
      setIsVideoProcessing(false);
      return;
    }
    
    try {
      // 确保视频已加载元数据
      if (video.readyState < 2) { // HAVE_CURRENT_DATA
        await new Promise<void>((resolve) => {
          const checkReady = () => {
            if (video.readyState >= 2) {
              resolve();
            } else {
              setTimeout(checkReady, 100);
            }
          };
          checkReady();
        });
      }
      
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      // 使用截取的时间范围（这是关键修改）
      const startTime = trimStart;
      const endTime = startTime + trimDuration;
      const duration = endTime - startTime;
      
      const fps = 1; // 降低到1fps
      const interval = 1 / fps;
      const framesCount = Math.max(1, Math.floor(duration * fps)); // 至少处理1帧
      
      console.log(`视频处理信息: 总时长=${duration}s, 起始时间=${startTime}s, 结束时间=${endTime}s, 帧数=${framesCount}`);
      
      // 注释掉处理开始提示弹窗
      // alert(`开始处理视频，共${framesCount}帧，请稍候...`);
      
      const attendanceRecords: {name: string, times: number[], positions: {time: number, box: any, confidence: number, recognition: any}[]}[] = [];
      const processedFrameResults: {time: number, imageData: string, detections: any[]}[] = [];
      
      // 逐帧处理
      for (let i = 0; i < framesCount; i++) {
        const time = startTime + (i * interval);
        video.currentTime = time;
        
        // 等待视频跳转完成
        await new Promise<void>((resolve) => {
          const checkReady = () => {
            if (video.readyState >= 2) { // HAVE_CURRENT_DATA
              resolve();
            } else {
              setTimeout(checkReady, 100);
            }
          };
          checkReady();
        });
        
        // 绘制当前帧到画布
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // 转换为Base64
        const base64Image = canvas.toDataURL('image/jpeg');
        
        try {
          // 发送到后端进行人脸检测
          const result = await detectFacesFromBase64(base64Image);
          
          console.log(`第 ${i + 1} 帧处理结果:`, result);
          
          if (result.success) {
            // 转换后端结果为前端格式
            const detectedFaces = result.faces.map(face => ({
              id: face.id,
              detectionScore: face.confidence,
              box: {
                x: face.bbox[0],
                y: face.bbox[1],
                width: face.bbox[2] - face.bbox[0],
                height: face.bbox[3] - face.bbox[1]
              },
              landmarks: face.landmarks.map(([x, y]) => ({ x, y })),
              recognition: face.recognition // 添加识别结果
            }));
            
            // 记录识别到的学生及其位置信息
            detectedFaces.forEach(face => {
              if (face.recognition) {
                const existingRecord = attendanceRecords.find(record => record.name === face.recognition!.name);
                if (existingRecord) {
                  existingRecord.times.push(time);
                  // 保存位置信息用于后续分析
                  existingRecord.positions.push({
                    time,
                    box: face.box,
                    confidence: face.detectionScore,
                    recognition: face.recognition
                  });
                  // 确保时间戳唯一
                  existingRecord.times = [...new Set(existingRecord.times)].sort((a: number, b: number) => a - b);
                } else {
                  attendanceRecords.push({
                    name: face.recognition!.name,
                    times: [time],
                    positions: [{
                      time,
                      box: face.box,
                      confidence: face.detectionScore,
                      recognition: face.recognition
                    }]
                  });
                }
              }
            });
            
            // 保存帧结果（包含绘制了人脸框的图像）
            const frameCanvas = document.createElement('canvas');
            frameCanvas.width = canvas.width;
            frameCanvas.height = canvas.height;
            const frameCtx = frameCanvas.getContext('2d');
            
            if (frameCtx) {
              // 绘制原始帧
              frameCtx.drawImage(canvas, 0, 0);
              
              // 绘制人脸检测结果
              drawFaceOverlaysOnCanvas(frameCtx, detectedFaces);
              
              // 保存绘制了人脸框的图像
              const frameImageData = frameCanvas.toDataURL('image/jpeg');
              processedFrameResults.push({
                time,
                imageData: frameImageData,
                detections: detectedFaces
              });
            }
          }
        } catch (error) {
          console.error(`处理第 ${i + 1} 帧时出错:`, error);
        }
        
        // 更新进度
        const progress = Math.round(((i + 1) / framesCount) * 100);
        setVideoProcessingProgress(progress);
        console.log(`处理进度: ${progress}% (${i + 1}/${framesCount})`);
      }
      
      console.log('视频帧处理完成，开始生成出勤名单');
      
      // 基于多帧数据分析，为每个位置确定最可能的学生
      const positionBasedAttendance = analyzePositionsForAttendance(attendanceRecords, students);
      
      // 生成综合结果图
      const summaryImageData = await generateSummaryImage(processedFrameResults, positionBasedAttendance);
      
      // 生成出勤名单
      const attendanceList = positionBasedAttendance.map(record => ({
        name: record.name,
        status: record.isPresent ? '出勤' : '缺勤',
        times: record.times,
        count: record.times.length,
        position: record.position // 添加位置信息
      }));
      
      setAttendanceData(attendanceList);
      setFrameResults(processedFrameResults); // 保存帧结果
      
      // 保存综合结果图
      if (summaryImageData) {
        // 可以将综合结果图保存到状态中，供后续使用
        console.log('综合结果图已生成');
      }
      
      // 默认选择第一帧进行预览
      if (processedFrameResults.length > 0) {
        setSelectedFrame(processedFrameResults[0]);
      }
      
      console.log('视频处理完成，出勤数据:', attendanceList);
      
      // 注释掉处理完成的弹窗提示
      // alert(`视频处理完成！识别到${attendanceList.filter(r => r.status === '出勤').length}名学生，${attendanceList.filter(r => r.status === '缺勤').length}名学生缺席。`);
      
    } catch (error) {
      console.error('视频处理出错:', error);
      // 注释掉错误弹窗，改为只在控制台输出错误
      // alert('视频处理过程中出现错误: ' + (error instanceof Error ? error.message : '未知错误'));
    } finally {
      setIsVideoProcessing(false);
    }
  };
  
  // 在canvas上绘制人脸检测结果
  const drawFaceOverlaysOnCanvas = (ctx: CanvasRenderingContext2D, detections: any[]) => {
    // 用于跟踪已绘制的学生，确保一个人只绘制一次
    const drawnStudents = new Set<string>();
    
    // Draw face detections
    detections.forEach((detection, index) => {
      const box = detection.box || detection.detection?.box;
      if (!box) return;
      
      // 检查是否已经有识别结果且该学生已经被绘制
      if (detection.recognition) {
        const studentName = detection.recognition.name;
        // 允许显示所有识别结果，但在标签上标注重复
        if (drawnStudents.has(studentName)) {
          // 如果该学生已经被绘制，添加重复标记
          detection.isDuplicate = true;
        } else {
          // 标记该学生已被绘制
          drawnStudents.add(studentName);
        }
      }
      
      // Draw bounding box with gradient color based on confidence
      const confidence = detection.detectionScore || detection.detection?.score || 0;
      let boxColor = '#10b981'; // Green for high confidence
      
      // Color coding based on confidence
      if (confidence < 0.3) {
        boxColor = '#ef4444'; // Red for low confidence
      } else if (confidence < 0.6) {
        boxColor = '#f59e0b'; // Yellow for medium confidence
      }
      
      ctx.strokeStyle = boxColor;
      ctx.lineWidth = 2;
      ctx.strokeRect(box.x, box.y, box.width, box.height);
      
      // Prepare label text
      let labelText = `人脸 ${index + 1} (${(confidence * 100).toFixed(1)}%)`;
      
      // Add recognition result if available
      if (detection.recognition) {
        const recognition = detection.recognition;
        labelText = `${recognition.name} (${(recognition.confidence * 100).toFixed(1)}%)`;
        // 如果是重复识别，添加标记
        if (detection.isDuplicate) {
          labelText += " [重复]";
        }
      }
      
      // Draw label background
      const textWidth = ctx.measureText(labelText).width;
      ctx.fillStyle = boxColor + '80'; // Add transparency
      ctx.fillRect(box.x, box.y - 25, textWidth + 15, 25); // 增大背景框高度以适应更大字体
      
      // Draw label text with larger font (+5px)
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 17px Arial'; // 从12px增加到17px
      ctx.fillText(labelText, box.x + 5, box.y - 7); // 调整y坐标以适应更大字体
      
      // Draw landmarks if available
      const landmarks = detection.landmarks || detection.landmarks?.positions;
      if (landmarks && Array.isArray(landmarks)) {
        ctx.fillStyle = '#f59e0b';
        landmarks.forEach(point => {
          if (point && typeof point.x === 'number' && typeof point.y === 'number') {
            ctx.beginPath();
            ctx.arc(point.x, point.y, 2, 0, Math.PI * 2);
            ctx.fill();
          }
        });
      }
      
      // Draw face dimensions with larger font (+5px)
      ctx.fillStyle = '#94a3b8';
      ctx.font = '15px Arial'; // 从10px增加到15px
      ctx.fillText(`${Math.round(box.width)}×${Math.round(box.height)}px`, box.x + 5, box.y + box.height + 20); // 调整y坐标
    });
  };
  
  // 打包下载所有帧结果
  const downloadFrameResults = async () => {
    if (frameResults.length === 0) {
      alert('没有帧结果可下载');
      return;
    }
    
    try {
      // 创建一个zip文件来包含所有图像
      const JSZip = (await import('jszip')).default;
      const zip = new JSZip();
      const imgFolder = zip.folder('frames');
      
      // 添加所有帧图像到zip
      for (let i = 0; i < frameResults.length; i++) {
        const frame = frameResults[i];
        const imageData = frame.imageData.split(',')[1]; // 移除data:image/jpeg;base64,前缀
        const binaryString = atob(imageData);
        const bytes = new Uint8Array(binaryString.length);
        
        for (let j = 0; j < binaryString.length; j++) {
          bytes[j] = binaryString.charCodeAt(j);
        }
        
        imgFolder?.file(`frame_${Math.round(frame.time)}s.jpg`, bytes);
      }
      
      // 生成zip文件
      const content = await zip.generateAsync({ type: 'blob' });
      
      // 下载zip文件
      const link = document.createElement('a');
      link.href = URL.createObjectURL(content);
      link.download = `video-frame-results-${new Date().getTime()}.zip`;
      link.click();
      
      alert(`成功打包下载${frameResults.length}帧处理结果！`);
    } catch (error) {
      console.error('打包下载失败:', error);
      alert('打包下载失败: ' + (error instanceof Error ? error.message : '未知错误'));
    }
  };
  
  // 基于位置分析的出勤判断函数
  const analyzePositionsForAttendance = (
    attendanceRecords: {name: string, times: number[], positions: {time: number, box: any, confidence: number, recognition: any}[]}[],
    students: Student[]
  ): {name: string, isPresent: boolean, times: number[], position?: any}[] => {
    // 创建位置到学生的映射
    const positionMap = new Map<string, {name: string, confidence: number, times: number[]}>();
    
    // 遍历所有识别记录
    attendanceRecords.forEach(record => {
      record.positions.forEach(position => {
        // 创建位置标识符（基于边界框坐标）
        const positionKey = `${Math.round(position.box.x/10)*10},${Math.round(position.box.y/10)*10},${Math.round(position.box.width/10)*10},${Math.round(position.box.height/10)*10}`;
        
        // 如果该位置还没有记录，或者当前识别的置信度更高，则更新
        if (!positionMap.has(positionKey) || positionMap.get(positionKey)!.confidence < position.confidence) {
          positionMap.set(positionKey, {
            name: record.name,
            confidence: position.confidence,
            times: [...record.times]
          });
        }
      });
    });
    
    // 为每个学生确定是否出勤
    const result = students.map(student => {
      // 查找该学生的所有位置记录
      const studentRecords = attendanceRecords.filter(record => record.name === student.name);
      
      if (studentRecords.length === 0) {
        // 学生从未被识别到
        return {
          name: student.name,
          isPresent: false,
          times: []
        };
      }
      
      // 计算学生出现在不同位置的次数
      const totalAppearances = studentRecords.reduce((sum, record) => sum + record.times.length, 0);
      
      // 如果学生出现在多个帧中，则认为出勤
      return {
        name: student.name,
        isPresent: totalAppearances > 0,
        times: studentRecords.flatMap(record => record.times)
      };
    });
    
    return result;
  };
  
  // 生成综合结果图
  const generateSummaryImage = async (frames: {time: number, imageData: string, detections: any[]}[], attendance: {name: string, isPresent: boolean, times: number[], position?: any}[]): Promise<string | null> => {
    if (frames.length === 0) return null;
    
    // 使用第一帧作为基础图像
    const firstFrame = frames[0];
    
    // 创建画布
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    
    // 创建图像对象
    const img = new Image();
    img.src = firstFrame.imageData;
    
    // 等待图像加载
    await new Promise<void>((resolve) => {
      img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        resolve();
      };
      img.onerror = () => resolve();
    });
    
    // 绘制标题
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(0, 0, canvas.width, 60);
    
    ctx.font = 'bold 24px Arial';
    ctx.fillStyle = 'white';
    ctx.fillText('人脸识别出勤分析结果', 20, 35);
    
    // 绘制统计信息
    const presentCount = attendance.filter(a => a.isPresent).length;
    const totalCount = attendance.length;
    
    ctx.font = '18px Arial';
    ctx.fillText(`出勤: ${presentCount}/${totalCount}`, 20, 90);
    
    // 绘制时间范围
    if (frames.length > 0) {
      const startTime = frames[0].time;
      const endTime = frames[frames.length - 1].time;
      ctx.fillText(`时间范围: ${startTime.toFixed(1)}s - ${endTime.toFixed(1)}s`, 20, 120);
    }
    
    // 绘制出勤学生列表
    ctx.font = '16px Arial';
    let yOffset = 160;
    const presentStudents = attendance.filter(a => a.isPresent);
    
    ctx.fillText('出勤学生:', 20, yOffset);
    yOffset += 30;
    
    // 每行显示5个学生
    for (let i = 0; i < presentStudents.length; i++) {
      const student = presentStudents[i];
      const row = Math.floor(i / 5);
      const col = i % 5;
      
      ctx.fillText(
        student.name, 
        20 + col * 150, 
        yOffset + row * 30
      );
    }
    
    return canvas.toDataURL('image/jpeg');
  };
  
  // 生成出勤报告Excel文件
  const generateAttendanceReport = () => {
    if (attendanceData.length === 0) {
      alert('没有出勤数据可导出');
      return;
    }
    
    // 准备Excel数据
    const worksheetData = [
      ['人脸识别出勤报告'],
      [`生成时间: ${new Date().toLocaleString('zh-CN')}`],
      [`总学生数: ${students.length}`, `出勤人数: ${attendanceData.filter(r => r.status === '出勤').length}`, `缺勤人数: ${attendanceData.filter(r => r.status === '缺勤').length}`],
      [], // 空行
      ['学生姓名', '出勤状态', '出现次数', '出现时间(秒)', '位置信息']
    ];
    
    // 添加学生数据
    attendanceData.forEach(record => {
      const timesStr = record.times.length > 0 
        ? record.times.map((t: number) => Math.round(t)).join(', ') 
        : 'N/A';
      
      // 添加位置信息（如果有的话）
      const positionStr = record.position 
        ? `x:${Math.round(record.position.box.x)}, y:${Math.round(record.position.box.y)}` 
        : 'N/A';
      
      worksheetData.push([
        record.name,
        record.status,
        record.count.toString(),
        timesStr,
        positionStr
      ]);
    });
    
    // 创建工作表
    const ws = XLSX.utils.aoa_to_sheet(worksheetData);
    
    // 设置列宽
    ws['!cols'] = [
      { wch: 15 }, // 学生姓名
      { wch: 10 }, // 出勤状态
      { wch: 10 }, // 出现次数
      { wch: 50 }, // 出现时间
      { wch: 20 }  // 位置信息
    ];
    
    // 创建工作簿
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, '出勤报告');
    
    // 生成文件名
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const excelFilename = `attendance-report-${timestamp}.xlsx`;
    
    // 导出Excel文件
    XLSX.writeFile(wb, excelFilename);
  };

  return (
    <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
      <div className="p-4 border-b border-slate-700">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <SquareDashedMousePointer className="w-5 h-5" />
          人脸识别
        </h2>
        <p className="text-slate-400 text-sm mt-1">
          上传图片或视频进行人脸识别检测
        </p>
      </div>
      
      <div className="p-4">
        {/* 文件上传区域 */}
        {!imageUrl && !videoUrl && (
          <div className="border-2 border-dashed border-slate-600 rounded-lg p-8 text-center hover:border-slate-500 transition-colors">
            <div className="flex justify-center gap-8">
              {/* 图片上传 */}
              <div className="text-center">
                <Upload className="w-12 h-12 text-slate-500 mx-auto mb-4" />
                <p className="text-slate-400 mb-4">上传图片文件进行人脸识别</p>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg font-medium transition-colors flex items-center gap-2 mx-auto"
                >
                  <Upload className="w-4 h-4" />
                  选择图片文件
                </button>
              </div>
              
              {/* 视频上传 */}
              <div className="text-center">
                <Video className="w-12 h-12 text-slate-500 mx-auto mb-4" />
                <p className="text-slate-400 mb-4">上传视频文件进行人脸识别</p>
                <button
                  onClick={() => videoFileInputRef.current?.click()}
                  className="px-4 py-2 bg-purple-600 hover:bg-purple-500 text-white rounded-lg font-medium transition-colors flex items-center gap-2 mx-auto"
                >
                  <Video className="w-4 h-4" />
                  选择视频文件
                </button>
              </div>
            </div>
            
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleImageUpload}
              accept="image/*"
              className="hidden"
            />
            
            <input
              type="file"
              ref={videoFileInputRef}
              onChange={handleVideoUpload}
              accept="video/*"
              className="hidden"
            />
          </div>
        )}
        
        {/* 图片或视频显示和结果 */}
        {(imageUrl || videoUrl) && (
          <div className="space-y-4">
            {/* 共用的大图框 - 显示图片或视频预览 */}
            {imageUrl && !selectedFrame && (
              <div className="relative inline-block max-w-full">
                {/* 视频预览 */}
                {videoUrl && (
                  <video
                    ref={videoRef}
                    src={videoUrl}
                    controls
                    className="max-w-full max-h-[500px] rounded-lg"
                    onLoadedMetadata={handleVideoLoadedMetadata}
                  />
                )}
                
                {/* 图片预览 */}
                {!videoUrl && (
                  <>
                    <img
                      ref={imageRef}
                      src={imageUrl}
                      alt="Uploaded"
                      className="max-w-full max-h-[500px] rounded-lg"
                      onLoad={() => {
                        // Set canvas dimensions to match image
                        if (imageRef.current && canvasRef.current && imageUrl) {
                          canvasRef.current.width = imageRef.current.naturalWidth;
                          canvasRef.current.height = imageRef.current.naturalHeight;
                        }
                      }}
                    />
                    <canvas
                      ref={canvasRef}
                      className="absolute top-0 left-0 w-full h-full pointer-events-none"
                      style={{ imageRendering: 'pixelated' }}
                    />
                  </>
                )}
              </div>
            )}
            
            {/* 视频帧预览 - 显示选中的视频帧 */}
            {selectedFrame && (
              <div className="relative inline-block max-w-full">
                <img
                  src={selectedFrame.imageData}
                  alt={`Selected frame at ${Math.round(selectedFrame.time)}s`}
                  className="max-w-full max-h-[500px] rounded-lg"
                />
                <div className="absolute top-2 left-2 bg-black/50 text-white text-sm px-2 py-1 rounded">
                  时间: {Math.round(selectedFrame.time)}s
                </div>
              </div>
            )}
            
            {/* 图片处理控制面板 */}
            {imageUrl && !videoUrl && (
              <div className="bg-slate-700/50 rounded-lg p-4">
                <div className="flex justify-between items-center">
                  <div>
                    <h3 className="font-medium text-white">检测结果</h3>
                    <p className="text-slate-400 text-sm">
                      检测到 <span className="text-green-400 font-bold">{detectionsCount}</span> 个人脸
                    </p>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <button
                      onClick={handleReDetect}
                      disabled={isProcessing || !lastBase64Image}
                      className="px-3 py-1.5 bg-amber-600 hover:bg-amber-500 disabled:opacity-50 text-white rounded-lg text-sm font-medium transition-colors flex items-center gap-1"
                    >
                      {isProcessing ? (
                        <>
                          <span className="inline-block w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
                          检测中...
                        </>
                      ) : (
                        <>
                          <RotateCcw className="w-4 h-4" />
                          重新检测
                        </>
                      )}
                    </button>
                    
                    <button
                      onClick={() => {
                        setImageUrl(null);
                        setImageFile(null);
                        setDetectionsCount(0);
                        setLastBase64Image(null);
                      }}
                      className="px-3 py-1.5 bg-slate-600 hover:bg-slate-500 text-white rounded-lg text-sm font-medium transition-colors"
                    >
                      重新上传
                    </button>
                    
                    <button
                      onClick={downloadResults}
                      disabled={detectionsCount === 0}
                      className="px-3 py-1.5 bg-green-600 hover:bg-green-500 disabled:opacity-50 text-white rounded-lg text-sm font-medium transition-colors flex items-center gap-1"
                    >
                      <Download className="w-4 h-4" />
                      下载结果
                    </button>
                  </div>
                </div>
              </div>
            )}
            
            {/* 视频处理控制面板 */}
            {videoUrl && (
              <div className="bg-slate-700/50 rounded-lg p-4">
                <div className="flex justify-between items-center">
                  <div>
                    <h3 className="font-medium text-white">视频处理</h3>
                    {isVideoProcessing && (
                      <div className="mt-2">
                        <p className="text-slate-400 text-sm">
                          处理进度: {videoProcessingProgress}%
                        </p>
                        <div className="w-full bg-slate-600 rounded-full h-2 mt-1">
                          <div 
                            className="bg-purple-600 h-2 rounded-full transition-all duration-300" 
                            style={{ width: `${videoProcessingProgress}%` }}
                          ></div>
                        </div>
                      </div>
                    )}
                    {!isVideoProcessing && attendanceData.length > 0 && (
                      <p className="text-green-400 text-sm mt-2">
                        处理完成！识别到 {attendanceData.filter(r => r.status === '出勤').length} 名学生
                      </p>
                    )}
                    {/* 添加截取状态指示器 */}
                    {isVideoTrimmed && !isVideoProcessing && attendanceData.length === 0 && (
                      <p className="text-blue-400 text-sm mt-2">
                        视频已截取: {trimStart.toFixed(1)}s - {(trimStart + trimDuration).toFixed(1)}s
                      </p>
                    )}
                  </div>
                  
                  <div className="flex items-center gap-2">
                    {!showVideoTrimControls && !isVideoProcessing && attendanceData.length === 0 && (
                      <button
                        onClick={processVideoFrames}
                        className="px-3 py-1.5 bg-purple-600 hover:bg-purple-500 text-white rounded-lg text-sm font-medium transition-colors flex items-center gap-1"
                      >
                        <Play className="w-4 h-4" />
                        开始识别 (1fps)
                      </button>
                    )}
                    
                    {!showVideoTrimControls && (isVideoProcessing || attendanceData.length > 0) && (
                      <>
                        {isVideoProcessing && (
                          <button
                            disabled
                            className="px-3 py-1.5 bg-purple-600 hover:bg-purple-500 disabled:opacity-50 text-white rounded-lg text-sm font-medium transition-colors flex items-center gap-1"
                          >
                            <span className="inline-block w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
                            识别中...
                          </button>
                        )}
                        
                        {attendanceData.length > 0 && (
                          <button
                            onClick={processVideoFrames}
                            disabled={isVideoProcessing}
                            className="px-3 py-1.5 bg-purple-600 hover:bg-purple-500 disabled:opacity-50 text-white rounded-lg text-sm font-medium transition-colors flex items-center gap-1"
                          >
                            <Play className="w-4 h-4" />
                            重新识别
                          </button>
                        )}
                      </>
                    )}
                    
                    <button
                      onClick={generateAttendanceReport}
                      disabled={attendanceData.length === 0}
                      className="px-3 py-1.5 bg-green-600 hover:bg-green-500 disabled:opacity-50 text-white rounded-lg text-sm font-medium transition-colors flex items-center gap-1"
                    >
                      <Download className="w-4 h-4" />
                      导出报告
                    </button>
                    
                    <button
                      onClick={() => {
                        setVideoUrl(null);
                        setVideoFile(null);
                        setAttendanceData([]);
                        setFrameResults([]);
                        setSelectedFrame(null);
                        // 重置截取状态
                        setIsVideoTrimmed(false);
                        setTrimStart(0);
                        setTrimDuration(30);
                      }}
                      className="px-3 py-1.5 bg-slate-600 hover:bg-slate-500 text-white rounded-lg text-sm font-medium transition-colors"
                    >
                      重新上传
                    </button>
                  </div>
                </div>
                
                {/* 视频截取控件 */}
                {showVideoTrimControls && (
                  <div className="mt-4">
                    <h4 className="font-medium text-white mb-2">
                      视频截取 (最长5分钟)
                    </h4>
                    <div className="space-y-4">
                      <div>
                        <div className="flex justify-between text-sm text-slate-300 mb-1">
                          <span>起始时间: {Math.round(trimStart)}s</span>
                          <span>时长: {Math.round(trimDuration)}s</span>
                          <span>总时长: {Math.round(videoDuration)}s</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-slate-400 text-sm">0s</span>
                          <input
                            type="range"
                            min="0"
                            max={videoDuration}
                            step="0.1"
                            value={trimStart}
                            onChange={handleTrimStartChange}
                            className="w-full h-2 bg-slate-600 rounded-lg appearance-none cursor-pointer"
                          />
                          <span className="text-slate-400 text-sm">{Math.round(videoDuration)}s</span>
                        </div>
                        <div className="flex items-center gap-2 mt-1">
                          <span className="text-slate-400 text-sm">0s</span>
                          <input
                            type="range"
                            min="0"
                            max={300}
                            step="0.1"
                            value={trimDuration}
                            onChange={handleTrimDurationChange}
                            className="w-full h-2 bg-slate-600 rounded-lg appearance-none cursor-pointer"
                          />
                          <span className="text-slate-400 text-sm">300s</span>
                        </div>
                      </div>
                      <div className="text-slate-400 text-sm">
                        当前截取时长: {Math.round(trimDuration)}秒 (最大支持300秒)
                      </div>
                      <div className="flex gap-2">
                        <button
                          onClick={handleTrimVideo}
                          className="px-4 py-2 bg-pink-600 hover:bg-pink-500 text-white rounded-lg font-medium transition-colors flex items-center gap-2"
                        >
                          <SkipForward className="w-4 h-4" />
                          确认截取
                        </button>
                        <button
                          onClick={() => setShowVideoTrimControls(false)}
                          className="px-4 py-2 bg-slate-600 hover:bg-slate-500 text-white rounded-lg font-medium transition-colors"
                        >
                          取消
                        </button>
                      </div>
                    </div>
                  </div>
                )}
                
                {/* 帧结果预览 */}
                {frameResults.length > 0 && (
                  <div className="mt-4">
                    <h4 className="font-medium text-white mb-2">
                      帧结果预览 ({frameResults.length} 帧)
                    </h4>
                    <div className="flex gap-2 overflow-x-auto pb-2">
                      {frameResults.map((frame, index) => (
                        <div 
                          key={index} 
                          className={`flex-shrink-0 cursor-pointer rounded border-2 transition-all ${
                            selectedFrame?.time === frame.time 
                              ? 'border-blue-500 ring-2 ring-blue-500/50' 
                              : 'border-slate-600 hover:border-slate-400'
                          }`}
                          onClick={() => setSelectedFrame(frame)}
                        >
                          <div className="text-xs text-slate-400 text-center p-1">
                            {Math.round(frame.time)}s
                          </div>
                          <img 
                            src={frame.imageData} 
                            alt={`Frame at ${Math.round(frame.time)}s`} 
                            className="w-24 h-18 object-cover"
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                
                {/* 出勤数据显示 */}
                {attendanceData.length > 0 && (
                  <div className="mt-4">
                    <div className="flex justify-between items-center mb-2">
                      <h4 className="font-medium text-white">
                        出勤统计 (总学生: {students.length}, 出勤: {attendanceData.filter(r => r.status === '出勤').length}, 缺勤: {attendanceData.filter(r => r.status === '缺勤').length})
                      </h4>
                      <button
                        onClick={downloadFrameResults}
                        disabled={frameResults.length === 0}
                        className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded-lg text-sm font-medium transition-colors flex items-center gap-1"
                      >
                        <Download className="w-4 h-4" />
                        下载帧结果 ({frameResults.length})
                      </button>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2 max-h-40 overflow-y-auto">
                      {attendanceData.map((record, index) => (
                        <div 
                          key={index} 
                          className={`p-2 rounded text-sm ${
                            record.status === '出勤' 
                              ? 'bg-green-900/30 text-green-400' 
                              : 'bg-red-900/30 text-red-400'
                          }`}
                        >
                          <div className="font-medium">{record.name}</div>
                          <div className="text-xs opacity-75">
                            {record.status} 
                            {record.times.length > 0 && ` (${record.times.length}次)`}
                          </div>
                          {record.times.length > 0 && (
                            <div className="text-xs opacity-60 mt-1">
                              时间: {record.times.map(t => Math.round(t)).join(', ')}s
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

      </div>
    </div>
  );
};

export default ImageAnalyzer;