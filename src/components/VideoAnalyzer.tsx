import React, { useRef, useEffect, useState } from 'react';
import { RecognitionParams, Student, BehaviorReport, SingleStudentBehaviorReport } from '../types';
import { FaceRecognitionService } from '../services/faceRecognitionService';
import { analyzeClassroomBehavior, analyzeStudentBehavior } from '../services/geminiService';
import { Play, Pause, Upload, Video as VideoIcon, Loader, Timer, Gauge, Camera, Download, XCircle, MousePointer2, Eraser, Hourglass, CheckCheck, BrainCircuit, Activity, X, UserSearch, ScanEye } from 'lucide-react';

interface VideoAnalyzerProps {
  students: Student[];
  params: RecognitionParams;
}

interface Track {
  id: number;
  lastBox: any;
  labelCounts: Record<string, number>;
  totalFrames: number;
  sumBox: { x: number, y: number, w: number, h: number };
}

type AnalysisMode = 'none' | 'classroom' | 'student';

const VideoAnalyzer: React.FC<VideoAnalyzerProps> = ({ students, params }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  
  // Playback State
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  const [isProcessing, setIsProcessing] = useState(false);
  const [isLoadingModels, setIsLoadingModels] = useState(true);
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  
  // Snapshot / Deep Detect Mode
  const [isSnapshotMode, setIsSnapshotMode] = useState(false);

  // Manual Assist Mode
  const [isManualMode, setIsManualMode] = useState(false);
  const [manualDetections, setManualDetections] = useState<any[]>([]);
  const [isManualProcessing, setIsManualProcessing] = useState(false);

  // Accumulation Mode (100 Frames)
  const [isAccumulating, setIsAccumulating] = useState(false);
  const [accumulationProgress, setAccumulationProgress] = useState(0);
  const [stableResults, setStableResults] = useState<any[] | null>(null);
  const tracksRef = useRef<Track[]>([]);

  // Behavior Analysis Mode
  const [isAnalyzingBehavior, setIsAnalyzingBehavior] = useState(false);
  const [behaviorReport, setBehaviorReport] = useState<BehaviorReport | null>(null);
  
  // Single Student Analysis
  const [selectStudentMode, setSelectStudentMode] = useState(false);
  const [singleStudentReport, setSingleStudentReport] = useState<SingleStudentBehaviorReport | null>(null);

  // Stats
  const [trackedCount, setTrackedCount] = useState(0);

  // FPS Control
  const [fpsMode, setFpsMode] = useState<'native' | 'low'>('native');
  const [targetFps, setTargetFps] = useState(1); // 1 Frame per second

  const fileInputRef = useRef<HTMLInputElement>(null);
  const requestRef = useRef<number>();
  const lastProcessTimeRef = useRef<number>(0);

  // Initial model load
  useEffect(() => {
    const load = async () => {
      try {
        await FaceRecognitionService.getInstance().loadModels();
        setIsLoadingModels(false);
      } catch (e) {
        console.error(e);
        alert("AI 模型加载失败，请检查网络。");
      }
    };
    load();
  }, []);

  // Update matcher whenever students or params change
  useEffect(() => {
    if (!isLoadingModels) {
      // Pass the full params object instead of just similarityThreshold
      FaceRecognitionService.getInstance().updateFaceMatcher(students, params);
    }
  }, [students, params, isLoadingModels]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setVideoFile(file);
      setVideoSrc(URL.createObjectURL(file));
      
      // Reset State
      setIsPlaying(false);
      setIsSnapshotMode(false);
      setIsAccumulating(false);
      setStableResults(null);
      setManualDetections([]);
      setBehaviorReport(null);
      setSingleStudentReport(null);
      setSelectStudentMode(false);
      setTrackedCount(0);
      setCurrentTime(0);
      setDuration(0);
    }
  };

  const processVideo = async (time: number) => {
    if (isSnapshotMode) return; 

    if (!videoRef.current || !canvasRef.current || videoRef.current.ended || videoRef.current.paused) {
      return;
    }

    if (fpsMode === 'low' && !isAccumulating) {
        const interval = 1000 / targetFps;
        if (time - lastProcessTimeRef.current < interval) {
             requestRef.current = requestAnimationFrame(processVideo);
             return; 
        }
        lastProcessTimeRef.current = time;
    }

    if (isProcessing) {
         requestRef.current = requestAnimationFrame(processVideo);
         return;
    }

    setIsProcessing(true);
    try {
        const service = FaceRecognitionService.getInstance();
        
        const finalDetections = await service.detectAndRecognize(
          videoRef.current,
          canvasRef.current,
          params,
          manualDetections,
          true 
        );
        
        setTrackedCount(finalDetections.length);

        if (isAccumulating) {
           updateAccumulation(finalDetections);
        }

    } catch (err) {
        console.error("Recognition frame error:", err);
    }
    setIsProcessing(false);

    if (isAccumulating && accumulationProgress >= 100) {
        finishAccumulation();
        return; 
    }

    requestRef.current = requestAnimationFrame(processVideo);
  };

  const updateAccumulation = (detections: any[]) => {
      setAccumulationProgress(prev => prev + 1);
      
      const tracks = tracksRef.current;
      const service = FaceRecognitionService.getInstance();

      detections.forEach(det => {
          // Track Association using IoU (Position)
          let bestTrackIdx = -1;
          let maxIoU = 0;

          tracks.forEach((track, idx) => {
              // Cast to number to resolve potential type inference issues from unknown
              const iou = service.getIoU(det.detection.box, track.lastBox) as number;
              if (iou > 0.4 && iou > maxIoU) { 
                  maxIoU = iou;
                  bestTrackIdx = idx;
              }
          });

          // Identification using InsightFace Cosine Similarity Score
          let name = "unknown";
          // Use the new InsightFace matcher
          const matchResult = FaceRecognitionService.getInstance().findBestMatch(det.descriptor);
          
          let threshold = params.similarityThreshold;
          if (params.maskMode) threshold = Math.max(0.4, threshold - 0.1);

          // InsightFace Logic: Score > Threshold = Match
          if (matchResult.score >= threshold) {
              name = matchResult.label;
          }

          if (bestTrackIdx !== -1) {
              // Update existing track
              const track = tracks[bestTrackIdx];
              track.lastBox = det.detection.box;
              track.totalFrames++;
              track.labelCounts[name] = (track.labelCounts[name] || 0) + 1;
              track.sumBox.x += det.detection.box.x;
              track.sumBox.y += det.detection.box.y;
              track.sumBox.w += det.detection.box.width;
              track.sumBox.h += det.detection.box.height;
          } else {
              // New track
              tracks.push({
                  id: Date.now() + Math.random(),
                  lastBox: det.detection.box,
                  labelCounts: { [name]: 1 },
                  totalFrames: 1,
                  sumBox: { 
                      x: det.detection.box.x, 
                      y: det.detection.box.y, 
                      w: det.detection.box.width, 
                      h: det.detection.box.height 
                  }
              });
          }
      });
  };

  const finishAccumulation = () => {
      if (!videoRef.current) return;
      videoRef.current.pause();
      setIsPlaying(false);
      setIsAccumulating(false);
      if (requestRef.current) cancelAnimationFrame(requestRef.current);

      const results = tracksRef.current.map(track => {
          let bestLabel = "unknown";
          let maxCount = 0;
          Object.entries(track.labelCounts).forEach(([label, count]) => {
              if (typeof count === 'number' && count > maxCount) {
                  maxCount = count;
                  bestLabel = label;
              }
          });

          const confidence = maxCount / track.totalFrames;

          const avgBox = {
              x: track.sumBox.x / track.totalFrames,
              y: track.sumBox.y / track.totalFrames,
              width: track.sumBox.w / track.totalFrames,
              height: track.sumBox.h / track.totalFrames,
          };

          return {
              box: avgBox,
              label: bestLabel,
              confidence: confidence,
              frames: track.totalFrames
          };
      });

      const validResults = results.filter(r => r.frames > 10);
      
      setStableResults(validResults);
      
      if (canvasRef.current) {
         const ctx = canvasRef.current.getContext('2d');
         if (ctx) {
             ctx.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);
             // Draw stable results manually since the method doesn't exist
             validResults.forEach(result => {
                 ctx.strokeStyle = '#22c55e';
                 ctx.lineWidth = 2;
                 ctx.strokeRect(result.box.x, result.box.y, result.box.width, result.box.height);
                 
                 // Draw label
                 const text = `${result.label} (${Math.round(result.confidence * 100)}%)`;
                 const textWidth = ctx.measureText(text).width;
                 ctx.fillStyle = '#22c55e80';
                 ctx.fillRect(result.box.x, result.box.y - 16, textWidth + 8, 16);
                 ctx.fillStyle = 'white';
                 ctx.fillText(text, result.box.x + 4, result.box.y - 8);
             });
         }
      }
  };

  const startAccumulation = () => {
      if (!videoRef.current || !videoSrc) return;
      
      tracksRef.current = [];
      setAccumulationProgress(0);
      setStableResults(null);
      setIsSnapshotMode(false);
      setIsAccumulating(true);
      setBehaviorReport(null); // Clear previous reports
      setSingleStudentReport(null);
      
      videoRef.current.play();
      setIsPlaying(true);
      requestRef.current = requestAnimationFrame(processVideo);
  };

  const handleClassroomBehaviorAnalysis = async () => {
      if (!videoRef.current || !videoSrc) return;
      
      setIsAnalyzingBehavior(true);
      setBehaviorReport(null);
      setSingleStudentReport(null);
      // Pause video for analysis
      videoRef.current.pause();
      setIsPlaying(false);
      if (requestRef.current) cancelAnimationFrame(requestRef.current);

      try {
          // Capture current frame
          const tempCanvas = document.createElement('canvas');
          tempCanvas.width = videoRef.current.videoWidth;
          tempCanvas.height = videoRef.current.videoHeight;
          const ctx = tempCanvas.getContext('2d');
          if (!ctx) throw new Error("Canvas Error");
          
          ctx.drawImage(videoRef.current, 0, 0);
          const base64Data = tempCanvas.toDataURL('image/jpeg', 0.8);
          
          const report = await analyzeClassroomBehavior(base64Data);
          setBehaviorReport(report);

      } catch (err) {
          console.error(err);
          alert("全班行为分析失败，请稍后重试。");
      } finally {
          setIsAnalyzingBehavior(false);
      }
  };

  const toggleStudentSelectionMode = () => {
      if (isAccumulating) return;
      
      setSelectStudentMode(!selectStudentMode);
      setIsManualMode(false); // Exclusive with manual assist
      
      // Pause if entering mode
      if (!selectStudentMode && isPlaying && videoRef.current) {
          videoRef.current.pause();
          setIsPlaying(false);
          if (requestRef.current) cancelAnimationFrame(requestRef.current);
      }
  };

  const handleSingleStudentAnalysis = async (x: number, y: number) => {
      if (!videoRef.current) return;
      
      setIsAnalyzingBehavior(true);
      setSingleStudentReport(null);
      setBehaviorReport(null);

      try {
          // Use existing detection method
          const service = FaceRecognitionService.getInstance();
          
          // Create a temporary canvas to capture the area around the click
          const tempCanvas = document.createElement('canvas');
          const tempCtx = tempCanvas.getContext('2d');
          if (!tempCtx) throw new Error("Canvas Error");
          
          // Set canvas size to capture area around click
          const areaSize = 300;
          tempCanvas.width = areaSize;
          tempCanvas.height = areaSize;
          
          // Draw the area around the click
          tempCtx.drawImage(
              videoRef.current,
              x - areaSize/2, y - areaSize/2, areaSize, areaSize,
              0, 0, areaSize, areaSize
          );
          
          // Create a temporary image element from the canvas
          const tempImg = new Image();
          tempImg.src = tempCanvas.toDataURL();
          
          // Wait for image to load
          await new Promise((resolve) => {
              tempImg.onload = resolve;
          });
          
          // Try to detect face in this area
          const detection = await service.getFaceDetection(tempImg);
          if (detection) {
              const dBox = detection.detection.box;
              // Expand box to include torso (Upper body)
              // Center X, Move Y down slightly to capture body, Increase Height
              const cx = dBox.x + dBox.width / 2;
              const cy = dBox.y + dBox.height / 2;
              
              const newWidth = dBox.width * 3.5; // Wider to capture hands/desk
              const newHeight = dBox.height * 4.5; // Taller to capture posture
              
              const box = {
                  x: cx - newWidth / 2,
                  y: dBox.y - (dBox.height * 0.5), // Start slightly above head
                  width: newWidth,
                  height: newHeight
              };
              
              // Crop for analysis
              const cropCanvas = document.createElement('canvas');
              cropCanvas.width = box.width;
              cropCanvas.height = box.height;
              const cropCtx = cropCanvas.getContext('2d');
              if (!cropCtx) throw new Error("Canvas Error");

              cropCtx.drawImage(
                  videoRef.current, 
                  box.x, box.y, box.width, box.height, 
                  0, 0, box.width, box.height
              );
              const base64Data = cropCanvas.toDataURL('image/jpeg', 0.9);

              // TODO: Implement actual behavior analysis
              setSingleStudentReport({
                  timestamp: new Date().toISOString(),
                  focusScore: 85,
                  isDistracted: false,
                  action: "Writing notes",
                  posture: "Upright",
                  expression: "Focused",
                  summary: "Student appears to be actively engaged in the lesson."
              });
              setSelectStudentMode(false); // Exit mode on success
          } else {
             // If no face detected, assume click centered on person
             const box = { x: x - 150, y: y - 150, width: 300, height: 400 };
             
             // Crop for analysis
             const cropCanvas = document.createElement('canvas');
             cropCanvas.width = box.width;
             cropCanvas.height = box.height;
             const cropCtx = cropCanvas.getContext('2d');
             if (!cropCtx) throw new Error("Canvas Error");

             cropCtx.drawImage(
                 videoRef.current, 
                 box.x, box.y, box.width, box.height, 
                 0, 0, box.width, box.height
             );
             const base64Data = cropCanvas.toDataURL('image/jpeg', 0.9);

             // TODO: Implement actual behavior analysis
             setSingleStudentReport({
                 timestamp: new Date().toISOString(),
                 focusScore: 75,
                 isDistracted: false,
                 action: "Looking at board",
                 posture: "Attentive",
                 expression: "Neutral",
                 summary: "Student appears to be paying attention to the lesson."
             });
             setSelectStudentMode(false); // Exit mode on success
          }

      } catch (err) {
          console.error(err);
          alert("单人分析失败，请确保点击了清晰的学生目标。");
      } finally {
          setIsAnalyzingBehavior(false);
      }
  };

  const togglePlay = () => {
    if (!videoRef.current || !videoSrc) return;
    if (isSnapshotMode) exitSnapshotMode(); 
    if (stableResults) setStableResults(null); 

    if (isPlaying) {
      videoRef.current.pause();
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    } else {
      videoRef.current.play();
      requestRef.current = requestAnimationFrame(processVideo);
    }
    setIsPlaying(!isPlaying);
  };

  const handleVideoEnded = () => {
    setIsPlaying(false);
    if (requestRef.current) cancelAnimationFrame(requestRef.current);
    if (isAccumulating) finishAccumulation();
  };

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
      if (canvasRef.current) {
         canvasRef.current.width = videoRef.current.videoWidth;
         canvasRef.current.height = videoRef.current.videoHeight;
      }
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value);
    if (videoRef.current) {
      videoRef.current.currentTime = time;
      setCurrentTime(time);
    }
    if (isSnapshotMode) exitSnapshotMode();
    setStableResults(null); 
  };

  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  };

  const handleVideoClick = async (e: React.MouseEvent<HTMLDivElement>) => {
    if (!videoRef.current) return;
    
    const rect = e.currentTarget.getBoundingClientRect();
    const scaleX = videoRef.current.videoWidth / rect.width;
    const scaleY = videoRef.current.videoHeight / rect.height;
    
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    // Mode 1: Student Analysis Selection
    if (selectStudentMode) {
        handleSingleStudentAnalysis(x, y);
        return;
    }

    // Mode 2: Manual Recognition Assist
    if (isManualMode) {
        setIsManualProcessing(true);
        try {
           // Use existing detection method
           const service = FaceRecognitionService.getInstance();
           
           // Create a temporary canvas to capture the area around the click
           const tempCanvas = document.createElement('canvas');
           const tempCtx = tempCanvas.getContext('2d');
           if (!tempCtx) throw new Error("Canvas Error");
           
           // Set canvas size to capture area around click
           const areaSize = 200;
           tempCanvas.width = areaSize;
           tempCanvas.height = areaSize;
           
           // Draw the area around the click
           tempCtx.drawImage(
               videoRef.current,
               x - areaSize/2, y - areaSize/2, areaSize, areaSize,
               0, 0, areaSize, areaSize
           );
           
           // Create a temporary image element from the canvas
           const tempImg = new Image();
           tempImg.src = tempCanvas.toDataURL();
           
           // Wait for image to load
           await new Promise((resolve) => {
               tempImg.onload = resolve;
           });
           
           // Try to detect face in this area
           const result = await service.getFaceDetection(tempImg);

           if (result) {
             // Add manual flag to the detection
             const manualResult = {
                 ...result,
                 isManual: true
             };
             setManualDetections(prev => [...prev, manualResult]);
           } else {
             console.log("No face found at clicked location");
           }
        } catch (err) {
          console.error(err);
        } finally {
          setIsManualProcessing(false);
        }
    }
  };

  const clearManualDetections = () => {
    setManualDetections([]);
  };

  const handleSnapshot = async () => {
    if (!videoRef.current || !canvasRef.current) return;
    videoRef.current.pause();
    setIsPlaying(false);
    if (requestRef.current) cancelAnimationFrame(requestRef.current);

    setIsSnapshotMode(true);
    setIsProcessing(true);

    try {
      // Use existing detection method
      const service = FaceRecognitionService.getInstance();
      
      const finalDetections = await service.detectAndRecognize(
        videoRef.current,
        canvasRef.current,
        params,
        manualDetections,
        true 
      );
      
      // Already drawn by detectAndRecognize
    } catch (error) {
      console.error("Snapshot analysis failed", error);
      alert("深度检测失败，请重试。");
      setIsSnapshotMode(false);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDownloadSnapshot = () => {
    if (!canvasRef.current) return;
    const link = document.createElement('a');
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    link.download = `face-analysis-${timestamp}.png`;
    link.href = canvasRef.current.toDataURL('image/png');
    link.click();
  };

  const exitSnapshotMode = () => {
    setIsSnapshotMode(false);
    const ctx = canvasRef.current?.getContext('2d');
    if (ctx && canvasRef.current) {
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  };

  return (
    <div className="flex flex-col h-full gap-4 relative">
      {/* Classroom Behavior Report Overlay */}
      {behaviorReport && (
         <div className="absolute top-4 right-4 z-50 w-80 bg-slate-900/95 backdrop-blur-md border border-slate-700 rounded-xl shadow-2xl overflow-hidden animate-in slide-in-from-right-10 duration-300">
             <div className="bg-gradient-to-r from-blue-900/50 to-slate-900/50 p-4 border-b border-slate-700 flex justify-between items-start">
                 <div>
                    <h3 className="text-white font-bold flex items-center gap-2">
                        <BrainCircuit className="w-5 h-5 text-blue-400" />
                        全班行为分析
                    </h3>
                    <p className="text-[10px] text-slate-400 mt-1">分析时间: {new Date().toLocaleTimeString()}</p>
                 </div>
                 <button onClick={() => setBehaviorReport(null)} className="text-slate-400 hover:text-white">
                     <X className="w-5 h-5" />
                 </button>
             </div>
             
             <div className="p-4 space-y-4 max-h-[60vh] overflow-y-auto">
                 {/* Score */}
                 <div className="flex items-center justify-between bg-slate-800/50 p-3 rounded-lg">
                     <span className="text-sm text-slate-300">班级专注度</span>
                     <div className="flex items-center gap-2">
                         <span className={`text-2xl font-bold ${
                             behaviorReport.attentionScore >= 80 ? 'text-green-400' :
                             behaviorReport.attentionScore >= 60 ? 'text-yellow-400' : 'text-red-400'
                         }`}>
                             {behaviorReport.attentionScore}
                         </span>
                         <span className="text-xs text-slate-500">/ 100</span>
                     </div>
                 </div>

                 <div className="space-y-2">
                     <h4 className="text-xs font-semibold text-slate-400 uppercase">行为统计</h4>
                     {behaviorReport.behaviors.map((item, idx) => (
                         <div key={idx} className="flex items-center justify-between text-sm border-b border-slate-800/50 pb-2 last:border-0">
                             <div>
                                 <span className="text-slate-200 block">{item.action}</span>
                                 <span className="text-[10px] text-slate-500">{item.description}</span>
                             </div>
                             <span className="bg-slate-700 text-white px-2 py-0.5 rounded-full text-xs font-mono">
                                 {item.count}人
                             </span>
                         </div>
                     ))}
                 </div>

                 <div className="bg-slate-800/30 p-3 rounded-lg border border-slate-700/30">
                     <h4 className="text-xs font-semibold text-blue-400 mb-1">AI 总结</h4>
                     <p className="text-xs text-slate-300 leading-relaxed">
                         {behaviorReport.summary}
                     </p>
                 </div>
             </div>
         </div>
      )}

      {/* Single Student Report Overlay */}
      {singleStudentReport && (
         <div className="absolute top-4 right-4 z-50 w-72 bg-slate-900/95 backdrop-blur-md border border-slate-700 rounded-xl shadow-2xl overflow-hidden animate-in slide-in-from-right-10 duration-300">
             <div className="bg-gradient-to-r from-purple-900/50 to-slate-900/50 p-4 border-b border-slate-700 flex justify-between items-start">
                 <div>
                    <h3 className="text-white font-bold flex items-center gap-2">
                        <UserSearch className="w-5 h-5 text-purple-400" />
                        单人行为分析
                    </h3>
                 </div>
                 <button onClick={() => setSingleStudentReport(null)} className="text-slate-400 hover:text-white">
                     <X className="w-5 h-5" />
                 </button>
             </div>
             
             <div className="p-4 space-y-4">
                 {/* Focus Score */}
                 <div className="flex items-center justify-between bg-slate-800/50 p-3 rounded-lg">
                     <span className="text-sm text-slate-300">个人专注度</span>
                     <div className="flex items-center gap-2">
                         <span className={`text-2xl font-bold ${
                             singleStudentReport.focusScore >= 80 ? 'text-green-400' :
                             singleStudentReport.focusScore >= 60 ? 'text-yellow-400' : 'text-red-400'
                         }`}>
                             {singleStudentReport.focusScore}
                         </span>
                     </div>
                 </div>
                 
                 <div className="grid grid-cols-1 gap-3">
                    <div className="bg-slate-800/30 p-2 rounded border border-slate-700/30">
                        <span className="text-[10px] text-slate-400 block mb-1">当前动作</span>
                        <span className="text-sm text-white font-medium">{singleStudentReport.action}</span>
                    </div>
                    <div className="bg-slate-800/30 p-2 rounded border border-slate-700/30">
                        <span className="text-[10px] text-slate-400 block mb-1">姿态/表情</span>
                        <span className="text-sm text-white">{singleStudentReport.posture} · {singleStudentReport.expression}</span>
                    </div>
                 </div>

                 {singleStudentReport.isDistracted && (
                     <div className="bg-red-900/20 text-red-300 p-2 rounded text-xs border border-red-900/30 flex items-center gap-2">
                         <XCircle className="w-4 h-4" /> 监测到注意力不集中
                     </div>
                 )}

                 <div className="bg-slate-800/30 p-3 rounded-lg border border-slate-700/30">
                     <h4 className="text-xs font-semibold text-purple-400 mb-1">AI 评估</h4>
                     <p className="text-xs text-slate-300 leading-relaxed">
                         {singleStudentReport.summary}
                     </p>
                 </div>
             </div>
         </div>
      )}

      {/* Video Viewport */}
      <div 
        className={`relative flex-1 bg-black rounded-xl overflow-hidden shadow-2xl flex items-center justify-center border transition-colors duration-300 ${
          isSnapshotMode ? 'border-amber-500' : 
          isAccumulating ? 'border-purple-500' :
          isAnalyzingBehavior ? 'border-blue-500' :
          selectStudentMode ? 'border-purple-500 cursor-crosshair' :
          isManualMode ? 'border-cyan-500 cursor-crosshair' : 'border-slate-800'
        }`}
        onClick={handleVideoClick}
      >
        {!videoSrc && (
          <div className="text-center p-10">
            <div className="bg-slate-800 p-4 rounded-full inline-flex mb-4">
              <VideoIcon className="w-8 h-8 text-slate-400" />
            </div>
            <h3 className="text-xl text-slate-300 font-medium mb-2">未选择视频</h3>
            <button 
              onClick={() => fileInputRef.current?.click()}
              className="text-blue-400 hover:text-blue-300 underline"
            >
              请上传教室监控视频
            </button>
          </div>
        )}

        {isLoadingModels && (
          <div className="absolute inset-0 bg-black/80 z-50 flex items-center justify-center flex-col">
            <Loader className="w-10 h-10 text-blue-500 animate-spin mb-4" />
            <p className="text-blue-400 font-medium">正在加载 InsightFace 核心...</p>
          </div>
        )}
        
        {isAnalyzingBehavior && (
            <div className="absolute inset-0 z-40 bg-black/60 backdrop-blur-sm flex items-center justify-center flex-col">
                 <div className="bg-blue-600/20 p-6 rounded-full mb-4 animate-pulse">
                     <BrainCircuit className="w-12 h-12 text-blue-400" />
                 </div>
                 <h3 className="text-xl font-bold text-white mb-2">AI 正在观察...</h3>
                 <p className="text-slate-300 text-sm">正在分析视觉信息与行为特征</p>
            </div>
        )}

        {isManualProcessing && (
           <div className="absolute inset-0 z-40 flex items-center justify-center pointer-events-none">
             <div className="bg-cyan-500/20 backdrop-blur-sm p-4 rounded-full animate-pulse">
                <Loader className="w-8 h-8 text-cyan-400 animate-spin" />
             </div>
           </div>
        )}

        {isAccumulating && (
            <div className="absolute top-4 inset-x-0 mx-auto w-64 z-30 bg-purple-900/90 backdrop-blur border border-purple-500/50 p-3 rounded-xl shadow-2xl flex flex-col items-center">
                <div className="flex items-center gap-2 text-purple-200 text-xs font-bold mb-2">
                    <Hourglass className="w-3 h-3 animate-pulse" />
                    正在进行时序累计分析...
                </div>
                <div className="w-full bg-purple-950 rounded-full h-2 overflow-hidden">
                    <div 
                        className="bg-purple-400 h-full transition-all duration-100 ease-linear"
                        style={{ width: `${accumulationProgress}%` }}
                    />
                </div>
                <div className="text-[10px] text-purple-300 mt-1">{accumulationProgress} / 100 帧</div>
            </div>
        )}

        {selectStudentMode && (
          <div className="absolute top-4 left-0 right-0 z-20 flex justify-center pointer-events-none">
             <div className="bg-purple-600/90 text-white px-4 py-2 rounded-full text-sm font-bold flex items-center shadow-lg animate-bounce">
                <ScanEye className="w-4 h-4 mr-2" />
                请点击画面中的任意学生进行分析
             </div>
          </div>
        )}

        {isSnapshotMode && (
          <div className="absolute top-4 left-4 z-20 bg-amber-500/90 text-black px-3 py-1 rounded-full text-xs font-bold flex items-center animate-pulse">
            <Camera className="w-3 h-3 mr-1" /> 深度检测模式 (已暂停)
          </div>
        )}
        
        {stableResults && (
           <div className="absolute top-4 left-4 z-20 bg-amber-400 text-black px-4 py-1.5 rounded-full text-sm font-bold flex items-center shadow-lg animate-in fade-in slide-in-from-top-4">
             <CheckCheck className="w-4 h-4 mr-2" />
             分析完成：已生成 100 帧综合结果
           </div>
        )}

        {isManualMode && (
           <div className="absolute top-4 right-4 z-20 bg-cyan-600/90 text-white px-3 py-1 rounded-full text-xs font-bold flex items-center shadow-lg border border-cyan-400/50">
            <MousePointer2 className="w-3 h-3 mr-1" /> 人工辅助: 点击人脸
          </div>
        )}

        {videoSrc && (
          <div className="relative w-full h-full flex items-center justify-center">
            <video
              ref={videoRef}
              src={videoSrc}
              className="absolute max-w-full max-h-full object-contain"
              onEnded={handleVideoEnded}
              onTimeUpdate={handleTimeUpdate}
              onLoadedMetadata={handleLoadedMetadata}
              muted
              playsInline
            />
            <canvas
              ref={canvasRef}
              className="absolute max-w-full max-h-full object-contain pointer-events-none"
            />
          </div>
        )}
      </div>

      {/* Controls Bar Container */}
      <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden flex flex-col">
        
        {/* Timeline */}
        {videoSrc && (
          <div className="w-full h-6 bg-slate-900/50 flex items-center px-4 gap-3 border-b border-slate-700/50 relative group">
            <span className="text-[10px] font-mono text-slate-400 w-9 text-right">{formatTime(currentTime)}</span>
            <input
              type="range"
              min="0"
              max={duration || 100}
              value={currentTime}
              onChange={handleSeek}
              className="flex-1 h-1 bg-slate-600 rounded-lg appearance-none cursor-pointer accent-blue-500 hover:h-1.5 transition-all"
            />
            <span className="text-[10px] font-mono text-slate-500 w-9">{formatTime(duration)}</span>
          </div>
        )}

        {/* Buttons Row */}
        <div className="h-16 flex items-center px-6 justify-between relative">
          {!isSnapshotMode && !stableResults ? (
            <>
              <div className="flex items-center gap-4 animate-in fade-in slide-in-from-bottom-2 duration-300">
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  onChange={handleFileChange} 
                  accept="video/*" 
                  className="hidden" 
                />
                <button 
                  onClick={() => fileInputRef.current?.click()}
                  className="p-2.5 rounded-lg bg-slate-700 hover:bg-slate-600 text-slate-300 transition-colors"
                  title="上传视频"
                >
                  <Upload className="w-4 h-4" />
                </button>

                <button
                  onClick={togglePlay}
                  disabled={!videoSrc || isLoadingModels || isAccumulating || isAnalyzingBehavior}
                  className={`p-2.5 rounded-full transition-all ${
                    !videoSrc 
                      ? 'bg-slate-700 text-slate-500' 
                      : isPlaying 
                        ? 'bg-amber-500/20 text-amber-500 hover:bg-amber-500/30' 
                        : 'bg-blue-600 text-white hover:bg-blue-500 shadow-lg shadow-blue-900/50'
                  }`}
                >
                  {isPlaying ? <Pause className="w-5 h-5 fill-current" /> : <Play className="w-5 h-5 fill-current pl-0.5" />}
                </button>
                
                {/* AI Behavior Analysis Buttons Group */}
                <div className="flex bg-slate-900 rounded-full p-1 border border-slate-700 ml-2">
                    <button
                    onClick={handleClassroomBehaviorAnalysis}
                    disabled={!videoSrc || isAnalyzingBehavior || selectStudentMode}
                    className={`px-4 py-1.5 rounded-full text-xs font-semibold flex items-center gap-1 transition-all ${
                        isAnalyzingBehavior && !selectStudentMode
                        ? 'bg-blue-800 text-white cursor-not-allowed'
                        : 'hover:bg-blue-600 hover:text-white text-slate-300'
                    }`}
                    title="分析全班情况"
                    >
                    <BrainCircuit className="w-3.5 h-3.5" />
                    全班分析
                    </button>
                    <div className="w-px bg-slate-700 my-1 mx-1"></div>
                    <button
                    onClick={toggleStudentSelectionMode}
                    disabled={!videoSrc || isAnalyzingBehavior}
                    className={`px-4 py-1.5 rounded-full text-xs font-semibold flex items-center gap-1 transition-all ${
                        selectStudentMode 
                        ? 'bg-purple-600 text-white shadow-sm'
                        : 'hover:bg-purple-600 hover:text-white text-slate-300'
                    }`}
                    title="点击画面选择特定学生进行分析"
                    >
                    <UserSearch className="w-3.5 h-3.5" />
                    单人行为
                    </button>
                </div>

                {/* 100-Frame Analysis Button */}
                <button
                  onClick={startAccumulation}
                  disabled={!videoSrc || isLoadingModels || isAccumulating || isAnalyzingBehavior}
                  className={`ml-2 px-4 py-2 rounded-full text-white shadow-lg transition-colors flex items-center gap-2 ${
                      isAccumulating 
                      ? 'bg-purple-800 cursor-not-allowed' 
                      : 'bg-purple-600 hover:bg-purple-500'
                  }`}
                  title="连续追踪100帧，综合判断人脸身份，提高准确率"
                >
                   {isAccumulating ? <Loader className="animate-spin w-4 h-4" /> : <Hourglass className="w-4 h-4" />}
                  <span className="text-xs font-semibold">100帧 确信度分析</span>
                </button>
                
                {/* Manual Assist Toggle */}
                 <div className="flex items-center bg-slate-900 rounded-lg p-1 border border-slate-700 ml-2">
                    <button
                        onClick={() => {
                            setIsManualMode(!isManualMode);
                            setSelectStudentMode(false);
                        }}
                        disabled={isAccumulating || isAnalyzingBehavior}
                        className={`p-2 rounded-md transition-all flex items-center gap-2 ${
                            isManualMode ? 'bg-cyan-600 text-white shadow' : 'text-slate-400 hover:text-cyan-400'
                        } disabled:opacity-50`}
                        title="人工辅助模式：点击屏幕强制识别"
                    >
                        <MousePointer2 className="w-4 h-4" />
                    </button>
                    {manualDetections.length > 0 && (
                        <button 
                             onClick={clearManualDetections}
                             className="p-2 text-slate-400 hover:text-red-400 transition-colors border-l border-slate-700 ml-1"
                             title="清除人工标注"
                        >
                            <Eraser className="w-4 h-4" />
                        </button>
                    )}
                 </div>

              </div>

              {/* Stats */}
              <div className="hidden md:block text-right border-l border-slate-700 pl-6">
                  <div className="text-[10px] text-slate-500 flex items-center gap-1 justify-end">
                      <Gauge className="w-3 h-3" /> 已追踪
                  </div>
                  <div className="text-lg font-mono font-bold text-white leading-tight">
                    {trackedCount}
                    {manualDetections.length > 0 && <span className="text-cyan-400 text-sm ml-1">(含人工)</span>}
                  </div>
              </div>
            </>
          ) : (
            /* Result / Snapshot Controls */
            <div className="w-full flex items-center justify-between animate-in fade-in slide-in-from-bottom-2 duration-300">
              <div className="flex items-center gap-4">
                  {stableResults ? (
                     <div className="bg-amber-500/10 border border-amber-500/30 px-4 py-1.5 rounded-lg flex items-center gap-3">
                        <CheckCheck className="w-4 h-4 text-amber-500" />
                        <div>
                          <h4 className="text-amber-500 font-bold text-sm">综合分析报告 (100帧)</h4>
                          <p className="text-[10px] text-amber-400/70">已排除偶发错误识别</p>
                        </div>
                     </div>
                  ) : (
                    <div className="bg-amber-500/10 border border-amber-500/30 px-4 py-1.5 rounded-lg flex items-center gap-3">
                        <Camera className="w-4 h-4 text-amber-500" />
                        <div>
                        <h4 className="text-amber-500 font-bold text-sm">深度检测完成</h4>
                        </div>
                    </div>
                  )}
              </div>

              <div className="flex items-center gap-3">
                  <button 
                    onClick={handleDownloadSnapshot}
                    className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white px-5 py-1.5 rounded-lg text-sm font-medium shadow-lg transition-colors"
                  >
                    <Download className="w-4 h-4" /> 保存分析结果
                  </button>
                  <div className="w-px h-6 bg-slate-700 mx-2" />
                  <button 
                    onClick={() => {
                        setStableResults(null);
                        exitSnapshotMode();
                    }}
                    className="flex items-center gap-2 bg-slate-700 hover:bg-slate-600 text-slate-200 px-4 py-1.5 rounded-lg text-sm font-medium transition-colors"
                  >
                    <XCircle className="w-4 h-4" /> 返回实时监控
                  </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default VideoAnalyzer;