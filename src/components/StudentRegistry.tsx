import React, { useState, useRef } from 'react';
import { Student, ImageQualityReport } from '../types';
import { FaceRecognitionService } from '../services/faceRecognitionService';
import { validateStudentImage } from '../services/geminiService';
import { Camera, CheckCircle, AlertTriangle, Upload, X, Loader2, FolderUp, FileCheck, FileWarning, UserPlus, Layers, Wand2, Eye } from 'lucide-react';

interface StudentRegistryProps {
  students: Student[];
  onAddStudent: (student: Student) => void;
  onRemoveStudent: (id: string) => void;
}

interface BatchLog {
  fileName: string;
  status: 'pending' | 'success' | 'error';
  message: string;
}

const StudentRegistry: React.FC<StudentRegistryProps> = ({ students, onAddStudent, onRemoveStudent }) => {
  const [mode, setMode] = useState<'single' | 'batch'>('single');
  
  // View Image State
  const [viewImage, setViewImage] = useState<string | null>(null);

  // Single Mode State
  const [name, setName] = useState('');
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [qualityReport, setQualityReport] = useState<ImageQualityReport | null>(null);
  const [isRegistering, setIsRegistering] = useState(false);
  const [isStandardized, setIsStandardized] = useState(false);
  
  // Batch Mode State
  const [batchFiles, setBatchFiles] = useState<File[]>([]);
  const [batchLogs, setBatchLogs] = useState<BatchLog[]>([]);
  const [isBatchProcessing, setIsBatchProcessing] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const batchInputRef = useRef<HTMLInputElement>(null);

  /**
   * Helper: Smart Crop & Standardize
   * Uses canvas transforms to perfectly center the face and prevent aspect ratio distortion.
   */
  const createStandardizedImage = (img: HTMLImageElement, detectionBox: any): string => {
    const canvas = document.createElement('canvas');
    const size = 300; // Standard output size
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    
    if (!ctx) throw new Error("Could not get canvas context");

    // Fill background with a nice slate color (handles edges if image is rotated or small)
    ctx.fillStyle = "#1e293b"; 
    ctx.fillRect(0, 0, size, size);

    const { x, y, width, height } = detectionBox;

    // Calculate Face Center
    const cx = x + width / 2;
    const cy = y + height / 2;

    // Determine scale: Make face occupy 90% of the canvas width (Requested by user for accuracy)
    // Previously 55%. 90% removes almost all background.
    const desiredFaceWidth = size * 0.90;
    const scale = desiredFaceWidth / width;

    // --- Transform Logic ---
    // 1. Move canvas origin to center (size/2, size/2)
    // 2. Scale the context
    // 3. Move origin "back" by the face center coordinate
    // Result: The face center ends up at the canvas center
    ctx.translate(size / 2, size / 2);
    ctx.scale(scale, scale);
    ctx.translate(-cx, -cy);

    // REMOVED ALL FILTERS: Keep original color and lighting for best matching accuracy
    ctx.filter = 'none';
    
    // Draw the entire image (transform handles the cropping/positioning)
    ctx.drawImage(img, 0, 0);

    // Reset transform
    ctx.setTransform(1, 0, 0, 1, 0, 0);

    return canvas.toDataURL('image/jpeg', 0.95);
  };

  // --- Single Mode Handlers ---
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImageFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setQualityReport(null);
      setIsStandardized(false);
    }
  };

  const handleStandardize = async () => {
    if (!previewUrl) return;

    setIsAnalyzing(true);
    try {
      // 1. Load Image
      const img = document.createElement('img');
      img.src = previewUrl;
      await new Promise((resolve) => { img.onload = resolve });

      // 2. Detect Face to get Box
      // Use getInstance() to ensure models are loaded
      const detection = await FaceRecognitionService.getInstance().getFaceDetection(img);
      
      if (!detection) {
        alert("未检测到人脸，或人脸过于模糊，无法标准化。请尝试更换照片。");
        setIsAnalyzing(false);
        return;
      }

      // 3. Process Image (Crop/Resize/Enhance)
      const standardizedDataUrl = createStandardizedImage(img, detection.detection.box);
      setPreviewUrl(standardizedDataUrl);
      setIsStandardized(true);

      // 4. Run Quality Check on the NEW Standardized Image
      const report = await validateStudentImage(standardizedDataUrl);
      setQualityReport(report);

    } catch (err) {
      console.error(err);
      alert("标准化处理失败。");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleRegister = async () => {
    if (!name || !previewUrl || (qualityReport && !qualityReport.isValid)) return;
    
    setIsRegistering(true);
    try {
      const img = document.createElement('img');
      img.src = previewUrl;
      await new Promise((resolve) => { img.onload = resolve });

      // We re-calculate descriptor on the standardized image to ensure the DB has the "clean" version
      const descriptor = await FaceRecognitionService.getInstance().getFaceDescriptor(img);
      
      if (descriptor) {
        const newStudent: Student = {
          id: Date.now().toString(),
          name: name.trim(),
          photoUrl: previewUrl, // Save the standardized image URL
          descriptors: [descriptor],
          createdAt: Date.now()
        };
        onAddStudent(newStudent);
        setName('');
        setImageFile(null);
        setPreviewUrl(null);
        setQualityReport(null);
        setIsStandardized(false);
        if (fileInputRef.current) fileInputRef.current.value = '';
      } else {
        alert("注册失败：无法从处理后的图像中提取特征。");
      }
    } catch (err) {
      console.error(err);
      alert("注册失败。");
    } finally {
      setIsRegistering(false);
    }
  };

  // --- Batch Mode Handlers ---
  const handleBatchSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const files = Array.from(e.target.files) as File[];
      setBatchFiles(files);
      // Initialize logs
      setBatchLogs(files.map(f => ({
        fileName: f.name,
        status: 'pending',
        message: '等待中...'
      })));
    }
  };

  const processBatch = async () => {
    if (batchFiles.length === 0) return;
    setIsBatchProcessing(true);

    const service = FaceRecognitionService.getInstance();
    
    // Process sequentially
    for (let i = 0; i < batchFiles.length; i++) {
      const file = batchFiles[i];
      const studentName = file.name.replace(/\.[^/.]+$/, ""); // Remove extension

      // Update log
      setBatchLogs(prev => {
        const newLogs = [...prev];
        newLogs[i] = { ...newLogs[i], message: '正在处理与注册...', status: 'pending' };
        return newLogs;
      });

      // Yield for UI
      await new Promise(r => setTimeout(r, 50));

      try {
        const url = URL.createObjectURL(file);
        const img = document.createElement('img');
        img.src = url;
        await new Promise((resolve, reject) => {
           img.onload = resolve;
           img.onerror = reject;
        });

        // 1. Detect
        const detection = await service.getFaceDetection(img);

        if (detection) {
           // 2. Standardize (Crop/Resize)
           const standardizedUrl = createStandardizedImage(img, detection.detection.box);
           
           // 3. Get Descriptor from Standardized Image
           const stdImg = document.createElement('img');
           stdImg.src = standardizedUrl;
           await new Promise((resolve) => { stdImg.onload = resolve });
           const descriptor = await service.getFaceDescriptor(stdImg);

           if (descriptor) {
             const newStudent: Student = {
              id: Date.now().toString() + Math.random().toString(36).substr(2, 5),
              name: studentName,
              photoUrl: standardizedUrl,
              descriptors: [descriptor],
              createdAt: Date.now()
            };
            onAddStudent(newStudent);
            
            setBatchLogs(prev => {
              const newLogs = [...prev];
              newLogs[i] = { ...newLogs[i], status: 'success', message: '已注册 (自动处理)' };
              return newLogs;
            });
           } else {
             throw new Error("特征提取失败");
           }
        } else {
           setBatchLogs(prev => {
            const newLogs = [...prev];
            newLogs[i] = { ...newLogs[i], status: 'error', message: '未检测到人脸' };
            return newLogs;
          });
        }
      } catch (error) {
         setBatchLogs(prev => {
            const newLogs = [...prev];
            newLogs[i] = { ...newLogs[i], status: 'error', message: '处理失败' };
            return newLogs;
          });
      }
    }
    setIsBatchProcessing(false);
  };

  return (
    <>
      {/* Full Screen Image Viewer */}
      {viewImage && (
        <div 
          className="fixed inset-0 z-[100] flex items-center justify-center bg-black/90 backdrop-blur-sm p-4 animate-in fade-in duration-200"
          onClick={() => setViewImage(null)}
        >
          <button className="absolute top-4 right-4 text-white/50 hover:text-white transition-colors">
            <X className="w-8 h-8" />
          </button>
          <img 
            src={viewImage} 
            alt="Full view" 
            className="max-w-full max-h-full object-contain rounded-lg shadow-2xl ring-1 ring-white/10" 
            onClick={(e) => e.stopPropagation()} 
          />
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full">
        {/* Registration Column */}
        <div className="lg:col-span-1 bg-slate-800 p-6 rounded-xl border border-slate-700 h-fit flex flex-col">
          <h2 className="text-xl font-bold mb-4 flex items-center">
            <Camera className="mr-2 text-purple-400" /> 学生注册
          </h2>

          {/* Mode Toggle */}
          <div className="flex bg-slate-900 p-1 rounded-lg mb-6">
            <button
              onClick={() => setMode('single')}
              className={`flex-1 py-2 text-sm font-medium rounded-md flex items-center justify-center gap-2 transition-all ${
                mode === 'single' ? 'bg-slate-700 text-white shadow' : 'text-slate-400 hover:text-white'
              }`}
            >
              <UserPlus className="w-4 h-4" /> 单人注册
            </button>
            <button
              onClick={() => setMode('batch')}
              className={`flex-1 py-2 text-sm font-medium rounded-md flex items-center justify-center gap-2 transition-all ${
                mode === 'batch' ? 'bg-slate-700 text-white shadow' : 'text-slate-400 hover:text-white'
              }`}
            >
              <Layers className="w-4 h-4" /> 批量导入
            </button>
          </div>

          {mode === 'single' ? (
            /* --- SINGLE MODE --- */
            <div className="space-y-4 animate-in fade-in duration-300">
              <div>
                <label className="block text-sm text-slate-400 mb-1">学生姓名</label>
                <input 
                  type="text" 
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-white focus:border-purple-500 focus:outline-none"
                  placeholder="例如：张三"
                />
              </div>

              <div>
                <label className="block text-sm text-slate-400 mb-1">基准照片</label>
                <div 
                  className={`border-2 border-dashed border-slate-600 rounded-lg p-4 flex flex-col items-center justify-center relative h-48 transition-colors ${!previewUrl ? 'cursor-pointer hover:bg-slate-750 hover:border-slate-500' : ''}`}
                  onClick={() => !previewUrl && fileInputRef.current?.click()}
                >
                  {previewUrl ? (
                    <div className="relative w-full h-full group">
                      <img 
                        src={previewUrl} 
                        alt="Preview" 
                        className="w-full h-full object-contain rounded cursor-zoom-in" 
                        onClick={(e) => { e.stopPropagation(); setViewImage(previewUrl); }}
                      />
                      {/* Change Image Button Overlay */}
                      <button
                        onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }}
                        className="absolute top-2 right-2 p-2 bg-slate-900/80 hover:bg-blue-600 text-white rounded-full opacity-0 group-hover:opacity-100 transition-all border border-slate-600 shadow-xl z-10"
                        title="更换照片"
                      >
                        <Upload className="w-4 h-4" />
                      </button>
                      
                      {isStandardized && <span className="absolute bottom-2 right-2 px-2 py-1 bg-black/60 text-xs rounded text-white font-mono pointer-events-none">已标准化</span>}
                    </div>
                  ) : (
                    <>
                      <Upload className="text-slate-500 mb-2" />
                      <span className="text-sm text-slate-500">点击上传照片</span>
                    </>
                  )}
                </div>
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  onChange={handleFileChange} 
                  accept="image/*" 
                  className="hidden" 
                />
              </div>

              <div className="flex gap-2">
                <button
                  onClick={handleStandardize}
                  disabled={!previewUrl || isAnalyzing}
                  className={`flex-1 text-white py-2 rounded font-medium text-sm transition-colors flex justify-center items-center ${
                    isStandardized ? 'bg-slate-600' : 'bg-blue-600 hover:bg-blue-500'
                  } disabled:bg-slate-700`}
                >
                  {isAnalyzing ? <Loader2 className="animate-spin w-4 h-4 mr-1"/> : (
                    <>
                      <Wand2 className="w-4 h-4 mr-1" />
                      {isStandardized ? "重新处理" : "一键标准化 (90%)"}
                    </>
                  )}
                </button>
                <button
                  onClick={handleRegister}
                  disabled={!previewUrl || !name || isRegistering}
                  className="flex-1 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-700 disabled:text-slate-500 text-white py-2 rounded font-medium text-sm transition-colors flex justify-center items-center"
                >
                  {isRegistering ? <Loader2 className="animate-spin w-4 h-4 mr-1"/> : "确认注册"}
                </button>
              </div>

              {/* Quality Report Card */}
              {qualityReport && (
                <div className={`mt-4 p-3 rounded text-sm border ${qualityReport.isValid ? 'bg-emerald-900/30 border-emerald-800' : 'bg-red-900/30 border-red-800'}`}>
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-semibold">{qualityReport.isValid ? "检测通过" : "检测失败"}</span>
                    <span className="font-mono text-xs px-2 py-0.5 bg-black/40 rounded">{qualityReport.score}/100</span>
                  </div>
                  <ul className="space-y-1 text-xs opacity-90">
                    {qualityReport.issues.length > 0 ? (
                      qualityReport.issues.map((issue, idx) => (
                        <li key={idx} className="flex items-start">
                          <AlertTriangle className="w-3 h-3 mr-1 mt-0.5 flex-shrink-0" /> {issue}
                        </li>
                      ))
                    ) : (
                      <li className="flex items-center"><CheckCircle className="w-3 h-3 mr-1" /> 照片符合标准。</li>
                    )}
                  </ul>
                </div>
              )}
            </div>
          ) : (
            /* --- BATCH MODE --- */
            <div className="space-y-4 animate-in fade-in duration-300 flex-1 flex flex-col">
              <div 
                  onClick={() => batchInputRef.current?.click()}
                  className="border-2 border-dashed border-slate-600 rounded-lg p-6 flex flex-col items-center justify-center cursor-pointer hover:bg-slate-750 hover:border-slate-500 transition-colors bg-slate-900/50"
                >
                  <FolderUp className="text-slate-400 w-10 h-10 mb-2" />
                  <span className="text-sm font-medium text-slate-300">批量上传图片</span>
                  <span className="text-xs text-slate-500 mt-1">文件名将自动作为学生姓名，并自动裁剪标准化</span>
              </div>
              <input 
                type="file" 
                ref={batchInputRef} 
                onChange={handleBatchSelect} 
                accept="image/*" 
                multiple
                className="hidden" 
              />

              {batchFiles.length > 0 && (
                <div className="flex-1 flex flex-col min-h-[200px]">
                  <div className="flex justify-between items-center mb-2">
                     <span className="text-sm font-semibold text-slate-400">上传队列 ({batchFiles.length})</span>
                     <button 
                       onClick={processBatch}
                       disabled={isBatchProcessing}
                       className="bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 text-white text-xs px-3 py-1 rounded"
                     >
                       {isBatchProcessing ? "处理中..." : "开始批量处理"}
                     </button>
                  </div>
                  
                  <div className="flex-1 overflow-y-auto bg-slate-900 rounded-lg p-2 space-y-2 border border-slate-700">
                    {batchLogs.map((log, idx) => (
                      <div key={idx} className="flex items-center justify-between text-xs p-2 bg-slate-800 rounded border border-slate-700/50">
                        <span className="truncate max-w-[150px] text-slate-300" title={log.fileName}>{log.fileName}</span>
                        <div className="flex items-center gap-2">
                          <span className={`
                            ${log.status === 'success' ? 'text-emerald-400' : ''}
                            ${log.status === 'error' ? 'text-red-400' : ''}
                            ${log.status === 'pending' ? 'text-slate-500' : ''}
                          `}>
                            {log.status === 'success' && <FileCheck className="w-3 h-3" />}
                            {log.status === 'error' && <FileWarning className="w-3 h-3" />}
                            {log.status === 'pending' && log.message.includes('Processing') && <Loader2 className="w-3 h-3 animate-spin" />}
                          </span>
                          <span className="text-slate-500 w-24 text-right truncate">{log.message}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Roster List */}
        <div className="lg:col-span-2 bg-slate-800 p-6 rounded-xl border border-slate-700 overflow-hidden flex flex-col">
          <h2 className="text-xl font-bold mb-4">已注册学生名单 ({students.length})</h2>
          <div className="overflow-y-auto flex-1 pr-2 space-y-2">
            {students.length === 0 && (
              <p className="text-slate-500 text-center mt-10">暂无已注册学生。</p>
            )}
            {students.map(student => (
              <div key={student.id} className="flex items-center justify-between bg-slate-900 p-3 rounded-lg border border-slate-700">
                <div className="flex items-center gap-3">
                  <div className="relative group/avatar">
                    <img 
                      src={student.photoUrl} 
                      alt={student.name} 
                      className="w-12 h-12 rounded-full object-cover border border-slate-600 cursor-zoom-in transition-all" 
                      onClick={() => setViewImage(student.photoUrl)}
                    />
                    <div className="absolute inset-0 bg-black/30 rounded-full flex items-center justify-center opacity-0 group-hover/avatar:opacity-100 pointer-events-none">
                      <Eye className="w-4 h-4 text-white" />
                    </div>
                  </div>
                  <div>
                    <h3 className="font-medium text-white">{student.name}</h3>
                    <p className="text-xs text-slate-500">ID: {student.id.slice(-6)}</p>
                  </div>
                </div>
                <button 
                  onClick={() => onRemoveStudent(student.id)}
                  className="text-slate-500 hover:text-red-400 p-2"
                  title="移除学生"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>
    </>
  );
};

export default StudentRegistry;