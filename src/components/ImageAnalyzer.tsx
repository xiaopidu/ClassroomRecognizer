import React, { useState, useRef, useEffect } from 'react';
import { RecognitionParams, Student } from '../types';
import { FaceRecognitionService } from '../services/faceRecognitionService';
import { enhanceImage } from '../services/geminiService';
import { Upload, Image as ImageIcon, Loader2, XCircle, FileImage, Wand2, Sparkles, Eraser, ScanFace } from 'lucide-react';

interface ImageAnalyzerProps {
  students: Student[];
  params: RecognitionParams;
}

const ImageAnalyzer: React.FC<ImageAnalyzerProps> = ({ students, params }) => {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isEnhancing, setIsEnhancing] = useState(false);
  const [detectionsCount, setDetectionsCount] = useState(0);

  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Re-run analysis if parameters change or after enhancement finishes
  useEffect(() => {
    // Only analyze if the image is actually loaded (complete) to avoid errors
    if (imageUrl && !isProcessing && !isEnhancing && imageRef.current?.complete) {
      analyzeImage();
    }
  }, [params, students, imageUrl, isEnhancing]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImageFile(file);
      const url = URL.createObjectURL(file);
      setImageUrl(url);
      setDetectionsCount(0);
    }
  };

  const analyzeImage = async () => {
    if (!imageRef.current || !canvasRef.current || !imageUrl) return;

    setIsProcessing(true);
    try {
        // Ensure models are ready
        const service = FaceRecognitionService.getInstance();
        await service.loadModels();
        service.updateFaceMatcher(students, params);

        // Analyze
        const detections = await service.detectAndRecognizeImage(
            imageRef.current,
            canvasRef.current,
            params
        );
        setDetectionsCount(detections.length);
    } catch (err) {
        console.error("Image analysis failed", err);
    } finally {
        setIsProcessing(false);
    }
  };

  const handleEnhance = async () => {
    if (!imageUrl) return;
    setIsEnhancing(true);
    
    try {
        // Get blob from current URL
        const response = await fetch(imageUrl);
        const blob = await response.blob();
        
        // Convert to Base64
        const reader = new FileReader();
        reader.readAsDataURL(blob);
        
        reader.onloadend = async () => {
            const base64 = reader.result as string;
            try {
                const enhancedBase64 = await enhanceImage(base64);
                if (enhancedBase64) {
                    setImageUrl(enhancedBase64);
                    // useEffect will trigger analyzeImage when isEnhancing becomes false
                } else {
                    alert("图片增强失败：AI 未返回有效的图像数据。");
                }
            } catch (e) {
                console.error(e);
                alert("AI 增强服务调用失败，请稍后重试。");
            } finally {
                setIsEnhancing(false);
            }
        };
    } catch (e) {
        console.error("Failed to read image data", e);
        setIsEnhancing(false);
        alert("读取当前图片数据失败。");
    }
  };

  const handleClearAnnotations = () => {
    if (canvasRef.current) {
        const ctx = canvasRef.current.getContext('2d');
        if (ctx) {
            ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        }
    }
  };

  const handleImageLoad = () => {
    if (!isEnhancing) {
        analyzeImage();
    }
  };

  const clearImage = () => {
    setImageFile(null);
    setImageUrl(null);
    setDetectionsCount(0);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div className="flex flex-col h-full gap-4">
      <div 
        className="relative flex-1 bg-black rounded-xl overflow-hidden shadow-2xl flex items-center justify-center border border-slate-800"
      >
        {!imageUrl ? (
          <div className="text-center p-10">
            <div className="bg-slate-800 p-4 rounded-full inline-flex mb-4">
              <ImageIcon className="w-8 h-8 text-slate-400" />
            </div>
            <h3 className="text-xl text-slate-300 font-medium mb-2">未选择图片</h3>
            <button 
              onClick={() => fileInputRef.current?.click()}
              className="text-blue-400 hover:text-blue-300 underline"
            >
              请上传静态图片进行分析
            </button>
            <input 
              type="file" 
              ref={fileInputRef} 
              onChange={handleFileChange} 
              accept="image/*" 
              className="hidden" 
            />
          </div>
        ) : (
          <div className="relative w-full h-full flex items-center justify-center bg-slate-900">
             <img 
               ref={imageRef}
               src={imageUrl} 
               alt="Analysis Target" 
               className="max-w-full max-h-full object-contain"
               onLoad={handleImageLoad}
             />
             <canvas 
               ref={canvasRef}
               className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 object-contain pointer-events-none"
               style={{
                   width: imageRef.current ? imageRef.current.width : 'auto',
                   height: imageRef.current ? imageRef.current.height : 'auto',
                   maxWidth: '100%',
                   maxHeight: '100%'
               }}
             />

             {/* Overlay Loading State */}
             {(isProcessing || isEnhancing) && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/40 backdrop-blur-sm z-10">
                    <div className="bg-slate-900/90 text-white px-5 py-4 rounded-lg flex flex-col items-center gap-3 shadow-xl border border-slate-700">
                        {isEnhancing ? (
                            <>
                                <Sparkles className="w-8 h-8 animate-pulse text-purple-400" />
                                <div className="text-center">
                                    <p className="font-medium text-purple-100">AI 正在增强画质...</p>
                                    <p className="text-xs text-purple-300/70 mt-1">去噪 · 锐化 · 超分辨率</p>
                                </div>
                            </>
                        ) : (
                            <>
                                <Loader2 className="w-6 h-6 animate-spin text-blue-500" />
                                <span>正在分析人脸特征...</span>
                            </>
                        )}
                    </div>
                </div>
             )}
          </div>
        )}
      </div>

      {/* Control Bar */}
      <div className="bg-slate-800 rounded-xl border border-slate-700 h-16 flex items-center px-6 justify-between overflow-x-auto">
          <div className="flex items-center gap-3">
            <button 
                onClick={() => fileInputRef.current?.click()}
                className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white px-3 py-2 rounded-lg text-sm font-medium transition-colors whitespace-nowrap"
                disabled={isEnhancing}
            >
                <Upload className="w-4 h-4" /> 上传
            </button>
            
            {imageUrl && (
                <>
                    <button 
                        onClick={handleEnhance}
                        disabled={isEnhancing || isProcessing}
                        className="flex items-center gap-2 bg-purple-600 hover:bg-purple-500 disabled:bg-slate-700 disabled:text-slate-500 text-white px-3 py-2 rounded-lg text-sm font-medium transition-colors shadow-lg shadow-purple-900/20 whitespace-nowrap"
                        title="使用 AI 提高图片清晰度，有助于识别模糊人脸"
                    >
                        {isEnhancing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Wand2 className="w-4 h-4" />}
                        AI 增强
                    </button>

                    <div className="w-px h-6 bg-slate-700 mx-1" />

                    <button 
                        onClick={analyzeImage}
                        disabled={isEnhancing || isProcessing}
                        className="flex items-center gap-2 bg-slate-700 hover:bg-slate-600 text-slate-200 px-3 py-2 rounded-lg text-sm font-medium transition-colors whitespace-nowrap"
                        title="重新运行人脸识别算法"
                    >
                        <ScanFace className="w-4 h-4" /> 识别
                    </button>

                    <button 
                        onClick={handleClearAnnotations}
                        disabled={isEnhancing || isProcessing}
                        className="flex items-center gap-2 bg-slate-700 hover:bg-slate-600 text-slate-200 px-3 py-2 rounded-lg text-sm font-medium transition-colors whitespace-nowrap"
                        title="清除画面上的文字和框"
                    >
                        <Eraser className="w-4 h-4" /> 清除标注
                    </button>

                    <div className="w-px h-6 bg-slate-700 mx-1" />
                    
                    <button 
                        onClick={clearImage}
                        disabled={isEnhancing}
                        className="flex items-center gap-2 bg-red-900/50 hover:bg-red-900 text-red-200 border border-red-900 px-3 py-2 rounded-lg text-sm font-medium transition-colors whitespace-nowrap"
                        title="移除当前图片"
                    >
                        <XCircle className="w-4 h-4" /> 移除
                    </button>
                </>
            )}
          </div>

          <div className="text-right pl-4 hidden md:block">
              <div className="text-[10px] text-slate-500 flex items-center gap-1 justify-end">
                  <FileImage className="w-3 h-3" /> 识别统计
              </div>
              <div className="text-lg font-mono font-bold text-white leading-tight">
                  {detectionsCount} <span className="text-sm font-normal text-slate-400">人脸</span>
              </div>
          </div>
      </div>
    </div>
  );
};

export default ImageAnalyzer;