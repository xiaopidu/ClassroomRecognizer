import React, { useState } from 'react';
import { AppTab, RecognitionParams, Student } from './types';
import VideoAnalyzer from './components/VideoAnalyzer';
import StudentRegistry from './components/StudentRegistry';
import ImageAnalyzer from './components/ImageAnalyzer';
import ParameterControls from './components/ParameterControls';
import { Users, ScanFace, Activity, Image as ImageIcon } from 'lucide-react';

console.log('App.tsx: Starting execution');

const App: React.FC = () => {
  console.log('App.tsx: App component rendering');
  
  const [activeTab, setActiveTab] = useState<AppTab>(AppTab.ANALYZE);
  const [students, setStudents] = useState<Student[]>([]);
  
  // Adjusted Default parameters for InsightFace (Cosine Similarity)
  // - similarityThreshold: 0.92 (High cosine similarity for strict matching)
  // - minConfidence: 0.2 (Detection threshold)
  // - networkSize: 608 (High res input)
  const [recognitionParams, setRecognitionParams] = useState<RecognitionParams>({
    minConfidence: 0.2, 
    similarityThreshold: 0.92, 
    minFaceSize: 20, 
    iouThreshold: 0.3,
    maskMode: false,
    networkSize: 608 
  });

  const handleAddStudent = (student: Student) => {
    setStudents(prev => [...prev, student]);
  };

  const handleRemoveStudent = (id: string) => {
    setStudents(prev => prev.filter(s => s.id !== id));
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Header */}
      <header className="h-16 border-b border-slate-800 bg-slate-900/50 backdrop-blur-md sticky top-0 z-10">
        <div className="container mx-auto px-4 h-full flex items-center justify-between">
          <div className="flex items-center gap-2">
            <ScanFace className="w-8 h-8 text-blue-500" />
            <div>
              <h1 className="text-lg font-bold tracking-tight text-white leading-tight">智慧教室人脸考勤</h1>
              <p className="text-[10px] text-slate-400 uppercase tracking-widest font-semibold">InsightFace 核心算法 (ArcFace Compatible)</p>
            </div>
          </div>
          
          <nav className="flex bg-slate-800 p-1 rounded-lg">
            <button
              onClick={() => setActiveTab(AppTab.ANALYZE)}
              className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all flex items-center gap-2 ${
                activeTab === AppTab.ANALYZE 
                  ? 'bg-blue-600 text-white shadow-lg' 
                  : 'text-slate-400 hover:text-white hover:bg-slate-700'
              }`}
            >
              <Activity className="w-4 h-4" /> 实时监控
            </button>
            <button
              onClick={() => setActiveTab(AppTab.IMAGE_RECOGNITION)}
              className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all flex items-center gap-2 ${
                activeTab === AppTab.IMAGE_RECOGNITION 
                  ? 'bg-blue-600 text-white shadow-lg' 
                  : 'text-slate-400 hover:text-white hover:bg-slate-700'
              }`}
            >
              <ImageIcon className="w-4 h-4" /> 图片识别
            </button>
            <button
              onClick={() => setActiveTab(AppTab.REGISTER)}
              className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all flex items-center gap-2 ${
                activeTab === AppTab.REGISTER 
                  ? 'bg-blue-600 text-white shadow-lg' 
                  : 'text-slate-400 hover:text-white hover:bg-slate-700'
              }`}
            >
              <Users className="w-4 h-4" /> 学生管理
            </button>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 container mx-auto px-4 py-6 overflow-hidden flex flex-col">
        <div className="flex flex-col lg:flex-row gap-6 h-full flex-1">
          
          {/* Left Column: Main View */}
          <div className="flex-1 min-h-[500px] flex flex-col">
            {activeTab === AppTab.ANALYZE && (
               <VideoAnalyzer 
                 students={students} 
                 params={recognitionParams} 
               />
            )}
            {activeTab === AppTab.IMAGE_RECOGNITION && (
               <ImageAnalyzer 
                 students={students} 
                 params={recognitionParams} 
               />
            )}
            {activeTab === AppTab.REGISTER && (
               <StudentRegistry 
                 students={students} 
                 onAddStudent={handleAddStudent} 
                 onRemoveStudent={handleRemoveStudent}
               />
            )}
          </div>

          {/* Right Column: Parameters (Visible on Analyze and Image Recognition tabs) */}
          {(activeTab === AppTab.ANALYZE || activeTab === AppTab.IMAGE_RECOGNITION) && (
            <div className="w-full lg:w-80 flex-shrink-0">
               <ParameterControls 
                 params={recognitionParams} 
                 onChange={setRecognitionParams} 
               />
               
               <div className="mt-6 bg-slate-900/50 p-4 rounded-xl border border-slate-800">
                  <h4 className="text-sm font-semibold text-slate-300 mb-2">InsightFace 算法说明</h4>
                  <div className="space-y-2 text-xs text-slate-400">
                    <p>当前系统采用 <span className="text-purple-400 font-bold">ArcFace 余弦相似度 (Cosine Similarity)</span> 进行人脸特征比对，相比欧氏距离更适合大规模人脸库。</p>
                    <p>底层使用 ResNet-34 神经网络提取 128 维特征向量，与 InsightFace 工业级模型架构保持一致。</p>
                  </div>
               </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

export default App;