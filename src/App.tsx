import React, { useState, useEffect } from 'react';
import { AppTab, RecognitionParams, Student } from './types';
import VideoAnalyzer from './components/VideoAnalyzer';
import StudentRegistry from './components/StudentRegistry';
import ImageAnalyzer from './components/ImageAnalyzer';
import ParameterControls from './components/ParameterControls';
import Pose2Analyzer from './components/pose2_PoseAnalyzer';
import Pose3VideoAnalyzer from './components/pose3_VideoAnalyzer';
import { Users, ScanFace, Activity, Image as ImageIcon, AlertTriangle, TestTube, Video } from 'lucide-react';
import { loadStudentsFromStorage, saveStudentsToStorage, loadParamsFromStorage, saveParamsToStorage } from './services/storageService';
import { getCurrentParams, updateRegisteredStudents } from './services/apiService';
console.log('App.tsx: Starting execution');

function App() {
  console.log('App.tsx: App component rendering');
  
  const [activeTab, setActiveTab] = useState<AppTab>(AppTab.ANALYZE);
  const [students, setStudents] = useState<Student[]>(() => {
    // Load students from localStorage on initial render
    return loadStudentsFromStorage();
  });
  
  // Optimized default parameters for classroom environment
  const [recognitionParams, setRecognitionParams] = useState<RecognitionParams>(() => {
    // Load parameters from localStorage or use optimized defaults
    const savedParams = loadParamsFromStorage();
    return savedParams || {
      minConfidence: 0.5, 
      similarityThreshold: 0.6, 
      minFaceSize: 20, 
      iouThreshold: 0.4,
      maskMode: false,
      networkSize: 640 
    };
  });
  
  // Snapshot state for passing data between tabs
  const [snapshotImageData, setSnapshotImageData] = useState<string | null>(null);

  // Save students to localStorage whenever they change
  useEffect(() => {
    saveStudentsToStorage(students);
  }, [students]);

  // Save parameters to localStorage whenever they change
  useEffect(() => {
    saveParamsToStorage(recognitionParams);
  }, [recognitionParams]);

  // Update backend with student data whenever it changes
  useEffect(() => {
    const updateBackendStudents = async () => {
      try {
        // Convert Float32Array descriptors to regular arrays for JSON serialization
        const studentsForBackend = students.map(student => ({
          ...student,
          descriptors: student.descriptors.map(desc => Array.from(desc))
        }));
        
        console.log('正在更新后端学生数据:', studentsForBackend);
        const result = await updateRegisteredStudents(studentsForBackend);
        console.log('后端学生数据更新结果:', result);
      } catch (error) {
        console.error('更新后端学生数据失败:', error);
      }
    };

    if (students.length > 0) {
      updateBackendStudents();
    }
  }, [students]);

  const handleAddStudent = (student: Student) => {
    setStudents(prev => [...prev, student]);
  };

  const handleRemoveStudent = (id: string) => {
    setStudents(prev => prev.filter(s => s.id !== id));
  };
  
  // Handle snapshot from video analyzer
  const handleVideoSnapshot = (imageData: string) => {
    const snapshotData = imageData || takeCurrentVideoSnapshot();
    if (snapshotData) {
      setSnapshotImageData(snapshotData);
      setActiveTab(AppTab.IMAGE_RECOGNITION);
    }
  };
  
  // Take snapshot from current video frame
  const takeCurrentVideoSnapshot = (): string | null => {
    // This would need to be implemented to get the current video frame
    // For now, we'll rely on the VideoAnalyzer component to provide the snapshot
    return null;
  };

  // 加载学生数据和参数
  useEffect(() => {
    const storedStudents = loadStudentsFromStorage();
    setStudents(storedStudents);
    
    const storedParams = loadParamsFromStorage();
    setRecognitionParams(storedParams);
    
    // 从后端获取当前参数
    fetchBackendParams();
  }, []);

  const fetchBackendParams = async () => {
    try {
      console.log('正在从后端获取参数...');
      const backendParams = await getCurrentParams();
      console.log('后端参数响应:', backendParams);
      
      if (backendParams.success) {
        // 更新本地参数
        const newParams: RecognitionParams = {
          ...recognitionParams,
          minConfidence: backendParams.params.min_confidence,
          networkSize: backendParams.params.network_size,
          minFaceSize: backendParams.params.min_face_size
        };
        setRecognitionParams(newParams);
        saveParamsToStorage(newParams);
        console.log('参数已更新到本地存储');
      } else {
        console.error('后端参数获取失败');
      }
    } catch (error) {
      console.error('获取后端参数失败:', error);
    }
  };

  const handleParamsChange = (newParams: RecognitionParams) => {
    setRecognitionParams(newParams);
    saveParamsToStorage(newParams);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
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
              <Activity className="w-4 h-4" /> 行为分析
            </button>
            <button
              onClick={() => setActiveTab(AppTab.POSE_TEST)}
              className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all flex items-center gap-2 ${
                activeTab === AppTab.POSE_TEST 
                  ? 'bg-blue-600 text-white shadow-lg' 
                  : 'text-slate-400 hover:text-white hover:bg-slate-700'
              }`}
            >
              <TestTube className="w-4 h-4" /> 行为分析（标定）
            </button>
            <button
              onClick={() => setActiveTab(AppTab.POSE_VIDEO)}
              className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all flex items-center gap-2 ${
                activeTab === AppTab.POSE_VIDEO 
                  ? 'bg-blue-600 text-white shadow-lg' 
                  : 'text-slate-400 hover:text-white hover:bg-slate-700'
              }`}
            >
              <Video className="w-4 h-4" /> 行为分析（视频）
            </button>
            <button
              onClick={() => setActiveTab(AppTab.IMAGE_RECOGNITION)}
              className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all flex items-center gap-2 ${
                activeTab === AppTab.IMAGE_RECOGNITION 
                  ? 'bg-blue-600 text-white shadow-lg' 
                  : 'text-slate-400 hover:text-white hover:bg-slate-700'
              }`}
            >
              <ImageIcon className="w-4 h-4" /> 人脸识别
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

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
        {/* 添加后端服务提示 */}
        <div className="bg-amber-900/30 border border-amber-700 rounded-lg p-4 mb-6">
          <div className="flex items-start">
            <AlertTriangle className="h-5 w-5 text-amber-400 mr-2 mt-0.5 flex-shrink-0" />
            <div>
              <h3 className="text-amber-200 font-medium">后端服务要求</h3>
              <p className="text-amber-100 text-sm mt-1">
                本应用需要启动后端 Python 服务才能进行人脸检测。请确保后端服务已在 http://localhost:5001 运行。
                <br />
                启动方法：进入 backend 目录，运行 ./start.sh 或按照 BACKEND_SETUP.md 文档操作。
              </p>
            </div>
          </div>
        </div>
        
        <div className={`grid grid-cols-1 gap-6 ${
          activeTab === AppTab.ANALYZE ? 'xl:grid-cols-4' : ''
        }`}>
          {/* Left Column: Main View */}
          <div className={`flex-1 min-h-[500px] grid grid-cols-1 gap-6 ${
            activeTab === AppTab.ANALYZE ? 'xl:col-span-3' : ''
          }`}>
            {activeTab === AppTab.ANALYZE && (
               <VideoAnalyzer 
                 students={students} 
                 params={recognitionParams} 
                 onSnapshotTaken={handleVideoSnapshot}
               />
            )}
            {activeTab === AppTab.POSE_TEST && (
               <Pose2Analyzer />
            )}
            {activeTab === AppTab.POSE_VIDEO && (
               <Pose3VideoAnalyzer />
            )}
            {activeTab === AppTab.IMAGE_RECOGNITION && (
               <ImageAnalyzer 
                 students={students} 
                 params={recognitionParams} 
                 snapshotImageData={snapshotImageData}
                 onSnapshotFromVideo={handleVideoSnapshot}
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

          {/* Right Column: Parameters (Visible only on Analyze tab) */}
          {activeTab === AppTab.ANALYZE && (
            <div className="xl:col-span-1 w-full flex-shrink-0">
               <ParameterControls 
                 params={recognitionParams} 
                 onChange={handleParamsChange} 
               />
               
               <div className="mt-6 bg-slate-900/50 p-4 rounded-xl border border-slate-800">
                  <h4 className="text-sm font-semibold text-slate-300 mb-2">InsightFace 算法说明</h4>
                  <div className="space-y-2 text-xs text-slate-400">
                    <p>当前系统采用 <span className="text-purple-400 font-bold">ArcFace 余弦相似度 (Cosine Similarity)</span> 进行人脸特征比对，相比欧氏距离更适合大规模人脸库。</p>
                    <p>底层使用深度卷积神经网络提取 512 维特征向量，与 InsightFace 工业级模型架构保持一致。</p>
                  </div>
               </div>            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;