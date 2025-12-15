import React from 'react';
import { RecognitionParams } from '../types';
import { Settings, Eye, Maximize, GitMerge, Shield, Cpu, BrainCircuit } from 'lucide-react';

interface ParameterControlsProps {
  params: RecognitionParams;
  onChange: (newParams: RecognitionParams) => void;
}

const ParameterControls: React.FC<ParameterControlsProps> = ({ params, onChange }) => {
  const handleChange = (key: keyof RecognitionParams, value: number | boolean) => {
    onChange({ ...params, [key]: value });
  };

  return (
    <div className="bg-slate-800 p-4 rounded-xl border border-slate-700 shadow-lg">
      <h3 className="text-lg font-semibold text-white flex items-center mb-4">
        <Settings className="w-5 h-5 mr-2 text-blue-400" />
        识别算法参数 (InsightFace)
      </h3>
      
      <div className="space-y-6">
        {/* Network Size (Upscaling / Accuracy Control) */}
        <div>
          <div className="flex justify-between text-sm text-slate-300 mb-1">
             <span className="flex items-center" title="决定了AI‘看’图的清晰度"><Cpu className="w-3 h-3 mr-1"/> 分析分辨率 (Input Size)</span>
             <span className="font-mono text-cyan-400">{params.networkSize}px</span>
          </div>
          <select 
            value={params.networkSize}
            onChange={(e) => handleChange('networkSize', parseInt(e.target.value))}
            className="w-full bg-slate-900 border border-slate-700 text-slate-200 text-xs rounded p-2 focus:border-cyan-500 outline-none"
          >
            <option value={224}>224 - 极速 (实时预览)</option>
            <option value={320}>320 - 快速</option>
            <option value={416}>416 - 平衡</option>
            <option value={512}>512 - 高精度</option>
            <option value={608}>608 - InsightFace 推荐 (教室场景)</option>
            <option value={800}>800 - 极限精度 (远距离)</option>
          </select>
          <p className="text-xs text-slate-500 mt-1">
            <span className="text-cyan-400">数值越大</span> = 相当于对人脸进行“放大”处理，能识别更远处、更模糊的人脸，但速度变慢。
          </p>
        </div>

        {/* Mask Optimization Mode */}
        <div className="bg-slate-900/50 p-3 rounded-lg border border-slate-700/50">
           <div className="flex items-center justify-between mb-2">
             <div className="flex items-center gap-2">
               <Shield className={`w-4 h-4 ${params.maskMode ? 'text-green-400' : 'text-slate-500'}`} />
               <span className="text-sm font-medium text-slate-200">戴口罩增强优化</span>
             </div>
             <label className="relative inline-flex items-center cursor-pointer">
                <input 
                  type="checkbox" 
                  checked={params.maskMode}
                  onChange={(e) => handleChange('maskMode', e.target.checked)}
                  className="sr-only peer" 
                />
                <div className="w-9 h-5 bg-slate-700 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-blue-800 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-green-600"></div>
             </label>
           </div>
           <p className="text-xs text-slate-500">
             开启后将放宽余弦相似度匹配阈值，并重点关注眼部特征。
           </p>
        </div>

        {/* Min Confidence */}
        <div>
          <div className="flex justify-between text-sm text-slate-300 mb-1">
            <span className="flex items-center"><Eye className="w-3 h-3 mr-1"/> 人脸检测阈值 (Det. Score)</span>
            <span className="font-mono text-blue-400">{params.minConfidence.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="0.05"
            max="0.9"
            step="0.05"
            value={params.minConfidence}
            onChange={(e) => handleChange('minConfidence', parseFloat(e.target.value))}
            className="w-full h-2 bg-slate-600 rounded-lg appearance-none cursor-pointer accent-blue-500"
          />
          <p className="text-xs text-slate-500 mt-1">
            <span className="text-yellow-500 font-bold">越低</span> = 能检测到更模糊的人脸 (建议 0.2)。
          </p>
        </div>

        {/* Similarity Threshold (Replaced Distance) */}
        <div className={params.maskMode ? "opacity-50 grayscale pointer-events-none" : ""}>
          <div className="flex justify-between text-sm text-slate-300 mb-1">
            <span className="flex items-center"><BrainCircuit className="w-3 h-3 mr-1"/> 相似度阈值 (Cosine Sim)</span>
            <span className="font-mono text-purple-400">{(params.similarityThreshold * 100).toFixed(0)}%</span>
          </div>
          <input
            type="range"
            min="0.4"
            max="0.99"
            step="0.01"
            value={params.similarityThreshold}
            onChange={(e) => handleChange('similarityThreshold', parseFloat(e.target.value))}
            className="w-full h-2 bg-slate-600 rounded-lg appearance-none cursor-pointer accent-purple-500"
          />
          <p className="text-xs text-slate-500 mt-1">
             InsightFace (ArcFace) 标准：<br/>
             <span className="text-purple-400">数值越大越严格</span>。建议 88%-93%。
          </p>
        </div>

        {/* Min Face Size */}
         <div>
          <div className="flex justify-between text-sm text-slate-300 mb-1">
            <span className="flex items-center"><Maximize className="w-3 h-3 mr-1"/> 最小人脸像素 (px)</span>
            <span className="font-mono text-emerald-400">{params.minFaceSize}px</span>
          </div>
          <input
            type="range"
            min="10"
            max="200"
            step="10"
            value={params.minFaceSize}
            onChange={(e) => handleChange('minFaceSize', parseInt(e.target.value))}
            className="w-full h-2 bg-slate-600 rounded-lg appearance-none cursor-pointer accent-emerald-500"
          />
          <p className="text-xs text-slate-500 mt-1">
             针对教室后排学生，建议设为 20px。
          </p>
        </div>
      </div>
    </div>
  );
};

export default ParameterControls;