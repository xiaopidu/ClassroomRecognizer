import { RecognitionParams, Student } from '../types';

// Use local models instead of CDN
const MODEL_URLS = [
  '/models',  // Local models in public/models directory
];

export class FaceRecognitionService {
  private static instance: FaceRecognitionService;
  private isModelsLoaded = false;
  // We store raw labeled descriptors instead of the FaceMatcher because we implement our own Cosine Matcher
  private labeledDescriptors: any[] = []; 
  // Store current parameters
  private currentParams: RecognitionParams | null = null;

  private constructor() {}

  public static getInstance(): FaceRecognitionService {
    if (!FaceRecognitionService.instance) {
      FaceRecognitionService.instance = new FaceRecognitionService();
    }
    return FaceRecognitionService.instance;
  }

  private async waitForFaceApi(): Promise<void> {
    let attempts = 0;
    while (!window.faceapi && attempts < 50) {
      await new Promise(resolve => setTimeout(resolve, 100));
      attempts++;
    }
    if (!window.faceapi) {
      throw new Error("核心组件 (face-api.js) 未加载。请检查网络连接或刷新页面。");
    }
  }

  /**
   * Helper to attempt loading from a specific URL
   */
  private async loadModelsFromUrl(url: string): Promise<void> {
      console.log(`Attempting to load models from: ${url}`);
      // Load specific nets required for the app
      // Promise.all is used to load them in parallel for the specific URL
      await Promise.all([
        window.faceapi.nets.ssdMobilenetv1.loadFromUri(url),
        window.faceapi.nets.faceLandmark68Net.loadFromUri(url),
        // Using FaceRecognitionNet (ResNet-34) which produces 128D vectors compatible with ArcFace-style logic
        window.faceapi.nets.faceRecognitionNet.loadFromUri(url)
      ]);
      console.log(`Successfully loaded models from: ${url}`);
  }

  public async loadModels(): Promise<void> {
    if (this.isModelsLoaded) return;
    
    try {
      await this.waitForFaceApi();

      // Iterate through sources until one works
      for (const url of MODEL_URLS) {
        try {
          console.log(`正在尝试加载 AI 模型 (源: ${url})...`);
          await this.loadModelsFromUrl(url);
          
          this.isModelsLoaded = true;
          console.log(`AI 模型加载成功 (来自: ${url})`);
          return; // Exit function on success
        } catch (err) {
          console.warn(`从 ${url} 加载模型失败，尝试下一个源...`, err);
          // Continue to next URL in the loop
        }
      }

      // If loop finishes without returning, all sources failed
      throw new Error("所有模型镜像源均无法连接");
    } catch (error) {
      console.error("Critical Model Load Error:", error);
      throw new Error("AI 模型初始化失败。请检查您的网络连接是否允许访问 GitHub Pages 或 jsDelivr CDN。");
    }
  }

  public async getFaceDetection(imageElement: HTMLImageElement): Promise<any | null> {
    if (!this.isModelsLoaded) await this.loadModels();
    if (!window.faceapi) throw new Error("FaceAPI not loaded");

    try {
      // Standard pass
      const detection = await window.faceapi
        .detectSingleFace(imageElement)
        .withFaceLandmarks()
        .withFaceDescriptor();
      if (detection) return detection;
    } catch (e) { 
      console.warn("Standard face detection failed.", e); 
    }

    // Fallback: Aggressive scan
    const options = new window.faceapi.SsdMobilenetv1Options({
      minConfidence: 0.1, 
      maxResults: 10,
    });

    const allDetections = await window.faceapi
      .detectAllFaces(imageElement, options)
      .withFaceLandmarks()
      .withFaceDescriptors();

    if (allDetections.length > 0) {
      return allDetections.reduce((prev: any, current: any) => {
        return (prev.detection.box.area > current.detection.box.area) ? prev : current;
      });
    }
    return null;
  }

  public async getFaceDescriptor(imageElement: HTMLImageElement): Promise<Float32Array | null> {
    const detection = await this.getFaceDetection(imageElement);
    if (!detection) return null;
    return detection.descriptor;
  }

  public updateFaceMatcher(students: Student[], params: RecognitionParams) {
    // Store current parameters
    this.currentParams = params;
    
    if (!window.faceapi || students.length === 0) {
      this.labeledDescriptors = [];
      return;
    }
    // Just store the data; matching is now dynamic
    this.labeledDescriptors = students.map((student) => {
      return new window.faceapi.LabeledFaceDescriptors(
        student.name,
        student.descriptors
      );
    });
  }

  public getIoU(box1: any, box2: any): number {
    const x1 = Math.max(box1.x, box2.x);
    const y1 = Math.max(box1.y, box2.y);
    const x2 = Math.min(box1.x + box1.width, box2.x + box2.width);
    const y2 = Math.min(box1.y + box1.height, box2.y + box2.height);

    const intersectionArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const box1Area = box1.width * box1.height;
    const box2Area = box2.width * box2.height;

    if (box1Area + box2Area - intersectionArea === 0) return 0;

    return intersectionArea / (box1Area + box2Area - intersectionArea);
  }

  /**
   * InsightFace Core: Cosine Similarity
   * Calculates the cosine similarity between two vectors.
   * Since face-api descriptors are normalized (L2 norm = 1), dot product equals cosine similarity.
   */
  private computeCosineSimilarity(descriptor1: Float32Array, descriptor2: Float32Array): number {
    let dotProduct = 0;
    for (let i = 0; i < descriptor1.length; i++) {
      dotProduct += descriptor1[i] * descriptor2[i];
    }
    return dotProduct;
  }

  /**
   * Finds the best match using Cosine Similarity (InsightFace Algorithm)
   * Returns { label, score } where score is 0.0 to 1.0 (1.0 is identical)
   */
  private findBestMatchInsightFace(queryDescriptor: Float32Array): { label: string; score: number } {
    if (this.labeledDescriptors.length === 0) {
      return { label: 'unknown', score: 0 };
    }

    let bestLabel = 'unknown';
    let maxSimilarity = -1;

    for (const labeledDesc of this.labeledDescriptors) {
      // A student might have multiple descriptors (multiple registered photos)
      // We find the max similarity among all their photos
      for (const descriptor of labeledDesc.descriptors) {
        const similarity = this.computeCosineSimilarity(queryDescriptor, descriptor);
        if (similarity > maxSimilarity) {
          maxSimilarity = similarity;
          bestLabel = labeledDesc.label;
        }
      }
    }

    return { label: bestLabel, score: maxSimilarity };
  }

  /**
   * Public method to find best match using Cosine Similarity (InsightFace Algorithm)
   * Returns { label, score } where score is 0.0 to 1.0 (1.0 is identical)
   */
  public findBestMatch(queryDescriptor: Float32Array): { label: string; score: number } {
    return this.findBestMatchInsightFace(queryDescriptor);
  }

  private drawFaceOverlays(ctx: CanvasRenderingContext2D, resizedDetections: any[], params: RecognitionParams) {
    ctx.font = '500 11px "Inter", sans-serif'; 
    ctx.textBaseline = 'middle';

    resizedDetections.forEach((d: any) => {
      const isManual = (d as any).isManual || false;

      // Use the new InsightFace matcher
      const matchResult = this.findBestMatchInsightFace(d.descriptor);
      
      let effectiveThreshold = params.similarityThreshold;
      
      // Mask mode optimization: Lower the similarity requirement slightly
      if (params.maskMode) {
        effectiveThreshold = Math.max(0.4, effectiveThreshold - 0.1);
      }

      let color = '#ef4444'; // Red (Unknown)
      let statusText = `${matchResult.label} (${(matchResult.score * 100).toFixed(0)}%)`;
      
      // Color coding based on confidence thresholds (ArcFace Logic)
      if (matchResult.score >= effectiveThreshold) {
        color = '#22c55e'; // Green (High Confidence Match)
      } else if (matchResult.score >= effectiveThreshold * 0.8) {
        color = '#eab308'; // Yellow (Low Confidence Match)
      }

      // Draw bounding box
      const box = d.detection.box;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(box.x, box.y, box.width, box.height);

      // Draw label background
      const textWidth = ctx.measureText(statusText).width;
      ctx.fillStyle = color + '80'; // 50% opacity
      ctx.fillRect(box.x, box.y - 16, textWidth + 8, 16);

      // Draw label text
      ctx.fillStyle = 'white';
      ctx.fillText(statusText, box.x + 4, box.y - 8);

      // Draw landmark dots if available
      if (d.landmarks) {
        ctx.fillStyle = color;
        const positions = d.landmarks.positions;
        positions.forEach((pos: any) => {
          ctx.beginPath();
          ctx.arc(pos.x, pos.y, 1.5, 0, 2 * Math.PI);
          ctx.fill();
        });
      }
    });
  }

  public async detectAndRecognize(
    videoElement: HTMLVideoElement,
    canvasElement: HTMLCanvasElement,
    params: RecognitionParams,
    manualDetections: any[] = [],
    drawOverlay: boolean = true
  ): Promise<any[]> {
    if (!this.isModelsLoaded) await this.loadModels();
    
    const ctx = canvasElement.getContext('2d');
    if (!ctx) throw new Error("Could not get canvas context");
    
    // Clear canvas
    ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    // Resize canvas to match video
    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;
    
    // Run detection with user-defined parameters
    const options = new window.faceapi.SsdMobilenetv1Options({
      minConfidence: params.minConfidence,
      maxResults: 20,
      inputSize: params.networkSize
    });
    
    // Combine automatic and manual detections
    const allDetections = [
      ...(await window.faceapi
        .detectAllFaces(videoElement, options)
        .withFaceLandmarks()
        .withFaceDescriptors()),
      ...manualDetections
    ];
    
    // Draw overlays if requested
    if (drawOverlay && allDetections.length > 0) {
      this.drawFaceOverlays(ctx, allDetections, params);
    }
    
    return allDetections;
  }

  public async detectAndRecognizeImage(
    imageElement: HTMLImageElement,
    canvasElement: HTMLCanvasElement,
    params: RecognitionParams
  ): Promise<any[]> {
    if (!this.isModelsLoaded) await this.loadModels();
    
    const ctx = canvasElement.getContext('2d');
    if (!ctx) throw new Error("Could not get canvas context");
    
    // Clear canvas
    ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    // Resize canvas to match image
    canvasElement.width = imageElement.naturalWidth;
    canvasElement.height = imageElement.naturalHeight;
    
    // Run detection with user-defined parameters
    const options = new window.faceapi.SsdMobilenetv1Options({
      minConfidence: params.minConfidence,
      maxResults: 20,
      inputSize: params.networkSize
    });
    
    const detections = await window.faceapi
      .detectAllFaces(imageElement, options)
      .withFaceLandmarks()
      .withFaceDescriptors();
    
    // Draw overlays
    if (detections.length > 0) {
      this.drawFaceOverlays(ctx, detections, params);
    }
    
    return detections;
  }
}