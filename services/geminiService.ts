import { GoogleGenAI, Type } from "@google/genai";
import { ImageQualityReport, BehaviorReport, SingleStudentBehaviorReport } from "../types";

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

export async function validateStudentImage(base64Image: string): Promise<ImageQualityReport> {
  const model = "gemini-2.5-flash";

  // Remove header if present (data:image/jpeg;base64,)
  const cleanBase64 = base64Image.split(',')[1] || base64Image;

  try {
    const response = await ai.models.generateContent({
      model: model,
      contents: {
        parts: [
          {
            inlineData: {
              mimeType: "image/jpeg",
              data: cleanBase64
            }
          },
          {
            text: `Analyze this image for use in a face recognition database for a classroom. 
            Evaluate lighting, face angle, and visibility.
            Return a JSON object assessing if it is valid for registration.
            IMPORTANT: Return the 'issues' list in Simplified Chinese (简体中文).`
          }
        ]
      },
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            isValid: { type: Type.BOOLEAN },
            score: { type: Type.NUMBER, description: "Quality score from 0 to 100" },
            issues: { 
              type: Type.ARRAY, 
              items: { type: Type.STRING },
              description: "List of specific issues in Chinese, e.g., '光线太暗', '侧脸严重'"
            },
            lighting: { type: Type.STRING, enum: ["good", "poor", "harsh"] },
            angle: { type: Type.STRING, enum: ["frontal", "profile", "tilted"] }
          },
          required: ["isValid", "score", "issues", "lighting", "angle"]
        }
      }
    });

    const text = response.text;
    if (!text) throw new Error("No response from Gemini");

    return JSON.parse(text) as ImageQualityReport;

  } catch (error) {
    console.error("Gemini Validation Error:", error);
    // Fallback if API fails
    return {
      isValid: false,
      score: 0,
      issues: ["AI 验证服务暂时不可用"],
      lighting: "poor",
      angle: "profile"
    };
  }
}

export async function enhanceImage(base64Image: string): Promise<string | null> {
  const model = "gemini-2.5-flash-image";
  const cleanBase64 = base64Image.split(',')[1] || base64Image;

  try {
    const response = await ai.models.generateContent({
      model: model,
      contents: {
        parts: [
          {
            inlineData: {
              mimeType: "image/jpeg",
              data: cleanBase64
            }
          },
          {
            text: "Enhance the quality of this image for face recognition. Sharpen details, reduce noise, and fix blurriness. Maintain the exact identity of the people in the image. Return the improved image."
          }
        ]
      }
    });

    if (response.candidates?.[0]?.content?.parts) {
      for (const part of response.candidates[0].content.parts) {
        if (part.inlineData && part.inlineData.data) {
          return `data:image/png;base64,${part.inlineData.data}`;
        }
      }
    }
    return null;
  } catch (error) {
    console.error("Gemini Enhance Error:", error);
    throw error;
  }
}

export async function analyzeClassroomBehavior(base64Image: string): Promise<BehaviorReport> {
  const model = "gemini-2.5-flash";
  const cleanBase64 = base64Image.split(',')[1] || base64Image;

  try {
    const response = await ai.models.generateContent({
      model: model,
      contents: {
        parts: [
          {
            inlineData: {
              mimeType: "image/jpeg",
              data: cleanBase64
            }
          },
          {
            text: `Role: Professional Classroom Behavior Analyst.
            Task: Analyze the provided classroom image frame. Identify student behaviors and the overall learning atmosphere.
            
            Detect and count the following specific behaviors if visible:
            - Looking at Blackboard/Teacher (Attentive)
            - Looking at Laptop/Computer Screen (Study)
            - Looking at Phone (Distracted)
            - Drinking Water/Eating
            - Chatting/Turning Head to Neighbor
            - Sleeping/Head Down
            
            Return a JSON object with a summary in Simplified Chinese.`
          }
        ]
      },
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            timestamp: { type: Type.STRING, description: "Current analysis time string" },
            studentCount: { type: Type.INTEGER, description: "Estimated total students visible" },
            attentionScore: { type: Type.INTEGER, description: "0-100 score representing overall class focus" },
            behaviors: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  action: { type: Type.STRING, description: "Name of the behavior in Chinese (e.g. 看黑板, 看手机)" },
                  count: { type: Type.INTEGER },
                  description: { type: Type.STRING, description: "Brief context in Chinese" }
                }
              }
            },
            summary: { type: Type.STRING, description: "A professional summary of the classroom atmosphere in Chinese." }
          }
        }
      }
    });

    const text = response.text;
    if (!text) throw new Error("No response from Gemini");
    
    return JSON.parse(text) as BehaviorReport;

  } catch (error) {
    console.error("Behavior Analysis Error:", error);
    throw error;
  }
}

export async function analyzeStudentBehavior(base64Image: string): Promise<SingleStudentBehaviorReport> {
  const model = "gemini-2.5-flash";
  const cleanBase64 = base64Image.split(',')[1] || base64Image;

  try {
    const response = await ai.models.generateContent({
      model: model,
      contents: {
        parts: [
          {
            inlineData: {
              mimeType: "image/jpeg",
              data: cleanBase64
            }
          },
          {
            text: `Role: Behavioral Psychologist.
            Task: Analyze the specific student in this image crop. The image focuses on one student's upper body.
            
            Determine:
            1. Focus Score (0-100): How attentive are they to the class?
            2. Action: What exactly are they doing? (e.g., Taking notes, Raising hand, Using phone, Sleeping, Daydreaming)
            3. Posture: Body language assessment.
            4. Expression: Facial expression assessment.
            
            Return JSON in Simplified Chinese.`
          }
        ]
      },
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            timestamp: { type: Type.STRING },
            focusScore: { type: Type.INTEGER, description: "0-100" },
            isDistracted: { type: Type.BOOLEAN },
            action: { type: Type.STRING, description: "Specific action in Chinese" },
            posture: { type: Type.STRING, description: "Body posture description in Chinese" },
            expression: { type: Type.STRING, description: "Facial expression in Chinese" },
            summary: { type: Type.STRING, description: "Brief psychological analysis in Chinese" }
          },
          required: ["focusScore", "isDistracted", "action", "posture", "expression", "summary"]
        }
      }
    });

    const text = response.text;
    if (!text) throw new Error("No response from Gemini");
    
    return JSON.parse(text) as SingleStudentBehaviorReport;

  } catch (error) {
    console.error("Student Analysis Error:", error);
    throw error;
  }
}