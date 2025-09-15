"use client";

import { CheckCircle, Circle, ArrowRight } from "lucide-react";

interface ProgressIndicatorProps {
  currentStep: 'topic' | 'outline' | 'report';
}

const steps = [
  { id: 'topic', label: '选择主题', description: '输入研究主题' },
  { id: 'outline', label: '生成大纲', description: '制定文章结构' },
  { id: 'report', label: '完成综述', description: '生成最终报告' }
];

export function ProgressIndicator({ currentStep }: ProgressIndicatorProps) {
  const currentIndex = steps.findIndex(step => step.id === currentStep);

  return (
    <div className="w-full max-w-3xl mx-auto mb-8">
      <div className="flex items-center justify-between">
        {steps.map((step, index) => (
          <div key={step.id} className="flex items-center">
            <div className="flex flex-col items-center">
              <div className={`
                flex items-center justify-center w-12 h-12 rounded-full border-2 transition-all duration-300
                ${index < currentIndex 
                  ? 'bg-green-500 border-green-500 text-white' 
                  : index === currentIndex 
                    ? 'bg-blue-600 border-blue-600 text-white' 
                    : 'bg-white border-gray-300 text-gray-400'
                }
              `}>
                {index < currentIndex ? (
                  <CheckCircle className="w-6 h-6" />
                ) : (
                  <span className="text-sm font-bold">{index + 1}</span>
                )}
              </div>
              <div className="mt-3 text-center">
                <div className={`text-sm font-medium ${
                  index <= currentIndex ? 'text-gray-900' : 'text-gray-400'
                }`}>
                  {step.label}
                </div>
                <div className={`text-xs ${
                  index <= currentIndex ? 'text-gray-600' : 'text-gray-400'
                }`}>
                  {step.description}
                </div>
              </div>
            </div>
            {index < steps.length - 1 && (
              <ArrowRight className={`mx-4 h-5 w-5 ${
                index < currentIndex ? 'text-green-500' : 'text-gray-300'
              }`} />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}