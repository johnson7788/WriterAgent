"use client";

import { BookOpen, Sparkles, Lightbulb } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { ProgressIndicator } from "~/components/ProgressIndicator";
import { Button } from "~/components/ui/button";
import { Textarea } from "~/components/ui/textarea";
import { useStore } from "~/core/store";

const exampleTopics = [
  {
    title: "血小板及其衍生物在糖尿病足溃疡治疗中的作用及机制研究进展",
    description: "糖尿病足溃疡",
    category: "医学"
  },
  {
    title: "Z世代消费者行为洞察：社交媒体与短视频内容对消费决策的影响研究",
    description: "Z世代，消费行为，社交媒体营销",
    category: "商业与金融"
  },
  {
    title: "移动学习环境下的认知负荷管理：基于元认知策略的应用与效果评估",
    description: "移动学习，认知负荷，元认知策略",
    category: "教育与心理学"
  },
  {
    title: "城市绿地对居民心理健康的积极影响：机制、量化与规划策略",
    description: "城市绿地，心理健康，城市规划",
    category: "环境科学"
  }
];

export default function TopicPage() {
  const router = useRouter();
  const { setTopic, setGenerationStep, setOutlineContent, setReviewContent } = useStore();
  const [localTopic, setLocalTopic] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!localTopic.trim() || isSubmitting) {
      return;
    }
    setIsSubmitting(true);
    // Reset state for a new review generation
    setTopic(localTopic);
    setOutlineContent("");
    setReviewContent("");
    setGenerationStep("FORM");

    router.push("/review/outline");
  };

  const handleExampleClick = (topic: string) => {
    setLocalTopic(topic);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      {/* Header */}
      <header className="sticky top-0 z-50 w-full border-b bg-white/95 backdrop-blur supports-[backdrop-filter]:bg-white/60">
        <div className="container mx-auto px-4 h-16 flex items-center">
          <div className="flex items-center space-x-2">
            <BookOpen className="h-8 w-8 text-blue-600" />
            <span className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
              AI ReviewGen
            </span>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <ProgressIndicator currentStep="topic" />
        
        <div className="max-w-4xl mx-auto">
          {/* Header Section */}
          <div className="text-center mb-12">
            <div className="inline-flex items-center rounded-full px-4 py-2 text-sm font-medium bg-blue-100 text-blue-800 ring-1 ring-blue-600/20 mb-4">
              <Sparkles className="h-4 w-4 mr-2" />
              第一步：选择您的研究主题
            </div>
            <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
              您想研究什么主题？
            </h1>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              输入您感兴趣的研究主题，或从我们精选的示例中选择一个开始。
            </p>
          </div>

          {/* Example Topics */}
          <div className="mb-12">
            <div className="flex items-center mb-6">
              <Lightbulb className="h-5 w-5 text-amber-500 mr-2" />
              <h2 className="text-2xl font-bold text-gray-900">热门主题示例</h2>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {exampleTopics.map((topic, index) => (
                <div
                  key={index}
                  onClick={() => handleExampleClick(topic.title)}
                  className="bg-white rounded-xl p-6 shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1 cursor-pointer border border-gray-100 group"
                >
                  <div className="flex items-start justify-between mb-3">
                    <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                      {topic.category}
                    </span>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      选择
                    </Button>
                  </div>
                  <h3 className="font-bold text-gray-900 mb-2 group-hover:text-blue-600 transition-colors">
                    {topic.title}
                  </h3>
                  <p className="text-gray-600 text-sm leading-relaxed">
                    {topic.description}
                  </p>
                </div>
              ))}
            </div>
          </div>

          {/* Topic Input Form */}
          <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">
              或者输入您的自定义主题
            </h2>
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="space-y-2">
                <label htmlFor="topic" className="text-sm font-medium text-gray-700">
                  研究主题 *
                </label>
                <Textarea
                  id="topic"
                  placeholder="请输入您想要研究的主题，例如：人工智能在教育领域的应用及其影响研究..."
                  value={localTopic}
                  onChange={(e) => setLocalTopic(e.target.value)}
                  rows={6}
                  className="w-full text-base resize-none border-2 border-gray-200 focus:border-blue-500 rounded-lg transition-colors"
                />
                <p className="text-sm text-gray-500">
                  提示：主题描述越详细，撰写的综述质量越高
                </p>
              </div>
              <Button
                type="submit"
                disabled={!localTopic.trim() || isSubmitting}
                className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1 py-4 text-lg"
                size="lg"
              >
                {isSubmitting ? "生成中..." : "开始撰写大纲"}
              </Button>
            </form>
          </div>
        </div>
      </main>
    </div>
  );
}