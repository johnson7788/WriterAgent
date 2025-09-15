"use client";

import { saveAs } from "file-saver";
import { BookOpen, Download, Edit3, Share2, RotateCcw, ArrowLeft, ArrowRight, Brain, ChevronDown, ChevronRight } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useState, useRef } from "react";
import { remark } from "remark";
import remarkGfm from "remark-gfm";
import remarkHtml from "remark-html";

import ReportEditor from "~/components/editor";
import { ProgressIndicator } from "~/components/ProgressIndicator";
import { Button } from "~/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "~/components/ui/collapsible";
import { generateReviewStream } from "~/core/api/review";
import { useStore } from "~/core/store";

import ReferencesPanel from "../components/references-panel";
import { StreamingMarkdownDisplay } from "../components/streaming-markdown-display";

export default function ReportPage() {
  const router = useRouter();
  const {
    topic,
    outlineContent,
    reviewContent,
    setReviewContent,
    appendReviewContent,
  } = useStore();
  const [isGenerating, setIsGenerating] = useState(true);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [references, setReferences] = useState<Record<string, any>>({});
  const [thoughtProcess, setThoughtProcess] = useState("");
  const [isThoughtProcessOpen, setIsThoughtProcessOpen] = useState(false);

  const chunkQueue = useRef<string[]>([]);
  const fullContentRef = useRef<string>("");

  useEffect(() => {
    const intervalId = setInterval(() => {
      if (chunkQueue.current.length > 0) {
        const chunkToAppend = chunkQueue.current.join("");
        appendReviewContent(chunkToAppend);
        chunkQueue.current = [];
      }
    }, 200);

    return () => clearInterval(intervalId);
  }, [appendReviewContent]);

  useEffect(() => {
    if (!topic || !outlineContent) {
      router.replace("/review/topic");
      return;
    }

    const generateReview = async () => {
      setIsGenerating(true);
      setReviewContent('');
      setReferences({});
      setThoughtProcess("");
      try {
        const stream = generateReviewStream(topic, outlineContent);
        for await (const event of stream) {
          const event_data = event.data;
          console.log("report输出的event_data:", event_data);
          console.log("event_data.type:", event_data.type);
          if (event_data.type === "text") {
            appendReviewContent(event_data.text.endsWith("\n") ? event_data.text : `${event_data.text}\n`);
          } else if (event_data.type === "metadata") {
            // Handle references
            console.log("触发了收集metadata", event_data.metadata);
            setReferences(event_data.metadata.references);
          } else if (event_data.type === "reference_item") {
            // Handle references
            setReferences((prev) => ({
              ...prev,
              [event_data.item.file_id]: event_data.item,
            }));
          } else if (event_data.type === "data") {
            if (event_data.data.type === "function_call") {
              const prettyCall = `
  **Function Call:**
  ${JSON.stringify(
                event_data.data,
                null,
                2,
              )}
  `;
              setThoughtProcess((prev) => prev + prettyCall);
            } else if (event_data.data.type === "function_response") {
              const prettyResponse = `
  **Function Response:**
  ${JSON.stringify(
                event_data.data,
                null,
                2,
              )}`;
              setThoughtProcess((prev) => prev + prettyResponse);
            }
          }
        }
      } catch (e) {
        console.error("Error generating review:", e);
      } finally {
        setIsGenerating(false);
      }
    };

    void generateReview();
  }, [topic, outlineContent, router, setReviewContent, appendReviewContent]);

  const handleDownload = async () => {
    console.log("Downloading content...");
    const processedHtml = await remark()
      .use(remarkGfm)
      .use(remarkHtml)
      .process(reviewContent);

    const htmlString = `<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Review</title>
</head>
<body>${String(processedHtml)}</body>
</html>`;

    const htmlToDocx = (await import("html-to-docx")).default;
    const fileBuffer = await htmlToDocx(htmlString);
    const blob = new Blob([fileBuffer as BlobPart]);
    saveAs(blob, "review.docx");
  };

  const handleShare = () => {
    console.log("Sharing content...");
    alert("分享功能即将上线！");
  };


  const handleRegenerate = () => {
    const generateReview = async () => {
      setIsGenerating(true);
      setReviewContent('');
      setReferences({});
      setThoughtProcess("");
      try {
        const stream = generateReviewStream(topic, outlineContent);
        for await (const event of stream) {
          const event_data = event.data;
          console.log("report输出的event_data:", event_data);
          if (event_data.type === "text") {
            appendReviewContent(event_data.text.endsWith("\n") ? event_data.text : `${event_data.text}\n`);
          } else if (event_data.type === "metadata") {
            // Handle references
            setReferences(event_data.metadata.references);
          } else if (event_data.type === "reference_item") {
            // Handle references
            setReferences((prev) => ({
              ...prev,
              [event_data.item.file_id]: event_data.item,
            }));
          } else if (event_data.type === "reference_item") {
            // Handle references
            setReferences((prev) => ({
              ...prev,
              [event_data.item.file_id]: event_data.item,
            }));
          } else if (event_data.type === "data") {
            if (event_data.data.type === "function_call") {
              const prettyCall = `
  **Function Call:**
  ${JSON.stringify(
                event_data.data,
                null,
                2,
              )}
  `;
              setThoughtProcess((prev) => prev + prettyCall);
            } else if (event_data.data.type === "function_response") {
              const prettyResponse = `
  **Function Response:**
  ${JSON.stringify(
                event_data.data,
                null,
                2,
              )}`;
              setThoughtProcess((prev) => prev + prettyResponse);
            }
          }
        }
      } catch (e) {
        console.error("Error regenerating review:", e);
      } finally {
        setIsGenerating(false);
      }
    };
    void generateReview();
  };

  // Thought Process Component
  const ThoughtProcessSection = ({ content, isGenerating }: { content: string; isGenerating?: boolean }) => {
    if (!content) return null;

    return (
      <div className="mb-6">
        <Collapsible open={isThoughtProcessOpen} onOpenChange={setIsThoughtProcessOpen}>
          <CollapsibleTrigger asChild>
            <Button 
              variant="ghost" 
              className="flex items-center justify-between w-full p-4 bg-amber-50 hover:bg-amber-100 border border-amber-200 rounded-lg transition-colors"
            >
              <div className="flex items-center space-x-2">
                <Brain className="h-4 w-4 text-amber-600" />
                <span className="text-sm font-medium text-amber-800">
                  {isGenerating ? "AI 思考过程 (实时)" : "AI 思考过程"}
                </span>
                {isGenerating && (
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-amber-500 rounded-full animate-pulse"></div>
                    <div className="w-2 h-2 bg-amber-500 rounded-full animate-pulse delay-75"></div>
                    <div className="w-2 h-2 bg-amber-500 rounded-full animate-pulse delay-150"></div>
                  </div>
                )}
              </div>
              {isThoughtProcessOpen ? (
                <ChevronDown className="h-4 w-4 text-amber-600" />
              ) : (
                <ChevronRight className="h-4 w-4 text-amber-600" />
              )}
            </Button>
          </CollapsibleTrigger>
          <CollapsibleContent className="mt-2">
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
              <div className="text-xs text-gray-500 mb-3 font-mono tracking-wide uppercase">
                内部推理过程 · 仅供参考
              </div>
              <div className="prose prose-sm max-w-none">
                <div 
                  className="font-mono text-xs text-gray-700 leading-relaxed whitespace-pre-wrap bg-white p-3 rounded border border-gray-100 max-h-96 overflow-y-auto"
                  style={{ 
                    fontFamily: 'Monaco, Menlo, "Ubuntu Mono", monospace',
                    fontSize: '11px',
                    lineHeight: '1.4'
                  }}
                >
                  {content}
                </div>
              </div>
            </div>
          </CollapsibleContent>
        </Collapsible>
      </div>
    );
  };

  if (isGenerating) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
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
          <ProgressIndicator currentStep="report" />
          <ThoughtProcessSection content={thoughtProcess} isGenerating={true} />
          <StreamingMarkdownDisplay content={reviewContent} title="正在撰写最终文章..." />
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
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
        <ProgressIndicator currentStep="report" />
        
        <div className="max-w-6xl mx-auto">
          {/* Header */}
          <div className="bg-white rounded-2xl shadow-xl p-8 mb-8 border border-gray-100">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div>
                <h1 className="text-3xl font-bold text-gray-900 mb-2">
                  文献综述已完成！
                </h1>
                <p className="text-gray-600">
                  您的综述已撰写完成，可以继续编辑或直接下载使用
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-3">
                <Button
                  variant="outline"
                  onClick={handleRegenerate}
                  className="flex items-center space-x-2"
                >
                  <RotateCcw className="h-4 w-4" />
                  <span>重新撰写</span>
                </Button>
                <Button
                  variant="outline"
                  onClick={handleShare}
                  className="flex items-center space-x-2"
                >
                  <Share2 className="h-4 w-4" />
                  <span>分享</span>
                </Button>
                <Button
                  onClick={handleDownload}
                  className="bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1 flex items-center space-x-2"
                >
                  <Download className="h-4 w-4" />
                  <span>下载 DOCX</span>
                </Button>
              </div>
            </div>
            
            {/* Topic and Stats */}
            <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-blue-50 rounded-lg p-4 border-l-4 border-blue-600">
                <div className="text-sm font-medium text-blue-800 mb-1">研究主题</div>
                <div className="text-blue-900 text-sm">{topic}</div>
              </div>
              <div className="bg-green-50 rounded-lg p-4 border-l-4 border-green-600">
                <div className="text-sm font-medium text-green-800 mb-1">字数统计</div>
                <div className="text-green-900 font-semibold">{reviewContent.length.toLocaleString()} 字符</div>
              </div>
              <div className="bg-purple-50 rounded-lg p-4 border-l-4 border-purple-600">
                <div className="text-sm font-medium text-purple-800 mb-1">生成时间</div>
                <div className="text-purple-900 font-semibold">{new Date().toLocaleDateString()}</div>
              </div>
            </div>
          </div>

          {/* Thought Process Section */}
          <ThoughtProcessSection content={thoughtProcess} />

          {/* Action Buttons */}
          <div className="flex justify-end items-center py-6 space-x-4">
            <Button
              variant="outline"
              onClick={() => router.push("/review/outline")}
              className="px-8 py-4 text-lg flex items-center"
            >
              <ArrowLeft className="mr-2 h-5 w-5" />
              返回大纲页
            </Button>
            <Button
              onClick={() => router.push("/review/topic")}
              className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1 px-10 py-4 text-lg"
            >
              开始新的综述 <ArrowRight className="ml-2 h-5 w-5" />
            </Button>
          </div>

          {/* Two-column layout for editor and references */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 max-w-6xl mx-auto">
            <div className="lg:col-span-2">
              <div className="bg-white rounded-2xl shadow-xl p-8 mb-8 border border-gray-100 h-full">
                <div className="flex items-center mb-6">
                  <Edit3 className="h-5 w-5 text-gray-600 mr-2" />
                  <h2 className="text-xl font-semibold text-gray-900">编辑最终报告</h2>
                </div>
                <div className="border-2 border-gray-200 rounded-lg overflow-hidden">
                  <ReportEditor
                    content={reviewContent}
                    onMarkdownChange={setReviewContent}
                  />
                </div>
              </div>
            </div>
            <div className="lg:col-span-1">
              <ReferencesPanel references={references} />
            </div>
          </div>

        </div>
      </main>
    </div>
  );
}
