import { ArrowRight, Users, Settings2, ShieldCheck, Sparkles, BookOpen, Zap } from "lucide-react";
import Link from "next/link";

import { Button } from "~/components/ui/button";

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      {/* Header */}
      <header className="sticky top-0 z-50 w-full border-b bg-white/95 backdrop-blur supports-[backdrop-filter]:bg-white/60">
        <div className="container mx-auto px-4 h-16 flex items-center">
          <div className="flex items-center space-x-2">
            <BookOpen className="h-8 w-8 text-blue-600" />
            <span className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
              AI Writer
            </span>
          </div>
        </div>
      </header>

      <main className="flex-grow">
        {/* Hero Section */}
        <section className="relative overflow-hidden py-20 md:py-32">
          <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 to-indigo-600/20 opacity-50"></div>
          <div className="container relative mx-auto px-4 text-center">
            <div className="flex justify-center mb-6">
              <div className="inline-flex items-center rounded-full px-4 py-2 text-sm font-medium bg-blue-100 text-blue-800 ring-1 ring-blue-600/20">
                <Sparkles className="h-4 w-4 mr-2" />
                AI驱动的学术写作工具
              </div>
            </div>
            <h1 className="text-4xl md:text-7xl font-bold tracking-tight bg-gradient-to-r from-gray-900 via-blue-800 to-indigo-800 bg-clip-text text-transparent mb-6">
              智能文献综述
              <br />
              <span className="text-blue-600">撰写</span>
            </h1>
            <p className="max-w-3xl mx-auto text-xl md:text-2xl text-gray-600 mb-12 leading-relaxed">
              从一个主题开始，我们的多智能体系统将为您协同完成
              <span className="font-semibold text-gray-800">高质量、可信赖</span>的学术综述。
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <Link href="/review/topic" passHref>
                <Button size="lg" className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1 group px-8 py-4 text-lg">
                  开始撰写 <ArrowRight className="ml-2 h-5 w-5 transition-transform group-hover:translate-x-1" />
                </Button>
              </Link>
              <Button variant="outline" size="lg" className="px-8 py-4 text-lg border-2 hover:bg-blue-50">
                查看示例
              </Button>
            </div>
          </div>
        </section>

        {/* Process Steps */}
        <section className="py-20 bg-white">
          <div className="container mx-auto px-4">
            <div className="text-center mb-16">
              <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
                三步完成专业综述
              </h2>
              <p className="text-xl text-gray-600 max-w-2xl mx-auto">
                简单三步，从主题到完整的学术综述
              </p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              <ProcessStep
                step="01"
                title="输入主题"
                description="输入您想要研究的主题，AI将为您分析并理解研究方向"
                icon={<BookOpen className="h-8 w-8" />}
              />
              <ProcessStep
                step="02"
                title="撰写大纲"
                description="AI自动撰写结构化大纲，您可以随时编辑和调整内容结构"
                icon={<Settings2 className="h-8 w-8" />}
              />
              <ProcessStep
                step="03"
                title="完成综述"
                description="基于大纲撰写完整的文献综述，包含引用和学术格式"
                icon={<Zap className="h-8 w-8" />}
              />
            </div>
          </div>
        </section>

        {/* Features Section */}
        <section className="py-20 bg-gradient-to-br from-gray-50 to-blue-50">
          <div className="container mx-auto px-4">
            <div className="text-center mb-16">
              <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
                为什么选择我们？
              </h2>
              <p className="text-xl text-gray-600 max-w-2xl mx-auto">
                专业的AI技术，为学术研究提供强大支持
              </p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              <FeatureCard
                icon={<Users className="h-12 w-12 text-blue-600" />}
                title="多Agent协同"
                description="多个专门的AI智能体（规划、研究、写作）分工协作，模拟专家团队的工作流程，确保综述的深度与广度。"
              />
              <FeatureCard
                icon={<Settings2 className="h-12 w-12 text-blue-600" />}
                title="实时可控"
                description="从大纲的初步撰写到最终内容的润色，您可以在每个关键步骤进行审查、编辑和确认，确保最终产出完全符合您的预期。"
              />
              <FeatureCard
                icon={<ShieldCheck className="h-12 w-12 text-blue-600" />}
                title="学术可信"
                description="我们的AI在撰写内容时，会注重信息的准确性和来源的可靠性，致力于提供有价值、可信赖的学术参考。"
              />
            </div>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
}

function ProcessStep({ step, title, description, icon }: { step: string, title: string, description: string, icon: React.ReactNode }) {
  return (
    <div className="relative">
      <div className="flex flex-col items-center text-center">
        <div className="relative mb-6">
          <div className="w-20 h-20 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-full flex items-center justify-center text-white shadow-lg">
            {icon}
          </div>
          <div className="absolute -top-2 -right-2 w-8 h-8 bg-white rounded-full flex items-center justify-center text-sm font-bold text-blue-600 shadow-md">
            {step}
          </div>
        </div>
        <h3 className="text-2xl font-bold text-gray-900 mb-4">{title}</h3>
        <p className="text-gray-600 leading-relaxed">{description}</p>
      </div>
    </div>
  );
}

function FeatureCard({ icon, title, description }: { icon: React.ReactNode, title: string, description: string }) {
  return (
    <div className="bg-white rounded-xl p-8 shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-2 border border-gray-100">
      <div className="mb-6">{icon}</div>
      <h3 className="text-2xl font-bold text-gray-900 mb-4">{title}</h3>
      <p className="text-gray-600 leading-relaxed">{description}</p>
    </div>
  );
}

function Footer() {
  return (
    <footer className="bg-gray-900 text-white py-12">
      <div className="container mx-auto px-4 text-center">
        <div className="flex items-center justify-center space-x-2 mb-4">
          <BookOpen className="h-6 w-6 text-blue-400" />
          <span className="text-2xl font-bold">AI ReviewGen</span>
        </div>
        <p className="text-gray-400">
          &copy; {new Date().getFullYear()} AI 文献综述撰写工具。让学术写作更简单。
        </p>
      </div>
    </footer>
  );
}