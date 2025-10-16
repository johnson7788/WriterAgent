import os
import unittest
import time
import httpx
from httpx import AsyncClient
class ReviewApiTestCase(unittest.IsolatedAsyncioTestCase):
    """
    Tests for the Markdown-based Review API
    - POST /api/review_outline (streaming markdown)
    - POST /api/review (streaming markdown for a single chapter or the full paper)
    """

    host = os.environ.get('host', 'http://127.0.0.1')
    port = os.environ.get('port', 7800)
    base_url = f"{host}:{port}"

    def test_generate_outline_stream(self):
        """测试生成大纲的内容"""
        url = f"{self.base_url}/api/outline"
        data = {
            "topic": "多模态大模型安全性",
            "language": "zh",
        }
        start_time = time.time()
        headers = {'content-type': 'application/json'}
        with httpx.stream("POST", url, json=data, headers=headers, timeout=None) as response:
            self.assertEqual(response.status_code, 200, "/api/review_outline stream should return 200")
            for chunk in response.iter_text():
                print(chunk, end="", flush=True)
        print(f"Outline stream test took: {time.time() - start_time}s")
        print(f"Server called: {self.host}")

    async def test_generate_content_stream(self):
        """测试生成正文内容"""
        url = f"{self.base_url}/api/review"
        data = {
            "topic": "多模态大模型安全性",
            "outline": """基于检索到的30+篇文献，我现在为您生成一份严谨的多模态大模型安全性综述大纲：\n\n# 多模态大模型安全性综述\n\n## 摘要\n- 背景：多模态大模型在视觉语言理解等任务中快速发展，但安全风险同步升级\n- 目的：系统梳理多模态大模型安全威胁、评测方法与防御策略\n- 方法：基于近5年30+篇核心文献的系统分析\n- 主要发现：视觉主导攻击成功率87%，跨模态威胁显著，传统防御机制覆盖不足\n- 结论与展望：需构建多模态一体化安全框架，加强真实世界风险评估\n- 关键词：多模态大模型；越狱攻击；安全评测；跨模态防御；视觉语言模型\n\n## 1. 引言\n### 1.1 多模态大模型发展现状与应用场景\n### 1.2 多模态安全问题的独特性与挑战\n### 1.3 研究范围与综述结构\n\n## 2. 多模态越狱攻击技术\n### 2.1 视觉主导攻击范式\n- VisCo攻击：图像引导伪造对话上下文，成功率显著提升\n- 视觉上下文放大安全风险，传统对齐机制覆盖不足\n### 2.2 跨模态协同攻击\n- PiCo攻击：图像代码情境化实现越狱\n- 多模态输入组合绕过单模态检测机制\n### 2.3 自动化攻击生成\n- Arondight框架：自动生成多模态越狱提示\n- 红队测试覆盖率与攻击多样性评估\n\n## 3. 安全风险评测基准\n### 3.1 综合评测数据集\n- MM-SafetyBench：5,040条多模态安全评测数据\n- 覆盖真实性、可控性、安全性与隐私性四个维度\n### 3.2 真实世界风险评估\n- RiOSWorld基准：MLLM代理在计算机操作中的风险剖析\n- GUI环境下智能体可信度评测框架MLA-Trust\n### 3.3 攻击成功率量化指标\n- 越狱攻击平均成功率87%，隐蔽度显著提升\n- 多模态攻击相比纯文本攻击成功率增加35%\n\n## 4. 检测与防御机制\n### 4.1 激活信号检测技术\n- HiddenDetect：无需训练，基于激活信号识别越狱攻击\n- 内心预警机制在推理阶段实时防护\n### 4.2 跨模态防御架构\n- BlueSuffix：基于强化微调的黑盒防御新架构\n- 在保持良性样本性能同时降低攻击成功率\n### 4.3 一体化安全解决方案\n- 蚁天鉴2.0：集成AI鉴真与深度伪造识别\n- 多模态内容真实性检测能力增强\n\n## 5. 方法学质量与标准化\n### 5.1 安全测评自动化\n- 智能体概念引入，实现大模型安全测评自动化\n- 34个核心维度的模块化评估体系\n### 5.2 行业标准与监管框架\n- 国内首次AI大模型安全\"体检\"：特有风险占比超60%\n- 多模态内容安全与对抗防御关键技术标准化\n\n## 6. 特定应用场景安全\n### 6.1 金融领域安全防护\n- 大模型安全网关：多模态预训练安全检测模型\n- 提示词注入防护与价值观内容过滤\n### 6.2 智能体操作安全\n- GUI环境下MLAs可信度评估\n- 持续性可信度评估技术支撑\n\n## 7. 新兴攻击向量\n### 7.1 后门攻击与木马植入\n- 计算机视觉领域后门攻击分类与防御\n- 触发器类型：补丁、混合/频率、语义、变换\n### 7.2 对抗样本迁移攻击\n- 传播修正攻击提升对抗性迁移能力\n- 异构模型间共性安全弱点分析\n\n## 8. 防御策略有效性比较\n### 8.1 传统防御机制局限性\n- 基于释义防御对文本攻击有效，但跨模态覆盖不足\n- 单一防御方法对抗多样攻击策略效果有限\n### 8.2 新型防御框架性能\n- ETA框架：多模态评估和双层对齐，降低不安全响应率\n- 通用防御框架对抗各种对抗策略的有效性\n\n## 9. 真实世界部署挑战\n### 9.1 产业应用安全实践\n- 天融信\"AI+安全\"技术：数据泄露、恶意注入风险防护\n- 从网络层到应用层的全栈智能防御\n### 9.2 供应链安全风险\n- API安全在AI驱动攻击下的升级需求\n- 业务逻辑滥用与供应链渗透新型威胁\n\n## 10. 未来研究方向\n### 10.1 技术发展趋势\n- 以模制模加固应用大模型抵御恶意攻击\n- 多模态分析技术提升安全检测精度\n### 10.2 标准化与伦理考量\n- 多模态大模型安全测评标准体系构建\n- 公平性、透明度与责任归属机制完善"""
        }
        start_time = time.time()
        headers = {'content-type': 'application/json'}
        async with AsyncClient() as client:
            async with client.stream("POST", url, json=data, headers=headers, timeout=None) as response:
                self.assertEqual(response.status_code, 200, "/api/review stream should return 200")
                md = ""
                async for chunk in response.aiter_text():
                    if not chunk:
                        continue
                    md += chunk
                    print(chunk, end="", flush=True)
        print(f"Section stream test took: {time.time() - start_time}s")
        print(f"Server called: {self.host}")

if __name__ == "__main__":
    unittest.main()
