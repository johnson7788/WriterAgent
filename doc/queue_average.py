#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
queue_average.py

模拟：后端较慢地产生段落内容，但前端逐字匀速输出。
核心：生产者只 enqueue 段落；独立 streamer 协程逐字发送（这里用 print 模拟 MQ 发给前端）。
"""

import asyncio
from typing import Optional
import random
import time

# ====== 可调参数 ======
BATCH_CHARS = 1        # 1 表示逐字；可调大为逐 N 字
INTERVAL_S = 0.02      # 逐批发送的间隔（前端打字机速度）
QUEUE_MAXSIZE = 200    # 背压阈值（队列满时合并，防止阻塞生成）
PARAGRAPH_DELAY_RANGE = (1.5, 4.0)  # 模拟后端生成每段之间的延时范围（秒）
# ====================


class CharStreamer:
    """
    将大段文本以“逐字/逐批”的方式异步、匀速发送（这里用 print 模拟发送至前端）。
    - 独立协程运行，不阻塞生产者
    - 内置节流：每 batch_chars 个字符发一次；每次发送后 sleep(interval_s)
    - 内置背压：队列满时合并文本，避免 await 阻塞生产者
    """
    def __init__(self, batch_chars: int = 1, interval_s: float = 0.02, queue_maxsize: int = 100):
        self.batch_chars = max(1, batch_chars)
        self.interval_s = max(0.0, interval_s)
        self.queue: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=queue_maxsize)
        self._task: Optional[asyncio.Task] = None
        self._closed = False
        self._start_ts = time.perf_counter()  # 仅用于打印时间戳

    def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="CharStreamer")

    def enqueue(self, text: str):
        """尽量无阻塞地入队；满了就合并，避免卡住上游生成。"""
        if not text or self._closed:
            return
        try:
            self.queue.put_nowait(text)
        except asyncio.QueueFull:
            # 合并策略：把队列现有内容尽可能取出并拼接，再塞回去
            merged = [text]
            try:
                while True:
                    merged.append(self.queue.get_nowait())
            except asyncio.QueueEmpty:
                pass
            big = "".join(merged)
            try:
                self.queue.put_nowait(big)
            except asyncio.QueueFull:
                # 极端情况下仍满：丢弃最早的一部分再放（也可改成写入临时文件缓冲）
                _ = self.queue.get_nowait()
                self.queue.put_nowait(big)

    async def close(self):
        """发送完所有剩余文本并优雅关闭。"""
        if self._closed:
            return
        self._closed = True
        await self.queue.put(None)  # 哨兵
        if self._task:
            await self._task
            self._task = None

    async def _run(self):
        buffer = ""
        try:
            while True:
                item = await self.queue.get()
                if item is None:
                    # 刷掉残留
                    if buffer:
                        await self._send_chunk(buffer)
                        buffer = ""
                    break
                buffer += item

                # 逐字/逐批发
                n = len(buffer)
                pos = 0
                while pos < n:
                    end = min(pos + self.batch_chars, n)
                    piece = buffer[pos:end]
                    await self._send_chunk(piece)
                    pos = end
                    if self.interval_s:
                        await asyncio.sleep(self.interval_s)
                buffer = ""
        finally:
            # 结束标记（模拟 [stop]）
            self._log("[END STREAM]\n")

    async def _send_chunk(self, text: str):
        """这里以 print 模拟“发送到前端”。你可改成 MQ 发送。"""
        # 为了视觉清晰，这里不换行，直接打印字符（可按需调整）
        # 也可以把 text 换成 {"type":2, "message":{...}} 的结构送 MQ
        print(text, end="", flush=True)

    def _log(self, msg: str):
        rel = time.perf_counter() - self._start_ts
        print(f"\n[{rel:7.3f}s] {msg}", flush=True)


async def backend_producer(streamer: CharStreamer):
    """
    模拟后端：生成多段内容；段落之间延时较长，但生成时不被前端发送速度阻塞。
    """
    paragraphs = [
        "摘要 城市化进程中居民心理健康问题日益突出，城市绿地作为重要干预手段，其心理健康效益的系统综述旨在整合多学科证据以阐明机制路径、量化方法与规划策略。基于神经机制、流行病学与规划设计视角的整合分析显示，城市绿地暴露1小时可有效降低大脑杏仁核活动，而长期接触绿色空间能够显著降低抑郁风险，相关研究报道风险降低幅度在28%至37%之间。神经影像学证据表明绿地环境通过调节边缘系统活动缓解压力反应，流行病学队列研究证实居住绿地暴露与抑郁焦虑发病率呈负相关关系。当前研究亟需建立标准化的暴露量化框架，推动健康导向的绿地规划政策制定，通过多尺度绿地系统优化提升城市居民心理健康水平。关键词： 城市绿地, 心理健康, 神经机制, 暴露量化, 健康城市规划 我将根据系统指令的要求，首先使用DocumentSearch工具检索相关文献，然后撰写引言章节。 我将根据系统指令的要求，首先进行文献检索，然后撰写引言章节的内容。 基于检索到的文献证据，现在开始撰写引言章节：",
        "1.1 城市化与心理健康挑战的流行病学背景城市化进程对心理健康构成显著威胁，预计到2030年全球城市覆盖面积将增加到190万平方公里，52亿人将生活在城市地区[30]。抑郁症，特别是在城市地区，患病的人数明显上升，成为中等或高收入国家中造成残疾的主要原因之一[46]。流行病学调研显示，较发达国家中有77.7%的人口居住在城市，这种趋势与城市化的加速存在关联[46]。到2050年，全球约70%的人口将生活在城市化环境中，流行病学数据表明城市居民的心理健康风险更高[53]。中国快速城市化进程中，城市空间中绿地所占比例急剧变化，正在逐渐成为影响公共健康的决定因素之一[47]。1.2 城市绿地作为心理健康干预的自然解决方案城市绿地作为重要的生态基础设施，通过影响居民行为方式、环境质量等机制影响其生理和心理健康[45]。大量流行病学证据表明，居住在绿化较好区域的居民，其精神健康状态好于居住在绿化较差区域的居民[46]。城市绿地促进人体健康的效应主要包括三个方面：减少不良人居环境对健康造成的负面影响、提供促进健康行为的场所、通过自然接触直接改善心理健康状态[3]。基于自然的解决方案（Nature-based Solutions, NbS）将城市绿色空间作为主要载体，其健康效应促进机制已成为科学前沿和实践热点[56]。社区内可及的绿地为社会互动创造积极环境，促进体力活动，其健康效益在社会经济地位较低或弱势群体中尤为显著[34]。1.3 研究目标与综述框架本研究旨在系统梳理城市绿地对居民心理健康积极影响的科学证据，分析其作用机制、量化评估方法及规划策略应用。综述框架首先探讨城市化背景下心理健康问题的流行病学特征，继而分析城市绿地作为自然解决方案的理论基础和作用路径。重点评述绿地暴露与心理健康指标的量化关系，包括不同绿地类型、空间配置和可达性对心理健康的差异化影响。最后综合现有证据，提出基于健康促进的城市绿地规划策略，为健康城市建设和公共卫生政策制定提供科学依据。通过整合多学科研究成果，本综述致力于建立城市绿地-心理健康关系的系统性认识框架，推动循证规划实践的发展。 基于检索到的文献证据，现在开始撰写引言章节：",
        "1.2 城市化与心理健康挑战的流行病学背景，全球城市化进程加速对居民心理健康构成了显著威胁。预计到2030年，全球城市覆盖面积将增加到190万平方公里，52亿人将生活在城市地区[30]。抑郁症，特别是在城市地区，患病率呈现明显上升趋势[30]。流行病学调查显示，全球约有3.5亿抑郁症患者，大多数国家的终生患病率在8%-12%之间[55,56]。中国国家精神障碍流行病学调查结果显示，成年人抑郁症患病率达到显著水平[62]。城市环境中，焦虑症和抑郁症的患病率增加与新冠病毒每日感染率增加、流动性降低等因素相关，已造成沉重的经济社会负担[61]。",
        "1.3 城市绿地作为心理健康干预的自然解决方案，城市绿地作为重要的生态基础设施，在促进居民心理健康方面发挥着关键作用。研究表明，城市绿地通过多种机制对心理健康产生积极影响，主要包括减少不良人居环境对健康的负面影响、提供生理和心理恢复的环境以及促进社会交往和体力活动[3]。实地试验证实，老年人参观不同城市绿地环境能够产生显著的生理和心理积极影响[39]。城市绿地的修复效益体现在生理修复、心理修复等多个维度，通过眼动分析等方法可量化其恢复效果[48]。基于自然解决方案的规划策略强调，城市绿地不仅提供生态效益，还为生物提供生境和避难所，为城市生态系统服务做出贡献[47]。",
        "1.4 研究目标与综述框架，本综述系统分析城市绿地对居民心理健康的积极影响机制、量化评估方法及规划策略应用。研究框架涵盖流行病学背景分析、多维度作用机制解析、量化指标体系构建以及基于实证研究的规划策略优化。通过整合风景园林学、环境行为学、环境医学等多学科视角，旨在为城市绿地规划提供科学依据，促进城市居民心理健康水平的提升。综述重点关注城市绿地作为自然解决方案在心理健康干预中的有效性证据，以及不同绿地类型、空间布局和质量特征对心理健康效益的影响差异。 我将根据系统指令的要求，首先使用DocumentSearch工具检索相关文献证据，然后撰写章节的内容。 我将根据系统指令的要求，首先进行文献检索来获取相关证据，然后撰写章节。 基于检索到的文献证据，我现在开始撰写章节的内容："
    ]

    for i, p in enumerate(paragraphs, 1):
        # 模拟“后端计算生成”耗时（段落间隔大）
        delay = random.uniform(*PARAGRAPH_DELAY_RANGE)
        await asyncio.sleep(delay)
        # 段落末尾补两个换行，模拟自然段
        streamer.enqueue(p + "\n\n")
        print(f"\n[PRODUCER] Enqueued paragraph {i} (len={len(p)}) after {delay:.1f}s", flush=True)

    # 所有段落都已经生产完毕，关闭 streamer（让其把剩余字发完）
    await streamer.close()
    print("\n[PRODUCER] All paragraphs produced. Requested graceful close.", flush=True)


async def main():
    print("[BOOT] Start steady streaming demo...\n")
    streamer = CharStreamer(
        batch_chars=BATCH_CHARS,
        interval_s=INTERVAL_S,
        queue_maxsize=QUEUE_MAXSIZE,
    )
    streamer.start()

    # 并发运行：生产者慢慢生产；streamer 独立匀速打印
    await backend_producer(streamer)

    print("\n[MAIN] Done.\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Exit.")
