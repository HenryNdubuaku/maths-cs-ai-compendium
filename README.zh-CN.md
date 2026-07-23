[English](README.md) | **简体中文** | [日本語](README.ja.md)

# 数学、计算机科学与人工智能知识汇编

<img src="images/logo.png" alt="Logo" style="border-radius: 30px; width: 100%;">

**在线阅读**：[henryndubuaku.github.io/maths-cs-ai-compendium](https://henryndubuaku.github.io/maths-cs-ai-compendium/)

<a href="https://trendshift.io/repositories/21344?utm_source=repository-badge&amp;utm_medium=badge&amp;utm_campaign=badge-repository-21344" target="_blank" rel="noopener noreferrer"><img src="https://trendshift.io/api/badge/repositories/21344" alt="HenryNdubuaku%2Fmaths-cs-ai-compendium | Trendshift" width="250" height="55"/></a>

## 概览
大多数教科书把好思想埋在密集的符号之下，略去直觉，假定你已经掌握一半内容，而且在人工智能等快速发展的领域很快就会过时。这是一本开放、非传统的教科书，从基础开始讲解数学、计算机科学和人工智能。它面向希望深入理解这些内容，而不只是应付考试或面试的好奇实践者。

## 背景
过去几年从事 AI/ML 工作时，我把以直觉为先、结合现实背景，并且不靠含糊说辞来解释数学、计算机科学和人工智能概念的内容记满了一本本笔记。2025 年，几位朋友用这些笔记准备 DeepMind、OpenAI、Nvidia 等公司的面试。他们都成功入职，目前在各自岗位上表现良好。与此同时，我去年也进入了 Y Combinator。因此，我决定把这些内容分享给大家。

## MCP 服务器
本仓库包含一个 MCP 服务器，让任何 AI 助手（Claude Code、Cursor、VS Code 等）都能将这份知识汇编作为知识库使用。它需要在本地克隆本仓库，并提供用于教学的工具和示例实现。

## 目录

| # | 章节 | 摘要 | 状态 |
|---|---------|---------|--------|
| 01 | [向量](chapter%2001%20-%20vectors/01.%20vector%20spaces.md) | 空间、大小、方向、范数、度量、点积/叉积/外积、基、对偶性 | 可用 |
| 02 | [矩阵](chapter%2002%20-%20matrices/01.%20matrix%20properties.md) | 性质、特殊类型、运算、线性变换、分解（LU、QR、SVD） | 可用 |
| 03 | [微积分](chapter%2003%20-%20calculus/01.%20differential%20calculus.md) | 导数、积分、多元微积分、泰勒近似、优化与梯度下降 | 可用 |
| 04 | [统计学](chapter%2004%20-%20statistics/01.%20fundamentals.md) | 描述性度量、抽样、中心极限定理、假设检验、置信区间 | 可用 |
| 05 | [概率论](chapter%2005%20-%20probability/01.%20counting.md) | 计数、条件概率、概率分布、贝叶斯方法、信息论 | 可用 |
| 06 | [机器学习](chapter%2006%20-%20machine%20learning/01.%20classical%20machine%20learning.md) | 经典机器学习、梯度方法、深度学习、强化学习、分布式训练 | 可用 |
| 07 | [计算语言学](chapter%2007%20-%20computational%20linguistics/01.%20linguistic%20foundations.md) | 句法、语义、语用学、NLP、语言模型、RNNs、CNNs、注意力、Transformer、文本扩散、文本 OCR、MoE、SSMs、现代 LLM 架构、NLP 评估 | 可用 |
| 08 | [计算机视觉](chapter%2008%20-%20computer%20vision/01.%20image%20fundamentals.md) | 图像处理、目标检测、分割、视频处理、SLAM、CNN、视觉 Transformer、扩散、流匹配、VR/AR | 可用 |
| 09 | [音频与语音](chapter%2009%20-%20audio%20and%20speech/01.%20digital%20signal%20processing.md) | DSP、ASR、TTS、语音与声学活动检测、说话人分离、声源分离、主动降噪、wavenet、conformer | 可用 |
| 10 | [多模态学习](chapter%2010%20-%20multimodal%20learning/01.%20multimodal%20representations.md) | 融合策略、对比学习、CLIP、VLMs、图像/视频标记化、跨模态生成、统一架构、世界模型 | 可用 |
| 11 | [自主系统](chapter%2011%20-%20autonomous%20systems/01.%20perception.md) | 感知、机器人学习、VLAs、自动驾驶汽车、太空机器人 | 可用 |
| 12 | [图神经网络](chapter%2012%20-%20graph%20neural%20networks/01.%20geometric%20deep%20learning.md) | 几何深度学习、图论、GNNs、图注意力、Graph Transformers、三维等变网络 | 可用 |
| 13 | [计算系统与操作系统](chapter%2013%20-%20computing%20and%20OS/01.%20discrete%20maths.md) | 离散数学、计算机体系结构、操作系统、并发、并行、编程语言 | 可用 |
| 14 | [数据结构与算法](chapter%2014%20-%20data%20structures%20and%20algorithms/00.%20foundations.md) | Big O、递归、回溯、动态规划、数组、哈希、链表、栈、树、图、排序、二分查找 | 可用 |
| 15 | [生产级软件工程](chapter%2015%20-%20production%20software%20engineering/01.%20linux%20and%20CMD.md) | Linux、Git、代码库设计、测试、CI/CD、Docker、模型服务、MLOps、监控、使用编码智能体的最佳方式 | 可用 |
| 16 | [SIMD 与 GPU 编程](chapter%2016%20-%20SIMD%20and%20GPU%20programming/00.%20why%20C%2B%2B%20and%20how%20ML%20frameworks%20work.md) | 面向 ML 的 C++、框架工作原理、硬件基础、ARM NEON/I8MM/SME2、x86 AVX、GPU/CUDA、Triton、TPUs、RISC-V、Vulkan、WebGPU | 可用 |
| 17 | [AI 推理](chapter%2017%20-%20AI%20inference/01.%20quantisation.md) | 量化、高效架构、服务与批处理、边缘推理、推测解码、成本优化 | 可用 |
| 18 | [ML 系统设计](chapter%2018%20-%20ML%20systems%20design/01.%20systems%20design%20fundamentals.md) | 系统基础、云计算、分布式系统、ML 生命周期、特征存储、A/B 测试、推荐/搜索/广告/欺诈检测设计示例 | 可用 |

## 前言

新生儿的大脑就像一个刚初始化的神经网络，从现实世界的数据和经验中训练，直到成年……并一直持续下去。对法语有出色的理解并带有无可挑剔的口音，意味着曾正确接触过出色的法语和无可挑剔的口音。同样，优秀的 AI 研究人员和工程师拥有出色的问题解决能力，也意味着他们吸收了高质量的知识，并接触过丰富的经验。

克瓦什切夫实验是一项长期的塞尔维亚研究，表明持续三年的高强度创造性问题解决训练可以显著提升智力，尤其是流体智力，能增加 10–15 点智商。天生拥有高智商这种现象确实存在，就像高质量的权重初始化会带来更好的训练一样；有关先天与后天的实验结果也证明了这一点。

不过，高智商者真正拥有的唯一优势，只是学习或识别模式更快。但通过反复使用某种模式，任何概念都完全可以学会。在老师和父亲眼中，查尔斯·达尔文只是一个非常普通、甚至低于平均水平的学生。他形容自己并不机敏，觉得自己像一个需要时间吸收数据的“慢速处理器”。

3–10 岁时，我学习成绩很好，无须记笔记或复习就能自然理解概念。11–13 岁时我有些自负，仍采用这种方法，结果跌到了 80 人班级的后半段。14–15 岁时，我开始像普通学生一样阅读，最终在中学最后一个学期取得了第一名。早期学校课程很适合依靠天生智力，但现实世界中的才能来自高质量的知识摄取和高强度的执行。

事实上，大多数学习成绩优异的学生只是更加勤奋，但教育体系是为学习速度快的人设计的。这份知识汇编提供全面且衔接良好的知识脉络，帮助世上的“达尔文们”更好地学习。你只需要掌握初等数学和基础 Python 编程，其余内容都能在过程中学会；只管阅读，并相信这个过程！

## 如何更好地学习

大学第一学期，我一次修了 17 门课，成绩并不理想，因此采用了下面的方法：

**阶段 1**：课后累积阅读
每次课后、睡觉前阅读相应材料。下一次上课时，再从头读到当前进度，然后通过额外研究填补知识空白。这样可以让大脑把各种模式联系起来。

**阶段 2**：考前影子阅读
阅读每张幻灯片或每份笔记的小标题，合上书，然后在脑中想象并写出对该概念的解释。只重读遗漏的内容，这类似于机器学习中的掩码语言建模。重读后，最终再用代码实现该概念，从而为每个概念形成肌肉记忆。

这种方法对那些不太自信的朋友非常有效。事实上，其中一位朋友在高等工程数学课程中的成绩超过了我，那门课涵盖了 Hessian 矩阵和优化。如今她在一家大型石油与天然气公司工作。心灵的意愿比我们所使用的身体更重要（罗森塔尔实验）。

## Henry Ndubuaku 是谁？

请阅读 GitHub 个人资料！

## 引用
```bibtex
@book{ndubuaku2025compendium,
  title     = {Maths, CS & AI Compendium},
  author    = {Henry Ndubuaku},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/HenryNdubuaku/maths-cs-ai-compendium}
}
```
