# Automatic-text-summarization
自动文本摘要

### Background

**应用：**搜索引擎、观点提取、新闻提取

**文本摘要方法分类：**

1.  抽取式（Extractive）：

- 从原文抽取字句形成摘要；
- 实现简单、保证句子可读性；
- 内容冗余、句子连贯性差；

2. 摘要（Abstractive）：

- 重新整理并生成新的句子
- 利用了句子压缩、重构、融合方法
- 生产句子质量高、困难大

**其他分类：**

- 单文档和多文档
- 单语言与多语言
- 有监督与无监督
- 通用摘要和个性化摘要

**难点：**

1. 缺乏标注数据
2. 文本理解复杂、提取重要内容难



### Basic Technologies

**PageRank：**

1. 用途：网页排序

2. 原理：计算网页的相关性和重要性。基于投票的思想，如果要计算网页A的**PR值（PageRank）**，需要得到网页A的入度，然后通过入度给网页A的投票来计算网页A的PR值。
   $$
   S(V_i) = (1-d) + d*\sum_{V_j∈In(V_i)}\frac{1}{|Out(V_j)|}S(V_j)
   $$
   Vi：表示某个网页

   Vj：表示连接到 Vi的网页（即Vi的入度）

   S（Vi）：表示网页Vi的PR值

   In（Vi）：表示Vi所有入度的集合

   Out（Vj）：表示Vj所有出度的集合

   |Out（Vj）|：表示集合的数量，用于平分Vj节点的PR值

   d：阻尼系数、用于处理没有入度的网页节点

**TextRrank：**

1. 用途：提取关键词、自动摘要

2. 原理：改进PageRank而来，比PageRank多了一个权重项**Wji**，用于表示两个节点之间连接的不同重要程度
   $$
   WS(V_i)=(1-d)+d*\sum_{V_j∈In(V_i)}{\frac{w_{ji}}{\sum_{V_k∈In(V_i)}{w_{jk}}}}WS(V_j)
   $$

3. TextRank-关键词提取

- 给定文本进行分句、只保留指定词性的单词，如名词、动词、形容词，作为候选关键词。
- 构建候选关键词图G=（V，E），其中V为关键词集合，然后采用共现关系构造任意两点之间的边，两个节点之间存在边当且仅当它们对应的词汇在长度为K的窗口中出现，K表示窗口大小，即最多共现K个单词。
- 根据公式，迭代传播，收敛
- 对节点权重进行倒排，选择前T个作为关键词

4. TextRank-生成摘要

- 将文本中的每个句子分别看做一个节点，如果两个句子有相似性，那么存在一个无向有权边。
  $$
  Similarity(S_i,S_j)=\frac{|\{w_k|w_k∈S_i\&w_k∈S_j\}|}{log(|S_i|)+log(|S_j|)}
  $$
  Si、Sj：分别表示两个句子

  Wk：表示句子中的词

  分子：同时出现在两个句子中的同一个词的个数

  分母：句子中词的个数求对数之和（遏制长句子在相似度计算上的优势）

- 根据阈值去掉两个节点之间相似度较低的边连接，然后计算TextRank值，倒排，选出TextRank值最高的几个节点作为摘要

**TF-IDF**

1. 用途：提取关键词（关键词重要性度量）

2. 原理：TF-IDF即考虑了词频、也兼顾了单词的普遍性

   TF表示单词w在文档Di中出现的频率；

   IDF（逆文档概率）反映单词的普遍程度，当一个词越普遍，其IDF值越低；反之，则IDF越低。
   $$
   TF_{w,D_i}=\frac{count(w)}{|D_i|}\\
   IDF_w=log\frac{N}{1+\sum_{i=1}^{N}I(w,D_i)}
   $$
   N：文档总数

   I（w，Di）：表示文档Di是否包含该单词，若包含则1，否则为0；

   平滑：分母加1，避免词w在所有文档都没出现过

   因此，单词w在文档Di的TF-IDF值为：
   $$
   TF-IDF_{w,D_i}=TF_{w,D_i}*IDF_w
   $$


**总结：**



### Reference：

TextRank:Bringing Order into Texts : https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf

TF-IDF：https://www.cnblogs.com/en-heng/p/5848553.html

TextRank：https://www.cnblogs.com/xueyinzhe/p/7101295.html