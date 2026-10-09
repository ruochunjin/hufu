# HuFu 虎符

> 虎符相合，方为可信。古代调兵凭虎符，两半合验方为真；**HuFu** 为数据做同样的"合验"——检查两份数据是否真正*一致*：查询与检索结果、记忆与记忆、陈述与文档。

[English README](README.md)

HuFu 是一个**轻量级一致性检查工具**。向量相似度只能说明两段文本"接近"，HuFu 判断它们是否"一致"——给出可解释证据、可证明的多项式时间复杂度。判定由符号结构（语言超图、超图模拟、质量规则）完成；小型神经模型（嵌入、NLI）只作为规则的可测量输入。

## 快速开始

```bash
pip install hufu            # 无依赖参考核心
pip install "hufu[models]"  # 文本级检查（spaCy、嵌入、NLI）
python -m spacy download en_core_web_trf
```

```python
import hufu

# 1. RAG 检索后检：每篇检索文档是否与查询一致？
verdicts = hufu.check_retrieval(
    "Where was the author of 'The Old Man and the Sea' born?",
    [
        "Ernest Hemingway, who wrote The Old Man and the Sea, was born in Oak Park, Illinois.",
        "Paris is the capital of France and a major European city.",
    ],
)
for v in verdicts:
    print(v.consistent, v.distance)

# 2. 向量记忆冲突：同一对象、相反动作？
conflicts = hufu.detect_memory_conflicts([
    ("打开客厅的灯", {"object": "light", "action": "on"}),
    ("关闭客厅的灯", {"object": "light", "action": "off"}),
])
print(conflicts[0].labels, conflicts[0].explanations)

# 3. 训练数据异常值：按下游实测影响三分类
from hufu.scenarios import classify_outliers
reports = classify_outliers(records, labels, evaluator=train_and_score)
```

每个判定都带证据溯源与完整决策链——HuFu 会解释*为什么*做出这个判定。

## 应用场景

1. **RAG 检索一致性后检**：挂在任意向量库检索与生成之间，将每篇检索文档标记为 `consistent / conflict`，附证据与距离。
2. **文档矛盾检测**：星型图规则检查文档自身陈述——类型化值槽交换、数值不匹配（含单位换算）、角色反转、反义词对、极性翻转。
3. **向量记忆冲突检测与修剪**：CDR 风格规则检测"开灯 vs 关灯"式冲突记忆；判定见证推断、证据级修剪与 HNSW 图索引级修剪。
4. **训练数据异常值管理**：检测器 + good / bad / ugly 三分类 + 修复方案族（KNN/RF 标签修复、值插补、删除）+ 流水线网格搜索。
5. **统一质量规则语言**：一套 JSON 规则模式（逻辑、NLI、相似度、数值/时态谓词），注册表可插拔；规则是数据——可手写、可版本化、可离线发现。

## 适用范围与边界

HuFu 刻意**精度优先**：确信结构性矛盾并给出可解释证据，不追求覆盖所有语义细微差别。

- **可靠子域（默认 `profile="strict"`）**：纯结构可证的冲突——类型化值槽交换（生于 X 与 Y、300 与 3000 人、5 月与 6 月）、非对称动词的角色反转、反义词对、相同框架上的极性翻转。
- **明确不做**：叙事张力、视角/情感变化、转述级分歧、时间*演进*（故事合法地在不同时间给同一属性不同值）——真实长篇叙事本身就携带与注入冲突相同的表面特征，这是方法边界，不是工程缺口。

一句话：用 HuFu 在结构化/半结构化内容（知识库、事实表、记录、整理过的文档）上*确信*矛盾、做 RAG 检索一致性后检、做向量记忆冲突检测，均附完整溯源。长篇叙事矛盾猎取明确不属于本工具范围。

## 安装

```bash
pip install hufu               # 无依赖参考核心（Python ≥ 3.10）
pip install "hufu[models]"     # + 文本级检查（spaCy、sentence-transformers、fastcoref）
pip install "hufu[outliers]"   # + sklearn 异常值分类与修复
python -m spacy download en_core_web_trf
pip install ./rust/simulation  # Rust 不动点加速器（需 Rust 工具链）
```

模型均通过环境变量配置：`HUFU_EMBED_MODEL` / `HUFU_NLI_MODEL` / `HUFU_COREF_MODEL` / `HUFU_SPACY_MODEL`。

## 学术来源

HuFu 整合了课题组多篇论文成果：

- *Catching Inconsistencies in Vector Similarity Retrieval*（HyperMatch 超图模拟一致性检测）
- *Conflict-Aware Memory for Embodied Agents: Enhancing Vector Data Quality via Detection Rules*（CDRs，ACL 2026）
- *Detecting Contradictions Within and Across Documents*（DocChecker / RICO 规则）
- *Outliers: The Good, the Bad and the Ugly*（OHunt / OMR 规则）

## 许可

Apache-2.0（见 LICENSE）。
