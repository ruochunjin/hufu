# HuFu 虎符

> 虎符相合，方为可信。古代调兵凭虎符，两半合验方为真；**HuFu** 为大模型时代的数据做同样的"合验"——检查两份数据是否真正*一致*：查询与检索结果、记忆与记忆、陈述与文档。

[English README](README.md)

HuFu 是一个**神经符号数据质量工具包**。向量相似度只能说明两段文本"接近"，HuFu 判断它们是否"一致"——给出可解释证据、可证明的多项式时间复杂度、判定路径上无需 LLM。神经模型（嵌入、NLI、可选 LLM）按需调用，最终判定由符号结构（语言超图、超图模拟、质量规则）完成。

## 四大应用场景

1. **RAG 检索一致性后检器**（v0.1）：挂在任意向量库检索与生成之间，将每篇检索文档标记为 `consistent / conflict / useless / irrelevant`，附证据。
2. **向量库/记忆冲突检测与修剪**（v0.2/v0.3）：CDR 风格规则检测"开灯 vs 关灯"式冲突记忆；证据级修剪（贪心最小覆盖）与 HNSW 图索引级修剪。
3. **文档内/跨文档矛盾检测**（v0.4）：RICO 风格规则 + NLI 兜底，每条发现带证据溯源（文档/句子位置 + 规则判定轨迹）。
4. **训练数据异常值管理**（v0.4/v0.5）：good / bad / ugly 三分类——v0.5 完整收编 OHunt 论文语义（gap 孤立检测器的 gap-of-gaps 归一化、hinge 三段分类、KNN 标签修复等五种修复方案、OLearner-lite 流水线网格搜索）。
5. **向量记忆冲突**（v0.5 强化）：CDR 规则升级为反义词对表语义（不再任意差异即冲突），NLI 兜底限定无结构属性记忆且修正为 softmax 概率；新增 Trimming 定理见证推断（同一 query 下判定分岐 ⇒ 互相冲突）与 Youden 阈值学习（纯 stdlib）。

## 统一质量规则语言

一套 JSON 规则模式融合 CDR / RICO / OMR 三种规则体系（逻辑谓词 + ML 谓词 + 相似度谓词），谓词注册表可插拔；规则可手写、可版本化，也可用 LLM 从标注样本中**自动发现**（覆盖率 + 准确率双阈值验证，运行期无需 LLM）。

## 适用范围与边界

HuFu 的文档检查器刻意**精度优先**：目标是*确信*结构性矛盾并给出可解释证据，而不是覆盖所有语义细微差别。基于 300 篇 golden 语料（`tests/e2e/golden_doc300.py`：手工规则族 + 分层 ContraDoc 正负例 + 合成硬负例）的实测结论：

- **可靠子域（默认 `profile="strict"`）**：纯结构可证的冲突——类型化值槽交换（生于 X 与 Y、300 与 3000 人、5 月与 6 月）、非对称动词的角色反转（sold/hired/bit）、反义词对（open/closed、increase/decrease）、相同框架上的极性翻转（attended 与 did not attend）。这些规则不调用任何模型，每个判定附带完整决策链。
- **刻意不纳入 strict 判定**：叙事张力、视角/情感变化、转述级分歧、以及时间*演进*（故事合法地在不同时间给同一属性不同值）。真实长篇叙事本身就含有大量与注入冲突相同的表面特征——干净的 ContraDoc 故事会高频触发结构规则——因此这些通道（相似度+NLI 线索、弱否定线索）仍计算并记录在决策链中，但不改变 strict 判定。
- **精度模式（`profile="full"` + `***REMOVED***=True`）**：线索通道由 LLM ***REMOVED******REMOVED***（默认本地 ***REMOVED*** 小模型；可接任何 OpenAI/Anthropic 兼容端点获得更高精度）。处理长篇叙事且更看重召回时使用。

一句话：用 HuFu 在结构化/半结构化内容（知识库、事实表、记录、整理过的文档）上*确信*矛盾并给出完整溯源；长篇叙事的结果应视为候选，交给 LLM ***REMOVED***模式裁决。

300 篇冻结语料实测（2026-10-07）：

| 模式 | 正确 | 精确率 | 召回率 | F1 | 耗时 |
|---|---|---|---|---|---|
| 纯规则（strict） | 203/300 | 0.636 | 0.669 | 0.652 | 12 分钟 |
| 本地 ***REMOVED*** ***REMOVED***（full） | 176/300 | 0.523 | 0.985 | 0.684 | 3.4 小时 |
| Claude API ***REMOVED***（full） | 201/300 | 0.593 | 0.868 | 0.704 | 6.3 小时 |

strict 是唯一误报可控的配置（合成硬负例 50/50、手工干净文档 14/15）；本地 ***REMOVED*** 判定器过度放行，是召回放大器而非精度闸门；API ***REMOVED***召回/F1 最高，但也无法修复结构规则在干净叙事文本上的信号过载——这一残余是方法边界，不是工程缺口。

*****REMOVED***已弃用（2026-10-08，实测结论）**：`full` + `***REMOVED***=True` 路径保留代码仅作复现留档，**不推荐使用**——本地 ***REMOVED*** 在叙事上过度放行（P=0.523），在线 API 慢（~76 秒/篇）且仍无法修复线索通道在干净叙事上的信号过载，两者对未见场景泛化都不好。strict 纯结构判定是唯一支持的判定路径。

一句话：用 HuFu 在结构化/半结构化内容（知识库、事实表、记录、整理过的文档）上*确信*矛盾、做 RAG 检索一致性后检、做向量记忆冲突检测，均附完整溯源。长篇叙事矛盾猎取明确不属于本工具范围。```python
import hufu

conflicts = hufu.detect_memory_conflicts([
    ("打开客厅的灯", {"object": "light", "action": "on"}),
    ("关闭客厅的灯", {"object": "light", "action": "off"}),
])
for c in conflicts:
    print(c.labels, c.explanations)
```

## 安装

```bash
pip install -e .            # 无依赖参考核心（Python ≥ 3.10）
pip install -e ".[models]"  # + 文本级检查（spaCy、sentence-transformers、fastcoref）
pip install -e ".[llm]"     # + LLM 辅助功能（OpenAI 兼容端点）
python -m spacy download en_core_web_trf
pip install ./rust/simulation   # Rust 不动点加速器（需 Rust 工具链）
```

模型均通过环境变量配置：`HUFU_EMBED_MODEL` / `HUFU_NLI_MODEL` / `HUFU_COREF_MODEL` / `HUFU_SPACY_MODEL`；LLM 端点：`HUFU_API_KEY` / `HUFU_BASE_URL` / `HUFU_MODEL`。

## 学术来源

HuFu 整合了课题组多篇论文成果：

- *Catching Inconsistencies in Vector Similarity Retrieval*（HyperMatch 超图模拟一致性检测）
- *Conflict-Aware Memory for Embodied Agents: Enhancing Vector Data Quality via Detection Rules*（CDRs，ACL 2026）
- *Detecting Contradictions Within and Across Documents*（DocChecker / RICO 规则）
- *Outliers: The Good, the Bad and the Ugly*（OHunt / OMR 规则）

## 许可

Apache-2.0（见 LICENSE）。
