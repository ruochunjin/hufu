# HuFu 虎符

> 虎符相合，方为可信。古代调兵凭虎符，两半合验方为真；**HuFu** 为大模型时代的数据做同样的"合验"——检查两份数据是否真正*一致*：查询与检索结果、记忆与记忆、陈述与文档。

[English README](README.md)

HuFu 是一个**神经符号数据质量工具包**。向量相似度只能说明两段文本"接近"，HuFu 判断它们是否"一致"——给出可解释证据、可证明的多项式时间复杂度、判定路径上无需 LLM。神经模型（嵌入、NLI、可选 LLM）按需调用，最终判定由符号结构（语言超图、超图模拟、质量规则）完成。

## 四大应用场景

1. **RAG 检索一致性后检器**（v0.1）：挂在任意向量库检索与生成之间，将每篇检索文档标记为 `consistent / conflict / useless / irrelevant`，附证据。
2. **向量库/记忆冲突检测与修剪**（v0.2/v0.3）：CDR 风格规则检测"开灯 vs 关灯"式冲突记忆；证据级修剪（贪心最小覆盖）与 HNSW 图索引级修剪。
3. **文档内/跨文档矛盾检测**（v0.4）：RICO 风格规则 + NLI 兜底，每条发现带证据溯源（文档/句子位置 + 规则判定轨迹）。
4. **训练数据异常值管理**（v0.4）：good / bad / ugly 三分类——按对下游模型的*实测影响*分类，ugly 修复而非删除，good 保留，bad 忽略。

## 统一质量规则语言

一套 JSON 规则模式融合 CDR / RICO / OMR 三种规则体系（逻辑谓词 + ML 谓词 + 相似度谓词），谓词注册表可插拔；规则可手写、可版本化，也可用 LLM 从标注样本中**自动发现**（覆盖率 + 准确率双阈值验证，运行期无需 LLM）。

```python
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
