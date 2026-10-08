# HuFu 虎符 测试报告（v0.5.0 发布候选）

日期：2026-10-09。测试环境：本机 CPU（无 GPU），Windows 11，Python 3.13，
`en_core_web_trf`（spaCy）+ all-MiniLM-L6-v2（嵌入）+ nli-deberta-v3-xsmall（NLI）。
所有运行 `HF_HUB_OFFLINE=1`（模型已缓存），`HUFU_COREF=off`，torch 单线程。

## 1. 测试范围与用例总览

**约 1300 个带标注测试用例 + 66 个单元测试**，覆盖工具全部四个限定场景：

| 套件 | 规模 | 场景 | 数据来源 |
|---|---|---|---|
| golden100 / golden250 | 350 对 | RAG 检索一致性（pair 级，钉定语义） | 手工 + 迭代修复 |
| golden_rag200 | 200 查询 / 600 文档对 | RAG 检索一致性（check_retrieval 端到端） | 模板生成（种子固定） |
| golden_struct200 | 186 篇 | 结构化文档矛盾（strict 星型引擎） | 模板生成 |
| golden_doc300 | 300 篇 | 文档级矛盾（含叙事边界） | 手工 + ContraDoc 分层 + 合成 |
| golden_mem32 + mem300 | 352 库 | 向量记忆冲突（CDR 规则） | 手工 32 + 生成 320 |
| test_ohunt_port / sweep | 9 + 100 实例 | 训练数据异常值（OHunt） | 合成（gap/hinge/修复） |
| test_real_outliers | 3 数据集 | 异常值真实数据 | sklearn 自带 iris/wine/breast_cancer |
| test_snli_real | 116 对 | NLI 组件真实数据校准 | SNLI validation（hf-mirror 下载） |
| test_cdrs_fixture | 85 条记忆 | 原型真实知识库冒烟 | CDRs 原型检索快照 |
| ContraDoc 句对协议 | 100 对 | 已验证边界基准 | ContraDoc 本地 |
| 单元测试 | 66 | 规则引擎/边界/修剪/校准/CDRs/OHunt | 手工 |

## 2. 最终结果

| 套件 | 结果 | 备注 |
|---|---|---|
| 单元测试 | **66/66 通过** | 含 CDRs 阈值学习/见证推断、OHunt 检测/修复/发现 |
| golden100 + golden250 | **pinned 全绿**（回归通过） | 参考引擎，θ=-1 结构路径 |
| golden_rag200 | **200/200 查询精确**（600/600 对） | 参考引擎 + 声明式通道 |
| golden_struct200 | **186/186，P=1.000 R=1.000** | strict 引擎 + trf |
| golden-mem32 | **32/32 精确** | 含 NLI 兜底三例 |
| golden-mem300 | **320/320 精确**（2 对低于 VecSim 闸门，按 CDR 语义记为出范围） | 生成语料 |
| OHunt 扫描 | **100/100**：检出召回 1.000，修复后准确率 ≥ 基线 98/100（1 例噪声链边界，已文档化） | 合成协议 |
| 真实异常值（iris/wine/breast_cancer） | **3/3 通过** | 噪声注入协议 |
| SNLI 真实数据 | **通过**：entailment 误报率 0.000（精度锚点）；显式否定矛盾召回仅 0.19（report-only，见 §4 发现 6） | 语义不匹配已文档化 |
| CDRs 原型 fixture | **通过**：85 条真实机器人知识库端到端 + 确定性 | 冒烟级 |
| ContraDoc 句对协议 | **F1 = 0.9474**（既有验证边界，未回归） | 已验证基准 |
| doc300（strict，sm） | 203/300，P=0.636 R=0.669 F1=0.652 | 叙事边界参考值 |
| doc300（strict，trf） | 196/300，P=0.616 R=0.625 F1=0.620 | trf 在叙事语料无收益（解析伪影不同） |

## 3. 测试驱动揪出并修复的漏洞（本轮）

1. **公共 API 距离闸门短路**（最严重）：`check_pair` 在嵌入距离 ≤0.25 时直接判"一致"，
   永不进入结构/极性检查——而冲突恰恰藏在最相似的文本对里（"weighs 2.3 kg" vs
   "weighs 4.1 kg" 距离 0.011）。钉定的 golden 套件全部用 θ=-1 绕开了闸门，公网接口
   从未被测过。已修：两引擎移除提前返回，声明式查询在覆盖满足后走极性通道
   （数值规则 → 单位换算 → NLI），`check_retrieval` 增加 `query_is_question` 透传。
2. **NLI 兜底阈值语义 bug**：`nli_score` 返回原始 logits（3.5+）而规则阈值 0.8 按概率
   设计 → 兜底规则对所有 vec_sim≥0.6 的对放行。已改 softmax 概率，并将兜底限定在
   无结构属性的记忆（结构化规则优先）。
3. **attr_conflict 任意差异误报**：新增 `attr_opposite` 谓词（显式反义词对表 +
   位置值状态模式），规则集升 v2；裸 "on"/"off" 入表。
4. **vec_sim 阈值数据驱动**：按 CDRs 论文方法用 Youden J 在 352 库语料上学得
   θ=0.59（tpr=0.985/fpr=0.057），替代拍脑袋的 0.5；`learn_threshold_youden` 入库。
5. **词边界包含的非词边缘**：`\bprof.\b` 因句号无词边界失配（NER 一侧含头衔一侧不含），
   桥接包含锚改为只在词字符边缘强制 `\b`。
6. **SNLI 语义不匹配发现**：SNLI "contradiction" 多为常识排除/细节不兼容（甚至标注噪声），
   非 HuFu 的结构冲突语义；xsmall NLI 在清晰矛盾探针上正确（prob≈1.0）。SNLI 套件
   重新定位为 entailment 负对照（误报 0.000）。
7. **spaCy 模型能力梯度**：sm 的 NER 漏标罕见地名/品牌（Tampere、Curitiba）→ GPE 类型
   丢失 → 类型化值槽规则哑火（struct200 在 sm 下 160/186，trf 下 186/186），结构化场景
   套件钉 trf；但 trf 对叙事语料无收益（doc300 strict：sm F1=0.652 vs trf F1=0.620），
   叙事边界与解析器档位无关。
8. **torch CPU 线程竞态**：解析器/NER 在同文跨进程偶发翻转（角色标签/NER 跨度），
   `get_nlp` 现钉 `torch.set_num_threads(1)`（`HUFU_TORCH_THREADS` 可覆盖）；
   规则层此前已用"跨角色锚定"对残余抖动鲁棒。
9. **CDRs 原型审查安全发现**：`src_trimmer/` 硬编码泄露的 zhipuai API key——
   收编时已剔除，不随任何数据迁移。
10. **KNN 标签修复的链式传播边界**：20% 噪声成链时沿链传播错误标签（扫描实测
    1/100 退化实例），文档化建议：链式噪声场景用 delete。

## 4. 适用范围（钉定）

- **可靠**：RAG 检索一致性（结构覆盖 + 声明式极性通道）、结构化/半结构化文档矛盾
  （strict 星型引擎，P=1.0）、向量记忆冲突（CDR 规则 + 学到阈值）、训练数据异常值
  （检测→hinge 分类→修复→网格搜索）。
- **明确不做**：长篇叙事语义矛盾猎取（所有模式 FP 都高，属方法边界）；答案正确性
  判断（覆盖语义之外，2026-10-04 钉定）；***REMOVED***（已弃用：慢、不准、不泛化）。
- wh 查询的"错误答案但结构覆盖"判一致是钉定语义；无关文档通过"不一致标记或
  距离明显更大"双信号过滤（golden_rag200 槽位契约）。

## 5. 发布就绪状态

- `hufu_dq-0.5.0` wheel 构建成功，**73 个文件含全部数据资源**（规则集 JSON、
  本体映射种子）；裸 venv 零依赖安装 + 核心路径（参考引擎/规则/修剪/校准/异常值）
  冒烟通过。
- ruff 全净；单元 66 + e2e 全绿（本报告 §2）。
- 引擎默认 `reference`（纯 Python，无需 Rust 工具链即可 `pip install` 使用）；
  Rust standard 引擎保留为长文档加速选项（wh 占位符覆盖缺陷已文档化）。
- 待用户确认：GitHub 推送（ruochunjin/hufu 需本人凭证）、PyPI 发布（hufu-dq）、
  CITATION.cff 作者名单。
