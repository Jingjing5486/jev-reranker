# Jev 决策模型 vs bge-reranker-v2-m3 对比方案（v4）

> 调研日期：2026-11-25
> 硬约束：reranker ≤ 8B，权重可下载，本地/Colab 仅推理。
> 前提：**现有生产 baseline = `BAAI/bge-reranker-v2-m3`**（内网服务 `http://bge-reranker-v2-m3...:31088/rerank`），本方案围绕"Jev 决策模型能否替代/超越它"展开。

---

## 一、Baseline 画像（对比锚点）

| 维度 | bge-reranker-v2-m3 |
|---|---|
| 参数 | **568M（0.56B）** |
| 架构 | XLM-RoBERTa **cross-encoder**（联合打分） |
| 上下文 | 8192 token |
| 语言 | 100+ 语言，**中文强项**（BAAI，中英混合优化） |
| 显存 | ~1.2GB（FP16） |
| License | Apache-2.0 / MIT |
| **BEIR nDCG@10** | **56.51** |
| **MIRACL 多语 nDCG@10** | **69.32** |
| 输出 | 相关度 logit（需定阈值） |

---

## 二、⚠️ 关键风险发现：中文能力

你的业务是 CICC 中文金融场景，但 ≤8B 的 Jev 候选里，**中文能力分化严重**，这是筛选的第一优先项：

| 候选 | 底座 | 中文能力 | 证据 |
|---|---|---|---|
| **KaLM-Reranker-V1**（Nano/Small/Large） | T5Gemma2（英文为主） | **偏弱** | 作者自己在 HF 开 issue 承认 "Chinese retrieval consistently weak" |
| **openjev**（0.8/2/4B） | Qwen3.5，但**英文 NLI 微调**（MNLI/SciTail 等） | **存疑** | 模型卡 language: en，评测全是英文基准 |
| **NeoHorse-Jev-4B** | Qwen3.5-4B + 国内团队 | **强** | 有中文基准 `zh-decision-bench` 实测：语音路由 0.950、客服路由 0.912 |

**结论**：如果你的测试语料以中文为主，**`NeoHorse-Jev-4B` 反而应提到最高优先级**（中文有实测证据），而不是只看英文 BEIR 数据高的 `KaLM-Reranker-V1`。

---

## 三、对比维度设计（7 个角度）

| # | 角度 | 怎么测 | 关键指标 |
|---|---|---|---|
| 1 | **排序质量（核心）** | 几百条带标注的 (query, 候选) 集 | nDCG@1/3/5/10、MRR、Recall@k、Top-1 命中率 |
| 2 | **中文/多语言（一票否决项）** | 中文 query 子集单独切出来 | 中文子集上的 nDCG/MRR，对比英文子集 |
| 3 | **难负例区分度** | 构造"看起来相关但不相关"的难负例 | 难负例的分数压制、正负分差 |
| 4 | **分数校准/阈值稳定性** | 看分数分布与阈值切分 | bge 是 logit、Jev 是 0-1 校准概率；跨 query 可比性 |
| 5 | **延迟/吞吐** | 几百条批量打分计时 | 单条 P50/P95、批量 docs/s、总耗时 |
| 6 | **长文本** | 长文档 chunk 子集 | bge 8k vs KaLM MEP 压缩 128× vs openjev 4k |
| 7 | **参数量-成本曲线** | 横向比显存/参数量 | 0.27B~4.66B 对 0.56B baseline 的性价比 |

---

## 四、选型清单（按中文场景重新排序）

**必测（核心对照，3 个）：**

1. **NeoHorse-Jev-4B**（4.66B，Apache-2.0，中文强）
   - 中文场景最该测的 Jev 决策模型；`zh-decision-bench` 中文实测（路由 0.950、客服 0.912）。
   - HF `TokenRhythm/NeoHorse-Jev-4B`（9.15GB），GGUF `DreamBlooms/NeoHorse-Jev-4B-GGUF`；用 Noul("该文档是否相关") 做 rerank。

2. **KaLM-Reranker-V1-Large**（4B，Apache-2.0，专门 reranker）
   - 英文 BEIR 62.87 **已超 bge-m3 的 56.51**；验证"专门 reranker 架构"的英文优势能否迁移到你的数据。
   - 但**中文弱**，务必单独看中文子集表现。HF `KaLM-Embedding/KaLM-Reranker-V1-Large`。

3. **openjev 4B**（MIT，cross-encoder 同类）
   - 与 bge-m3 **同架构**（都是交叉编码器），最公平的"决策模型化交叉编码器 vs 传统交叉编码器"对比。
   - HF `AlexWortega/openjev`（subfolder `qwen3.5-4b-nli-v5`），用 entailment 概率打分；中文存疑。

**降配对照（2 个，看规模-质量曲线）：**

4. **KaLM-Reranker-V1-Small**（1B）—— 1B 能否接近 0.56B 的 bge-m3。
5. **KaLM-Reranker-V1-Nano**（0.27B）—— 极致轻量（比 baseline 还小一半），测损失多少。

**可选：** `Kev-4B`（Qwen3.5 底座，中文可能较好，但无专门 BEIR 数据）。

---

## 五、BEIR/中文数据对照表

| 模型 | 参数 | BEIR nDCG@10 | MIRACL | 中文能力 |
|---|---|---|---|---|
| **bge-reranker-v2-m3（baseline）** | 0.56B | 56.51 | 69.32 | 强 |
| KaLM-Nano | 0.27B | 57.41 | 62.08 | 弱 |
| KaLM-Small | 1B | 60.01 | — | 弱 |
| KaLM-Large | 4B | **62.87** | **70.07** | 弱 |
| openjev 4B | 4B | 无 BEIR（有 NLI 基准） | — | 存疑 |
| NeoHorse-Jev-4B | 4.66B | 无 BEIR | — | 强（zh-bench） |

> 参考：Qwen3-Reranker-4B = 63.50、Qwen3-Reranker-8B = 65.11（同口径 BEIR，供感知上限）。

---

## 六、落地建议

1. **先明确语料语言分布**：中文为主 → `NeoHorse-Jev-4B` 权重最高；英文为主 → `KaLM-Reranker-V1-Large` 最可能赢。
2. **测试集分中文/英文子集**，分别报指标，别混在一起（中文风险会被英文高分掩盖）。
3. **固定打分原语**：Jev 模型统一用 Noul（"该文档是否回答该 query"），openjev 用 entailment，bge-m3 用 logit，都归一化到 0-1 再比排序。
4. **先跑通 3 个必测 + baseline**，几百条在 A100 40GB / 本地 24GB 卡上都能原版直接跑。

---

## 七、参考来源

- bge-reranker-v2-m3 官方模型卡与 MTEB/BEIR 公开数据（BEIR 56.51、MIRACL 69.32）
- KaLM-Reranker-V1 / Lychee-Reranker 论文（arXiv:2606.22807）与作者 HF discussion（中文弱）
- `AlexWortega/openjev` 模型卡（language: en）
- `NeoHorse-Jev-4B` 官方与 `zh-decision-bench` 中文基准（github.com/CodyQin/zh-decision-bench）
