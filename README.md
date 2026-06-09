# Coursework
# UniTE (Determine-Then-Ensemble) 与 AdaWC 改进

本文代码基于论文 [Determine-Then-Ensemble: Necessity of Top-k Union for Large Language Model Ensembling](https://arxiv.org/abs/2410.03777) 的官方实现，并在其 UNIon Top-k Ensembling (UNITE) 方法上提出了 **AdaWC（Adaptive Weight Change，动态自适应权重变化）** 改进。

## 1. 项目背景

大语言模型（LLMs）在不同任务上表现各异，各有长短。模型集成（Model Ensembling）旨在融合多个模型的互补优势，但现有的 LLM 集成方法往往忽视模型间的兼容性，并且在全词表概率对齐上开销巨大。

论文通过实证研究发现，影响集成效果的关键因素包括：**模型性能、词表大小以及回答风格**。其中模型之间的兼容性对集成效果至关重要。基于此，论文提出了 **UNIon Top-k Ensembling（UNITE）**：通过聚焦每个模型 top-k token 的并集来高效融合模型，避免对全词表做对齐，显著降低了计算开销。

## 2. AdaWC 改进（本文工作）

虽然 UNITE 在静态权重（默认 λ = 0.5，即两个模型等权融合）下已经取得不错效果，但在实际推理中我们发现：

- 不同模型在不同生成步骤的“可靠性”并不一致；
- 当两个模型 top-1 token 完全一致时，做加权和融合是冗余计算；
- 固定的 0.5 权重在一方模型明显占优的场景下，会拖累整体质量。

为此，我们在 UNITE 的融合阶段引入了 **AdaWC（Adaptive Weight Change）动态自适应权重变化** 机制。AdaWC 包含以下两个核心组件：

### 2.1 动态自适应权重（Adaptive Weight）

为每条样本维护一个长度为 `history_window` 的滑动窗口，分别记录模型 1 与模型 2 在最近若干步中“被选中的 token 在自身 top-k 词表中对应的归一化概率”作为该模型的历史得分。

每一步的融合权重按下式动态计算：

$$
\bar{s}_1 = \frac{1}{W}\sum_{t \in \text{hist}_1} s_1^{(t)},\quad
\bar{s}_2 = \frac{1}{W}\sum_{t \in \text{hist}_2} s_2^{(t)}
$$

$$
w_1 = \mathrm{clip}\!\left(\frac{\bar{s}_1}{\bar{s}_1 + \bar{s}_2 + \epsilon},\; \text{min\_weight},\; 1-\text{min\_weight}\right),\quad
w_2 = 1 - w_1
$$

其中 `min_weight` 用于防止任一模型被完全压制（例如设为 0.2 或 0.45）。这使得在生成过程中表现更稳定、置信度更高的模型能自动获得更大的融合权重。

### 2.2 冗余融合跳过（Redundant Fusion Skip）

当两个模型在当前步的 top-1 token 完全一致时，说明双方对当前续写已经形成强共识，再做加权融合并不会改变最终结果，反而引入了不必要的计算开销。此时 AdaWC 会跳过 `average_and_sample` 的加权步骤，直接采用该共同 top-1 token 作为输出，并复用当前权重更新历史得分。

### 2.3 AdaWC 带来的优势

- **质量提升**：动态权重能根据生成过程中模型的实时表现自适应调整，让更强的模型在关键时刻获得更大话语权；
- **效率提升**：冗余融合跳过去除了大量无意义的加权计算；
- **可调节性**：通过 `history_window` 与 `min_weight` 两个超参，可以在不同任务（短答案 QA、长链推理、CoT）上灵活平衡稳定性与自适应能力。

## 3. 代码结构

```
UniTE-main/
├── README.md                 # 本文档
├── unite2.py                 # 原版 2 模型 UNITE
├── unite3.py                 # 原版 3 模型 UNITE
├── unite2_pro.py             # 加入 AdaWC 的 2 模型版本（TriviaQA）
├── unite2_pro_mmlu.py        # 加入 AdaWC 的 2 模型版本（MMLU）
├── unite2_pro_simplemath.py  # 加入 AdaWC 的 2 模型版本（SimpleMath）
├── unite2_pro_triviaqa.py    # 加入 AdaWC 的 2 模型版本（TriviaQA 变体）
├── unite2_mmlu.py            # 原版 2 模型 MMLU
├── unite2_simplemath.py      # 原版 2 模型 SimpleMath
├── unite2_triviaqa.py        # 原版 2 模型 TriviaQA
├── unite_mmlu.py             # 原始 MMLU 评估脚本
├── utils/                    # 答案抽取与后处理工具
├── datasets/                 # 评测数据
│   ├── ARC-challenge/
│   ├── GSM/
│   ├── MMLU/
│   ├── NaturalQuestions/
│   ├── PIQA/
│   ├── simplemath/
│   └── TriviaQA/
├── figures/                  # 论文配图
└── result*/                  # 评测输出
```

带 `_pro_` 后缀的脚本即为本工作加入 AdaWC 之后的改进版本。

## 4. 使用方式

### 4.1 环境

- Python ≥ 3.9
- PyTorch（CUDA）
- `transformers`, `accelerate`, `datasets`, `tqdm`, `numpy`

```bash
pip install torch transformers accelerate datasets tqdm numpy
```

### 4.2 运行原版 UNITE

2 模型集成：

```bash
python unite2.py
```

3 模型集成：

```bash
python unite3.py
```

### 4.3 运行带 AdaWC 的改进版本（推荐）

以 MMLU 为例（其余任务类似，将脚本名替换即可）：

```bash
python unite2_pro_mmlu.py \
    --model_path1 /path/to/Qwen3-4B \
    --model_path2 /path/to/Llama-3.2-3B-Instruct \
    --output_file ./result_pro_mmlu.jsonl \
    --history_window 2 \
    --min_weight 0.45
```

各 `_pro_` 脚本通用参数说明：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model_path1` | Qwen3-4B | 模型 1 路径 |
| `--model_path2` | Llama-3.2-3B-Instruct | 模型 2 路径 |
| `--output_file` | `./result_pro_*.jsonl` | 结果输出文件 |
| `--per_device_batch_size` | 1 | 单设备 batch size |
| `--max_new_tokens` | 任务相关 | 最大生成 token 数 |
| `--history_window` | 任务相关 | AdaWC 滑动窗口大小 |
| `--min_weight` | 任务相关 | 任一模型权重的下限 |

> 经验值参考：`TriviaQA/short-answer` 可用 `history_window=2, min_weight=0.2`；`MMLU/CoT` 可用 `history_window=2, min_weight=0.45`，以避免长链推理过程中某一模型被完全压制。

## 5. 实验与结果

带 AdaWC 的改进版本已在 `MMLU`、`TriviaQA`、`SimpleMath` 等多个基准上进行了评测，对应结果分别保存为：

- `result_pro_mmlu.jsonl`
- `result_pro_triviaqa.jsonl`
- `result_pro_simplemath.jsonl`

可直接与未带 `_pro_` 的原版结果文件进行对比，验证 AdaWC 的有效性。

## 6. 致谢与引用

本工作基于以下论文的实现开展：

```bibtex
@misc{yao2024determinethenensemblenecessitytopkunion,
      title={Determine-Then-Ensemble: Necessity of Top-k Union for Large Language Model Ensembling},
      author={Yuxuan Yao and Han Wu and Mingyang Liu and Sichun Luo and Xiongwei Han and Jie Liu and Zhijiang Guo and Linqi Song},
      year={2024},
      eprint={2410.03777},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.03777},
}
```

## 7. 本工作贡献

- 在 UNITE 的融合阶段引入 **AdaWC（Adaptive Weight Change）**：基于滑动窗口的得分历史动态调整两模型融合权重；
- 加入 **冗余融合跳过** 机制，进一步提升推理效率；
- 在多个基准上验证了 AdaWC 相对于固定权重的有效性，且几乎不增加额外超参调优成本。
