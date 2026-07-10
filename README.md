# MyClone

轻量级聊天风格克隆工具。把你的微信/QQ 聊天记录训练成一个 LoRA 适配器，让 LLM 学会你的说话方式。

## 完整流程

```
原始聊天记录 → process_data.py → 训练数据 → train.py → LoRA 适配器
```

---

## 一、数据处理（本地运行）

### 1. 准备原始数据

从 **WeChatMsg**（微信）或 **NapCat**（QQ）导出聊天记录：

```
# 目录结构
../WeClone/my-chat-history/
├── wxchat/texts/
│   ├── 私聊_联系人A/     # 每个联系人一个文件夹
│   │   ├── msg_1.csv
│   │   └── ...
│   └── 私聊_联系人B/
│       └── ...
└── qqchat/
    ├── 12345678.json      # NapCat 导出（文件名即 QQ 号）
    └── ...
```

### 2. 編輯 `config.json`

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `raw_data_dir` | `../WeClone/my-chat-history` | 原始聊天导出路径 |
| `system_prompt` | `请你扮演一名人类，不要说自己是人工智能` | 每条对话的系统提示词 |
| `combine_time_window` | `10` | 同一人连续发言合并的时间窗口（分钟） |
| `qa_match_time_window` | `10` | 问答配对的间隔上限（分钟），超过则截断为新对话 |
| `max_combine_messages` | `4` | 连续发言最大合并条数，防止刷屏变成长篇大论 |
| `add_empty_think` | `true` | 为 Qwen3/3.5 等推理模型添加空 `<think>` 块；非推理模型请设为 `false` |
| `qq_self_qq` | `0` | 你的 QQ 号，`0` 表示跳过 QQ |
| `exclude_contacts` | `["文件传输助手"]` | 排除的联系人 |
| `blocked_words` | `[]` | 包含这些词的消息会被过滤 |
| `contact_relations` | `{}` | 联系人→关系标签，如 `{"妈妈": "母亲"}`，会追加到系统提示词中 |

### 3. 运行

```bash
pip install pandas
python process_data.py
```

输出到 `dataset/sft/`：
- `sft-my.json` — ShareGPT 格式的训练数据
- `dataset_info.json` — LLaMA-Factory 描述文件

---

## 二、训练（Kaggle / 本地 GPU）

### 环境

```bash
pip install -r requirements.txt
```

依赖：`transformers>=5.4.0`、`peft`、`trl`、`bitsandbytes`、`accelerate`、`datasets`。

### 上传数据集到 Kaggle

把 `dataset/` 文件夹打包上传为 Kaggle Dataset，Notebook 中挂载即可。

### 方式一：Kaggle 一键脚本（推荐）

新建 Notebook，把 `run_kaggle.py` 的内容粘贴到一个 cell 中运行。按需修改顶部的 CONFIG：

```python
MODEL = "Qwen/Qwen3.5-9B"     # 或 Qwen/Qwen3.5-4B 等
MODEL_PARALLEL = True          # 9B 模型单卡装不下，用 True 把模型切到两张 T4
DATASET = ""                   # 留空自动搜索已挂载的数据集
EPOCHS = 1
```

> **9B vs 4B 选型：**
> - `Qwen/Qwen3.5-4B` — 单卡 T4 装得下，`MODEL_PARALLEL=False`，走 DDP 两卡加速
> - `Qwen/Qwen3.5-9B` — 单卡 T4 OOM，`MODEL_PARALLEL=True`，用 device_map="auto" 切到两卡（串行，无加速但能跑）

### 方式二：手动运行

**单卡 / 本地：**
```bash
python train.py --dataset /path/to/sft
```

**Kaggle 双卡 DDP（用于 4B 等小模型）：**
```bash
torchrun --nproc_per_node=2 train.py --dataset /kaggle/input/xxx/sft
```

> 用 `torchrun` 而非 `accelerate launch`，因为 Kaggle 的 accelerate CLI 会导入 timm/torchvision 导致版本冲突报错。

**模型并行（用于 9B 等大模型）：**
```bash
python train.py --model-parallel --model Qwen/Qwen3.5-9B --dataset /kaggle/input/xxx/sft
```

⚠️ `--model-parallel` 必须用 `python train.py` 直接运行，不可用 torchrun / accelerate launch。

### 全部参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | 自动检测 | 包含 `sft-my.json` 的目录 |
| `--model` | `Qwen/Qwen3.5-9B` | HuggingFace 模型名或本地路径 |
| `--output` | `./output` 或 Kaggle 自动 | 输出目录 |
| `--epochs` | `1` | 训练轮数（风格克隆 1 轮足够，多轮易过拟合） |
| `--batch-size` | `1` | 每设备 batch size |
| `--grad-accum` | `4` | 梯度累积步数 |
| `--lr` | `3e-5` | 学习率 |
| `--lora-rank` | `16` | LoRA 秩 |
| `--lora-dropout` | `0.05` | Dropout 正则化 |
| `--eval-ratio` | `0.05` | 验证集比例（设 0 关闭验证） |
| `--eval-steps` | `50` | 验证/保存间隔步数 |
| `--early-stopping-patience` | `3` | 验证 loss 不改善时提前停止 |
| `--no-assistant-only-loss` | 关闭 | 对所有 token 都计算 loss（默认只算助手的） |
| `--model-parallel` | 关闭 | 跨卡切分大模型（串行） |

有效 batch size = `batch-size × grad-accum × GPU 数`（DDP 时）。

### 输出

训练完成后，`--output` 目录下保存 LoRA 适配器。在聊天框架中加载 `base_model` + 此 LoRA 适配器即可使用。

---

## 三、推理

训练出来的 LoRA 适配器和基座模型配合使用。以原始模型加载并用 LoRA 合并：

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-9B", device_map="auto")
model = PeftModel.from_pretrained(base, "./output")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
```

推理时建议关掉思考（reasoning）以确保风格纯净；训练数据带有的空 `<think>` 块已让模型学会空想直接答。

---

## 原理简介

| 阶段 | 做了什么 |
|------|----------|
| **数据处理** | 读取微信/QQ 导出 → 过滤非文本消息 → 合并连续发言 → 严格交替生成 `user → assistant` 对话对 → 输出 ShareGPT 格式 |
| **训练** | QLoRA (4-bit NF4) 微调，只对 assistant 回复计算 loss。内置早停、验证集、LoRA dropout 防过拟合 |
| **输出** | 一个轻量 LoRA 适配器（~几 MB），合并到基座模型即可模仿你的聊天风格 |
