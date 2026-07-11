# MyClone

轻量级聊天风格克隆工具。把你的微信/QQ 聊天记录训练成一个 LoRA 适配器，让 LLM 学会你的说话方式。

## 完整流程

```
原始聊天记录 → process_data.py → 质量清洗后的训练数据 → train.py → 文本 LoRA
                                                        ↓
原始视觉基座 + LoRA GGUF → Ollama 多模态模型
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
| `raw_data_dir` | `./data/raw-chat-history` | 原始聊天导出路径；相对路径固定以仓库目录为基准 |
| `output_dir` | `./data/dataset` | 处理后数据集目录 |
| `system_prompt` | `请你扮演一名人类，不要说自己是人工智能` | 每条对话的系统提示词 |
| `combine_time_window` | `10` | 同一人连续发言合并的时间窗口（分钟） |
| `qa_match_time_window` | `10` | 问答配对的间隔上限（分钟），超过则截断为新对话 |
| `max_combine_messages` | `4` | 连续发言最大合并条数，防止刷屏变成长篇大论 |
| `include_self_initiated` | `false` | 是否训练没有用户输入的主动发言；默认关闭，避免伪造问题和答案泄漏 |
| `deduplicate_conversations` | `true` | 删除完全重复的对话 |
| `max_short_reply_chars` | `4` | 对高频短回复执行去偏置的字符阈值 |
| `max_short_reply_occurrences` | `30` | 同一短回复最多保留多少个不同上下文 |
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

输出到 `data/dataset/sft/`：
- `sft-my.json` — ShareGPT 格式的训练数据
- `dataset_info.json` — LLaMA-Factory 描述文件
- `quality-report.json` — 答案泄漏、重复短回复和丢弃样本统计

处理器不会把“自己主动发的话”改写成 `<begin_chat>你应该说：正确答案`。这种写法会让训练 loss 很好看，但模型只学会从提示中复制答案，换成真实问法后语义表现会明显下降。

训练前可单独运行零依赖质量门禁：

```bash
python validate_dataset.py --dataset ./data/dataset/sft
```

门禁会检查角色交替、答案泄漏、媒体占位符、重复对话、空回复和高频短回复上限，任一失败都会返回非零退出码。

---

## 二、训练（Kaggle / 本地 GPU）

### 环境

```bash
pip install -r requirements.txt
```

依赖：`transformers>=5.4.0`、`peft`、`trl`、`bitsandbytes`、`accelerate`、`datasets`。

### 上传数据集到 Kaggle

推荐先生成包含“当前本地代码 + 当前清洗数据”的自包含训练包，避免 Kaggle 再次从远端仓库拉到旧版本：

```bash
python prepare_kaggle_bundle.py
```

上传 `dist/myclone-kaggle-bundle-v3.1.zip` 为 Kaggle Dataset，然后在 Notebook 中运行 `run_kaggle.py`。v3.1 会清除引用、图片、转账、链接、@ 等导出伪文本，并把每个助手回复转换为显式 prompt-completion 样本；脚本会先执行 `validate_dataset.py`，再开始训练。

Kaggle Notebook 的第一个 cell 可以直接使用下面的启动器，无论平台把 zip 保留原样还是自动解包都能运行：

```python
import glob, zipfile

scripts = glob.glob("/kaggle/input/**/run_kaggle.py", recursive=True)
if scripts:
    code = open(scripts[0], encoding="utf-8").read()
else:
    bundle = sorted(glob.glob("/kaggle/input/**/myclone-kaggle-bundle*.zip", recursive=True))[-1]
    with zipfile.ZipFile(bundle) as z:
        code = z.read("run_kaggle.py").decode("utf-8")
exec(compile(code, "run_kaggle.py", "exec"))
```

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
| `--lr` | `2e-5` | 学习率；较保守，减少对基座语义能力的扰动 |
| `--lora-rank` | `8` | LoRA 秩；风格克隆不需要过高容量 |
| `--lora-alpha` | `4` | LoRA 强度，默认 rank/2，减少覆盖基座语义能力 |
| `--lora-dropout` | `0.08` | Dropout 正则化 |
| `--eval-ratio` | `0.08` | 验证集比例（设 0 关闭验证） |
| `--eval-strategy` | `chronological` | 默认按联系人分别保留最新对话，避免随机切分泄漏且覆盖不同关系 |
| `--multiline-oversample` | `1` | 每个干净多行回复额外加入一个聚焦样本，让模型学习连续聊天气泡而不过度放大 |
| `--max-ultrashort-share` | `0.20` | 训练集中一至两个字的单行回复最多占 20%，避免 `在`、`有`、`嗯` 主导输出 |
| `--eval-steps` | `50` | 验证/保存间隔步数 |
| `--early-stopping-patience` | `3` | 验证 loss 不改善时提前停止 |
| `--no-assistant-only-loss` | 关闭 | 对完整 prompt 和 completion 计算 loss（默认仅对助手 completion 计算） |
| `--model-parallel` | 关闭 | 跨卡切分大模型（串行） |

有效 batch size = `batch-size × grad-accum × GPU 数`（DDP 时）。

### 输出

训练完成后，`--output` 目录下保存 LoRA 适配器。在聊天框架中加载 `base_model` + 此 LoRA 适配器即可使用。

---

## 三、保留多模态能力的 Ollama 部署

不要把完整的 Qwen3.5 VLM 合并后直接转成一个语言 GGUF。该路径会只导出语言模型，视觉塔会消失；表现为模型从约 9.7B 变成 9.0B，`ollama show` 的能力列表也不再包含 `vision`。

正确做法是保留 Ollama 原始的视觉基座，只把文本 LoRA 转成 GGUF adapter：

```powershell
# 默认读取 kaggle_output-latest，并创建 myclone-vl:latest
# 对旧的 alpha=rank 适配器会自动限制为 alpha<=rank/2，减少语义退化
powershell -ExecutionPolicy Bypass -File .\model_process\build_ollama_vl.ps1

# 已经转换过 adapter 时可跳过转换
powershell -ExecutionPolicy Bypass -File .\model_process\build_ollama_vl.ps1 -SkipConvert

ollama show myclone-vl:latest
```

验证输出的 `Capabilities` 必须同时包含 `vision` 和 `thinking`。部署端应发送 Ollama `/api/chat` 的 `message.images` 字段，并设置 `think=false`；本项目的 QQ 插件已经按此方式处理图片。

部署前后用固定用例进行 A/B 对比，不再只看训练 loss：

```powershell
python .\evaluate_ollama.py `
  --models qwen3.5:9b myclone:latest myclone-vl:latest `
  --image "你的测试图片.jpg" `
  --output eval-report.json
```

报告覆盖否定理解、多轮省略、歧义澄清、情绪回应、用户纠错和真实图片请求。自动门禁检查空回复、媒体占位符与视觉能力，语义质量通过同一用例的并排回复人工判断。

## 四、Transformers 推理

训练出来的 LoRA 适配器和基座模型配合使用。以原始模型加载并用 LoRA 合并：

```python
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

base = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-9B", device_map="auto")
model = PeftModel.from_pretrained(base, "./output")
processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B")
```

推理时建议关掉思考（reasoning）以确保风格纯净；训练数据带有的空 `<think>` 块已让模型学会空想直接答。

---

## 原理简介

| 阶段 | 做了什么 |
|------|----------|
| **数据处理** | 读取微信/QQ 导出 → 严格交替生成对话 → 删除答案泄漏与完全重复样本 → 限制高频一字回复偏置 |
| **训练** | QLoRA 只训练语言层 LoRA、只计算 assistant loss；视觉层保持冻结；按时间留出验证集 |
| **部署** | 原始视觉基座保持不变，只挂载 LoRA GGUF，因此同时保留聊天风格和图片理解能力 |
