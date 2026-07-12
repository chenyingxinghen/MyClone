# 手把手教你克隆自己的聊天风格：从 QQ/微信导出到私人风格 Bot 全流程

## 写在前面

去年某个深夜，我看着自己和女朋友、好哥们几千页的聊天记录，突然冒出一个念头：如果这些聊天记录能教会一个 AI 模仿我的说话方式，那不就能有个"我分身"来处理日常消息了吗？经过踩坑无数，我最终搭出了一套完整流水线。今天就把整个过程拆开揉碎了分享给你。

**最终效果预览**：朋友给你发 QQ 消息 -> Bot 自动回复，语气像你、用词像你、连表情节奏都像你 —— 但不刻意模仿口头禅而忽略问题本身。

全流程分为七个阶段，每个阶段都可以独立复用：

```
聊天记录 → 数据清洗 → QLoRA 微调 → GGUF 部署 → 消息接收 → 上下文+记忆 → Bot 上线
```

---

## 一、数据从哪里来：QQ 与微信消息导出

### QQ 端：手机备份 + SQLCipher 解密

网上大多数 QQ 导出教程依赖 PC 客户端，但我发现更稳定的路径是从手机备份入手：

1. 用手机自带的备份工具备份 QQ 数据（华为用"手机助手"，其他品牌同理）
2. 解压备份文件，在 `com.tencent.mobileqq/files/` 下找到你的 uid
3. 通过 `md5(md5(uid) + "nt_kernel")` 计算出 QQ 数据库 hash 目录
4. 在 `databases/nt_db/` 下找到 `nt_msg.db`——这是 SQLCipher 加密的 SQLite
5. 用 sqlcipher 命令行解密并导出为明文 SQLite
6. 使用 [QQNT-Database-Export-Tool](https://github.com/star-picker/QQNT-Database-Export-Tool) 导出为 JSON

> 完整的图文步骤参考：[QQDecrypt 文档](https://qqbackup.github.io/QQDecrypt/)

导出后的 JSON 结构类似：

```json
[
  {
    "msgId": 12345,
    "msgTime": 1700000000,
    "senderQQNum": 3512024837,
    "sendType": 0,
    "elements": ["今天吃什么？"]
  }
]
```

### 微信端：WeFlow 导出

微信端我用了 [WeFlow](https://github.com/StrayMeteor3337/WeFlow) 这个开源工具。它可以直接扫描手机微信备份文件，导出每个联系人的 CSV 消息记录，字段包括时间、发送方、消息类型和内容。

微信导出的目录结构：

```
raw-chat-history/
├── wxchat/texts/
│   ├── 私聊_女朋友/
│   │   ├── msg_1.csv
│   │   └── msg_2.csv
│   └── 私聊_好哥们/
│       └── msg_1.csv
└── qqchat/
    └── 3512024837.json
```

**避坑提醒**：微信导出时选择"仅文字"，避免混入大量图片/语音占位符增加清洗难度。QQ 端如果有多媒体消息，导出脚本会自动包含 `[图片]` `[表情]` 之类的占位符，后续清洗阶段会被过滤掉。（当然，如果你想让自己的bot更强，也可以加入多模态训练计划，不过那样的话，你就需要改动一些训练和数据处理相关代码）。

---

## 二、数据变黄金：process_data.py 清洗流程

拿到原始聊天记录后，直接扔给模型训练是不行的——微信和 QQ 的消息结构不同、系统通知混在其中、连续刷屏会破坏对话逻辑。我写了一个清洗脚本 `process_data.py`，位于 [MyClone](https://github.com/chenyingxinghen/MyClone) 仓库。

它的处理管线如下：

```
原始 JSON/CSV → 加载解析 → 过滤垃圾消息 → 同人合并 → QA 配对 → 质量报告
```

### 2.1 加载与合并

脚本统一了微信 CSV 和 QQ JSON 的读取接口，每条消息抽象为 `ChatMessage`（普通文本）和 `CutMessage`（需要打断对话的信号，如音视频通话）。

**同人合并**：你一分钟内发了五六条消息（"在吗""跟你说个事""今天遇到了一个超好笑的事""..."），如果不合并，模型会认为回应一句话后就应该断掉。`combine_time_window` 控制合并窗口（默认 10 分钟），`max_combine_messages` 限制最多合并 4 条，避免刷屏变成长篇大论。

### 2.2 严格 QA 配对

这一步是核心。脚本按时间窗口（`qa_match_time_window`，默认 10 分钟）把对话拆成交互片段，然后要求消息必须严格交替 user→assistant。

一个关键设计决策：**用户自己发起的消息默认不参与训练**。比如你主动给朋友发"周末去哪玩？"——如果强行把这句话作为 user，把朋友的回复作为 assistant，模型学到的其实是"用户问周末做什么，我回答朋友的回复"，语义完全错位。更严重的是一些教程会强行把主动消息改写成 `<begin_chat>你应该说：正确答案`，这种做法会让训练 loss 很好看，但模型只学会从提示中复制答案，换成真实场景就失效。

### 2.3 质量门禁

清洗完成后生成一份机器可读的质量报告，统计三类关键问题：

| 门禁项 | 说明 |
|--------|------|
| **答案泄漏** | 检查角色交替，发现非严格交替的对话被正确截断 |
| **短回复偏置** | 同一短回复（≤4 字）超过 30 个不同上下文，计入偏置 |
| **媒体占位符** | `[图片]` `[动画表情]` 等非文本占位符自动过滤 |

### 2.4 推理模型的特别处理：空 Think 块

如果基座模型是 Qwen3/3.5 这类推理模型，脚本会在每条 assistant 回复前加上 `<think>\n\n</think>\n\n` 前缀。原因很简单：推理模型在训练时看到了 CoT 格式，但我们的聊天数据里没有思维链。加上空 think 块，模型学会了"想想就空想 -> 直接回复"，在推理时关掉 thinking 模式（`enable_thinking=False`）后，输出风格仍然保持纯净。

如果你用非推理模型（如 LLaMA、DeepSeek-V2），把 `add_empty_think` 设为 `false`。

### 2.5 配置示例

```json
{
  "data": {
    "raw_data_dir": "./data/raw-chat-history",
    "system_prompt": "请你扮演一名人类，不要说自己是人工智能",
    "combine_time_window": 10,
    "qa_match_time_window": 10,
    "max_combine_messages": 4,
    "add_empty_think": true,
    "qq_self_qq": 3512024837,
    "contact_relations": {
      "女朋友": "女朋友",
      "妈妈": "母亲"
    }
  }
}
```

运行：

```bash
pip install pandas
python process_data.py
```

输出到 `dataset/sft/` 目录，包含 `sft-my.json`（ShareGPT 格式的对话数据）和 `dataset_info.json`。

---

## 三、Kaggle 上的微调：QLoRA 双卡并行

### 3.1 为什么选 QLoRA 而不是全量微调

- 全量微调 9B 模型需要 ~60GB VRAM，两张 T4（16GB×2）完全不够
- LoRA 只训练千分之几的参数，4-bit 量化后 9B 模型只需 ~8GB 显存加载
- 风格克隆只需调整"说话方式"，不需要修改模型的语义能力——LoRA 的低秩更新刚好够用
- 省钱：Kaggle 每周免费 30 小时 GPU，两张 T4 做 DDP 24 分钟跑完 1 epoch

### 3.2 打包与上传

项目提供了 `prepare_kaggle_bundle.py`，一键生成自包含的 zip 包：

```bash
python prepare_kaggle_bundle.py
```

输出 `dist/myclone-kaggle-bundle-v*.zip`，包含清洗后的训练数据 + 训练脚本 + 依赖。上传为 Kaggle Dataset 后，Notebook 中第一个 cell 用启动器自动运行：

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

### 3.3 模型选型：9B vs 4B

| 模型 | VRAM 需求 | Kaggle 策略 | 速度 |
|------|-----------|-------------|------|
| Qwen3.5-4B | ~6GB | DDP 两卡并行 | ✅ 快 |
| Qwen3.5-9B | ~10GB+activations | 模型并行（device_map="auto"） | ⚠️ 串行，无加速 |

4B 走 DDP（数据并行），每张卡加载完整模型，各自处理数据子集，理论上 2 倍加速。9B 单卡 16GB T4 装不下（activations 会撑爆），只能走模型并行——把模型不同层分散到两张卡，GPU 串行执行，无吞吐加速但能跑。

### 3.4 训练参数解读

核心参数及其设计意图：

| 参数 | 值 | 为什么这么设 |
|------|-----|-------------|
| `--epochs 1` | 1 个 epoch | 风格克隆 1 轮足够，多轮必然过拟合到特定措辞 |
| `--lr 2e-5` | 较低学习率 | 减少对基座语义能力的扰动，只微调表达方式 |
| `--lora-rank 8` | 低秩 | 风格适配不需要高容量，rank=8 刚好够 |
| `--lora-alpha 4` | 保守适配器强度 | rank/2，减少 LoRA 覆盖基座能力 |
| `--lora-dropout 0.08` | 正则化 | 应对小数据集的过拟合风险 |
| `--eval-ratio 0.08` | 8% 验证集 | 按联系人保留最新对话作为验证，避免随机切分泄漏 |
| `--eval-steps 50` | 每 50 步验证 | Kaggle 时间有限，尽早检测过拟合 |
| `--multiline-oversample 1` | 多行回复增强 | 给每条多行回复额外造一个聚焦样本，让模型学会连续聊天气泡 |
| `--max-ultrashort-share 0.20` | ≤20% 短回复 | 避免"嗯""好的""在"主导输出 |

### 3.5 启动训练

```bash
# Kaggle 双卡（4B 模型）：
torchrun --nproc_per_node=2 train.py --dataset /kaggle/input/xxx/sft

# Kaggle 模型并行（9B 模型）：
python train.py --model-parallel --model Qwen/Qwen3.5-9B --dataset /kaggle/input/xxx/sft

# 单卡（本地调试）：
python train.py --dataset /path/to/sft
```

**注意**：训练默认只对 assistant 的 completion token 计算 loss，忽略 system prompt 和 user 输入。配合 `--grad-accum 4` 和 DDP（双卡），有效 batch size = 1 × 4 × 2 = 8。

**训练输出**：`/kaggle/working/output/` 目录下的 LoRA adapter，包含 `adapter_config.json` 和 `adapter_model.safetensors`。

---

## 四、从 LoRA 到 GGUF：本地部署

Kaggle 训练出来的是 HuggingFace 格式的 LoRA adapter，不能直接让 Ollama 加载。需要两个步骤：转 GGUF + Ollama 导入。

### 4.1 关键原则：不要合并视觉模型

如果你用的是 Qwen3.5-9B（原生多模态模型），千万**不要**做 `base_model + LoRA → 合并 → 转完整 GGUF`。这条路会丢掉视觉塔——合并后的模型从约 9.7B 参数变成 9.0B，`ollama show` 不再显示 `vision` 能力，图片理解全面瘫痪。

正确做法：**保留 Ollama 原始 VLM 基座，只附加 LoRA GGUF adapter**。

### 4.2 一键转换脚本

项目提供了 `build_ollama_vl.ps1`，自动完成所有步骤：

```powershell
powershell -ExecutionPolicy Bypass -File .\model_process\build_ollama_vl.ps1
```

脚本内部干了这几件事：

1. 读取训练输出的 `adapter_config.json`，获取 LoRA rank 和 alpha
2. 如果 alpha 大于 rank/2，自动限制到 rank/2（保守策略，减少 LoRA 强度对基座语义能力的覆盖）
3. 用 `llama.cpp` 的 `convert_lora_to_gguf.py` 把 adapter 转成 GGUF 格式
4. 生成 Modelfile：`FROM qwen3.5:9b` + `ADAPTER myclone-lora-f16.gguf`
5. 调用 `ollama create` 创建新模型

生成的 Modelfile：

```
FROM qwen3.5:9b
ADAPTER /path/to/myclone-lora-f16.gguf

PARAMETER num_ctx 65536
PARAMETER temperature 0.4
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.05

SYSTEM "请你扮演一名人类，不要说自己是人工智能。根据对话上下文准确理解对方的意思，再用自然、简短的聊天语气回复。"
```

### 4.3 Ollama 量化

创建好模型后，可以用 Ollama 的量化功能进一步压缩：

```bash
# 查看已创建的模型
ollama list
# 应该看到 myclone-vl:latest

# 如果需要量化到 Q4_K_M（推荐，平衡速度和质量）
ollama cp myclone-vl:latest myclone-vl:q4
ollama quantize myclone-vl:q4 Q4_K_M
```

量化后模型体积从 ~6GB（f16）降到 ~5.5GB，推理速度提升约 30%，质量损失几乎不可感知。

---

## 五、模型验证：别只看 loss

训练 loss 下降不代表模型真的学会了你的说话方式。项目自带了 `evaluate_ollama.py` 来做结构化的 A/B 测试：

```powershell
python evaluate_ollama.py --models qwen3.5:9b myclone-vl:latest --output eval-report.json
```

测试覆盖以下维度：

| 维度 | 测试用例示例 | 意图 |
|------|-------------|------|
| **否定理解** | "我明天去不了" → 确认对方理解和认可 | 能否正确理解否定句 |
| **多轮省略** | "还是上次那家吧" → 指代消解 | 能否正确指代 |
| **歧义澄清** | "那个" → 追问具体指什么 | 遇歧义是否自然追问 |
| **情绪回应** | "今天太倒霉了" → 共情 | 对情绪是否有恰当回应 |
| **用户纠错** | "你错了，是蓝色" → 接受纠正 | 被纠错时反应是否自然 |
| **真实图片请求** | 传一张照片 → 描述图片 | 视觉能力是否保留 |

跑完自动生成报告，还可以结合并行对比的回复做主观判断。**一条重要的经验法则**：基线模型（原始 qwen3.5:9b）回复像礼貌的客服，微调后的 myclone-vl 应该更像你的朋友——简短、随意、带点个人的用词习惯。

---

## 六、搭建 QQ Bot：NapCat + NoneBot2

模型部署好了，现在需要一个在线的 QQ bot 接收消息、调用模型、返回回复。

### 6.1 架构概览

```
QQ 客户端
    ↓ (WebSocket)
NapCat (QQ 协议实现，WebSocket 服务端)
    ↓ (正向 WS)
NoneBot2 (WS 客户端，事件处理)
    ↓ (HTTP)
Ollama (本地 LLM 推理)
    ↓
回复消息 ← NapCat → QQ
```

[NapCat](https://github.com/NapNeko/NapCatQQ) 是一个基于 QQ NT 协议的轻量级框架，提供 OneBot v11 标准的 WebSocket API。Bot 端使用 [NoneBot2](https://nonebot.dev/) 作为框架，连接 NapCat 的正向 WebSocket 接收消息事件。

### 6.2 启动 NapCat

NapCat Shell 版本以命令行方式运行 QQ 协议端，不需要手动登录 QQ（会自动读取已登录的 QQ 会话）。启动后监听 `ws://localhost:8080` 提供 OneBot API。

### 6.3 Bot 端实现

Bot 核心代码在 `bot.py`：

```python
import nonebot
from nonebot.adapters.onebot.v11 import Adapter as OneBotV11Adapter

nonebot.init()
driver = nonebot.get_driver()
driver.register_adapter(OneBotV11Adapter)
nonebot.load_from_toml("pyproject.toml")
nonebot.run()
```

插件 `chat_style` 负责处理私聊消息。在 `plugins/chat_style/__init__.py` 中：

1. **接收消息**：监听 PrivateMessageEvent，过滤自己发的消息
2. **处理图片**：把 OneBot 的图片 URL 下载转 base64
3. **构建上下文**：维护 per-user 的对话历史，超时（2 小时）自动清空
4. **召回记忆**：通过 embedding 向量召回 + 词面召回（RFF 融合）检索相关长期记忆
5. **调用 Ollama**：发送完整 system prompt + 上下文 + 记忆 + 时间感知到 `/api/chat`
6. **拆分回复**：长回复按标点拆成多条消息分别发送，模拟真实打字节奏
7. **延迟模拟**：根据消息长度随机延迟 1~4 秒，更像真人

WebSocket 配置在 `.env`：

```ini
HOST=0.0.0.0
PORT=7001
ONEBOT_V11_ACCESS_TOKEN=xxx

# Ollama 配置
LLM_API_BASE=http://localhost:11434/v1
LLM_MODEL=myclone-vl:latest
LLM_THINK=false
LLM_TEMPERATURE=0.5
```

启动：

```bash
start.bat
```

---

## 七、记忆系统：让 Bot 不只学语气，还能记住事

风格克隆只是第一步。如果你的朋友问"你不是说下周去成都吗？"，一个只学语气的 bot 可能会回答"哈哈对"，但一个带记忆的 bot 可以回答"哦对，下周成都，票订好了没？"

### 7.1 记忆架构

记忆系统分为三个层次：

```
消息流水（messages 表） → 会话处理（preprocessed） → 长期认知（memories 表）
    ↑                              ↓                              ↓
运行时消息                    预筛选质量评估              + 词面索引 (FTS5)
（在线交互）                  离线处理脚本                + 向量索引 (embedding)
```

- **message**：原始消息记录，来源可以是导入的历史 JSON 或运行时交互
- **preprocessed**：经过质量筛选的会话片段，保留高信息密度的对话
- **memory**：从会话中提取的长期认知，分为四种类型

### 7.2 记忆类型

| 类型 | subject_id | 示例 |
|------|-----------|------|
| **self**（自我） | 自己的 QQ | "我喜欢吃辣但胃不好""我养了一只橘猫叫年糕" |
| **person**（对象） | 对方 QQ | "她今年考研""他上周搬家到上海了" |
| **relationship**（关系） | relationship:self:partner | "我们在一起两年了""我们约定轮流请客" |
| **episode**（事件） | relationship:self:partner | "去年春节一起去哈尔滨看雪" |

### 7.3 检索逻辑：混合召回 + RFF 融合

当用户发来消息时，bot 同时使用两种方式召回相关记忆：

1. **词面召回**（FTS5）：用 trigram tokenizer 做中文全文检索 + 自定义相关性打分 `_relevance()`，会对关键词命中给予更高权重
2. **向量召回**（Embedding）：用 `qwen3-embedding:0.6b` 把记忆转成向量，与用户查询向量做 dot product 余弦相似度

两种召回的排名通过 **Reciprocal Rank Fusion** 融合：

```python
def fused(memory_id):
    score = 0
    if memory_id in vector_rank:
        score += 1.0 / (60 + vector_rank[memory_id])
    if memory_id in lexical_rank:
        score += 0.85 / (60 + lexical_rank[memory_id])
    # 时间敏感性
    if has_temporal_intent(query):
        score += 0.35 / (60 + temporal_rank[memory_id])
    return score
```

### 7.4 在线记忆抽取

Bot 在每次对话后自动触发记忆抽取。当积累足够多消息后，调用 LLM 从最近对话中提取认知，写入记忆库：

```python
prompt = """从对话中只提取有长期价值且有证据的认知：
- self：自我的身份、经历、偏好、价值观、行为及情绪模式
- person：对象的身份、经历、偏好、边界及行为模式
- relationship：双方称呼、关系状态、共同经历
- episode：值得长期保留的具体事件
输出严格 JSON 数组..."""
```

置信度控制机制防止误提取：
- 多次确认的事实：0.90-0.95
- 单次明确陈述：0.80-0.90
- 行为推断模式：0.55-0.75
- 不确定推断：不保存

### 7.5 记忆冲突与版本管理

当旧记忆和新记忆冲突时（比如"他在北京"变成"他到上海了"），系统会：

1. 标记旧记忆为 `historical`，设置 `valid_to` 时间戳
2. 创建新记忆为 `active`，通过 `supersedes_id` 关联旧版本
3. 触发冲突解决作业，由 LLM 判断是新情况覆盖，还是不同事实需要并存

### 7.6 导入历史数据

用预置脚本导入历史聊天记录并批量提取记忆：

```bash
# 导入历史消息
python scripts/import_memory.py qqchat/1422105979.json --peer 1422105979

# 预处理会话质量
python scripts/preprocess_memory.py --peer 1422105979

# 批量抽取长期认知
python scripts/extract_memories.py --partner 1422105979
```

---

## 八、人设系统：让 Bot 知道"你是谁"

模型虽然学过你的说话风格，但它不知道自己的身份设定。训练用的 system prompt 只是笼统的"扮演一个人类"，进入线上部署后需要更具体的人设。

我维护了一个 `style_prompt.txt`，每次推理时注入到 system prompt：

自我认知：
```
请你扮演一名人类，不要说自己是人工智能。
你叫xxx，是个牛马。先结合完整对话和图片准确理解对方的意思，再自然回复；
不要为了模仿口头禅而忽略问题本身。
对方身份需要根据上下文判断，可能是女朋友、家人、同学或朋友，不要固定假设。
回复保持简短随意，但信息要完整；确实无法判断时可以自然地追问一句。
输出换行分割的句子，将分别发送。
```
时间感知：
```
当前时间：2026-07-12 星期日 14:30
[距上次对话 3 小时后]

以下是你对自己、交谈对象及双方关系形成的内部认知：
【当前话题相关】
- [person，发生于 2026-07-10，约2天前] 对方说这周末想去爬山（可信度 0.85）
```

这样 bot 既知道自己是谁、知道当前跟谁聊天、知道最近发生过什么，再配合学到的语气，回复就非常自然了。

---

## 九、效果验收

最后是实战检验。以下是一些与 bot 的真实对话片段：

**场景 1：日常闲聊**

```
朋友：今天好累啊
Bot：咋了，加班了？
朋友：没，打了三个小时球
Bot：可以啊，在哪打的，下次叫上我
```

**场景 2：指代理解**

```
朋友：上次那家火锅不错
Bot：确实，他家毛肚很新鲜，啥时候再去
```

**场景 3：带图片**

```
朋友：[发送了一张猫咪照片]
Bot：哈哈这小猫啥表情，一脸嫌弃
```

**场景 4：对方纠错**

```
朋友：我说的是周三不是周五
Bot：哦哦记错了，那就周三见
```

（上面这些看起来像对话，但确实是 bot 的独立输出，没有预设脚本。）

### 对比 base 模型

同样的输入，原始 Qwen3.5-9B 的回复往往是：

```
我今天过得还不错，希望你也一样！😊
有什么我可以帮你的吗？
```

而微调后的 myclone-vl 回复：

```
还行吧，今天满课，你呢
```

前者像礼貌的客服，后者像朋友——这就是数据清洗 + 风格微调 + 人设 + 记忆共同作用的结果。

---

## 总结

整个流水线的核心理念是：**让数据说话，不要替模型编答案**。

这条流水线可以帮你从一个想法走到一个可以实际使用的私人 bot：

| 阶段 | 关键工具 | 耗时 |
|------|---------|------|
| QQ/微信导出 | QQDecrypt / WeFlow | 20 分钟 |
| 数据清洗 | MyClone process_data.py | 5 分钟 |
| 微调训练 | Kaggle T4 × 2 | 60 分钟 |
| GGUF 转换 | llama.cpp | 10 分钟 |
| Ollama 部署 | Ollama | 5 分钟 |
| Bot 搭建 | NapCat + NoneBot2 | ... 分钟 |
| 记忆系统 | SQLite + Embedding | 集成完成 |
| **总计** | | **~1 小时** |

所有代码开源在 [github.com/chenyingxinghen/MyClone](https://github.com/chenyingxinghen/MyClone)，欢迎 Star 和 PR。

---