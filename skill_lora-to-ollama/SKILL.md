---
name: lora-to-ollama
description: 将 HuggingFace/ModelScope LoRA adapter 合并到基座模型、转 GGUF、量化并导入 Ollama 的完整部署管线。当用户完成 LoRA 微调后想部署到本地 Ollama 时使用，尤其适用于中国大陆网络环境（ModelScope 替代 HF、git clone llama.cpp）和缺少编译工具（无 cmake/gcc）的场景。
version: 1.0.0
---

# LoRA to Ollama 部署管线

将 LoRA adapter（safetensors/bin 格式）合并到基座模型，转换为 GGUF，量化后导入 Ollama 本地推理。

## 多模态模型警告（Qwen3.5-VL 等）

下面“合并完整模型再转 GGUF”的传统流程只适合纯文本模型。对带视觉塔的模型这样做，转换器可能只导出语言部分：模型参数量会变小，`ollama show` 中的 `vision` 能力也会消失。

多模态模型必须保留 Ollama 原始视觉基座，只转换和挂载 LoRA：

```powershell
python llama.cpp-src/convert_lora_to_gguf.py ./adapter `
  --base ./base_model `
  --outfile ./adapter-f16.gguf `
  --outtype f16
```

```dockerfile
FROM qwen3.5:9b
ADAPTER ./adapter-f16.gguf
PARAMETER num_ctx 65536
PARAMETER temperature 0.4
```

```powershell
ollama create myclone-vl -f Modelfile
ollama show myclone-vl
```

最终 `Capabilities` 必须包含 `vision`。Ollama 的 Safetensors LoRA 导入目前不直接支持 Qwen，因此先用 llama.cpp 转成 GGUF adapter；不要把整个 VLM 合并成单一语言 GGUF。

## 前置条件

- Python 3.11+（3.13 也支持）、Ollama 已安装
- LoRA adapter 文件：adapter_model.safetensors（或 .bin）+ adapter_config.json
- 磁盘空间：工作分区需至少 3 倍模型大小（7B 约需 45GB+）
- 内存：合并 7B 模型峰值约 28-30GB RAM

## Step 1: 环境准备

创建独立 venv，安装依赖：

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv/Scripts/activate

# 中国大陆（清华镜像）
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  transformers peft safetensors huggingface_hub modelscope \
  accelerate sentencepiece tiktoken gguf
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 海外直连
pip install transformers peft safetensors huggingface_hub modelscope \
  accelerate sentencepiece tiktoken gguf torch
```

注意：不要升级 venv 自带的 pip，某些环境升级 pip 会导致 venv 损坏。

## Step 2: 下载基座模型

从 adapter_config.json 的 base_model_name_or_path 确认基座模型名称。

方案 1 - ModelScope（首选，国内快）：

```python
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen2.5-7B-Instruct', cache_dir='./base_model_cache')
print(f'Downloaded to: {model_dir}')
```

下载路径：`cache_dir/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master/`

方案 2 - HuggingFace 镜像：`HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./base_model`

方案 3 - HuggingFace 直连：`huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./base_model`

## Step 3: 合并 LoRA Adapter（仅纯文本基座）

```python
import gc, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "./base_model"
ADAPTER    = "./adapter"
OUTPUT     = "./merged_model"

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True, device_map="cpu")
model = PeftModel.from_pretrained(base, ADAPTER)
model = model.merge_and_unload()
model = model.to(torch.float16)
del base; gc.collect()
model.save_pretrained(OUTPUT, max_shard_size="5GB")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
tokenizer.save_pretrained(OUTPUT)
```

关键：low_cpu_mem_usage 减少内存峰值；max_shard_size 分片保存；转 float16 减半磁盘。
RAM 不足时（<32GB），可在 merge_and_unload 后手动 del base + gc.collect() 释放基座模型内存再保存。

## Step 4: 获取 llama.cpp 转换工具

只需 convert_hf_to_gguf.py 脚本，不需要编译 llama-quantize。

方案 A - git clone（推荐，最可靠）：

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git llama.cpp-src
```

脚本在 llama.cpp-src/convert_hf_to_gguf.py。

方案 B - 单独下载脚本（git 不可用时）：

```bash
curl -sL https://raw.githubusercontent.com/ggml-org/llama.cpp/master/convert_hf_to_gguf.py -o convert_hf_to_gguf.py
```

GitHub 被墙时尝试镜像代理：

```bash
curl -sL https://ghfast.top/https://raw.githubusercontent.com/ggml-org/llama.cpp/master/convert_hf_to_gguf.py -o convert_hf_to_gguf.py
```

## Step 5: 转换为 GGUF

```bash
python llama.cpp-src/convert_hf_to_gguf.py ./merged_model --outtype f16 --outfile ./my-model-f16.gguf
```

输出 fp16 GGUF 约 15GB（7B），量化完成后可删。依赖：gguf、sentencepiece、tiktoken（Step 1 已装）。

## Step 6: 量化并导入 Ollama

### 方案 B（默认推荐）：Ollama 内置量化

适用于无编译工具（cmake/gcc/MSVC）的环境。Ollama 自带量化引擎，效果与 llama-quantize 一致。

1. 创建 Modelfile：

```dockerfile
FROM ./my-model-f16.gguf
PARAMETER num_ctx 4096
PARAMETER temperature 0.8
SYSTEM "You are a helpful assistant."
```

**不要在 Modelfile 中写 QUANTIZE 指令** — 旧版 Ollama 不支持，会报错 `command must be one of "from", "license", "template"...`。量化通过命令行参数完成。

2. 创建模型（Ollama 自动量化）：

```bash
ollama create my-model -f Modelfile --quantize q4_K_M
```

`--quantize` 可选值：q4_0、q4_1、**q4_K_M（推荐）**、q4_K_S、q5_0、q5_1、q5_K_M、q5_K_S、q8_0、q2_K、q3_K_M、q3_K_S、q6_K。Q4_K_M 是中文场景质量和速度的最佳平衡点，7B 模型量化后约 4.7GB。

### 方案 A：llama-quantize 手动量化

有编译工具时可用。先编译 llama.cpp：

```bash
cd llama.cpp-src
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target llama-quantize
./build/bin/llama-quantize ./my-model-f16.gguf ./my-model-q4km.gguf q4_K_M
```

Modelfile 指向已量化文件，不需要 --quantize 参数：

```dockerfile
FROM ./my-model-q4km.gguf
PARAMETER num_ctx 4096
PARAMETER temperature 0.8
SYSTEM "You are a helpful assistant."
```

```bash
ollama create my-model -f Modelfile
```

### Chat Template

GGUF 文件通常已内嵌 chat template 元数据，Ollama 会自动识别。如果对话输出格式异常，手动在 Modelfile 中添加 TEMPLATE 指令。

Qwen2.5 ChatML 格式（Go template 语法），大多数情况下无需手动设置。如需自定义，参考 Ollama 官方 Modelfile 文档中的 TEMPLATE 指令。

## Step 7: 验证

```bash
ollama list                        # 确认模型存在
ollama run my-model "你好"          # 简单对话测试
ollama show my-model --modelfile   # 查看模型配置
```

通过 OpenAI 兼容 API 测试（用于接入其他服务）：

```bash
curl http://localhost:11434/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"my-model","messages":[{"role":"user","content":"你好"}],"max_tokens":64}'
```

## Pitfalls

### 网络问题（中国大陆）
- HuggingFace 直连失败：用 ModelScope snapshot_download 替代
- GitHub release zip 下载卡住：git clone --depth 1 比 curl 更可靠
- pip 安装慢：使用 -i https://pypi.tuna.tsinghua.edu.cn/simple

### Modelfile 中不要写 QUANTIZE
旧版 Ollama 的 Modelfile 不支持 QUANTIZE 指令，会报 parser 错误。解决方案：移除 Modelfile 中的 QUANTIZE 行，改用命令行 `ollama create --quantize q4_K_M`。

### C 盘空间不足
Ollama 默认模型目录在 C 盘。设置环境变量迁移：
- Windows（PowerShell）：`[System.Environment]::SetEnvironmentVariable('OLLAMA_MODELS', 'G:\ollama_models', 'User')`
- Linux/macOS：`export OLLAMA_MODELS="/path/to/models"`
- 设置后需重启 Ollama。Windows 上必须杀掉系统托盘的 `ollama app.exe`，仅杀 `ollama.exe` 不够。

### Ollama 上下文长度
Ollama 默认 num_ctx 为 2048。在 Modelfile 中设置 `PARAMETER num_ctx 65536`，或设用户级环境变量 `OLLAMA_NUM_CTX` 全局生效。

### Windows Bash 注意事项
- taskkill 参数用 `//f` 而非 `/f`（避免 MSYS2 路径转换）
- 后台进程写法：`command & sleep N;` 而非 `command &;`
- 环境变量路径中 `/` 也能被 Ollama 识别

## Disk Space Reference

以 Qwen2.5-7B-Instruct 为例：

| 阶段 | 产物 | 大小 | 可清理 |
|------|------|------|--------|
| 基座模型 | base_model_cache/ | ~15GB | 合并后可删 |
| LoRA adapter | kaggle_output/ | ~150MB | 保留备份 |
| 合并模型 | merged_model/ | ~15GB | GGUF 转换后可删 |
| fp16 GGUF | *-f16.gguf | ~15GB | 量化后可删 |
| Ollama 模型 | ~/.ollama/models/ | ~5GB | 最终产物 |
| llama.cpp 源码 | llama.cpp-src/ | ~200MB | 可删 |
| venv | .venv/ | ~1-2GB | 可删 |

峰值磁盘占用约 60GB，及时清理后最终约 5GB。
