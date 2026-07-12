# 记忆系统

## 目录

```text
components/  可复用的存储、向量、抽取和在线记忆组件
scripts/     导入、预处理、抽取、回填、冲突消解和测试入口
docs/        记忆系统文档
data/        SQLite 数据库及风格提示词
plugins/     NoneBot 插件
```

## 初始化

```powershell
ollama pull qwen3-embedding:0.6b
python scripts/backfill_embeddings.py --db data/memory.db
```

`scripts/backfill_embeddings.py` 可重复运行，只处理缺失向量或正文已变化的记忆。
默认通过 `/api/embed` 调用 Ollama，并传入 `num_gpu=0`，因此 embedding 在 CPU 上运行。

## 召回

聊天插件首先生成查询向量，然后用 Reciprocal Rank Fusion 合并：

- Qwen3 embedding 语义排名
- 中文词面及关键词排名
- 时间意图、记忆类型和置信度排名

Ollama embedding 不可用时自动退回词面召回，不影响回复。召回结果包含
`valid_from/valid_to` 的绝对日期和相对时间。

## 在线记忆

运行期间的收发消息写入 `messages`，互动时间写入 `interaction_state`。
每个联系人积累到 `MEMORY_AUTO_EXTRACT_MIN_MESSAGES` 条新消息后，后台自动抽取长期记忆；
抽取失败可重试，embedding 失败则保留正文，稍后由回填脚本补齐。

## 冲突消解

```powershell
python scripts/resolve_memory_conflicts.py --db data/memory.db `
  --embedding-model qwen3-embedding:0.6b --model myclone:latest
```

脚本先按主体、类型和向量相似度生成 `conflict_jobs`，再批量判定：

- `duplicate` / `supports`：合并证据，重复项标记为 `duplicate`
- `supersedes`：旧记忆转为 `historical` 并写入 `valid_to`
- `contradicts`：双方标记为 `disputed`，召回时明确提示存在冲突
- `coexists` / `unrelated`：保留原状态

任务和结果分别保存在 `conflict_jobs`、`memory_conflicts`，可中断后继续执行。
