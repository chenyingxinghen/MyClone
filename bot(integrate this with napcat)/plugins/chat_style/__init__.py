"""
chat_style 插件：用 LLM 模仿用户说话风格回复私聊消息
"""

import asyncio
import base64
import random
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import httpx
from nonebot import get_driver, logger, on_message
from nonebot.adapters.onebot.v11 import Bot, PrivateMessageEvent
from nonebot.rule import Rule
from components.embedding_client import EmbeddingCircuitBreaker, OllamaEmbeddingClient
from components.memory_store import MemoryStore
from components.online_memory import OnlineMemoryExtractor

# ─── 配置加载 ───────────────────────────────────────────

driver = get_driver()
config = driver.config

LLM_API_BASE: str = getattr(config, "llm_api_base", "https://api-inference.modelscope.cn/v1/")
LLM_API_KEY: str = getattr(config, "llm_api_key", "")
LLM_MODEL: str = getattr(config, "llm_model", "Qwen/Qwen3-235B-A22B-Instruct")
LLM_VISION_MODEL: str = getattr(config, "llm_vision_model", LLM_MODEL)
BOT_QQ: int = int(getattr(config, "bot_qq", "0"))
SELF_QQ: str = str(getattr(config, "memory_self_qq", BOT_QQ))
MAX_CONTEXT: int = int(getattr(config, "max_context", "20"))
REPLY_DELAY_MIN: float = float(getattr(config, "reply_delay_min", "1"))
REPLY_DELAY_MAX: float = float(getattr(config, "reply_delay_max", "4"))
LLM_TEMPERATURE: float = float(getattr(config, "llm_temperature", "0.4"))
MAX_REPLY_CHARS: int = int(getattr(config, "max_reply_chars", "80"))
MAX_IMAGES: int = int(getattr(config, "max_images", "4"))
MAX_IMAGE_BYTES: int = int(getattr(config, "max_image_bytes", str(10 * 1024 * 1024)))
EMBEDDING_API_BASE: str = getattr(config, "embedding_api_base", "http://127.0.0.1:11434")
EMBEDDING_API_KEY: str = getattr(config, "embedding_api_key", "")
EMBEDDING_MODEL: str = getattr(config, "embedding_model", "qwen3-embedding:0.6b")
EMBEDDING_TIMEOUT: float = float(getattr(config, "embedding_timeout", "30"))
EMBEDDING_NUM_GPU: int = int(getattr(config, "embedding_num_gpu", "0"))
EMBEDDING_KEEP_ALIVE: str = getattr(config, "embedding_keep_alive", "30m")
LLM_KEEP_ALIVE: str = getattr(config, "llm_keep_alive", "30m")
CONTEXT_TIMEOUT: int = int(getattr(config, "context_timeout", str(3600 * 2)))
MEMORY_AUTO_EXTRACT_MIN_MESSAGES: int = int(getattr(
    config, "memory_auto_extract_min_messages", "8"))


def _as_bool(value, default=False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


LLM_THINK: bool = _as_bool(getattr(config, "llm_think", False), False)
MEMORY_AUTO_EXTRACT: bool = _as_bool(getattr(config, "memory_auto_extract", True), True)

# 加载风格 prompt
_style_prompt_path = Path(__file__).parent.parent.parent / "data" / "style_prompt.txt"
_default_memory_path = Path(__file__).parent.parent.parent / "data" / "memory.db"
MEMORY_DB = Path(str(getattr(config, "memory_db", _default_memory_path)))
memory_store = MemoryStore(MEMORY_DB)
embedding_client = OllamaEmbeddingClient(
    EMBEDDING_API_BASE, EMBEDDING_MODEL, EMBEDDING_TIMEOUT, EMBEDDING_API_KEY,
    EMBEDDING_NUM_GPU, EMBEDDING_KEEP_ALIVE)
embedding_breaker = EmbeddingCircuitBreaker(60)
online_extractor = OnlineMemoryExtractor(
    store=memory_store,
    self_id=SELF_QQ,
    llm_api_base=LLM_API_BASE,
    llm_model=LLM_MODEL,
    llm_api_key=LLM_API_KEY,
    embedding_client=embedding_client,
    min_messages=MEMORY_AUTO_EXTRACT_MIN_MESSAGES,
    llm_keep_alive=LLM_KEEP_ALIVE,
)
if _style_prompt_path.exists():
    STYLE_PROMPT = _style_prompt_path.read_text(encoding="utf-8").strip()
    logger.info(f"已加载风格 prompt（{len(STYLE_PROMPT)} 字符）")
else:
    STYLE_PROMPT = "你是一个普通的 QQ 用户，说话简短随意。"
    logger.warning("未找到 style_prompt.txt，使用默认 prompt")

# ─── 会话管理 ───────────────────────────────────────────

# user_id -> list of {"role": "user"|"assistant", "content": str}
conversations: dict[str, list[dict]] = defaultdict(list)
conversation_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
online_extraction_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
background_tasks: set[asyncio.Task] = set()

# 记录上次活跃时间，超时清空上下文
last_active: dict[str, float] = {}


def _get_conversation(user_id: str) -> list[dict]:
    """获取某用户的对话上下文，超时自动清空"""
    now = time.time()
    if user_id in last_active and (now - last_active[user_id]) > CONTEXT_TIMEOUT:
        conversations[user_id].clear()
    last_active[user_id] = now
    return conversations[user_id]


def _add_message(user_id: str, role: str, content: str):
    """添加一条消息到上下文，保持长度在 MAX_CONTEXT 内"""
    ctx = _get_conversation(user_id)
    ctx.append({"role": role, "content": content})
    # 保留最近的完整 user→assistant 对，避免截断后以上下文中的 assistant 开头。
    while len(ctx) > MAX_CONTEXT:
        ctx.pop(0)
    while ctx and ctx[0]["role"] == "assistant":
        ctx.pop(0)


# ─── 时间感知 ───────────────────────────────────────────

_WEEKDAYS = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


def _now_str() -> str:
    """当前时间的中文描述，用于注入 system prompt"""
    now = datetime.now()
    return f"{now:%Y-%m-%d} {_WEEKDAYS[now.weekday()]} {now:%H:%M}"


def _gap_str(user_id: str) -> str:
    """距上次互动的时间间隔描述；无历史或间隔很短则返回空串"""
    state = memory_store.get_interaction_state(user_id)
    if not state or not state["last_user_at"]:
        return ""
    gap = time.time() - int(state["last_user_at"])
    if gap < 120:                       # 2 分钟内视为连续对话，不标注
        return ""
    if gap < 3600:
        return f"[距上次对话 {int(gap // 60)} 分钟后] "
    if gap < 3600 * 24:
        return f"[距上次对话 {int(gap // 3600)} 小时后] "
    return f"[距上次对话 {int(gap // 86400)} 天后] "


# ─── LLM 调用 ──────────────────────────────────────────

async def _extract_images(event: PrivateMessageEvent) -> list[str]:
    """把 OneBot 图片段下载并转成 Ollama 接受的 base64。"""
    image_sources = []
    for segment in event.message:
        if segment.type != "image":
            continue
        source = segment.data.get("url") or segment.data.get("file")
        if source:
            image_sources.append(str(source))
        if len(image_sources) >= MAX_IMAGES:
            break

    if not image_sources:
        return []

    encoded = []
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        for source in image_sources:
            try:
                if source.startswith("data:image/") and "," in source:
                    payload = source.split(",", 1)[1]
                    raw = base64.b64decode(payload, validate=True)
                elif source.startswith(("http://", "https://")):
                    response = await client.get(source)
                    response.raise_for_status()
                    raw = response.content
                else:
                    local = Path(source.removeprefix("file:///").replace("/", "\\"))
                    raw = local.read_bytes()
                if not raw or len(raw) > MAX_IMAGE_BYTES:
                    logger.warning(f"忽略大小异常的图片（{len(raw)} bytes）")
                    continue
                encoded.append(base64.b64encode(raw).decode("ascii"))
            except Exception as exc:
                logger.warning(f"读取图片失败: {type(exc).__name__}: {exc}")
    return encoded


def _split_reply(reply: str) -> list[str]:
    """按自然边界拆分长回复，不再把每行粗暴截断到 30 字。"""
    chunks = []
    for paragraph in (part.strip() for part in reply.splitlines() if part.strip()):
        rest = paragraph
        while len(rest) > MAX_REPLY_CHARS:
            window = rest[:MAX_REPLY_CHARS + 1]
            cuts = [window.rfind(mark) + 1 for mark in "。！？!?，," if window.rfind(mark) >= 0]
            cut = max(cuts, default=MAX_REPLY_CHARS)
            chunks.append(rest[:cut].strip())
            rest = rest[cut:].strip()
        if rest:
            chunks.append(rest)
    return [chunk for chunk in chunks if chunk][:6]


async def call_llm(user_id: str, user_text: str, images: list[str] | None = None) -> list[str]:
    """调用 LLM 生成回复，返回一条或多条消息（每行一条）"""
    # 时间感知：间隔要在 _get_conversation 刷新 last_active 之前读取
    gap_prefix = _gap_str(user_id)

    _add_message(user_id, "user", gap_prefix + user_text)
    memory_store.touch_interaction(user_id, user_at=int(time.time()))
    ctx = _get_conversation(user_id)

    # 当前消息常是“然后呢/还记得吗”之类的承接句；带上最近几轮用户话题，
    # 避免只检索这一句而丢掉真正的关键词。
    recent_user_text = [m["content"] for m in ctx[-8:] if m["role"] == "user"]
    memory_query = "\n".join(reversed(recent_user_text[-4:])) or user_text
    query_embedding = None
    if embedding_breaker.available():
        try:
            vectors = await embedding_client.embed_async([
                "为当前私人聊天检索相关的长期记忆：\n" + memory_query
            ])
            query_embedding = vectors[0]
            embedding_breaker.success()
        except Exception as exc:
            embedding_breaker.failure()
            logger.warning(f"向量召回暂不可用，已降级到词面召回: {type(exc).__name__}: {exc}")
    memory = memory_store.search_context(
        memory_query, SELF_QQ, user_id,
        query_embedding=query_embedding,
        embedding_model=EMBEDDING_MODEL,
    )
    memory_prompt = ""
    if memory:
        memory_prompt = ("\n\n以下是你对自己、交谈对象及双方关系形成的内部认知。"
                         "它只用于理解和判断，不要复述、引用或提及记忆系统；"
                         "若与对方当前明确陈述冲突，以当前信息为准：\n" + memory)
    system_prompt = f"{STYLE_PROMPT}{memory_prompt}\n\n当前时间：{_now_str()}"
    messages = [{"role": "system", "content": system_prompt}] + [dict(m) for m in ctx]
    if images and messages and messages[-1]["role"] == "user":
        messages[-1]["images"] = images

    # 使用 Ollama 原生接口，显式关闭 thinking。
    _ollama_root = LLM_API_BASE.rstrip("/")
    if _ollama_root.endswith("/v1"):
        _ollama_root = _ollama_root[: -len("/v1")]
    is_local_ollama = any(host in _ollama_root.lower()
                          for host in ("127.0.0.1", "localhost", "::1"))
    if (not is_local_ollama
            and (not LLM_API_KEY or LLM_API_KEY == "你的魔搭token")):
        logger.error("远程 LLM 未配置 LLM_API_KEY")
        _add_message(user_id, "assistant", "嗯")
        return ["嗯"]

    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY and LLM_API_KEY != "你的魔搭token":
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{_ollama_root}/api/chat",
                headers=headers,
                json={
                    "model": LLM_VISION_MODEL if images else LLM_MODEL,
                    "keep_alive": LLM_KEEP_ALIVE,
                    "messages": messages,
                    "think": LLM_THINK,
                    "stream": False,
                    "options": {
                        "temperature": LLM_TEMPERATURE,
                        "top_p": 0.9,
                        "repeat_penalty": 1.05,
                        "num_predict": 160,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()

        reply = (data.get("message", {}).get("content") or "").strip()
        # 去掉可能残留的思考标签
        if "</think>" in reply:
            reply = reply.split("</think>", 1)[1].strip()
        # 清理可能的格式残留
        reply = reply.replace("**", "").replace("*", "").replace("`", "")
        lines = _split_reply(reply)
        if not lines:
            logger.warning(f"LLM 返回空正文，done_reason={data.get('done_reason')}，请检查模型/思考设置")
            lines = ["嗯"]
        # 上下文里记录完整回复（多条合并为一条 assistant 消息）
        _add_message(user_id, "assistant", "\n".join(lines))
        logger.info(f"LLM 回复 [{user_id}]: {' | '.join(lines)}")
        return lines

    except httpx.HTTPStatusError as e:
        logger.error(f"LLM 调用失败: {e.response.status_code} {e.response.text}")
        _add_message(user_id, "assistant", "嗯")
        return ["嗯"]
    except Exception as e:
        logger.error(f"LLM 调用失败: {e}")
        _add_message(user_id, "assistant", "嗯")
        return ["嗯"]


async def _run_online_extraction(user_id: str) -> None:
    if not MEMORY_AUTO_EXTRACT:
        return
    async with online_extraction_locks[user_id]:
        try:
            count = await online_extractor.maybe_extract(user_id)
            if count:
                logger.info(f"在线记忆抽取 [{user_id}]：写入/更新 {count} 条")
        except Exception as exc:
            logger.warning(f"在线记忆抽取失败 [{user_id}]: {type(exc).__name__}: {exc}")


# ─── 消息处理 ───────────────────────────────────────────

async def _is_private(event: PrivateMessageEvent) -> bool:
    """只处理私聊消息"""
    return True


async def _not_self(event: PrivateMessageEvent) -> bool:
    """过滤自己发的消息"""
    return str(event.user_id) != str(BOT_QQ)


private_chat = on_message(
    rule=Rule(_is_private, _not_self),
    priority=10,
    block=True,
)


@private_chat.handle()
async def handle_private_msg(bot: Bot, event: PrivateMessageEvent):
    """处理私聊消息并回复"""
    user_id = str(event.user_id)
    text = event.get_plaintext().strip()

    # 同一联系人串行处理，避免短时间连发时上下文和回复顺序交叉。
    async with conversation_locks[user_id]:
        images = await _extract_images(event)
        if not text:
            text = "请看看这张图片并自然回复。" if images else "[发送了一条非文字消息]"
        logger.info(f"收到私聊 [{user_id}]: {text}（图片 {len(images)} 张）")
        received_at = int(time.time())
        memory_store.save_runtime_message(
            f"onebot:{event.message_id}", user_id, user_id, received_at, text, 1)

        estimated_chars = max(3, min(len(text) // 2 + random.randint(2, 6), 15))
        typing_delay = random.uniform(REPLY_DELAY_MIN, REPLY_DELAY_MAX) + estimated_chars * random.uniform(0.08, 0.2)
        await asyncio.sleep(typing_delay)

        replies = await call_llm(user_id, text, images)

        for i, reply in enumerate(replies):
            if not reply:
                continue
            if i > 0:
                gap = random.uniform(REPLY_DELAY_MIN, REPLY_DELAY_MAX) + len(reply) * random.uniform(0.08, 0.2)
                await asyncio.sleep(gap)
            await bot.send(event, reply)
            sent_at = int(time.time())
            memory_store.save_runtime_message(
                f"runtime:bot:{uuid.uuid4().hex}", SELF_QQ, user_id, sent_at, reply, 2)
            memory_store.touch_interaction(user_id, bot_at=sent_at)

        task = asyncio.create_task(_run_online_extraction(user_id))
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)
