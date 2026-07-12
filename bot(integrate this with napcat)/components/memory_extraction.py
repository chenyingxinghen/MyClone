from __future__ import annotations

import json


PROMPT = """你在构建一个人的长期认知，而不是总结聊天或模仿措辞。
自我QQ={self_id}，对象QQ={partner_id}。
从对话中只提取有长期价值且有证据的认知：
- self：自我的身份、经历、偏好、价值观、行为及情绪模式
- person：对象的身份、经历、偏好、边界及行为模式
- relationship：双方称呼、关系状态、共同经历、约定、矛盾与相处模式
- episode：值得长期保留的具体事件
区分明确事实和推断；玩笑、反讽、假设、转述不要当作确定事实。不要只描述说话风格。
输出严格 JSON 数组，每项字段：kind, subject_id, object_id, summary, keywords,
confidence(0到1), valid_from(Unix秒或null), valid_to(null)。
subject_id 规则：self={self_id}，person={partner_id}，relationship 和 episode 都必须为
relationship:{self_id}:{partner_id}。不要自行创造人物或事件ID。
置信度必须校准：反复明确确认的稳定事实最高0.95；单次明确陈述0.80-0.90；
从行为推断的模式0.55-0.75；不确定推断不要提取，禁止使用1.0。
日常偏好、习惯、重要近况、相处模式和有后续影响的事件都有长期价值；
有充分证据时通常提取3到{max_items}条，最多只能输出{max_items}条；达到上限立即结束，
不要为了覆盖所有细节继续增加条目。只输出紧凑 JSON，不要 Markdown、解释或思考过程。
没有值得保存的认知才输出 []。

对话：
{dialogue}"""

RETRY_SUFFIX = """

重要：上一次生成的 JSON 不完整。请重新从原对话提取，不要续写旧输出。
最多输出 {max_items} 条，每个 summary 保持简洁，只输出一个完整、可解析的 JSON 数组。"""


def parse_json_array(content: str) -> list:
    content = content.strip()
    if "</think>" in content:
        content = content.split("</think>", 1)[1].strip()
    if "```" in content:
        blocks = content.split("```")
        candidates = [block.removeprefix("json").strip() for block in blocks[1::2]]
        content = next((block for block in candidates if "[" in block and "]" in block), content)
    start, end = content.find("["), content.rfind("]")
    if start < 0 or end < start:
        raise ValueError("模型响应中没有 JSON 数组")
    result = json.loads(content[start:end + 1])
    if not isinstance(result, list):
        raise ValueError("模型输出不是 JSON 数组")
    return result


def salvage_json_array(content: str) -> list:
    """从被截断的数组中保留已经完整生成的顶层对象。"""
    content = content.strip()
    if "</think>" in content:
        content = content.split("</think>", 1)[1].strip()
    start = content.find("[")
    if start < 0:
        return []
    decoder = json.JSONDecoder()
    items = []
    pos = start + 1
    while pos < len(content):
        while pos < len(content) and (content[pos].isspace() or content[pos] == ","):
            pos += 1
        if pos >= len(content) or content[pos] == "]":
            break
        try:
            item, pos = decoder.raw_decode(content, pos)
        except json.JSONDecodeError:
            break
        if isinstance(item, dict):
            items.append(item)
    return items


async def request_extraction(client, *, url: str, headers: dict, prompt: str,
                             model: str, is_ollama: bool,
                             max_output_tokens: int) -> tuple[str, str]:
    if is_ollama:
        payload = {"model": model, "stream": False, "think": False,
                   "messages": [{"role": "user", "content": prompt}],
                   "options": {"temperature": .1, "num_predict": max_output_tokens}}
    else:
        payload = {"model": model, "stream": False,
                   "messages": [{"role": "user", "content": prompt}],
                   "temperature": .1, "max_tokens": max_output_tokens}
    response = await client.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    if is_ollama:
        return (data.get("message", {}).get("content", "").strip(),
                str(data.get("done_reason") or "unknown"))
    choice = data.get("choices", [{}])[0]
    return (choice.get("message", {}).get("content", "").strip(),
            str(choice.get("finish_reason") or "unknown"))
