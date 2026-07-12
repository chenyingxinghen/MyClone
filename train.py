"""
MyClone - Fine-tune on chat data using QLoRA, data-parallel across 2 GPUs (DDP).

Stack: transformers + peft + bitsandbytes + trl, launched under accelerate.
Each process loads the full 4-bit model onto ITS OWN GPU and trains on a shard
of the data (DistributedDataParallel). This is real acceleration — unlike
device_map="auto", which only splits one model across cards and runs serially.

Kaggle quickstart (2x T4):
  !pip install -r requirements.txt
  !git clone <your-repo> MyClone && cd MyClone
  # Use torchrun, NOT `accelerate launch` — on Kaggle the accelerate CLI imports
  # timm/torchvision at startup and dies on version skew before training runs.
  # train.py reads torchrun's env via accelerate.PartialState() all the same.
  !torchrun --nproc_per_node=2 train.py --dataset /kaggle/input/YOUR_DATASET/sft

Big model that won't fit one 16GB T4 (e.g. Qwen3.5-9B) — shard across both cards:
  # ONE process, model-parallel (device_map="auto"). Do NOT use torchrun here.
  !python train.py --model-parallel --model Qwen/Qwen3.5-9B \
      --dataset /kaggle/input/YOUR_DATASET/sft
  # Fits by pooling ~32GB, but GPUs run serially (no throughput speedup).

Single GPU (falls back automatically):
  !python train.py --dataset /kaggle/input/YOUR_DATASET/sft
"""

import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# Reduce fragmentation OOM on 16GB T4/P100 (the error message suggests this).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from accelerate import PartialState
from datasets import Dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

DEFAULT_MAX_SEQ_LENGTH = 2048


def find_dataset_dir():
    for local in (Path("./data/dataset/sft"), Path("./dataset/sft")):
        if (local / "sft-my.json").exists():
            return str(local)

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for d in kaggle_input.iterdir():
            if (d / "sft" / "sft-my.json").exists():
                return str(d / "sft")
            if (d / "sft-my.json").exists():
                return str(d)

    return "./data/dataset/sft"


def load_sft_dataset(dataset_dir):
    data_file = os.path.join(dataset_dir, "sft-my.json")
    if not os.path.exists(data_file):
        print(f"ERROR: {data_file} not found")
        print("Use --dataset to specify the directory containing sft-my.json")
        sys.exit(1)

    with open(data_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    conversations = []
    for item in raw:
        msgs = []
        if item.get("system"):
            msgs.append({"role": "system", "content": item["system"]})
        msgs.extend(item["messages"])
        conversations.append({
            "messages": msgs,
            "time": item.get("time"),
            "source": item.get("source", ""),
            "id": item.get("id", ""),
        })

    return Dataset.from_list(conversations)


def audit_dataset(dataset, allow_leaky_data=False):
    """Fail fast on structures that make the training loss misleading."""
    invalid = 0
    answer_leaks = 0
    assistant_turns = 0
    format_artifacts = 0
    copied_user_replies = 0
    for row in dataset:
        msgs = row.get("messages", [])
        start = 1 if msgs and msgs[0].get("role") == "system" else 0
        body = msgs[start:]
        if (not body or len(body) % 2
                or any(m.get("role") != ("user" if i % 2 == 0 else "assistant")
                       for i, m in enumerate(body))):
            invalid += 1
        for msg in body:
            if msg.get("role") == "assistant":
                assistant_turns += 1
            if msg.get("role") == "user" and "<begin_chat>你应该说：" in msg.get("content", ""):
                answer_leaks += 1
        for i in range(0, len(body) - 1, 2):
            user = body[i].get("content", "").strip()
            reply = strip_thinking(body[i + 1].get("content", ""))
            format_artifacts += len(re.findall(r"\[[^\]\n]{1,240}\]", user))
            format_artifacts += len(re.findall(r"\[[^\]\n]{1,240}\]", reply))
            format_artifacts += sum(
                line.strip().startswith("@") for line in reply.splitlines()
            )
            copied_user_replies += bool(len(user) >= 4 and user in reply)

    if invalid:
        raise ValueError(f"Dataset has {invalid} conversations with invalid role alternation.")
    if answer_leaks and not allow_leaky_data:
        raise ValueError(
            f"Dataset has {answer_leaks} user turns containing the target answer "
            "(<begin_chat>你应该说：...). Re-run process_data.py with the current "
            "config, or explicitly pass --allow-leaky-data to reproduce an old run."
        )
    if format_artifacts:
        raise ValueError(
            f"Dataset has {format_artifacts} exporter artifacts such as [引用], "
            "[图片] or standalone @ mentions. Re-run process_data.py."
        )
    if copied_user_replies:
        raise ValueError(
            f"Dataset has {copied_user_replies} assistant replies that copy the full "
            "user message. Re-run process_data.py."
        )
    return {"conversations": len(dataset), "assistant_turns": assistant_turns,
            "answer_leaks": answer_leaks, "format_artifacts": format_artifacts,
            "copied_user_replies": copied_user_replies}


def split_dataset(dataset, eval_ratio, strategy):
    if not eval_ratio or not 0 < eval_ratio < 1 or len(dataset) < 20:
        return dataset, None

    if strategy == "chronological" and "time" in dataset.column_names:
        # Hold out each contact/source's newest conversations. A single global
        # cutoff can accidentally put only the most recently active contact in
        # eval, which says little about generalization across relationships.
        groups = defaultdict(list)
        has_source = "source" in dataset.column_names
        for index in range(len(dataset)):
            source = dataset[index].get("source", "") if has_source else "all"
            groups[source or "unknown"].append(index)

        train_indices, eval_indices = [], []
        for indices in groups.values():
            ordered = sorted(indices, key=lambda i: str(dataset[i].get("time") or ""))
            if len(ordered) < 2:
                train_indices.extend(ordered)
                continue
            count = max(1, int(round(len(ordered) * eval_ratio)))
            count = min(count, len(ordered) - 1)
            train_indices.extend(ordered[:-count])
            eval_indices.extend(ordered[-count:])
        if train_indices and eval_indices:
            return dataset.select(train_indices), dataset.select(eval_indices)

    split = dataset.train_test_split(test_size=eval_ratio, seed=3407)
    return split["train"], split["test"]


def get_dataset_system_prompt(dataset):
    if not len(dataset):
        return None
    for msg in dataset[0].get("messages", []):
        if msg.get("role") == "system" and msg.get("content"):
            return msg["content"]
    return None


_THINKING_KW_WARNED = False


def render_chat(tokenizer, messages, add_generation_prompt):
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        # Older/other templates don't accept enable_thinking. Warn once so the
        # revert isn't silent — think-suppression then relies solely on the empty
        # <think></think> blocks baked into the training data by process_data.py.
        global _THINKING_KW_WARNED
        if not _THINKING_KW_WARNED:
            print("WARNING: apply_chat_template does not accept enable_thinking; "
                  "relying on trained empty-think blocks to keep thinking OFF.")
            _THINKING_KW_WARNED = True
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def tokenize_chat(tokenizer, messages):
    """Tokenize one complete conversation with thinking disabled when supported."""
    kwargs = dict(tokenize=True, add_generation_prompt=False)
    try:
        encoded = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        encoded = tokenizer.apply_chat_template(messages, **kwargs)

    # transformers 5.x may return BatchEncoding instead of a bare list even
    # without return_tensors. Iterating BatchEncoding yields field names such as
    # "input_ids", which previously made every full/empty sequence look equal.
    if hasattr(encoded, "get"):
        encoded = encoded.get("input_ids")
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if (isinstance(encoded, list) and len(encoded) == 1
            and isinstance(encoded[0], list)):
        encoded = encoded[0]
    if not isinstance(encoded, list) or not all(isinstance(token, int) for token in encoded):
        raise TypeError(
            "apply_chat_template returned unsupported token data: "
            f"{type(encoded).__name__}"
        )
    return encoded


def strip_thinking(reply):
    # Peel off any leading think blocks (a prefilled empty one plus, defensively,
    # one the model may still emit) so only the final reply is shown.
    text = reply.strip()
    while text.startswith("<think>") and "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text


def generate_test(model, tokenizer, prompts, system_prompt=None, max_new_tokens=64,
                  temperature=0.2):
    model.eval()
    for prompt in prompts:
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": prompt})
        text = render_chat(tokenizer, msgs, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        torch.manual_seed(3407)
        generate_args = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.eos_token_id,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generate_args.update(temperature=temperature, top_p=0.9)
        with torch.no_grad():
            outputs = model.generate(**inputs, **generate_args)
        reply = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        reply = strip_thinking(reply)
        print(f"\nUser: {prompt}")
        print(f"Model ({reply.count(chr(10)) + 1} message line(s)): {reply}")


def expand_prompt_completion(dataset):
    """Turn each assistant reply into an explicit masked completion example."""
    examples = []
    for row_index, row in enumerate(dataset):
        messages = row.get("messages", [])
        system = messages[:1] if messages and messages[0].get("role") == "system" else []
        body = messages[len(system):]
        history = list(system)
        for pair_index in range(0, len(body) - 1, 2):
            user, assistant = body[pair_index:pair_index + 2]
            examples.append({
                "prompt": history + [user],
                "completion": [assistant],
                "time": row.get("time"),
                "source": row.get("source", ""),
                "id": f"{row.get('id', row_index)}:completion:{pair_index}",
            })
            history.extend([user, assistant])
    return Dataset.from_list(examples)


def oversample_multiline_completions(dataset, extra_copies, log):
    """Duplicate only clean multiline completions in the training split."""
    if extra_copies <= 0:
        return dataset, 0

    focused = []
    for row_index, row in enumerate(dataset):
        completion = row.get("completion", [])
        reply = strip_thinking(completion[0].get("content", "")) if completion else ""
        if "\n" not in reply:
            continue
        for copy_index in range(extra_copies):
            clone = dict(row)
            clone["id"] = f"{row.get('id', row_index)}:multiline:{copy_index}"
            focused.append(clone)

    if not focused:
        log("Multiline oversampling requested, but no multiline completions were found.")
        return dataset, 0
    augmented = concatenate_datasets([dataset, Dataset.from_list(focused)])
    log(f"Multiline oversampling: added {len(focused)} clean completion examples "
        f"({extra_copies} extra copies per multiline completion).")
    return augmented, len(focused)


def cap_ultrashort_completions(dataset, max_share, log):
    """Limit single-line 1-2 character completions in the training split."""
    if max_share >= 1:
        return dataset, 0
    max_share = max(0.0, max_share)
    ultrashort, other = [], []
    for index, row in enumerate(dataset):
        completion = row.get("completion", [])
        reply = strip_thinking(completion[0].get("content", "")) if completion else ""
        target = ultrashort if 0 < len(reply) <= 2 and "\n" not in reply else other
        target.append(index)
    if not ultrashort:
        return dataset, 0

    allowed = int((max_share / max(1e-9, 1 - max_share)) * len(other))
    allowed = min(len(ultrashort), allowed)
    ranked = sorted(
        ultrashort,
        key=lambda i: hashlib.sha256(
            str(dataset[i].get("id", i)).encode("utf-8")
        ).hexdigest(),
    )
    keep = sorted(other + ranked[:allowed])
    removed = len(ultrashort) - allowed
    filtered = dataset.select(keep)
    log(f"Ultra-short balancing: kept {allowed}/{len(ultrashort)} single-line "
        f"1-2 character completions; removed {removed} from train only.")
    return filtered, removed


def tokenize_completion_examples(dataset, tokenizer, max_length, completion_only, log):
    """Build labels from one full-template tokenization per example.

    TRL normally tokenizes a conversational prompt with
    ``add_generation_prompt=True`` and then tokenizes prompt+completion again.
    Qwen3.5's template can add thinking/special tokens only in the former, so
    those sequences are not prefixes and TRL's completion mask is shifted.  We
    instead compare a full answer with the same conversation containing an
    empty assistant answer.  Both take the exact same template path.
    """
    tokenized = []
    boundary_fallbacks = 0
    total = len(dataset)
    log(f"Building explicit token labels for {total} examples...")
    for row_index, row in enumerate(dataset, 1):
        prompt = list(row.get("prompt", []))
        completion = list(row.get("completion", []))
        if not completion:
            continue
        full_ids = list(tokenize_chat(tokenizer, prompt + completion))
        empty_answer = dict(completion[0])
        empty_answer["content"] = ""
        empty_ids = list(tokenize_chat(tokenizer, prompt + [empty_answer]))

        boundary = 0
        for full_token, empty_token in zip(full_ids, empty_ids):
            if full_token != empty_token:
                break
            boundary += 1
        if boundary >= len(full_ids):
            boundary_fallbacks += 1
            continue

        labels = list(full_ids)
        if completion_only:
            labels[:boundary] = [-100] * boundary

        # Keep the end so the assistant target is never truncated away by long
        # history. The prompt's most recent context is retained as well.
        if max_length and len(full_ids) > max_length:
            full_ids = full_ids[-max_length:]
            labels = labels[-max_length:]
        if completion_only and all(label == -100 for label in labels):
            boundary_fallbacks += 1
            continue
        tokenized.append({"input_ids": full_ids, "labels": labels})
        if row_index % 500 == 0 or row_index == total:
            log(f"Explicit token labels progress: {row_index}/{total}")

    if boundary_fallbacks:
        raise RuntimeError(
            f"Could not derive a non-empty assistant token span for "
            f"{boundary_fallbacks} examples; refusing ambiguous loss masks."
        )
    log(f"Explicit token labels: built {len(tokenized)} examples with "
        f"{'prompt tokens masked' if completion_only else 'all tokens trainable'}; "
        "TRL prompt-prefix inference bypassed.")
    return Dataset.from_list(tokenized)


def load_tokenizer(model_id, log):
    """Return a plain text tokenizer.

    Qwen3.5 and other VLMs may resolve to a multimodal *processor*, which wraps
    the tokenizer. Passing that straight to SFTTrainer makes apply_chat_template
    treat string content as vision dicts and crash, so unwrap to `.tokenizer`.
    """
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if not hasattr(tok, "apply_chat_template") and hasattr(tok, "tokenizer"):
        log("Loaded a multimodal processor; using its inner text tokenizer")
        return tok.tokenizer
    return tok


def load_base_model(model_id, bnb_config, device_map, log):
    """Load the 4-bit base model.

    Qwen2/Qwen3 dense text models are plain CausalLM. Qwen3.5 (and other VLMs)
    ship as `*ForConditionalGeneration`, which is NOT in the CausalLM auto-map —
    from_pretrained raises ValueError. Fall back to the image-text-to-text class;
    LoRA still targets the q/k/v/o + gate/up/down projections in its text stack.

    device_map is either {"": local_rank} (DDP: full model per GPU) or "auto"
    (model-parallel: one model sharded across all GPUs, for models too big for
    a single card).
    """
    common = dict(
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **common)
    except (ValueError, KeyError) as e:
        log(f"CausalLM load failed ({type(e).__name__}: {e}); "
            f"retrying with AutoModelForImageTextToText (VLM path)")
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(model_id, **common)


def build_trainer(model, tokenizer, train_dataset, eval_dataset, sft_config, callbacks):
    """SFTTrainer's tokenizer kwarg was renamed to processing_class; support both."""
    common = dict(model=model, train_dataset=train_dataset,
                  eval_dataset=eval_dataset, args=sft_config, callbacks=callbacks)
    try:
        return SFTTrainer(processing_class=tokenizer, **common)
    except TypeError:
        return SFTTrainer(tokenizer=tokenizer, **common)


def cast_trainable_params_to_fp32(model, log):
    """Keep GradScaler away from bf16/fp16 LoRA gradients.

    Some model/PEFT combinations initialize adapter weights in the base model's
    dtype (often bf16). fp16 training uses GradScaler, whose CUDA unscale kernel
    does not accept bf16 gradients. Casting only trainable adapter params to fp32
    is the standard QLoRA stability path and leaves the frozen 4-bit base alone.
    """
    counts = {}
    casted = 0
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        counts[str(param.dtype)] = counts.get(str(param.dtype), 0) + param.numel()
        if param.dtype in (torch.float16, torch.bfloat16):
            param.data = param.data.float()
            casted += param.numel()
    if casted:
        log(f"Casted {casted:,} trainable adapter parameters to fp32 "
            f"to avoid AMP unscale dtype errors. Original trainable dtypes: {counts}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None,
                        help="Directory containing sft-my.json")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=None,
                        help="LoRA alpha. Defaults to rank/2 so style tuning "
                             "does not overpower the base model.")
    parser.add_argument("--lora-dropout", type=float, default=0.08)
    parser.add_argument("--eval-ratio", type=float, default=0.08,
                        help="Fraction held out for validation (0 disables eval).")
    parser.add_argument("--eval-strategy", choices=("chronological", "random"),
                        default="chronological",
                        help="Use newest conversations for eval by default; this "
                             "better measures future-chat generalization than a "
                             "random near-duplicate split.")
    parser.add_argument("--eval-steps", type=int, default=50,
                        help="Run validation every N steps (also the save cadence).")
    parser.add_argument("--early-stopping-patience", type=int, default=3,
                        help="Stop after N evals without val-loss improvement.")
    parser.add_argument("--no-assistant-only-loss", action="store_true",
                        help="Disable completion-only masking and train prompt tokens too. "
                             "Unsafe for chat cloning; retained only for debugging.")
    parser.add_argument("--allow-leaky-data", action="store_true",
                        help="Allow legacy samples that contain their target answer "
                             "inside the user turn. Not recommended.")
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--multiline-oversample", type=int, default=1,
                        help="Extra focused training copies per assistant reply "
                             "containing message-boundary newlines (0 disables).")
    parser.add_argument("--max-ultrashort-share", type=float, default=0.20,
                        help="Maximum share of single-line 1-2 character completion "
                             "examples before multiline oversampling (0-1).")
    parser.add_argument("--model-parallel", action="store_true",
                        help="Shard ONE model across all visible GPUs "
                             "(device_map='auto') instead of DDP. Use for models "
                             "too big for a single 16GB T4 (e.g. Qwen3.5-9B). "
                             "Slower (GPUs run serially) but fits. Run with plain "
                             "`python train.py` — do NOT use torchrun/accelerate "
                             "launch, which would spawn conflicting processes.")
    parser.add_argument("--system-prompt", type=str, default=None,
                        help="System prompt used for before/after test generation. "
                             "Defaults to the first dataset system prompt.")
    parser.add_argument("--test-max-new-tokens", type=int, default=64,
                        help="Max new tokens for before/after test generations.")
    parser.add_argument("--test-temperature", type=float, default=0.2,
                        help="Stable sampling temperature for before/after tests; "
                             "0 uses greedy decoding.")
    parser.add_argument("--test-prompts", type=str, nargs="*", default=[
        "在干嘛",
        "今天累死了",
        "周末有空吗",
        "我刚下课，又饿又困，外面还下雨了",
        "我其实没生气，就是刚才语气有点急",
    ], help="Test prompts for before/after comparison")
    args = parser.parse_args()

    # DDP context. Under `accelerate launch` this reflects each process; run
    # plainly (single process) and it degrades to one GPU.
    state = PartialState()
    is_main = state.is_main_process
    local_rank = state.local_process_index

    def log(msg):
        if is_main:
            print(msg)

    dataset_dir = args.dataset or find_dataset_dir()
    is_kaggle = os.path.exists("/kaggle")
    output_dir = args.output or ("/kaggle/working/output" if is_kaggle else "./output")

    dataset = load_sft_dataset(dataset_dir)
    audit = audit_dataset(dataset, args.allow_leaky_data)
    test_system_prompt = args.system_prompt or get_dataset_system_prompt(dataset)
    log(f"Loaded {len(dataset)} conversations from {dataset_dir}")
    log(f"Dataset audit: {audit['assistant_turns']} assistant turns, "
        f"answer leaks={audit['answer_leaks']}")
    log(f"World size: {state.num_processes} GPU(s)")
    if test_system_prompt:
        log(f"Test system prompt: {test_system_prompt}")

    # Two ways to use 2x T4:
    #   DDP (default)           — device_map={"": rank}: a FULL model per GPU.
    #                             Real throughput scaling, but each card must fit
    #                             the whole model. 9B in 4-bit + activations OOMs
    #                             a single 16GB T4.
    #   Model-parallel (--model-parallel) — device_map="auto": ONE model sharded
    #                             across both cards (~32GB pooled). Fits 9B, but
    #                             runs serially (no speedup). Single process only.
    if args.model_parallel:
        if state.num_processes > 1:
            print("ERROR: --model-parallel needs a single process, but "
                  f"{state.num_processes} were launched. Run `python train.py "
                  "--model-parallel ...` directly — not under torchrun/accelerate "
                  "launch (each process would grab all GPUs and OOM).")
            sys.exit(1)
        device_map = "auto"
        log("Model-parallel: sharding one model across all visible GPUs "
            "(serial, no throughput speedup, but fits big models).")
    else:
        device_map = {"": local_rank}

    # 4-bit NF4 quantization. T4 (Turing) has no bf16 → fp16 compute.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    log(f"Loading model: {args.model} (4-bit QLoRA, device_map={device_map})")
    model = load_base_model(args.model, bnb_config, device_map, log)
    model.config.use_cache = False

    tokenizer = load_tokenizer(args.model, log)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)
    lora_alpha = args.lora_alpha or max(1, args.lora_rank // 2)
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, peft_config)
    vision_trainable = [
        name for name, param in model.named_parameters()
        if param.requires_grad and ("vision" in name.lower() or "visual" in name.lower())
    ]
    if vision_trainable:
        raise RuntimeError(
            "LoRA unexpectedly attached to vision modules. Keep the visual tower "
            f"frozen for text-only style tuning. Examples: {vision_trainable[:5]}"
        )
    cast_trainable_params_to_fp32(model, log)
    if is_main:
        model.print_trainable_parameters()

    # We construct explicit labels below instead of relying on either Jinja
    # generation masks or TRL's prompt-prefix inference.
    import inspect
    sft_params = inspect.signature(SFTConfig.__init__).parameters
    use_completion_loss = not args.no_assistant_only_loss

    raw_train, raw_eval = split_dataset(dataset, args.eval_ratio, args.eval_strategy)
    train_conversation_count = len(raw_train)
    eval_conversation_count = len(raw_eval) if raw_eval is not None else 0
    train_completions = expand_prompt_completion(raw_train)
    eval_completions = expand_prompt_completion(raw_eval) if raw_eval is not None else None
    train_completion_count = len(train_completions)
    train_completions, ultrashort_examples_removed = cap_ultrashort_completions(
        train_completions, args.max_ultrashort_share, log)
    train_completion_count_after_short_balance = len(train_completions)
    train_completions, multiline_examples_added = oversample_multiline_completions(
        train_completions, args.multiline_oversample, log)
    train_dataset = tokenize_completion_examples(
        train_completions, tokenizer, args.max_seq_length, use_completion_loss, log)
    eval_dataset = (tokenize_completion_examples(
        eval_completions, tokenizer, args.max_seq_length, use_completion_loss, log)
        if eval_completions is not None else None)
    log("Completion-only loss: " +
        ("ON (explicit labels; prompt/user/system tokens masked)"
         if use_completion_loss else "OFF - DEBUG ONLY"))
    if eval_dataset is not None:
        eval_sources = len(set(raw_eval["source"])) if "source" in raw_eval.column_names else 0
        log(f"Split: {len(train_dataset)} train / {len(eval_dataset)} eval "
            f"({args.eval_ratio:.0%}, {args.eval_strategy}, "
            f"sources={eval_sources or 'n/a'})")
    else:
        log("Eval disabled (eval-ratio=0 or dataset too small)")

    # Before-training sample (main process only, to avoid duplicate output).
    if is_main:
        print("\n" + "=" * 60)
        print("BEFORE TRAINING - Test generations")
        print("=" * 60)
        generate_test(model, tokenizer, args.test_prompts,
                      test_system_prompt, args.test_max_new_tokens,
                      args.test_temperature)

    log(f"\nTraining: {args.epochs} epochs, per-device bs={args.batch_size}x{args.grad_accum}, "
        f"lr={args.lr}, rank={args.lora_rank}, alpha={lora_alpha}")
    log(f"Effective batch = {args.batch_size} x {args.grad_accum} x {state.num_processes} "
        f"= {args.batch_size * args.grad_accum * state.num_processes}")
    log(f"Output: {output_dir}\n")

    # SFTConfig's sequence-length arg was renamed max_seq_length -> max_length
    # in newer TRL (>=0.20). Pass whichever the installed version accepts.
    seq_len_kwarg = "max_seq_length" if "max_seq_length" in sft_params else "max_length"

    extra_cfg = {}
    # Labels are already explicit. Disable TRL's prompt/completion inference so
    # Qwen3.5 cannot produce shifted masks or prefix-mismatch warning floods.
    if "completion_only_loss" in sft_params:
        extra_cfg["completion_only_loss"] = False

    callbacks = []
    if eval_dataset is not None:
        # Evaluate on a schedule and keep the checkpoint with the lowest val
        # loss — the overfitting guardrail the previous run lacked. save/eval
        # strategy + steps must match for load_best_model_at_end.
        eval_every = max(10, args.eval_steps)
        extra_cfg.update(
            eval_strategy="steps",
            eval_steps=eval_every,
            save_strategy="steps",
            save_steps=eval_every,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            per_device_eval_batch_size=args.batch_size,
        )
        if args.early_stopping_patience > 0:
            from transformers import EarlyStoppingCallback
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience))
    else:
        extra_cfg.update(save_strategy="steps", save_steps=100)

    sft_config = SFTConfig(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        weight_decay=0.1,
        logging_steps=10,
        save_total_limit=2,
        output_dir=output_dir,
        optim="paged_adamw_8bit",
        seed=3407,
        # The model and bitsandbytes compute already run in fp16. Keep Trainer
        # AMP off so Accelerate does not create a GradScaler; some Kaggle
        # torch/PEFT stacks surface bf16 adapter grads, and GradScaler cannot
        # unscale bf16 CUDA gradients.
        fp16=False,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        dataset_num_proc=1,
        report_to="none",
        **extra_cfg,
        **{seq_len_kwarg: args.max_seq_length},
    )

    trainer = build_trainer(model, tokenizer, train_dataset, eval_dataset,
                            sft_config, callbacks)
    trainer.train()

    # Save + after-training sample on main process only (DDP-safe).
    if is_main:
        print("\n" + "=" * 60)
        print("AFTER TRAINING - Test generations")
        print("=" * 60)
        generate_test(model, tokenizer, args.test_prompts,
                      test_system_prompt, args.test_max_new_tokens,
                      args.test_temperature)

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        with open(os.path.join(output_dir, "training_manifest.json"), "w", encoding="utf-8") as f:
            json.dump({
                "base_model": args.model,
                "text_only_lora": True,
                "vision_modules_trainable": False,
                "dataset": dataset_dir,
                "dataset_audit": audit,
                "eval_strategy": args.eval_strategy,
                "hyperparameters": {
                    "epochs": args.epochs,
                    "learning_rate": args.lr,
                    "lora_rank": args.lora_rank,
                    "lora_alpha": lora_alpha,
                    "lora_dropout": args.lora_dropout,
                    "max_seq_length": args.max_seq_length,
                    "multiline_oversample": args.multiline_oversample,
                    "max_ultrashort_share": args.max_ultrashort_share,
                },
                "training_rows": {
                    "train_conversations": train_conversation_count,
                    "eval_conversations": eval_conversation_count,
                    "completion_examples_before_multiline_oversample": train_completion_count,
                    "ultrashort_examples_removed": ultrashort_examples_removed,
                    "completion_examples_after_ultrashort_balance":
                        train_completion_count_after_short_balance,
                    "focused_multiline_examples_added": multiline_examples_added,
                    "completion_examples_after_multiline_oversample": len(train_completions),
                    "eval_completion_examples": len(eval_completions)
                    if eval_completions is not None else 0,
                },
                "loss_mode": "completion_only" if use_completion_loss else "all_tokens",
                "deployment_note": (
                    "For multimodal Ollama deployment, convert only the LoRA adapter "
                    "to GGUF and attach it to the original vision-capable base model. "
                    "Do not merge the full VLM into a text-only GGUF."
                ),
            }, f, ensure_ascii=False, indent=2)
        print(f"\nDone! LoRA adapter saved to: {output_dir}")
        if is_kaggle:
            print("Download from the notebook's Output tab.")


if __name__ == "__main__":
    main()
