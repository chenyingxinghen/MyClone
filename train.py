"""
MyClone - Fine-tune on chat data using QLoRA, data-parallel across 2 GPUs (DDP).

Stack: transformers + peft + bitsandbytes + trl, launched under accelerate.
Each process loads the full 4-bit model onto ITS OWN GPU and trains on a shard
of the data (DistributedDataParallel). This is real acceleration — unlike
device_map="auto", which only splits one model across cards and runs serially.

Kaggle quickstart (2x T4):
  !pip install -r requirements.txt
  !git clone <your-repo> MyClone && cd MyClone
  !accelerate launch --multi_gpu --num_processes 2 train.py \
      --dataset /kaggle/input/YOUR_DATASET/sft

Single GPU (falls back automatically):
  !python train.py --dataset /kaggle/input/YOUR_DATASET/sft
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Reduce fragmentation OOM on 16GB T4/P100 (the error message suggests this).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from accelerate import PartialState
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MAX_SEQ_LENGTH = 2048


def find_dataset_dir():
    local = Path("./dataset/sft")
    if (local / "sft-my.json").exists():
        return str(local)

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for d in kaggle_input.iterdir():
            if (d / "sft" / "sft-my.json").exists():
                return str(d / "sft")
            if (d / "sft-my.json").exists():
                return str(d)

    return str(local)


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
        conversations.append({"messages": msgs})

    return Dataset.from_list(conversations)


def generate_test(model, tokenizer, prompts):
    model.eval()
    for prompt in prompts:
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        reply = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"\nUser: {prompt}")
        print(f"Model: {reply.strip()}")


def build_trainer(model, tokenizer, dataset, sft_config):
    """SFTTrainer's tokenizer kwarg was renamed to processing_class; support both."""
    try:
        return SFTTrainer(model=model, train_dataset=dataset,
                          processing_class=tokenizer, args=sft_config)
    except TypeError:
        return SFTTrainer(model=model, train_dataset=dataset,
                          tokenizer=tokenizer, args=sft_config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None,
                        help="Directory containing sft-my.json")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--test-prompts", type=str, nargs="*", default=[
        "你平时喜欢做什么？",
        "今天心情怎么样？",
        "帮我推荐一部电影吧",
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
    log(f"Loaded {len(dataset)} conversations from {dataset_dir}")
    log(f"World size: {state.num_processes} GPU(s)")

    # 4-bit NF4 quantization. T4 (Turing) has no bf16 → fp16 compute.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    log(f"Loading model: {args.model} (4-bit QLoRA on GPU {local_rank})")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        # Pin the whole model to THIS process's GPU — required for DDP.
        device_map={"": local_rank},
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, peft_config)
    if is_main:
        model.print_trainable_parameters()

    # Pre-render each conversation with the chat template into a "text" column.
    def formatting_func(examples):
        return {"text": [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in examples["messages"]
        ]}

    with state.main_process_first():
        dataset = dataset.map(formatting_func, batched=True,
                              remove_columns=dataset.column_names)

    # Before-training sample (main process only, to avoid duplicate output).
    if is_main:
        print("\n" + "=" * 60)
        print("BEFORE TRAINING - Test generations")
        print("=" * 60)
        generate_test(model, tokenizer, args.test_prompts)

    log(f"\nTraining: {args.epochs} epochs, per-device bs={args.batch_size}x{args.grad_accum}, "
        f"lr={args.lr}, rank={args.lora_rank}")
    log(f"Effective batch = {args.batch_size} x {args.grad_accum} x {state.num_processes} "
        f"= {args.batch_size * args.grad_accum * state.num_processes}")
    log(f"Output: {output_dir}\n")

    sft_config = SFTConfig(
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        weight_decay=0.1,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        output_dir=output_dir,
        optim="paged_adamw_8bit",
        seed=3407,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        dataset_num_proc=1,
        report_to="none",
    )

    trainer = build_trainer(model, tokenizer, dataset, sft_config)
    trainer.train()

    # Save + after-training sample on main process only (DDP-safe).
    if is_main:
        print("\n" + "=" * 60)
        print("AFTER TRAINING - Test generations")
        print("=" * 60)
        generate_test(model, tokenizer, args.test_prompts)

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"\nDone! LoRA adapter saved to: {output_dir}")
        if is_kaggle:
            print("Download from the notebook's Output tab.")


if __name__ == "__main__":
    main()
