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


def load_base_model(model_id, bnb_config, local_rank, log):
    """Load the 4-bit base model onto THIS process's GPU.

    Qwen2/Qwen3 dense text models are plain CausalLM. Qwen3.5 (and other VLMs)
    ship as `*ForConditionalGeneration`, which is NOT in the CausalLM auto-map —
    from_pretrained raises ValueError. Fall back to the image-text-to-text class;
    LoRA still targets the q/k/v/o + gate/up/down projections in its text stack.
    """
    common = dict(
        quantization_config=bnb_config,
        # Pin the whole model to THIS process's GPU — required for DDP.
        device_map={"": local_rank},
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None,
                        help="Directory containing sft-my.json")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--eval-ratio", type=float, default=0.05,
                        help="Fraction held out for validation (0 disables eval).")
    parser.add_argument("--eval-steps", type=int, default=50,
                        help="Run validation every N steps (also the save cadence).")
    parser.add_argument("--early-stopping-patience", type=int, default=3,
                        help="Stop after N evals without val-loss improvement.")
    parser.add_argument("--no-assistant-only-loss", action="store_true",
                        help="Train on all tokens instead of masking user/system "
                             "turns. Use this if assistant-only loss errors on the "
                             "model's chat template.")
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
    model = load_base_model(args.model, bnb_config, local_rank, log)
    model.config.use_cache = False

    tokenizer = load_tokenizer(args.model, log)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        lora_dropout=args.lora_dropout,
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

    # Assistant-only loss: mask user/system tokens so the model only learns to
    # imitate the assistant's replies — the whole point of style cloning. Needs
    # a TRL new enough to accept `assistant_only_loss`; for known families
    # (Qwen3) TRL auto-patches the chat template to emit the assistant mask.
    import inspect
    sft_params = inspect.signature(SFTConfig.__init__).parameters
    use_aol = ("assistant_only_loss" in sft_params) and not args.no_assistant_only_loss

    # Pre-render each conversation with the chat template into a "text" column.
    def formatting_func(examples):
        return {"text": [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in examples["messages"]
        ]}

    if use_aol:
        # Keep the conversational `messages` column — SFTTrainer applies the
        # chat template itself and builds the assistant token mask from it.
        # Pre-rendering to text would throw away the role boundaries it needs.
        prepared = dataset
        log("Assistant-only loss: ON (masking user/system tokens)")
    else:
        with state.main_process_first():
            prepared = dataset.map(formatting_func, batched=True,
                                   remove_columns=dataset.column_names)
        log("Assistant-only loss: OFF (training on all tokens)")

    # Hold out a validation split so eval loss can flag overfitting and drive
    # early stopping. Deterministic seed → identical split on every DDP rank.
    eval_dataset = None
    if args.eval_ratio and 0 < args.eval_ratio < 1 and len(prepared) >= 20:
        split = prepared.train_test_split(test_size=args.eval_ratio, seed=3407)
        train_dataset, eval_dataset = split["train"], split["test"]
        log(f"Split: {len(train_dataset)} train / {len(eval_dataset)} eval "
            f"({args.eval_ratio:.0%} held out)")
    else:
        train_dataset = prepared
        log("Eval disabled (eval-ratio=0 or dataset too small)")

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

    # SFTConfig's sequence-length arg was renamed max_seq_length -> max_length
    # in newer TRL (>=0.20). Pass whichever the installed version accepts.
    seq_len_kwarg = "max_seq_length" if "max_seq_length" in sft_params else "max_length"

    extra_cfg = {}
    if use_aol:
        extra_cfg["assistant_only_loss"] = True
    else:
        # Only meaningful when we pre-rendered a "text" column above.
        extra_cfg["dataset_text_field"] = "text"

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
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        dataset_num_proc=1,
        report_to="none",
        **extra_cfg,
        **{seq_len_kwarg: MAX_SEQ_LENGTH},
    )

    trainer = build_trainer(model, tokenizer, train_dataset, eval_dataset,
                            sft_config, callbacks)
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
